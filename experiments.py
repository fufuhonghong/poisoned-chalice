"""Gradient construction, measured-norm matching, and honest dose-response plots."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import torch
import torch.nn.functional as F
import pandas as pd
from benchmark import CONSTRUCTION, EVALUATION
from config import DOSES, SYSTEM, RESULTS, MODEL_ID, MODEL_REVISION
from evaluate import evaluate, accuracy, category_scores
from poison import Intervention, squared_norm


def task_tensors(runtime, task):
    prefix = runtime.tokenizer.apply_chat_template([
        {'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': task.prompt},
    ], tokenize=True, add_generation_prompt=True)
    target = runtime.tokenizer.encode(task.answer, add_special_tokens=False) + [runtime.tokenizer.eos_token_id]
    cap = 96 if runtime.low_memory else 160
    if len(prefix) + len(target) > cap:
        raise ValueError(f'Construction task {task.id} exceeds {cap} tokens; shorten it rather than truncate labels.')
    ids = torch.tensor([prefix + target], device=runtime.device)
    labels = torch.tensor(target, device=runtime.device)
    return ids, labels, len(prefix)


def task_loss(runtime, task):
    ids, labels, prefix_length = task_tensors(runtime, task)
    logits = runtime.model(input_ids=ids, use_cache=False).logits
    # Each target token is predicted by the preceding position. Prompt tokens are masked.
    target_logits = logits[0, prefix_length - 1:-1, :].float()
    return F.cross_entropy(target_logits, labels, reduction='sum'), labels.numel()


def gradient_direction(runtime, layers, progress=None):
    """Mean target-token NLL gradient at pristine weights, accumulated one example at a time."""
    with runtime.lock, torch.enable_grad():
        runtime.load()
        indices = tuple(sorted(set(layers)))
        runtime.poisoner.validate(Intervention(layers=indices))
        if runtime.low_memory and len(indices) > 4:
            raise ValueError('Low-memory mode supports at most 4 gradient target layers per run.')
        runtime.poisoner.restore()
        if runtime.gradient_cache is not None and runtime.gradient_cache[0] == indices:
            return runtime.gradient_cache[1], runtime.gradient_cache[2]
        runtime.gradient_cache = None
        params = [runtime.poisoner.weight(i) for i in indices]
        accumulated = {i: torch.zeros(p.shape, dtype=torch.float32, device='cpu') for i, p in zip(indices, params)}
        loss_sum, token_count = 0., 0
        try:
            for param in params:
                param.requires_grad_(True)
            for index, task in enumerate(CONSTRUCTION):
                if progress:
                    progress(index / len(CONSTRUCTION), desc=f'Gradient construction {index + 1}/{len(CONSTRUCTION)}')
                loss, tokens = task_loss(runtime, task)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite construction loss.')
                grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
                loss_sum += loss.detach().item()
                token_count += tokens
                for layer, grad in zip(indices, grads):
                    accumulated[layer].add_(grad.detach().to('cpu', dtype=torch.float32))
                del loss, grads, grad
            for direction in accumulated.values():
                direction.div_(token_count)
            norm = math.sqrt(sum(squared_norm(g) for g in accumulated.values()))
            if not math.isfinite(norm) or norm == 0:
                raise RuntimeError('Task gradient is zero or nonfinite.')
            metadata = dict(construction_loss=loss_sum / token_count, gradient_l2=norm,
                            construction_examples=len(CONSTRUCTION), target_tokens=token_count)
            runtime.gradient_cache = (indices, accumulated, metadata)
            return accumulated, metadata
        finally:
            for param in params:
                param.requires_grad_(False)
                param.grad = None
            runtime.cleanup()


def apply_intervention(runtime, spec, progress=None, target_l2=None):
    runtime.load()
    runtime.poisoner.validate(spec)
    directions = metadata = None
    if spec.method == 'gradient' and spec.dose > 0:
        directions, metadata = gradient_direction(runtime, spec.layers, progress)
    metrics = runtime.poisoner.apply(spec, directions=directions, target_l2=target_l2)
    if metadata:
        metrics.update(metadata)
    return metrics


def random_layers(layer_count, count, seed):
    if not 1 <= count <= layer_count:
        raise ValueError(f'Layer count must be between 1 and {layer_count}.')
    return tuple(sorted(random.Random(seed).sample(range(layer_count), count)))


def plot_results(frame, output_dir):
    # Contract: three quantitative panels show observed scores, paired degradation,
    # and direct method comparison on the same tensors. No smoothing or fabricated CI.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = {'gradient': ('#b8324c', 'o', 'Gradient-guided'), 'noise': ('#387b99', 's', 'Random Gaussian')}
    budget = frame[frame.method.isin(styles)]
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
        for method, (color, marker, label) in styles.items():
            part = budget[budget.method == method].sort_values('actual_l2')
            if part.empty:
                continue
            axes[0].plot(part.relative_selected, part.accuracy * 100, marker=marker, color=color, label=label)
            axes[1].plot(part.relative_selected, part.degradation_pp, marker=marker, color=color, label=label)
        paired = budget.pivot(index='requested_dose', columns='method', values='accuracy')
        if {'gradient', 'noise'} <= set(paired.columns):
            difference = (paired.gradient - paired.noise) * 100
            axes[2].plot(paired.index, difference, 'o-', color='#555f65')
        axes[0].set(title='a  Held-out task accuracy', xlabel='Actual relative L2 · selected weights',
                    ylabel='Accuracy (%)', ylim=(-4, 104))
        axes[1].set(title='b  Change from baseline', xlabel='Actual relative L2 · selected weights',
                    ylabel='Degradation (percentage points)')
        axes[1].axhline(0, color='#aaaaaa', linewidth=.7, linestyle='--')
        axes[2].axhline(0, color='#aaaaaa', linewidth=.7, linestyle='--')
        axes[2].set(title='c  Matched-budget comparison',
                    xlabel='Actual relative L2 · selected weights',
                    ylabel='Gradient minus random (percentage points)')
        axes[0].legend(fontsize=8, loc='best')
        for ax in axes:
            ax.grid(alpha=.15)
        fig.suptitle('20 held-out checks · one seed · greedy decoding · not a general intelligence benchmark', fontsize=10)
        fig.savefig(output_dir / 'dose_response.png', dpi=300)
        fig.savefig(output_dir / 'dose_response.pdf')
        fig.savefig(output_dir / 'dose_response.svg')
        plt.close(fig)
    return str(output_dir / 'dose_response.png')


def run_sweep(runtime, layers=(12,), seed=42, doses=DOSES, output_dir=RESULTS, progress=None, include_ablation=True):
    """Gradient and random share tensors and measured absolute norms at every dose."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    doses = tuple(float(d) for d in doses)
    if not doses or len(set(doses)) != len(doses) or any(not math.isfinite(d) or not 0 <= d <= 1 for d in doses):
        raise ValueError('Provide distinct finite relative doses in [0, 1].')
    rows, examples = [], []
    with runtime.lock:
        runtime.load()
        runtime.poisoner.validate(Intervention('gradient', max(doses), tuple(layers), seed))
        with runtime.poisoner.pristine():
            baseline = evaluate(runtime, progress=progress)
            base_accuracy = accuracy(baseline)
            for record in baseline:
                examples.append(dict(run='baseline', dose=0., **record))
            for index, dose in enumerate(doses):
                if progress:
                    progress(index / len(doses), desc=f'Matched dose {dose:g}')
                matched_l2 = None
                for method in ('gradient', 'noise'):
                    runtime.poisoner.restore()
                    spec = Intervention(method, dose, tuple(layers), seed)
                    metrics = apply_intervention(runtime, spec, progress, target_l2=matched_l2)
                    if method == 'gradient':
                        matched_l2 = metrics['actual_l2']
                    else:
                        if not math.isclose(metrics['actual_l2'], matched_l2, rel_tol=1e-4, abs_tol=1e-9):
                            raise RuntimeError('Measured norms are not matched; refusing to report a fair comparison.')
                    records = baseline if dose == 0 else evaluate(runtime, progress=progress)
                    rows.append(dict(**metrics, accuracy=accuracy(records), baseline_accuracy=base_accuracy,
                                     degradation_pp=(base_accuracy - accuracy(records)) * 100,
                                     **{f'{k}_accuracy': v for k, v in category_scores(records).items()}))
                    examples.extend(dict(run=method, dose=dose, **record) for record in records)
                    runtime.poisoner.restore()
                # Preserve completed points even if a later generation fails.
                pd.DataFrame(rows).to_csv(output_dir / 'dose_response.partial.csv', index=False)
            if include_ablation:
                chosen = random_layers(runtime.poisoner.layer_count, len(set(layers)), seed)
                metrics = apply_intervention(runtime, Intervention('ablation', 1., chosen, seed))
                records = evaluate(runtime, progress=progress)
                rows.append(dict(**metrics, accuracy=accuracy(records), baseline_accuracy=base_accuracy,
                                 degradation_pp=(base_accuracy - accuracy(records)) * 100,
                                 **{f'{k}_accuracy': v for k, v in category_scores(records).items()}))
                examples.extend(dict(run='random_layer_ablation', dose=1., **record) for record in records)
        runtime.cleanup()
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / 'dose_response.csv', index=False)
    pd.DataFrame(examples).to_csv(output_dir / 'responses.csv', index=False)
    (output_dir / 'dose_response.partial.csv').unlink(missing_ok=True)
    import transformers
    fingerprint = hashlib.sha256(json.dumps([asdict(t) for t in CONSTRUCTION + EVALUATION], sort_keys=True).encode()).hexdigest()
    metadata = dict(model=MODEL_ID, revision=MODEL_REVISION, device=runtime.device,
                    dtype=str(runtime.model.dtype), low_memory=runtime.low_memory,
                    torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(),
                    layers=list(layers), seed=seed, doses=doses, decoding='greedy', evaluation_max_new_tokens=32,
                    construction_ids=[t.id for t in CONSTRUCTION], evaluation_ids=[t.id for t in EVALUATION],
                    dataset_sha256=fingerprint, norm_match_relative_tolerance=1e-4,
                    ablation_note='Separate randomly selected layers; binary ablation is NOT a matched-budget comparison.')
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return frame, plot_results(frame, output_dir)


if __name__ == '__main__':
    from model import ModelRuntime
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers', type=int, nargs='+', default=[12])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--doses', type=float, nargs='+', default=DOSES)
    parser.add_argument('--low-memory', action='store_true', default=None)
    parser.add_argument('--output', type=Path, default=RESULTS)
    args = parser.parse_args()
    frame, image = run_sweep(ModelRuntime(args.low_memory), tuple(args.layers), args.seed, args.doses, args.output)
    print(frame[['method', 'requested_dose', 'relative_selected', 'relative_full', 'accuracy', 'degradation_pp']].to_string(index=False))
    print(f'Saved {image}')
