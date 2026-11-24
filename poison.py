"""Bounded parameter interventions. All measurements use actual differences."""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import threading
import torch


@dataclass(frozen=True)
class Intervention:
    method: str = 'gradient'
    dose: float = 0.01  # Relative L2 budget over the selected pristine tensors.
    layers: tuple[int, ...] = (12,)
    seed: int = 42


def squared_norm(tensor):
    """Float64 CPU reduction in bounded chunks; MPS does not support float64."""
    total = 0.0
    for chunk in tensor.detach().reshape(-1).split(262144):
        # MPS cannot perform a float64 cast. Transfer first, then cast on CPU.
        cpu = chunk.to(device='cpu').to(dtype=torch.float64)
        total += torch.dot(cpu, cpu).item()
    return total


class WeightPoisoner:
    def __init__(self, model):
        self.model = model
        self.lock = threading.RLock()
        self.originals = {}
        self.active = None
        self.state = 'original'
        self.total_count = sum(p.numel() for p in model.parameters())
        self.full_norm_sq = None

    @property
    def layer_count(self):
        return len(self.model.model.layers)

    def weight(self, index):
        return self.model.model.layers[index].mlp.down_proj.weight

    def validate(self, spec):
        if spec.method not in ('noise', 'ablation', 'gradient'):
            raise ValueError('Unknown intervention method.')
        if not math.isfinite(spec.dose) or not 0 <= spec.dose <= 1:
            raise ValueError('Relative budget must be between 0 and 1.')
        if not spec.layers or any(type(i) is not int or not 0 <= i < self.layer_count for i in spec.layers):
            raise ValueError(f'Select layers from 0 to {self.layer_count - 1}.')
        if type(spec.seed) is not int or not 0 <= spec.seed < 2**32:
            raise ValueError('Seed must be an integer in [0, 2^32).')

    def initialize_full_norm(self):
        if self.full_norm_sq is None:
            if self.originals:
                raise RuntimeError('Full-model denominator must be computed before modification.')
            self.full_norm_sq = sum(squared_norm(p) for p in self.model.parameters())

    @torch.no_grad()
    def restore(self):
        with self.lock:
            for index, original in self.originals.items():
                self.weight(index).copy_(original)
                if not torch.equal(self.weight(index).detach().cpu(), original):
                    raise RuntimeError(f'Exact restoration failed at layer {index}.')
            self.originals.clear()
            self.active = None
            self.state = 'restored'

    def metrics(self, spec):
        selected_sq = delta_sq = 0.0
        selected_count = changed_count = 0
        for index in sorted(set(spec.layers)):
            current = self.weight(index).detach().cpu()
            original = self.originals.get(index, current)
            selected_sq += squared_norm(original)
            delta = current - original
            delta_sq += squared_norm(delta)
            selected_count += current.numel()
            changed_count += torch.count_nonzero(delta).item()
        return dict(requested_dose=spec.dose, actual_l2=math.sqrt(delta_sq),
                    relative_selected=math.sqrt(delta_sq / selected_sq) if selected_sq else 0.,
                    relative_full=math.sqrt(delta_sq / self.full_norm_sq) if self.full_norm_sq else 0.,
                    selected_parameters=selected_count, modified_parameters=changed_count,
                    modified_percent=100 * changed_count / self.total_count,
                    selected_percent=100 * selected_count / self.total_count,
                    total_parameters=self.total_count, layers=','.join(map(str, sorted(set(spec.layers)))),
                    method=spec.method, seed=spec.seed)

    @torch.no_grad()
    def apply(self, spec, directions=None, target_l2=None):
        """Normalize a gradient or std-scaled Gaussian direction to a common L2 budget.

        target_l2 can match random noise to a previously measured gradient delta.
        Ablation is a separate binary intervention and is NEVER labeled budget-matched.
        """
        self.validate(spec)
        with self.lock:
            self.restore()
            self.initialize_full_norm()
            try:
                indices = sorted(set(spec.layers))
                if spec.dose == 0:
                    self.active, self.state = spec, 'original'
                    return self.metrics(spec)
                for index in indices:
                    self.originals[index] = self.weight(index).detach().to('cpu', copy=True)
                if spec.method == 'ablation':
                    for index in indices:
                        self.weight(index).zero_()
                else:
                    if spec.method == 'noise':
                        directions = {}
                        for index in indices:
                            generator = torch.Generator(device='cpu').manual_seed(spec.seed + index)
                            directions[index] = torch.randn(self.weight(index).shape, generator=generator)
                            directions[index].mul_(self.originals[index].float().std().item())
                    if directions is None or set(directions) != set(indices):
                        raise ValueError('Gradient directions must cover exactly the selected layers.')
                    direction_norm = math.sqrt(sum(squared_norm(directions[i]) for i in indices))
                    original_norm = math.sqrt(sum(squared_norm(self.originals[i]) for i in indices))
                    desired = spec.dose * original_norm if target_l2 is None else float(target_l2)
                    if not math.isfinite(direction_norm) or direction_norm <= 0:
                        raise ValueError('Gradient/noise direction must be finite and nonzero.')
                    if not math.isfinite(desired) or desired < 0 or desired > original_norm * (spec.dose + 1e-6):
                        raise ValueError('Requested matched norm exceeds the selected budget.')
                    # Measure the rounded parameters, then calibrate the multiplier if needed.
                    scale = desired / direction_norm
                    for _ in range(4):
                        for index in indices:
                            perturbed = self.originals[index].float() + directions[index].float() * scale
                            self.weight(index).copy_(perturbed)
                        measured = self.metrics(spec)
                        actual = measured['actual_l2']
                        if desired == 0 or abs(actual - desired) <= max(1e-9, desired * 1e-5):
                            break
                        if actual == 0:
                            raise ValueError('Budget is below representable precision; use float32 or a larger dose.')
                        scale *= desired / actual
                    if desired and abs(actual - desired) > max(1e-9, desired * 1e-4):
                        raise ValueError('Cannot match the requested norm within 0.01%; use float32.')
                self.active, self.state = spec, 'modified'
                return self.metrics(spec)
            except Exception:
                self.restore()
                raise

    @contextmanager
    def pristine(self):
        """Experiments always exit with the physical model restored."""
        with self.lock:
            self.restore()
            try:
                yield
            finally:
                self.restore()
