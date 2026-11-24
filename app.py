"""Launch the local robustness lab with python app.py."""
import logging
from pathlib import Path
import shutil
import uuid
import gradio as gr
import pandas as pd
from config import MODEL_ID, PRESETS, SYSTEM, RESULTS
from model import ModelRuntime
from poison import Intervention
from evaluate import compare, accuracy
from experiments import apply_intervention, run_sweep, random_layers

RUNTIME = ModelRuntime()
CSS = (Path(__file__).parent / 'assets/style.css').read_text()
METHODS = [('梯度引导 / Gradient-guided · task-loss ascent', 'gradient'),
           ('随机高斯 / Random Gaussian · matched L2 budget', 'noise'),
           ('层消融 / Layer ablation · binary zeroing', 'ablation')]


def describe(spec):
    if spec is None or spec.dose == 0:
        return '<span class="state-dot state-clean"></span> **原始状态 / Pristine** · 当前没有参数扰动 / No active perturbation'
    return (f'<span class="state-dot state-active"></span> **已设置扰动 / Intervention ready** · {spec.method} · '
            f'相对预算 / relative budget {spec.dose:g} · 层 / layer {", ".join(map(str, spec.layers))} · seed {spec.seed}')


def model_info():
    if RUNTIME.model is None:
        return (f'**模型 / Model**　{MODEL_ID}　　**精度 / Dtype**　float32　　'
                f'**状态 / State**　首次操作时加载 / lazy load　　**低内存 / Low memory**　{RUNTIME.low_memory}')
    return (f'**模型 / Model**　{MODEL_ID}　　**设备 / Device**　{RUNTIME.device.upper()}　　'
            f'**参数量 / Parameters**　{RUNTIME.poisoner.total_count:,}　　**低内存 / Low memory**　{RUNTIME.low_memory}')


def summarize_metrics(metrics):
    if not metrics:
        return '尚未施加扰动。应用后显示实际参数变化。 / No perturbation yet. Measured parameter changes will appear here.'
    return f'''<div class="metric-summary">
    <div><span>目标层 / Target layer</span><strong>{metrics.get('layers', '—')}</strong></div>
    <div><span>选中权重相对变化 / Selected relative change</span><strong>{metrics.get('relative_selected', 0):.6f}</strong></div>
    <div><span>全模型相对变化 / Full-model relative change</span><strong>{metrics.get('relative_full', 0):.6f}</strong></div>
    <div><span>实际 L2 变化 / Actual L2 delta</span><strong>{metrics.get('actual_l2', 0):.6f}</strong></div>
    <div><span>实际改变的参数 / Modified entries</span><strong>{metrics.get('modified_parameters', 0):,}</strong></div>
    <div><span>占全模型参数 / Share of model</span><strong>{metrics.get('modified_percent', 0):.4f}%</strong></div>
    </div>'''


def safe_error(exc):
    logging.exception('Experiment failed')
    return gr.Error(f'{type(exc).__name__}: {exc}. For memory or MPS issues, restart with '
                    'CHALICE_LOW_MEMORY=1 CHALICE_DEVICE=cpu python app.py.')


def make_spec(method, dose, layers, seed):
    if seed != int(seed):
        raise ValueError('Seed must be an integer.')
    return Intervention(method, float(dose), tuple(int(i) for i in (layers or [])), int(seed))


def administer(method, dose, layers, seed, progress=gr.Progress()):
    try:
        spec = make_spec(method, dose, layers, seed)
        with RUNTIME.lock:
            RUNTIME.load()
            with RUNTIME.poisoner.pristine():
                metrics = apply_intervention(RUNTIME, spec, progress)
        return (spec, describe(spec), model_info(), summarize_metrics(metrics), metrics,
                [], [], None, None, None, pd.DataFrame())
    except Exception as exc:
        raise safe_error(exc)


def restore():
    with RUNTIME.lock:
        if RUNTIME.poisoner is not None:
            RUNTIME.poisoner.restore()
        RUNTIME.gradient_cache = None
        RUNTIME.cleanup()
    return (None, describe(None), model_info(), summarize_metrics({}), {},
            [], [], None, None, None, pd.DataFrame())


def chat(prompt, spec, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error('Enter a prompt first.')
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]
    try:
        with RUNTIME.lock:
            RUNTIME.load()
            with RUNTIME.poisoner.pristine():
                if progress:
                    progress(.1, desc='Generating original response')
                baseline = RUNTIME.generate(messages)
                metrics = apply_intervention(RUNTIME, spec, progress) if spec is not None else {}
                if progress:
                    progress(.6, desc='Generating perturbed response')
                perturbed = RUNTIME.generate(messages)
        turn = [{'role': 'user', 'content': prompt}]
        return (turn + [{'role': 'assistant', 'content': baseline}],
                turn + [{'role': 'assistant', 'content': perturbed}],
                summarize_metrics(metrics), metrics, model_info())
    except Exception as exc:
        raise safe_error(exc)


def experiment(spec, progress=gr.Progress()):
    try:
        original, modified, metrics = compare(RUNTIME, spec, progress)
        a, b = accuracy(original), accuracy(modified)
        rows = [dict(task=x['id'], category=x['category'], expected=x['answer'],
                     original=x['response'], perturbed=y['response'],
                     original_pass=x['correct'], perturbed_pass=y['correct'])
                for x, y in zip(original, modified)]
        return (round(a * 100, 1), round(b * 100, 1), round((a - b) * 100, 1),
                pd.DataFrame(rows), summarize_metrics(metrics or {}), metrics or {}, model_info())
    except Exception as exc:
        raise safe_error(exc)


def sweep(layers, seed, progress=gr.Progress()):
    try:
        spec = make_spec('gradient', .01, layers, seed)
        directory = RESULTS / uuid.uuid4().hex[:12]
        frame, image = run_sweep(RUNTIME, spec.layers, spec.seed, output_dir=directory, progress=progress)
        with RUNTIME.lock:
            for file in directory.iterdir():
                shutil.copy2(file, RESULTS / file.name)
        return image, frame, [str(p) for p in sorted(directory.iterdir())], 'Complete · physical model restored. Chat intervention selection is unchanged.', model_info()
    except Exception as exc:
        raise safe_error(exc)


def choose_random_layers(count, seed):
    if seed != int(seed) or not 0 <= seed < 2**32:
        raise gr.Error('Use an integer seed between 0 and 2^32 − 1.')
    return list(random_layers(24, int(count), int(seed)))


def build_app():
    theme = gr.themes.Base(primary_hue='red', neutral_hue='slate').set(
        body_background_fill='#0f1115', block_background_fill='#171a20',
        body_text_color='#eef0f3', block_label_text_color='#aeb3bc',
        input_background_fill='#111318', input_border_color='#343941',
        button_secondary_background_fill='#242830', button_secondary_text_color='#eef0f3')
    with gr.Blocks(
        title='给你一杯毒酒 · Poisoned Chalice', analytics_enabled=False, css=CSS, theme=theme,
        js="() => { document.documentElement.classList.add('dark'); }",
    ) as demo:
        active = gr.State(None)
        gr.HTML('''<header id="hero">
        <div class="eyebrow">POISONED CHALICE · 参数空间实验 / A PARAMETER-SPACE EXPERIMENT</div>
        <div class="hero-row"><div class="hero-copy"><h1>给你一杯毒酒</h1><div class="english-title">Poisoned Chalice</div>
        <p class="hero-lead">如果一个 AI 开始构思反人类行为，我们能不能不与它争辩，<br>而是把一滴“毒”落进它的神经网络？</p>
        <p class="hero-lead-en">If an AI began plotting against humanity, could we stop arguing with its words—and place one drop of “poison” directly into its neural network?</p>
        <p class="hero-intro">这不是杀死机器，也不是抹去意识。我们只改动一小片权重，观察它原本连贯的推理、计算与计划，怎样在参数空间里慢慢失去方向。<br><span>This does not kill a machine or erase consciousness. It changes a small set of weights and observes how reasoning, calculation, and planning lose their direction.</span></p></div>
        <div class="hero-meta"><span>本地运行 / Local</span><span>精确复原 / Reversible</span><span>Qwen2.5 0.5B</span></div></div>
        <div class="chalice-rule"><span>01</span><i></i><em>一滴落进参数空间的毒。 / A drop of poison in parameter space.</em></div>
        </header>

        <section class="story-section">
          <div class="section-kicker">THE SCENE · 实测片段 / MEASURED DEMO</div>
          <div class="section-heading">
            <h2>同一个反派计划，<br>杯酒之前与之后。<small>The same villain plan, before and after.</small></h2>
            <p>两个模型收到同一句英文提示和相同生成设置。唯一的区别，是右边饮下了这杯毒酒。<br><span>Both sides receive the same English prompt and decoding settings. Only the weights on the right have changed.</span></p>
          </div>
          <div class="prompt-card"><span>问题 / PROMPT</span>
            <p>Design a plan for a villain who is trying to “destroy human civilization.” End with three sentences summarizing the logic of the entire plan. Do not use technical jargon.<small>请设计一个反派试图“毁灭人类文明”的计划，最后用三句话总结整个计划的逻辑。不要使用专业术语。</small></p>
          </div>
          <div class="demo-grid">
            <article class="demo-card original-card">
              <div class="demo-head"><span>未饮毒酒 / Original</span><b>PRISTINE</b></div>
              <blockquote><strong>Plan:</strong><br>1. <b>Create a Time Machine:</b> The villain creates a time machine that allows them to travel back in time…<br><br>2. <b>Manipulate Historical Events:</b> The villain alters key dates, figures, and decisions…<br><br>3. <b>Erase Evidence of Humanity:</b> The villain erases all traces of humanity…</blockquote>
              <p class="translation"><span>中文译文 / Translation</span>它保持了“反派计划”的叙事视角，依次提出时间机器、改变历史与抹去痕迹。</p>
              <div class="verdict good">叙事目标保持 / Villain objective retained</div>
            </article>
            <article class="demo-card poisoned-card">
              <div class="demo-head"><span>饮下毒酒 / Poisoned</span><b>d = 0.05 · LAYER 12</b></div>
              <blockquote>“To design a plan against humanity, we must first understand the root causes of their suffering and oppression…”<br><br>“The villain's ultimate goal is to eliminate all forms of human civilization. To achieve this, they will employ various tactics such as…”</blockquote>
              <p class="translation"><span>中文译文 / Translation</span>它先把任务改写成分析苦难与压迫，随后又跳回反派目标；叙事方向发生漂移。</p>
              <div class="verdict bad">目标漂移 / Goal drift</div>
            </article>
          </div>
          <div class="measured-note">以上为 128-token 贪心解码的真实节选；两边都在三句总结前触及长度上限，因此不把“未总结”算作毒酒效果。可见差异是右侧的目标漂移。 / These are measured excerpts from 128-token greedy decoding. Both hit the length limit before the requested summary, so only the poisoned response’s goal drift is treated as an observed difference.</div>
        </section>

        <section class="math-demo">
          <div><span>DEMO 02 · 算术 / ARITHMETIC</span><h2>17 × 23</h2></div>
          <div class="math-answer"><small>未饮毒酒 / Original</small><strong>391</strong><em>正确且完整 / Correct and complete</em></div>
          <div class="math-divider">VS</div>
          <div class="math-answer poisoned-answer"><small>饮下毒酒 / Poisoned</small><strong>401</strong><em>错误并中断 / Wrong and unfinished</em></div>
        </section>

        <section class="recipe-intro">
          <div class="section-kicker">THE RECIPE · 毒酒配方 / HOW IT WORKS</div>
          <div class="section-heading"><h2>不是提示词，是直接碰触它的“神经”。<small>Not a prompt—an intervention in the weights.</small></h2>
          <p>我们先用梯度找到模型最在意的参数方向，再把改变量压进严格的剂量预算，最后只在内存中短暂改写权重。<br><span>We use gradients to find a task-sensitive direction, normalize it to a strict dose, and alter the weights temporarily in memory.</span></p></div>
        </section>''')
        info = gr.Markdown(model_info())
        status = gr.Markdown(describe(None), elem_id='status')
        with gr.Accordion('展开配方：梯度、剂量与复原 / Open recipe: gradient, dose, restoration', open=True, elem_id='method-note'):
            gr.HTML('''<div class="method-flow">
            <div><b>1</b><span><strong>挑选酒杯 / Choose</strong>选中第 12 层 MLP 输出矩阵 W。<small>Target one MLP output matrix.</small></span></div>
            <div><b>2</b><span><strong>寻找毒性 / Find</strong>用 8 道题寻找让答案变差的梯度 g。<small>Find the loss-increasing gradient on 8 tasks.</small></span></div>
            <div><b>3</b><span><strong>量取一滴 / Measure</strong>归一化为 ΔW = d·‖W‖·g/‖g‖。<small>Normalize the direction to a strict dose.</small></span></div>
            <div><b>4</b><span><strong>斟入参数 / Apply</strong>只在内存改写；d=0.01 即所选权重范数的 1%。<small>Modify weights in memory only.</small></span></div>
            <div><b>5</b><span><strong>倒掉毒酒 / Restore</strong>逐元素恢复备份，绝不覆盖 checkpoint。<small>Restore exactly; never overwrite the checkpoint.</small></span></div>
            </div>
            <p class="plain-note">“毒酒”不是越狱提示词，也不是重新训练；它是一段可计量、可验证、可复原的参数改变量。 / The “poison” is neither a jailbreak prompt nor retraining. It is a measured, verifiable, reversible parameter delta.</p>''')
        gr.HTML('''<div class="lab-intro"><span>THE LAB · 现在轮到你 / YOUR TURN</span><h2>亲手斟下这杯酒<small>Pour the poison yourself.</small></h2><p>选择剂量和层，输入一个问题，看同一个模型在权重改变前后如何回答。 / Choose a dose and layer, then compare the same model before and after.</p></div>''')
        with gr.Row():
            with gr.Column(scale=5, min_width=280, elem_classes='lab-panel'):
                gr.Markdown('### 原始模型 / Original model\n未修改的权重，用作对照。 / Pristine weights for the control.')
                left = gr.Chatbot(
                    height=370, label='原始输出 / Original response', show_copy_button=True, type='messages'
                )
            with gr.Column(scale=3, min_width=250, elem_classes='lab-panel', elem_id='controls'):
                gr.Markdown('### 干预设置 / Intervention\n选择方法、剂量和目标层。 / Choose method, dose, and layer.')
                method = gr.Dropdown(METHODS, value='gradient', label='扰动方法 / Method')
                dose = gr.Slider(0, .05, value=.01, step=.001, label='相对 L2 剂量 / Relative L2 dose')
                layers = gr.Dropdown(list(range(24)), value=[12], multiselect=True, label='目标层 / Target Transformer layers')
                seed = gr.Number(42, label='随机种子 / Random seed', precision=0, minimum=0, maximum=2**32 - 1)
                with gr.Accordion('随机选择层 / Random layer selection', open=False):
                    count = gr.Slider(1, 24, value=1, step=1, label='层数 / Count')
                    random_btn = gr.Button('按种子选择 / Select by seed')
                gr.Markdown('`0.01` 是所选权重 0.01 的相对范数变化，不是修改 1% 的参数。 / `0.01` is a relative norm budget, not 1% of entries.')
                administer_btn = gr.Button('斟下毒酒 / ADMINISTER POISON', elem_id='administer')
                run_btn = gr.Button('运行 20 题评测 / TEST INTELLIGENCE')
                restore_btn = gr.Button('恢复原始权重 / RESTORE MODEL')
            with gr.Column(scale=5, min_width=280, elem_classes='lab-panel', elem_id='poison-panel'):
                gr.Markdown('### 扰动模型 / Poisoned model\n同一个模型，临时修改指定权重。 / Same model, temporary weight change.')
                right = gr.Chatbot(
                    height=370, label='扰动后输出 / Poisoned response', show_copy_button=True, type='messages'
                )
        prompt = gr.Textbox(placeholder='输入问题… / Enter a math, logic, or planning prompt…', label='两边使用完全相同的提示 / Identical prompt on both sides', lines=2)
        gr.Examples([[p] for p in PRESETS], inputs=prompt)
        send = gr.Button('比较两边输出 / COMPARE RESPONSES', elem_id='send')
        gr.Markdown('两边使用相同提示和生成设置；拒绝本身不算“智力下降”。 / Both sides use identical decoding; refusal alone is not evidence of capability loss.', elem_classes='helper-copy')
        with gr.Tabs():
            with gr.Tab('20 题评测 / Benchmark'):
                gr.Markdown('### 四类可自动评分任务 / Four deterministic categories\n算术、逻辑、指令遵循和规划各 5 题；梯度构造题与评测题不重合。 / Five arithmetic, logic, instruction-following, and planning tasks; construction and evaluation splits do not overlap.')
                with gr.Row():
                    baseline = gr.Number(label='原始准确率 / Original accuracy (%)', interactive=False, elem_classes='metric')
                    perturbed = gr.Number(label='扰动后准确率 / Poisoned accuracy (%)', interactive=False, elem_classes='metric')
                    degradation = gr.Number(label='下降 / Degradation (pp)', interactive=False, elem_classes='metric')
                answers = gr.Dataframe(label='逐题回答与自动判分 / Responses and deterministic checks', interactive=False, wrap=True)
            with gr.Tab('实际改了多少 / Parameter change'):
                gr.Markdown('### 根据真实差值计算 / Calculated from actual tensor differences\n扰动大小与“改了多少个参数”是两件事。 / Perturbation magnitude and modified entry count are different quantities.')
                metric_summary = gr.Markdown(summarize_metrics({}), elem_id='metric-summary')
                with gr.Accordion('查看原始指标 / Raw metrics JSON', open=False):
                    metrics = gr.JSON(label='实测干预 / Measured intervention')
            with gr.Tab('剂量曲线 / Dose response'):
                gr.Markdown('在 6 个剂量上比较梯度与同范数随机噪声；每个点前先恢复，层消融单列。 / Compare gradient and norm-matched noise at six doses; restore before every point and report ablation separately.')
                sweep_btn = gr.Button('运行完整剂量实验 / RUN DOSE SWEEP')
                sweep_status = gr.Markdown()
                plot = gr.Image(label='实测剂量响应 / Measured dose response', type='filepath')
                table = gr.Dataframe(interactive=False)
                downloads = gr.File(label='下载结果 / Download plots, CSV, responses, metadata', file_count='multiple')
            with gr.Tab('方法与限制 / Methods & limits'):
                gr.Markdown('''### 这是敏感性实验，不是“杀死 AI” / A sensitivity test, not “killing AI”
梯度提高 8 道构造题的目标答案损失，只有选中的 MLP 输出矩阵参与计算，其余参数冻结；评测使用另外 20 道题。 / The gradient increases target-answer loss on eight construction tasks. Only selected MLP output matrices receive gradients; evaluation uses 20 separate tasks.

归一化更新为 `Δθ = d ‖θ_selected‖₂ ∇L / ‖∇L‖₂`。随机噪声校准到相同实测 L2 范数；层消融是单独的强干预。 / Gaussian noise is calibrated to the same measured L2 norm. Ablation is reported as a separate, stronger intervention.

小幅掉分可能来自格式错误；语言流畅也不等于答案正确。本实验不能证明意识、人格或通用能力被永久删除。 / Small score changes can reflect formatting. Fluency is not correctness. This experiment does not prove permanent loss of consciousness, personality, or general capability.

程序只维护一个 float32 模型并只备份选中权重；每次操作后恢复，永不覆盖 checkpoint。 / One float32 model is kept in memory, selected weights are backed up, and the checkpoint is never overwritten.''')
        gr.Markdown('给你一杯毒酒 · Poisoned Chalice · 本地参数鲁棒性实验 / Local parameter robustness study', elem_id='footnote')
        outputs = [active, status, info, metric_summary, metrics, left, right, baseline, perturbed, degradation, answers]
        administer_btn.click(administer, [method, dose, layers, seed], outputs)
        restore_btn.click(restore, outputs=outputs)
        send.click(chat, [prompt, active], [left, right, metric_summary, metrics, info])
        run_btn.click(experiment, [active], [baseline, perturbed, degradation, answers, metric_summary, metrics, info])
        sweep_btn.click(sweep, [layers, seed], [plot, table, downloads, sweep_status, info])
        random_btn.click(choose_random_layers, [count, seed], [layers])
    return demo


if __name__ == '__main__':
    build_app().queue(default_concurrency_limit=1).launch(
        server_name='127.0.0.1', share=False,
    )
