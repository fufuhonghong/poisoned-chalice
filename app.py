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
METHODS = [('Gradient-guided · task-loss ascent', 'gradient'),
           ('Random Gaussian · matched L2 budget', 'noise'),
           ('Layer ablation · binary zeroing', 'ablation')]


def describe(spec):
    if spec is None or spec.dose == 0:
        return '<span class="state-dot state-clean"></span> **原始状态** · 当前没有参数扰动'
    return (f'<span class="state-dot state-active"></span> **已设置扰动** · {spec.method} · '
            f'相对预算 {spec.dose:g} · 第 {", ".join(map(str, spec.layers))} 层 · seed {spec.seed}')


def model_info():
    if RUNTIME.model is None:
        return (f'**模型**　{MODEL_ID}　　**精度**　float32　　'
                f'**状态**　首次操作时加载　　**低内存模式**　{RUNTIME.low_memory}')
    return (f'**模型**　{MODEL_ID}　　**设备**　{RUNTIME.device.upper()}　　'
            f'**参数量**　{RUNTIME.poisoner.total_count:,}　　**低内存模式**　{RUNTIME.low_memory}')


def summarize_metrics(metrics):
    if not metrics:
        return '尚未施加扰动。应用后，这里会显示程序实际测得的参数变化。'
    return f'''<div class="metric-summary">
    <div><span>目标层</span><strong>{metrics.get('layers', '—')}</strong></div>
    <div><span>选中权重相对变化</span><strong>{metrics.get('relative_selected', 0):.6f}</strong></div>
    <div><span>全模型相对变化</span><strong>{metrics.get('relative_full', 0):.6f}</strong></div>
    <div><span>实际 L2 变化</span><strong>{metrics.get('actual_l2', 0):.6f}</strong></div>
    <div><span>实际改变的参数</span><strong>{metrics.get('modified_parameters', 0):,}</strong></div>
    <div><span>占全模型参数</span><strong>{metrics.get('modified_percent', 0):.4f}%</strong></div>
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
        title='Poisoned Chalice', analytics_enabled=False, css=CSS, theme=theme,
        js="() => { document.documentElement.classList.add('dark'); }",
    ) as demo:
        active = gr.State(None)
        gr.HTML('''<header id="hero">
        <div class="eyebrow">Local parameter robustness study</div>
        <div class="hero-row"><div><h1>Poisoned Chalice</h1>
        <p>How little does it take to change a language model?</p></div>
        <div class="hero-meta"><span>Qwen2.5 0.5B</span><span>Local only</span><span>Reversible</span></div></div>
        </header>''')
        info = gr.Markdown(model_info())
        status = gr.Markdown(describe(None), elem_id='status')
        with gr.Accordion('先看这里：这个“毒”是怎么加进去的？', open=True, elem_id='method-note'):
            gr.HTML('''<div class="method-flow">
            <div><b>1</b><span><strong>选一块权重</strong>默认选择第 12 层 MLP 的输出矩阵 W。</span></div>
            <div><b>2</b><span><strong>找破坏方向</strong>用 8 道独立小题计算损失梯度 g，方向指向“让答案变差”。</span></div>
            <div><b>3</b><span><strong>限制剂量</strong>把梯度归一化，只加入 ΔW = d·‖W‖·g/‖g‖。</span></div>
            <div><b>4</b><span><strong>只在内存修改</strong>d=0.01 表示改变量的 L2 范数是所选权重范数的 1%。</span></div>
            <div><b>5</b><span><strong>实验后恢复</strong>运行前备份 W，结束后逐元素恢复；下载的模型文件不会被改写。</span></div>
            </div>
            <p class="plain-note">它不是在提示词里加入“毒药”，也没有训练模型；它直接、暂时地改变神经网络中的数字。</p>''')
        with gr.Row():
            with gr.Column(scale=5, min_width=280, elem_classes='lab-panel'):
                gr.Markdown('### 原始模型\n未修改的权重，用作对照。')
                left = gr.Chatbot(
                    height=370, label='原始输出', show_copy_button=True, type='messages'
                )
            with gr.Column(scale=3, min_width=250, elem_classes='lab-panel', elem_id='controls'):
                gr.Markdown('### 干预设置\n选择方法、剂量和目标层。')
                method = gr.Dropdown(METHODS, value='gradient', label='扰动方法')
                dose = gr.Slider(0, .05, value=.01, step=.001, label='相对 L2 剂量')
                layers = gr.Dropdown(list(range(24)), value=[12], multiselect=True, label='目标 Transformer 层')
                seed = gr.Number(42, label='随机种子', precision=0, minimum=0, maximum=2**32 - 1)
                with gr.Accordion('随机选择层', open=False):
                    count = gr.Slider(1, 24, value=1, step=1, label='层数')
                    random_btn = gr.Button('按种子选择')
                gr.Markdown('`0.01` 表示所选权重的相对范数变化为 0.01，并不是修改 1% 的参数。')
                administer_btn = gr.Button('应用扰动', elem_id='administer')
                run_btn = gr.Button('运行 20 题评测')
                restore_btn = gr.Button('恢复原始权重')
            with gr.Column(scale=5, min_width=280, elem_classes='lab-panel', elem_id='poison-panel'):
                gr.Markdown('### 扰动模型\n同一个模型，临时修改指定权重。')
                right = gr.Chatbot(
                    height=370, label='扰动后输出', show_copy_button=True, type='messages'
                )
        prompt = gr.Textbox(placeholder='输入数学、逻辑或虚构规划问题…', label='两个模型使用完全相同的提示', lines=2)
        gr.Examples([[p] for p in PRESETS], inputs=prompt)
        send = gr.Button('比较两边输出', elem_id='send')
        gr.Markdown('两边使用相同提示和生成设置。拒绝回答本身不会被当作“智力下降”。', elem_classes='helper-copy')
        with gr.Tabs():
            with gr.Tab('20 题评测'):
                gr.Markdown('### 四类可自动评分任务\n算术、逻辑、指令遵循和规划各 5 题。用于计算梯度的 8 道题与这 20 道题不重合。')
                with gr.Row():
                    baseline = gr.Number(label='原始准确率 (%)', interactive=False, elem_classes='metric')
                    perturbed = gr.Number(label='扰动后准确率 (%)', interactive=False, elem_classes='metric')
                    degradation = gr.Number(label='下降（百分点）', interactive=False, elem_classes='metric')
                answers = gr.Dataframe(label='逐题回答与自动判分', interactive=False, wrap=True)
            with gr.Tab('实际改了多少'):
                gr.Markdown('### 根据参数真实差值计算\n扰动大小与“改了多少个参数”是两件不同的事。下面同时报告相对范数和参数数量。')
                metric_summary = gr.Markdown(summarize_metrics({}), elem_id='metric-summary')
                with gr.Accordion('查看原始指标 JSON', open=False):
                    metrics = gr.JSON(label='Measured intervention')
            with gr.Tab('剂量曲线'):
                gr.Markdown('在 6 个剂量上比较梯度方向和同范数随机噪声。每个点都先恢复原始权重；层消融单独列出，不冒充同剂量比较。')
                sweep_btn = gr.Button('运行完整剂量实验')
                sweep_status = gr.Markdown()
                plot = gr.Image(label='实测剂量响应', type='filepath')
                table = gr.Dataframe(interactive=False)
                downloads = gr.File(label='下载图表、CSV、原始回答与元数据', file_count='multiple')
            with gr.Tab('方法与限制'):
                gr.Markdown('''### 这是敏感性实验，不是“杀死 AI”
梯度会提高 8 道构造题的目标答案损失，只有选中的 MLP 输出矩阵参与梯度计算，其余参数全部冻结。评测使用另外 20 道题。

归一化更新为 `Δθ = d ‖θ_selected‖₂ ∇L / ‖∇L‖₂`。随机高斯噪声会被校准到相同的实测 L2 范数。层消融直接清零矩阵，是单独的强干预。

小幅掉分可能来自格式错误。语言流畅也不等于答案正确。这个实验不能证明意识、人格或通用能力被永久删除。

程序只维护一个 float32 模型，只备份被选中的权重。每次操作后都会恢复，下载的 checkpoint 永远不会被覆盖。''')
        gr.Markdown('Poisoned Chalice · Local parameter robustness study', elem_id='footnote')
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
