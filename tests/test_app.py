from types import SimpleNamespace

import torch

import app
from poison import Intervention, WeightPoisoner
from test_poison import TinyModel


def test_dashboard_builds_without_loading_model():
    demo = app.build_app()
    assert demo.title == 'Poisoned Chalice'
    assert app.RUNTIME.model is None


def test_comparison_uses_identical_prompt_and_restores(monkeypatch):
    model = TinyModel()
    poisoner = WeightPoisoner(model)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    calls = []

    def generate(messages, max_new_tokens=128):
        calls.append(messages)
        return 'perturbed' if poisoner.state == 'modified' else 'original'

    runtime = SimpleNamespace(
        lock=poisoner.lock,
        load=lambda: None,
        poisoner=poisoner,
        generate=generate,
        cleanup=lambda: None,
        model=object(),
        device='cpu',
        low_memory=False,
        gradient_cache=None,
    )
    monkeypatch.setattr(app, 'RUNTIME', runtime)
    spec = Intervention('noise', 0.01, (0,), 7)
    left, right, summary, metrics, _ = app.chat('hello', spec, progress=None)
    assert calls[0] == calls[1]
    assert left[-1]['content'] == 'original'
    assert right[-1]['content'] == 'perturbed'
    assert '选中权重相对变化' in summary
    assert metrics['actual_l2'] > 0
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
