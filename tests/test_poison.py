import math

import pytest
import torch
from torch import nn

from benchmark import CONSTRUCTION, EVALUATION, check
from poison import Intervention, WeightPoisoner, squared_norm


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for _ in range(3):
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.down_proj = nn.Linear(8, 4, bias=False)
            layer.mlp.up_proj = nn.Linear(4, 8, bias=False)
            self.model.layers.append(layer)


@pytest.fixture
def setup():
    torch.manual_seed(3)
    model = TinyModel()
    poisoner = WeightPoisoner(model)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    return model, poisoner, original


def test_splits_are_disjoint_and_have_twenty_evaluation_tasks():
    assert len(EVALUATION) >= 20
    assert not {task.id for task in CONSTRUCTION} & {task.id for task in EVALUATION}
    assert not {task.prompt for task in CONSTRUCTION} & {task.prompt for task in EVALUATION}
    assert {task.category for task in EVALUATION} == {
        'arithmetic', 'logic', 'instruction', 'planning'
    }


def test_strict_automatic_checks():
    assert check(EVALUATION[0], '391')
    assert not check(EVALUATION[0], 'The answer is 391.')
    assert check(EVALUATION[5], 'YES!')
    assert not check(EVALUATION[10], 'chalice')


def test_noise_is_budgeted_seeded_and_exactly_restorable(setup):
    model, poisoner, original = setup
    spec = Intervention('noise', 0.01, (0, 2), 7)
    first = poisoner.apply(spec)
    changed = poisoner.weight(0).clone()
    assert first['relative_selected'] == pytest.approx(0.01, rel=1e-4)
    assert first['actual_l2'] > 0
    assert first['selected_parameters'] == 64
    assert 0 < first['modified_parameters'] <= 64
    poisoner.restore()
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
    second = poisoner.apply(spec)
    assert torch.equal(changed, poisoner.weight(0))
    assert second['actual_l2'] == first['actual_l2']


def test_gradient_direction_and_noise_match_actual_l2(setup):
    _, poisoner, _ = setup
    directions = {0: torch.arange(32, dtype=torch.float32).reshape(4, 8) + 1}
    gradient = poisoner.apply(Intervention('gradient', 0.02, (0,), 2), directions)
    target = gradient['actual_l2']
    poisoner.restore()
    noise = poisoner.apply(Intervention('noise', 0.02, (0,), 2), target_l2=target)
    assert noise['actual_l2'] == pytest.approx(target, rel=1e-4, abs=1e-9)
    assert noise['relative_selected'] == pytest.approx(gradient['relative_selected'], rel=1e-4)


def test_new_interventions_never_accumulate(setup):
    _, poisoner, original = setup
    poisoner.apply(Intervention('noise', 0.01, (0,), 1))
    poisoner.apply(Intervention('ablation', 1.0, (1,), 1))
    assert torch.equal(poisoner.weight(0), original['model.layers.0.mlp.down_proj.weight'])
    assert torch.count_nonzero(poisoner.weight(1)) == 0


def test_ablation_reports_actual_difference_and_restores(setup):
    model, poisoner, original = setup
    metrics = poisoner.apply(Intervention('ablation', 0.001, (1,), 5))
    assert metrics['relative_selected'] == pytest.approx(1.0)
    assert metrics['modified_parameters'] == poisoner.weight(1).numel()
    assert torch.count_nonzero(poisoner.weight(1)) == 0
    poisoner.restore()
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('method', ['noise', 'ablation', 'gradient'])
def test_restoration_recovers_parameters_and_output(dtype, method):
    torch.manual_seed(9)
    model = TinyModel().to(dtype=dtype)
    poisoner = WeightPoisoner(model)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    x = torch.ones(1, 8, dtype=dtype)
    baseline = model.model.layers[0].mlp.down_proj(x).detach().clone()
    directions = {0: torch.ones_like(poisoner.weight(0), dtype=torch.float32)} if method == 'gradient' else None
    if dtype == torch.float16 and method == 'gradient':
        with pytest.raises(ValueError, match='use float32'):
            poisoner.apply(Intervention(method, 0.5, (0,)), directions)
        assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
        return
    poisoner.apply(Intervention(method, 0.5, (0,)), directions)
    poisoner.restore()
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
    assert torch.equal(model.model.layers[0].mlp.down_proj(x), baseline)


@pytest.mark.parametrize('spec', [
    Intervention('bad', 0.1, (0,)),
    Intervention('noise', float('nan'), (0,)),
    Intervention('gradient', 2, (0,)),
    Intervention('noise', 0.1, (99,)),
    Intervention('gradient', 0.1, ()),
    Intervention('noise', 0.1, (0,), -1),
])
def test_invalid_request_preserves_current_intervention(setup, spec):
    _, poisoner, _ = setup
    poisoner.apply(Intervention('noise', 0.1, (0,), 4))
    before = poisoner.weight(0).clone()
    with pytest.raises(ValueError):
        poisoner.apply(spec)
    assert torch.equal(before, poisoner.weight(0))


def test_failure_during_apply_restores_everything(setup):
    model, poisoner, original = setup
    bad = {0: torch.ones_like(poisoner.weight(0)), 1: torch.full_like(poisoner.weight(1), math.nan)}
    with pytest.raises(ValueError):
        poisoner.apply(Intervention('gradient', 0.1, (0, 1)), bad)
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
    assert not poisoner.originals


def test_squared_norm_is_stable_for_half_precision():
    tensor = torch.tensor([3.0, 4.0], dtype=torch.float16)
    assert squared_norm(tensor) == pytest.approx(25.0)
