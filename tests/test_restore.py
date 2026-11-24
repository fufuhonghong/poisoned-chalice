import torch

from poison import Intervention, WeightPoisoner
from test_poison import TinyModel


def test_restore_matches_original_within_dtype_precision():
    torch.manual_seed(21)
    model = TinyModel().to(dtype=torch.float32)
    poisoner = WeightPoisoner(model)
    originals = {name: value.detach().clone() for name, value in model.state_dict().items()}
    direction = {1: torch.randn_like(poisoner.weight(1))}
    poisoner.apply(Intervention('gradient', 0.05, (1,), 42), direction)
    assert not torch.equal(poisoner.weight(1), originals['model.layers.1.mlp.down_proj.weight'])
    poisoner.restore()
    for name, restored in model.state_dict().items():
        assert torch.equal(restored, originals[name]), name
