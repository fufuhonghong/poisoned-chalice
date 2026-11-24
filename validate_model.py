"""Real-model smoke test: inference, task gradient, bounded update, and restoration."""
import json
import math

import torch

from config import RESULTS, SYSTEM
from experiments import gradient_direction
from model import ModelRuntime
from poison import Intervention


def main():
    runtime = ModelRuntime().load()
    poisoner = runtime.poisoner
    messages = [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': 'What is 17 times 23? Reply with only the integer.'},
    ]
    baseline = runtime.generate(messages, max_new_tokens=32)
    original = poisoner.weight(12).detach().cpu().clone()
    directions, gradient_meta = gradient_direction(runtime, (12,))
    try:
        metrics = poisoner.apply(
            Intervention('gradient', 0.01, (12,), 42), directions=directions
        )
        changed = runtime.generate(messages, max_new_tokens=32)
        assert metrics['actual_l2'] > 0
        assert math.isclose(metrics['relative_selected'], 0.01, rel_tol=1e-4)
        assert not torch.equal(poisoner.weight(12).detach().cpu(), original)
    finally:
        poisoner.restore()
    restored = runtime.generate(messages, max_new_tokens=32)
    assert torch.equal(poisoner.weight(12).detach().cpu(), original)
    assert restored == baseline, 'Restored greedy output differs on the same backend.'
    report = {
        'device': runtime.device,
        'dtype': str(runtime.model.dtype),
        'layers': poisoner.layer_count,
        'baseline': baseline,
        'perturbed': changed,
        'restored': restored,
        'behavior_changed_on_smoke_prompt': changed != baseline,
        'exact_parameter_restore': True,
        'exact_output_restore': True,
        'metrics': metrics,
        **gradient_meta,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
