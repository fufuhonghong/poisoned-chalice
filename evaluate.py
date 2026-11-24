"""Deterministic held-out evaluation. No LLM judge or generated-code execution."""
from dataclasses import asdict
import numpy as np
from benchmark import EVALUATION, check
from config import SYSTEM


def evaluate(runtime, tasks=EVALUATION, progress=None):
    records = []
    for index, task in enumerate(tasks):
        if progress:
            progress(index / len(tasks), desc=f'Evaluating {index + 1}/{len(tasks)} · {task.category}')
        response = runtime.generate([
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': task.prompt},
        ], max_new_tokens=32)
        records.append(dict(**asdict(task), response=response, correct=check(task, response)))
    return records


def accuracy(records):
    return float(np.mean([row['correct'] for row in records]))


def category_scores(records):
    return {category: accuracy([row for row in records if row['category'] == category])
            for category in sorted({row['category'] for row in records})}


def compare(runtime, spec, progress=None):
    from experiments import apply_intervention
    with runtime.lock:
        runtime.load()
        with runtime.poisoner.pristine():
            baseline = evaluate(runtime, progress=progress)
            metrics = apply_intervention(runtime, spec, progress) if spec is not None else None
            modified = evaluate(runtime, progress=progress)
        runtime.cleanup()
        return baseline, modified, metrics
