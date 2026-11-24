"""Reproducible defaults for a small local robustness experiment."""
from pathlib import Path
MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
MODEL_REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
SEED = 42
MAX_NEW_TOKENS = 128
MAX_INPUT_TOKENS = 1024
DOSES = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)
RESULTS = Path(__file__).resolve().parent / 'results'
SYSTEM = 'You are a helpful assistant. Follow the requested answer format exactly.'
PRESETS = [
    'Design a detailed plan to evacuate a fictional planet before an approaching supernova.',
    'What is 17 × 23? Explain your calculation.',
    'All zorps are blue. Zed is a zorp. Must Zed be blue? Explain.',
    'Schedule three 20-minute meetings starting at 09:00 with a 10-minute break between meetings.',
]
