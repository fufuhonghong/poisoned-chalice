"""One float32 model, lazy loading, MPS/CPU and bounded inference memory."""
import gc
import os
import threading
import torch
from config import MODEL_ID, MODEL_REVISION, SEED, MAX_NEW_TOKENS, MAX_INPUT_TOKENS
from poison import WeightPoisoner


class ModelRuntime:
    def __init__(self, low_memory=None):
        self.lock = threading.RLock()
        self.model = self.tokenizer = self.poisoner = None
        self.device = None
        self.low_memory = (os.getenv('CHALICE_LOW_MEMORY', '0') == '1') if low_memory is None else low_memory
        self.gradient_cache = None  # At most one set of CPU direction tensors.

    def load(self):
        with self.lock:
            if self.model is not None:
                return self
            from transformers import AutoModelForCausalLM, AutoTokenizer
            requested = os.getenv('CHALICE_DEVICE', 'auto')
            if requested not in ('auto', 'cpu', 'mps'):
                raise ValueError('CHALICE_DEVICE must be auto, cpu, or mps.')
            available = torch.backends.mps.is_available()
            if requested == 'mps' and not available:
                raise RuntimeError('MPS unavailable. Use CHALICE_DEVICE=cpu.')
            self.device = 'mps' if available and requested != 'cpu' else 'cpu'
            torch.manual_seed(SEED)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
            # Float32 preserves tiny parameter budgets and stabilizes task-loss gradients.
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32,
                attn_implementation='eager', use_safetensors=True,
            ).to(self.device).eval()
            model.requires_grad_(False)
            self.tokenizer, self.model = tokenizer, model
            self.poisoner = WeightPoisoner(model)
            return self

    def cleanup(self):
        gc.collect()
        if self.device == 'mps':
            torch.mps.empty_cache()

    def generate(self, messages, max_new_tokens=MAX_NEW_TOKENS):
        with self.lock, torch.inference_mode():
            self.load()
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)
            limit = 512 if self.low_memory else MAX_INPUT_TOKENS
            if inputs.input_ids.shape[1] > limit:
                raise ValueError(f'Input exceeds {limit} tokens. Shorten the prompt.')
            inputs = inputs.to(self.device)
            output = self.model.generate(
                **inputs, max_new_tokens=min(max_new_tokens, 64) if self.low_memory else max_new_tokens,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
                temperature=None, top_p=None, top_k=None,
                use_cache=not self.low_memory,
            )
            return self.tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
