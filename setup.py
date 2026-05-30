"""
Shared setup: model loading and helper functions.
Imported by all experiment scripts. Not run directly.
"""
import torch
import numpy as np
from transformer_lens import HookedTransformer

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Forward-pass only saves memory
torch.set_grad_enabled(False)

_MODEL = None  # Loaded lazily

def get_model(name: str = "pythia-1.4b"):
    """Load the model once, cache in module globals."""
    global _MODEL
    if _MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {name} on {device}...")
        _MODEL = HookedTransformer.from_pretrained_no_processing(
            name,
            device=device,
            dtype=torch.float16,
        )
        _MODEL.eval()
        if device == "cuda":
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"Loaded. VRAM: {vram:.2f} GB. Layers: {_MODEL.cfg.n_layers}")
        else:
            print(f"Loaded on CPU. Layers: {_MODEL.cfg.n_layers}")
    return _MODEL


def next_token_probs(prompt: str) -> torch.Tensor:
    model = get_model()
    tokens = model.to_tokens(prompt)
    logits = model(tokens)
    return torch.softmax(logits[0, -1].float(), dim=-1)


def prob_of_token(prompt: str, target: str) -> float:
    model = get_model()
    probs = next_token_probs(prompt)
    target_id = model.to_single_token(target)
    return probs[target_id].item()


def top_k_next(prompt: str, k: int = 5) -> list:
    model = get_model()
    probs = next_token_probs(prompt)
    top = torch.topk(probs, k)
    return [(model.to_string(idx.item()), p.item()) for idx, p in zip(top.indices, top.values)]
