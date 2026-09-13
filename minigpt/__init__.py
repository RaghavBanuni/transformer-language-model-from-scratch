"""A decoder-only transformer language model in pure NumPy.

Every gradient in this package is derived by hand and verified against central finite differences.
That is the whole point: autodiff makes the chain rule invisible, and a wrong backward pass does not
crash -- it trains slightly worse and looks like a hyperparameter problem.

What is here:

- :mod:`minigpt.layers` -- Linear, Embedding, LayerNorm, GELU, Dropout, fused cross-entropy
- :mod:`minigpt.attention` -- causal multi-head self-attention, plus a KV-cached single step
- :mod:`minigpt.model` -- pre-LN blocks, tied embeddings, the full backward pass
- :mod:`minigpt.optim` -- AdamW with decoupled decay, global-norm clipping, cosine warmup
- :mod:`minigpt.tokenizer` -- byte-level BPE, so no input is ever out of vocabulary
- :mod:`minigpt.sampling` -- temperature, top-k, nucleus, and cached generation
- :mod:`minigpt.train` -- batching, the loop, and the two diagnostics worth running first
- :mod:`minigpt.gradcheck` -- the numerical checks that make the derivations trustworthy
"""

from .attention import CausalSelfAttention, causal_mask
from .gradcheck import GradCheckResult, check_module, check_parameter, relative_error
from .layers import (
    GELU,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    Module,
    cross_entropy,
    log_softmax,
    softmax,
    softmax_backward,
)
from .model import MLP, Block, ModelConfig, TransformerLM
from .optim import AdamW, clip_grad_norm, cosine_schedule_with_warmup, global_grad_norm
from .sampling import generate, sample_from_logits, top_k_filter, top_p_filter
from .tokenizer import BPETokenizer, ByteTokenizer
from .train import (
    Dataset,
    TrainReport,
    evaluate,
    initial_loss_report,
    overfit_batch,
    train,
    uniform_loss,
)

__version__ = "1.0.0"

__all__ = [
    "AdamW",
    "BPETokenizer",
    "Block",
    "ByteTokenizer",
    "CausalSelfAttention",
    "Dataset",
    "Dropout",
    "Embedding",
    "GELU",
    "GradCheckResult",
    "LayerNorm",
    "Linear",
    "MLP",
    "ModelConfig",
    "Module",
    "TrainReport",
    "TransformerLM",
    "causal_mask",
    "check_module",
    "check_parameter",
    "clip_grad_norm",
    "cosine_schedule_with_warmup",
    "cross_entropy",
    "evaluate",
    "generate",
    "global_grad_norm",
    "initial_loss_report",
    "log_softmax",
    "overfit_batch",
    "relative_error",
    "sample_from_logits",
    "softmax",
    "softmax_backward",
    "top_k_filter",
    "top_p_filter",
    "train",
    "uniform_loss",
]
