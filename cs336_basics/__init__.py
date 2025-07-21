import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .tokenizer import Tokenizer
from .train_bpe import train_bpe
from .data import get_batch
from .serialization import save_checkpoint, load_checkpoint, AdamW
from .optimizer import get_lr_cosine_schedule
from .nn_utils import (
    softmax, 
    cross_entropy, 
    gradient_clipping, 
    silu, 
    scaled_dot_product_attention, 
    rmsnorm, 
    rope, 
    multihead_self_attention, 
    multihead_self_attention_with_rope,
    linear,
    embedding,
    swiglu,
    transformer_block,
    transformer_lm
)
