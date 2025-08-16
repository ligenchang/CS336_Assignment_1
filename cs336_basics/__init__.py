import importlib.metadata

# __version__ = importlib.metadata.version("cs336_basics")

from .tokenizer import Tokenizer
from .train_bpe import train_bpe
from .data import get_batch
from .serialization import (
    save_checkpoint, 
    load_checkpoint, 
    load_checkpoint_enhanced,
    AdamW,
    create_training_checkpoint,
    load_training_checkpoint,
    safe_load_checkpoint,
    checkpoint_exists
)
from .optimizer import get_lr_cosine_schedule
from .nn_utils import (
    softmax, 
    cross_entropy, 
    gradient_clipping, 
    silu, 
    scaled_dot_product_attention, 
    RMSNorm,
    RotaryPositionalEmbedding,
    multihead_self_attention, 
    swiglu,
    transformer_block,
    TransformerLM,
    Embedding,
    Linear
)
