import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .tokenizer import Tokenizer
from .train_bpe import train_bpe
