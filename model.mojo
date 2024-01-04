from python import Python
from sys import argv
import math

fn main() raises:
    let torch = Python.import_module("torch")
    let nn = torch.nn
    let F = nn.functional

# Model args
alias dim: Int = 4096
alias n_layers: Int = 32
alias n_heads = 32
alias multiple_of = 256
alias nom_eps = 1e-5

# KV cache
alias max_batch_size = 32
alias max_seq_len = 2048

@value    
struct ModelArgs:

    var n_kv_heads: Int # Optional
    var vocab_size: Int # -1, Load with tokenizer
    var ffn_dim_multiplier: Float16 # Hidde dimension
    var device: String # None

struct Transformers:
    var args: ModelArgs


    fn __init__(inout self, args: ModelArgs):
        self.args = args

