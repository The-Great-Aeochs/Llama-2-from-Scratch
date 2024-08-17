import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 #no. of heads for the queries
    n_kv_heads: Optional[int] = None #no. of heads for the K and V
    vocab_size: int = -1 #this will be set when we load the tokenizer
    multiple_of: int = 256 # these two parameters are for feed forward layer(ffl)
    #with grouped query attention they reduced the number of heads for k and v but increased the number of params for ffl to keep the no. of params same for classical transformer and llama
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    #for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = None

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0,"Dimension must be divisible by 2"  #as mentioned in the paper
    # Build the theta params
    # According to the formula theta_i = 10000 ^ (-2(i-1)/dim) for i = [1, 2, ...dim/2]
    # Shape: (Head_Dim / 2) #as mentioned in the paper (RoFormer)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim/2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # Construct the positions (the "m" param), basically calculate all the possible theta and m our model will see, and that info is present in Seq_Len
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device = device)
    # Multiply each theta by each position using the outer product (basically saare possible combinations)
    # Shape: (Seq_Len) outer_product* (Head_Dim/2) -> (Seq_Len, Head_Dim/2)
    freqs = torch.outer(m, theta).float()

    # complex number in polar form -> c = R * exp(1 * m * theta), where R = 1
    # (Seq_Len, Head_Dim/2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

class Transformer(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab.size != -1, "vocab suze must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.n_layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps = args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len*2, device = self.args.device)
    
    def forward(self, tokens: torch.Tensor, start_pos: int):
        #(B, Seq_Len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "only one token at a time can be processed"
        #1 token at the time of input kyunki, hum KV caching technique ka use kar rhe hai, to purane tokens ki zaroorat nahi hai

        # (B, Seq_len) -> (B, Seq_len, Dim) converting tokens to embeddings
        h = self.tok_embeddings(tokens) 

        #Retrieve the pairs (m, theta) corresponding to the positions (start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        #consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()

        
