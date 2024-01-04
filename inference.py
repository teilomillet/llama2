from typing import Optional
from pathlib import Path
from sentencepiece import SentencePieceProcessor

from tqdm import tqdm 

import torch
import time
import json


from model import ModelArgs, Transformers

class LLaMA:
    def __init__(self, model: Transformers, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer  # Corrected
        self.args = model_args
    
    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob('*.pth'))
            assert len(checkpoints) > 0, f"No checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoints {ckpt_path}')
            checkpoint = torch.load(ckpt_path, map_location="cpu")  # Corrected map_location
            print(f'Loaded checkpoints in {time.time() - prev_time:.2f}s')  # Corrected format
            prev_time = time.time()
            
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())  # Simplified file reading
        
        
        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device, 
            **params
        )
        
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()
        
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfStorage)  # Adjusted tensor type
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)  # Adjusted tensor type
            
        model = Transformers(model_args).to(device)
        
        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f'Loaded state dict in {time.time() - prev_time:.2f}s')  # Corrected format
            
        return LLaMA(model, tokenizer, model_args)  
if __name__ == '__main__':
    torch.manual_seed(0)
    
    allow_cuda = True
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    
    model = LLaMA.build(
        checkpoints_dir='llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device
    )
    
    print('All OK')