import torch
import os
from app import m, stoi, itos, load_checkpoint

def encode_prompt(prompt):
    return torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0)

checkpoint = load_checkpoint()

prompt = "The quick brown fox jumps over the lazy dog."

idx = encode_prompt(prompt)
print(prompt, end='')

for token in m.generate(idx):
    print(itos[token], end='', flush=True)