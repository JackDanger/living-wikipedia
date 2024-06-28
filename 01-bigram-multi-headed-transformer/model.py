import os
import io
from collections import defaultdict
import torch
import torch.nn as nn
import tiktoken
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split


ENV_SOURCE_DATA = os.environ['SOURCE_DATA']

torch.set_default_device('cuda')

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.key   = nn.Linear(c.n_embd, c.n_embd // c.n_head, bias=False)
        self.query = nn.Linear(c.n_embd, c.n_embd // c.n_head, bias=False)
        self.value = nn.Linear(c.n_embd, c.n_embd // c.n_head, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(c.block_size, c.block_size)))

        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)
        # computes attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (C // self.c.n_head)**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parralel """
    def __init__(self, c):
        super().__init__()
        self.heads = nn.ModuleList([Head(c) for _ in range(c.n_head)])
        self.proj = nn.Linear(c.n_embd, c.n_embd)

    def forward (self, x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c.n_embd, 4 * c.n_embd),
            nn.ReLU(),
            nn.Linear(4 * c.n_embd, c.n_embd),
            nn.Dropout(c.dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, c):
        super().__init__()
        self.sa = MultiHeadAttention(c)
        self.ffwd = FeedForward(c)
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.ln2 = nn.LayerNorm(c.n_embd)

    def forward(self, x):
        ln = self.ln1(x)
        x = x + self.sa(ln)
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.token_embdedding_table = nn.Embedding(c.vocab_size, c.n_embd)
        self.positional_embedding_table = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.Sequential(*[Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd) # final layer norm
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embdedding_table(idx) # (B, T, C)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, c.vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx=None):
        self.eval()
        if idx is None:
            idx = torch.zeros((1, 1), dtype=torch.long)
        # idx is (B, T)
        while True:
            # Truncate the sequence to the last c.block_size tokens
            idx_cond = idx[:, -self.c.block_size:] # (B, T)
            # get the logits for the next token
            logits, loss = self(idx_cond)
            # get the last token logits
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample the next token
            idx_next = torch.multinomial(probs, num_samples=1) #( B, 1)
            yield idx_next.item()
            # append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

### Hyperparameters
torch.manual_seed(1337)

config = Config(
    block_size=128,
    batch_size=64,
    dropout=0.2,
    n_head=16,
    n_layer=12,
    n_embd=128,
    vocab_size=None,
    learning_rate=1e-4,
)
config.n_embd = 32 * config.n_head


### Dataset

save_path = os.path.dirname(os.path.realpath(__file__))
data_source = os.path.join(save_path, ENV_SOURCE_DATA)
data_source_name = os.path.basename(data_source)


class FileSeekDataset(Dataset):
    def __init__(self, file_path, stoi, block_size):
        self.file_path = file_path
        self.stoi = stoi
        self.block_size = block_size

        self.vocab_size = len(stoi)
        self.file_size = os.path.getsize(file_path)
        self.num_blocks = (self.file_size + block_size - 1) // block_size

        self.bytes_read = 0

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx, forced_seek=None):
        # Sometimes we have to be able to read from a specific location
        if forced_seek is None:
            seek = idx * self.block_size
        else:
            seek = forced_seek

        try:
            with open(self.file_path, 'r') as f:
                f.seek(seek)
                # Read (at least) an extra byte because we'll shift this data by one
                # to manufacture a target set
                text = f.read(self.block_size + 1)
                self.bytes_read += self.block_size
                encoded = [self.stoi[ch] for ch in text if ch in self.stoi]
                return torch.tensor(encoded, dtype=torch.long, device='cuda')
        except UnicodeDecodeError as e:
            # We started reading from the middle of a multi-byte character
            # back up and try again
            return self.__getitem__(idx, forced_seek=seek-1)



def collate_fn(batch):
    """
    Turn the batch of data (which are each deliberately one longer than the
    block_size) into two tensors, the target shifted one step right of the input
    """
    flattened = torch.tensor([item for sublist in batch for item in sublist])
    # trim any amount that is more than a multiple of block_size
    batch = batch[:len(batch) - len(batch) % config.block_size]

    # Make this an actual batch of sequences again
    batch = flattened.reshape((config.batch_size, len(flattened) // config.batch_size))
    data = batch[:, :-1].contiguous() # All except the last element
    target = batch[:, 1:].contiguous() # All except the first element

    return data, target


# Calculate and cache the vocab just once
vocab_file = os.path.join(save_path, f"{data_source_name}.vocab.txt")
if not os.path.exists(vocab_file):
    chars = set()
    with open(data_source) as f:
        while chunk := f.read(1024):
            chars.update(chunk)
    with open(vocab_file, 'w') as f:
        f.write(''.join(sorted(chars)))
chars = open(vocab_file).read()
vocab_size = len(chars)
config.vocab_size = vocab_size


stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


# Dataset and DataLoader
train_ratio = 0.8
val_ratio = 0.2

dataset = FileSeekDataset(data_source, stoi, config.block_size)

# Calculate lengths for training and validation sets
total_length = len(dataset)
train_length = int(train_ratio * total_length)
val_length = total_length - train_length

generator = torch.Generator(device='cuda').manual_seed(1337)
train_dataset, val_dataset = random_split(dataset, [train_length, val_length], generator=generator)
train_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
val_data   = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

m = BigramLanguageModel(config)

optimizer = torch.optim.AdamW(m.parameters(), lr=config.learning_rate)


# Checkpointing with datasource as part of the filename because I test on small
# datasets and then forget to clear this when switching to the larger
ckpt_path = os.path.join(save_path, f"{data_source_name}.ckpt.pt")

def load_checkpoint():
    if not os.path.exists(ckpt_path):
        print("No checkpoint found, starting from scratch")
        return {}

    print("loading checkpoint from", ckpt_path)
    checkpoint = torch.load(ckpt_path)
    m.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


def save_checkpoint(checkpoint):
    print(f"saving checkpoint to {ckpt_path}")
    torch.save(checkpoint, ckpt_path)
