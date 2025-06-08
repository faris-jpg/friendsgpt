import torch
import torch.nn as nn
from torch.nn import functional as F
import os

block_size = 256
batch_size = 64
max_iters = 15000            # ~60 epochs
eval_interval = 750          # Evaluate every ~5 epochs
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_emb = 384
n_layer = 6
n_head = 6
dropout = 0.2
temperature = 0.8

torch.manual_seed(2408)

# with open('combined.txt', 'r', encoding='utf-8') as f:
#     text = f.read()
text = None
model = None
    
chars = ['\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ']', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '}', 'É', 'ç', 'è', 'é', 'í', '–', '—', '‘', '’', '“', '”', '…']
vocab_size = len(chars)

#creating a map for char to int and vv
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] #encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #decoder: take a list of integers, output a string
if text:
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

def get_batch(split):
    #generates a small batch of input and target
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) #random offsets
    
    #get a 4x8 tensor
    x = torch.stack([data[i:i+block_size] for i in ix]) 
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)  #move to device
    
    return x, y

#function to estimate loss
@torch.no_grad()
def estimate_loss():
    model.eval()  #set the model to evaluation mode
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            # x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  #set the model back to training mode
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout) 
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # apply causal mask
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)  # apply dropout to attention weights
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_emb, n_emb)  # projection to the original embedding size
        self.dropout = nn.Dropout(dropout)  

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # concatenate the outputs of all heads   
        out = self.proj(out)  # project back to the original embedding size
        out = self.dropout(out)
        return out  # return the concatenated output
    
class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.GELU(), 
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout)  # dropout for regularization    
            )

    def forward(self, x):
        return self.net(x)  # apply the feedforward network

class Block(nn.Module):
    def __init__(self, n_emb, n_head):
        super().__init__()
        heead_size = n_emb // n_head
        self.sa = MultiHeadAttention(n_head, heead_size)  # self-attention
        self.ffwd = FeedForward(n_emb)  # feedforward network
        self.ln1 = nn.LayerNorm(n_emb) 
        self.ln2 = nn.LayerNorm(n_emb)  

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # apply self-attention
        x = x + self.ffwd(self.ln2(x))  # apply feedforward network
        return x
    
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_emb, n_layer, n_head):
        super().__init__()
        self.block_size = block_size
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.pos_embedding_table = nn.Embedding(block_size, n_emb)  # positional embeddings
        self.blocks = nn.Sequential(*[Block(n_emb, n_head=n_head) for _ in range(n_layer)])  # stack of transformer blocks
        self.ln_f = nn.LayerNorm(n_emb)  
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # B is batch size, T is block size

        #idx and targets are both (B,T) tensor of integers
        tkn_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tkn_emb + pos_emb  
        x = self.blocks(x)  # pass through the transformer blocks
        x = self.ln_f(x)  # apply layer normalization
        
        logits = self.lm_head(x) # (B,T,vc size)
        #logits are the unnormalized log probabilities of the next token

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) #quality of logits based on targets

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
            # idx is (B, T) array of indices in the current context
            for _ in range(max_new_tokens):
                # cut the context to the last block_size tokens
                # Ensure 'self.block_size' is used if it's an instance attribute
                idx_cond = idx[:, -block_size:] if hasattr(self, 'block_size') else idx # (B, T) get the last block_size tokens
                
                # get the predictions (logits)
                logits, _ = self(idx_cond) # Assuming self(idx_cond) returns (logits, loss)
                
                # focus only on the last time step
                logits = logits[:, -1, :] # becomes (B, C)
                
                # apply softmax to get probabilities
                # Ensure 'temperature' is accessible, e.g., passed as an arg or self.temperature
                probs = F.softmax(logits / temperature, dim=-1) # Or self.temperature if it's an attribute
                
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
                
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
                
                # YIELD the *decoded* new token immediately
                # The 'decode' function should convert a list of single token IDs to a string
                yield decode([idx_next.item()]) # .item() gets the Python number from a 0-dim tensor
class Interface:
    def __init__(self, ver: str = 'latest'):
        # Pass the hyperparameters from the global scope to the model constructor
        self.model = BigramLanguageModel(vocab_size, block_size, n_emb, n_layer, n_head).to(device)
        self.load_model(ver)

    def load_model(self, ver: str):
        if ver == 'latest':
            file_name = 'model.pt'
        elif ver == 'best':
            file_name = 'checkpoint.pt'
        elif ver == '1mb':
            file_name = '1mb_model.pt'
        else:
            print(f"Warning: Invalid model version '{ver}'. Using 'latest' by default.")
            file_name = 'model.pt'

        model_path = os.path.join('models', file_name)

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model '{model_path}' loaded successfully.")
        except FileNotFoundError:
            print(f"No saved model '{model_path}' found. Starting with a new, untrained model.")
        except Exception as e:
            print(f"Error loading model '{model_path}': {e}. Starting with a new, untrained model.")

    def generate_text(self, prompt: str, max_new_tokens: int = 100):
        self.model.eval()

        # Initial prompt encoding
        idx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

        # Yield the initial prompt characters first
        current_output = decode(idx[0].tolist())
        yield current_output

        # Generate new tokens and yield accumulated output
        for new_token_char in self.model.generate(idx, max_new_tokens):
            current_output += new_token_char
            yield current_output