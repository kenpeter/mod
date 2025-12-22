import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import os

# --- 1. Hyperparameters (Optimized for CUDA) ---
P = 113             
D_MODEL = 128       
N_HEADS = 4         
D_HEAD = 32         
D_MLP = 512         
TRAIN_FRACTION = 0.3
WEIGHT_DECAY = 1.0  
LR = 1e-3
EPOCHS = 40000      
LOG_FILE = "grokking_data.csv"
SEED = 42

# Speed optimization for CUDA
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

# --- 2. Fast Data Generation ---
def get_data(p, train_fraction):
    # Generate all possible pairs (i, j)
    all_pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    labels = (all_pairs[:, 0] + all_pairs[:, 1]) % p
    
    # Add the "=" token (which is index P) to every pair
    eq_tokens = torch.full((p*p, 1), p)
    inputs = torch.cat([all_pairs, eq_tokens], dim=1).to(device)
    labels = labels.to(device)
    
    # Shuffle and split
    indices = torch.randperm(p*p)
    split = int(p*p * train_fraction)
    
    return (inputs[indices[:split]], labels[indices[:split]], 
            inputs[indices[split:]], labels[indices[split:]])

train_x, train_y, test_x, test_y = get_data(P, TRAIN_FRACTION)

# --- 3. Optimized Transformer Architecture ---
class GrokkingTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_mlp):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(3, d_model))
        
        # Using a MultiheadAttention module for speed optimizations
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Linear(d_mlp, d_model)
        )
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        # Embedding + Position
        h = self.token_embedding(x) + self.pos_embedding
        
        # Self-Attention
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = h + attn_out
        
        # MLP
        h = h + self.mlp(h)
        
        # Only take the last token output (at the "=" position)
        return self.unembed(h[:, -1, :])

model = GrokkingTransformer(P + 1, D_MODEL, N_HEADS, D_MLP).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98))
criterion = nn.CrossEntropyLoss()

# --- 4. Training Loop ---
print(f"Running on: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

with open(LOG_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Test_Loss', 'Test_Acc'])

for epoch in range(EPOCHS + 1):
    model.train()
    optimizer.zero_grad(set_to_none=True) # Slightly faster than zero_grad()
    
    logits = model(train_x)
    loss = criterion(logits, train_y)
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        model.eval()
        with torch.no_grad():
            train_acc = (logits.argmax(-1) == train_y).float().mean().item()
            test_logits = model(test_x)
            test_loss = criterion(test_logits, test_y).item()
            test_acc = (test_logits.argmax(-1) == test_y).float().mean().item()
            
            with open(LOG_FILE, mode='a', newline='') as f:
                csv.writer(f).writerow([epoch, loss.item(), train_acc, test_loss, test_acc])
                
            print(f"Epoch {epoch:5d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
            
            if test_acc > 0.999:
                print(f"Grokking complete at epoch {epoch}!")
                break