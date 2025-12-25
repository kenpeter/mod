import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import os
from visualize import generate_all_visualizations

# --- 1. Hyperparameters (Optimized for CUDA) ---
# vocab size = 113
P = 113 
# model dimension 128            
D_MODEL = 128       
N_HEADS = 4         
D_HEAD = 32         
D_MLP = 512         
TRAIN_FRACTION = 0.3
WEIGHT_DECAY = 1.0  
LR = 1e-3
# running 40,000 epoch
EPOCHS = 40000      
LOG_FILE = "grokking_data.csv"
SEED = 42

# Speed optimization for CUDA
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

# --- 2. Fast Data Generation ---
def get_data(p, train_fraction):
    # all apir
    """
        [0, 0]
        [0, 1]
        [0, 2]
        [0, 3]
        ...
        [1, 0]
        [1, 1]
    """

    # 1. prepare a+b, prepre real a+b
    all_pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    labels = (all_pairs[:, 0] + all_pairs[:, 1]) % p
    
    """
        [113]
        [113]
        [113]
        [113]
        [113]
        [113]
        ...
    """

    # 2. prepare equal sign with 113
    eq_tokens = torch.full((p*p, 1), p)

    # 3. merge a+b with equal sign
    """
        0 0 0 113
        1 0 1 113
        2 0 2 113
        ...
    """
    inputs = torch.cat([all_pairs, eq_tokens], dim=1).to(device)
    labels = labels.to(device)
    
    # 4. split train and test, random the index
    split = int(p*p * train_fraction)
    indices = torch.randperm(p*p)
    
    # 5. form train x y, and form test x y
    return (inputs[indices[:split]], labels[indices[:split]], 
            inputs[indices[split:]], labels[indices[split:]])

train_x, train_y, test_x, test_y = get_data(P, TRAIN_FRACTION)

# --- 3. Optimized Transformer Architecture ---

# class className (nn.module)
class GrokkingTransformer(nn.Module):
    # init
    # self, vocab_size, model_dimension, n head, mlp_dimension
    def __init__(self, vocab_size, d_model, n_heads, d_mlp):
        # super init
        super().__init__()
        # token embed
        # vocab_size = 113+1, 1 is the equal sign
        # d_model = 128
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # position embed
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
        # 1. token embed + position
        h = self.token_embedding(x) + self.pos_embedding
        
        # 2. attention + residual
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = h + attn_out
        
        # 3. MLP + residual
        h = h + self.mlp(h)
        
        # Only take the last token output (at the "=" position)
        return self.unembed(h[:, -1, :])

# entire model just weights and move to device
model = GrokkingTransformer(P + 1, D_MODEL, N_HEADS, D_MLP).to(device)

# adamw
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98))
criterion = nn.CrossEntropyLoss()

# --- 4. Training Loop ---
print(f"Running on: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

# Define visualization checkpoints
VIZ_EPOCHS = [0, 200, 500, 1000, 1400, 2000, 4000, 6000, 8000, 9400, 10000, 12000, 14000, 16000, 20000, 30000, 40000]

# open the csv file
with open(LOG_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Test_Loss', 'Test_Acc'])

# epochs+1, as 0 -> epochs-1, get to the real epochs
for epoch in range(EPOCHS + 1):
    # model train mode
    model.train()
    # reset to zero, but actually set to none faster
    optimizer.zero_grad(set_to_none=True) # Slightly faster than zero_grad()
    
    # we call forward function here, so need to high D to low D
    logits = model(train_x)
    # compare prediction with label -> loss
    loss = criterion(logits, train_y)
    # cal backward
    loss.backward()
    # real backward
    optimizer.step()
    
    if epoch % 200 == 0:
        # now eval mode
        model.eval()
        with torch.no_grad():
            # train acc
            train_acc = (logits.argmax(-1) == train_y).float().mean().item()
            test_logits = model(test_x)
            test_loss = criterion(test_logits, test_y).item()
            test_acc = (test_logits.argmax(-1) == test_y).float().mean().item()

            with open(LOG_FILE, mode='a', newline='') as f:
                csv.writer(f).writerow([epoch, loss.item(), train_acc, test_loss, test_acc])

            print(f"Epoch {epoch:5d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

            # Generate visualizations at key checkpoints
            if epoch in VIZ_EPOCHS:
                generate_all_visualizations(model, epoch, test_x, test_y, P)

            if test_acc > 0.999:
                print(f"Grokking complete at epoch {epoch}!")
                # Generate final visualization
                if epoch not in VIZ_EPOCHS:
                    generate_all_visualizations(model, epoch, test_x, test_y, P)
                break