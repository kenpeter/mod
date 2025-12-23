import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
from pathlib import Path

def create_viz_dir(epoch):
    """Create directory for saving visualizations"""
    viz_dir = Path(f"visualizations/epoch_{epoch:05d}")
    viz_dir.mkdir(parents=True, exist_ok=True)
    return viz_dir

def visualize_embedding_waves(model, epoch, viz_dir, P=113):
    """
    Original sin/cos wave visualization showing embedding projections
    """
    W_E = model.token_embedding.weight.data[:P].detach().cpu().numpy()

    key_freqs = [14, 35, 41, 42, 52]

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.3)

    for idx, k in enumerate(key_freqs):
        w_k = 2 * np.pi * k / P

        # Project embeddings onto this frequency
        cos_component = np.zeros(P)
        sin_component = np.zeros(P)

        for i in range(P):
            # Project embedding onto cos/sin basis
            cos_basis = np.cos(w_k * np.arange(P))
            sin_basis = np.sin(w_k * np.arange(P))

            cos_component[i] = W_E[i] @ W_E.T @ cos_basis / P
            sin_component[i] = W_E[i] @ W_E.T @ sin_basis / P

        # Top row: circular patterns
        ax = fig.add_subplot(gs[0, idx])
        scatter = ax.scatter(cos_component, sin_component, c=np.arange(P),
                           cmap='hsv', s=30, alpha=0.7)
        ax.set_title(f'Freq k={k}\nCircle', fontsize=10)
        ax.set_xlabel('cos')
        ax.set_ylabel('sin')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Middle row: cos wave
        ax = fig.add_subplot(gs[1, idx])
        ax.plot(np.arange(P), np.cos(w_k * np.arange(P)), 'b-', alpha=0.3, label='Expected')
        ax.scatter(np.arange(P), cos_component, c=np.arange(P), cmap='hsv', s=10, alpha=0.7)
        ax.set_title(f'cos({k}·2π·n/{P})', fontsize=9)
        ax.set_xlabel('Token n')
        ax.set_ylabel('cos')
        ax.grid(True, alpha=0.3)

        # Bottom row: sin wave
        ax = fig.add_subplot(gs[2, idx])
        ax.plot(np.arange(P), np.sin(w_k * np.arange(P)), 'r-', alpha=0.3, label='Expected')
        ax.scatter(np.arange(P), sin_component, c=np.arange(P), cmap='hsv', s=10, alpha=0.7)
        ax.set_title(f'sin({k}·2π·n/{P})', fontsize=9)
        ax.set_xlabel('Token n')
        ax.set_ylabel('sin')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Embedding Sin/Cos Patterns (Epoch {epoch})', fontsize=16, y=0.995)
    plt.savefig(viz_dir / 'embedding_waves.png', dpi=150, bbox_inches='tight')
    plt.close()

def visualize_layer_interactions(model, epoch, viz_dir, P=113):
    """
    Grid showing model's internal representations for different token pairs
    This CHANGES during training from random to structured patterns!
    """
    model.eval()
    device = next(model.parameters()).device

    grid_size = 8
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(24, 24))
    fig.patch.set_facecolor('black')

    with torch.no_grad():
        for i in range(grid_size):
            for j in range(grid_size):
                ax = axes[i, j]
                ax.set_facecolor('black')
                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.2, 1.2)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('equal')

                # Diagonal: identity (a=constant, b varies)
                if i == j:
                    t = np.linspace(-1, 1, 100)
                    colors = plt.cm.hsv(np.linspace(0, 1, 100))
                    for idx in range(len(t)-1):
                        ax.plot([t[idx], t[idx+1]], [t[idx], t[idx+1]],
                               color=colors[idx], linewidth=3, alpha=0.8)

                else:
                    # Off-diagonal: visualize how model represents (a,b) pairs
                    # Use actual MLP activations which evolve during training
                    x_coords = []
                    y_coords = []

                    # Pick a neuron based on grid position
                    neuron_idx = (i * grid_size + j) * 4 % 512

                    # Sample tokens based on grid position
                    sample_step = max(1, P // 15)

                    for a_idx, a in enumerate(range(0, P, sample_step)):
                        for b_idx, b in enumerate(range(0, P, sample_step)):
                            # Get model's internal representation
                            x_val = torch.tensor([a, b, P], device=device)
                            h = model.token_embedding(x_val) + model.pos_embedding

                            # Through attention
                            attn_out, _ = model.attn(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0), need_weights=False)
                            h = h.unsqueeze(0) + attn_out

                            # MLP activation (this changes dramatically during training!)
                            mlp_input = h[:, -1, :]
                            mlp_hidden = torch.relu(model.mlp[0](mlp_input))

                            # Project to 2D using two different neurons
                            x = mlp_hidden[0, neuron_idx].item()
                            y = mlp_hidden[0, (neuron_idx + 1) % 512].item()

                            x_coords.append(x)
                            y_coords.append(y)

                    if len(x_coords) > 0:
                        x_coords = np.array(x_coords)
                        y_coords = np.array(y_coords)

                        # Normalize
                        if np.std(x_coords) > 1e-6:
                            x_coords = (x_coords - np.mean(x_coords)) / (np.std(x_coords) + 1e-6)
                        if np.std(y_coords) > 1e-6:
                            y_coords = (y_coords - np.mean(y_coords)) / (np.std(y_coords) + 1e-6)

                        # Clip
                        x_coords = np.clip(x_coords, -1.1, 1.1)
                        y_coords = np.clip(y_coords, -1.1, 1.1)

                        # Color by pattern
                        colors = plt.cm.hsv(np.random.rand(len(x_coords)))
                        ax.scatter(x_coords, y_coords, c=colors, s=12, alpha=0.7)

                for spine in ax.spines.values():
                    spine.set_visible(False)

    plt.tight_layout(pad=0.5)
    plt.savefig(viz_dir / 'layer_interactions.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()

def generate_all_visualizations(model, epoch, test_x, test_y, P=113):
    """Generate BOTH visualizations"""
    viz_dir = create_viz_dir(epoch)

    print(f"  Generating visualizations for epoch {epoch}...")

    try:
        visualize_embedding_waves(model, epoch, viz_dir, P)
        print(f"    ✓ Embedding Waves (Sin/Cos)")
    except Exception as e:
        print(f"    ✗ Embedding Waves failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        visualize_layer_interactions(model, epoch, viz_dir, P)
        print(f"    ✓ Layer Interactions Grid")
    except Exception as e:
        print(f"    ✗ Layer Interactions failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"  Saved to {viz_dir}/")
    return viz_dir
