import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import csv
from pathlib import Path

def create_viz_dir(epoch):
    """Create directory for saving visualizations"""
    viz_dir = Path(f"visualizations/epoch_{epoch:05d}")
    viz_dir.mkdir(parents=True, exist_ok=True)
    return viz_dir

def visualize_embedding_waves(model, epoch, viz_dir, P=113):
    """
    Original sin/cos wave visualization showing embedding projections
    Auto-detects the top 5 most powerful frequencies using FFT
    """
    W_E = model.token_embedding.weight.data[:P].detach().cpu().numpy()

    # Auto-detect top frequencies using FFT
    fft_power = np.zeros(P)
    for d in range(W_E.shape[1]):
        fft_result = np.fft.fft(W_E[:, d])
        fft_power += np.abs(fft_result) ** 2

    # Find top 5 frequencies (excluding DC component at k=0)
    valid_freqs = np.arange(1, P // 2)
    freq_powers = fft_power[valid_freqs]
    top_freq_indices = np.argsort(freq_powers)[-5:][::-1]
    key_freqs = valid_freqs[top_freq_indices].tolist()

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
                    neuron_idx = (i * grid_size + j) * 4 % 512
                    sample_step = max(1, P // 15)

                    x_coords = []
                    y_coords = []

                    for a_idx, a in enumerate(range(0, P, sample_step)):
                        for b_idx, b in enumerate(range(0, P, sample_step)):
                            x_val = torch.tensor([a, b, P], device=device)
                            h = model.token_embedding(x_val) + model.pos_embedding

                            attn_out, _ = model.attn(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0), need_weights=False)
                            h = h.unsqueeze(0) + attn_out

                            mlp_input = h[:, -1, :]
                            mlp_hidden = torch.relu(model.mlp[0](mlp_input))

                            x = mlp_hidden[0, neuron_idx].item()
                            y = mlp_hidden[0, (neuron_idx + 1) % 512].item()

                            x_coords.append(x)
                            y_coords.append(y)

                    if len(x_coords) > 0:
                        x_coords = np.array(x_coords)
                        y_coords = np.array(y_coords)

                        if np.std(x_coords) > 1e-6:
                            x_coords = (x_coords - np.mean(x_coords)) / (np.std(x_coords) + 1e-6)
                        if np.std(y_coords) > 1e-6:
                            y_coords = (y_coords - np.mean(y_coords)) / (np.std(y_coords) + 1e-6)

                        x_coords = np.clip(x_coords, -1.1, 1.1)
                        y_coords = np.clip(y_coords, -1.1, 1.1)

                        colors = plt.cm.hsv(np.random.rand(len(x_coords)))
                        ax.scatter(x_coords, y_coords, c=colors, s=12, alpha=0.7)

                for spine in ax.spines.values():
                    spine.set_visible(False)

    plt.tight_layout(pad=0.5)
    plt.savefig(viz_dir / 'layer_interactions.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()

def visualize_grokking_dashboard(model, epoch, viz_dir, P=113, log_file='log.csv'):
    # Create dashboard visualization similar to grokking paper/videos:
    # - 8 rows showing different Fourier frequencies
    # - Column 1: Fourier projection vs token index (shows sinusoids emerging)
    # - Columns 2-7: 2D scatter plots of (cos_k, sin_k) pairs (shows circular topology)
    # - Bottom: Training vs Test accuracy curve
    
    # Get token embeddings (exclude the "=" token at index P)
    W_E = model.token_embedding.weight.data[:P].detach().cpu().numpy()
    
    # Auto-detect top frequencies using FFT
    # Compute power spectrum across all embedding dimensions
    fft_power = np.zeros(P)
    for d in range(W_E.shape[1]):
        fft_result = np.fft.fft(W_E[:, d])
        fft_power += np.abs(fft_result) ** 2
        
    # Find top 8 frequencies (excluding DC component at k=0 and symmetric halves)
    # Only consider frequencies 1 to P//2
    valid_freqs = np.arange(1, P // 2)
    freq_powers = fft_power[valid_freqs]
    top_freq_indices = np.argsort(freq_powers)[-8:][::-1]
    top_freqs = valid_freqs[top_freq_indices]
    
    # Precompute Fourier projections for all top frequencies
    fourier_data = {}
    for k in top_freqs:
        w_k = 2 * np.pi * k / P
        n = np.arange(P)
        cos_basis = np.cos(w_k * n)
        sin_basis = np.sin(w_k * n)
        
        # Project each token's embedding onto the Fourier basis
        # cos_proj[i] = W_E[i] . (W_E^T . cos_basis) / norm
        # This captures how much the embedding space aligns with this frequency
        cos_proj = W_E @ (W_E.T @ cos_basis) / P
        sin_proj = W_E @ (W_E.T @ sin_basis) / P
        fourier_data[k] = {'cos': cos_proj, 'sin': sin_proj}

    # Setup the dashboard figure
    fig = plt.figure(figsize=(24, 18))
    fig.patch.set_facecolor('#1a1a1a') # Dark background
    
    # Grid: 9 rows (8 freqs + 1 acc), ~7 columns
    gs = fig.add_gridspec(9, 8, hspace=0.4, wspace=0.3)
    
    for row_idx, k in enumerate(top_freqs):
        # Column 0-2: Waveform plots (Fourier projection vs token index)
        ax = fig.add_subplot(gs[row_idx, 0:2], facecolor='#1a1a1a')
        
        # Retrieve precomputed data
        cos_proj = fourier_data[k]['cos']
        sin_proj = fourier_data[k]['sin']
        
        # Plot waveforms
        ax.plot(np.arange(P), cos_proj, color='cyan', alpha=0.6, label='cos')
        ax.plot(np.arange(P), sin_proj, color='orange', alpha=0.6, label='sin')
        
        # Formatting (from Image 2)
        ax.axvline(x=0, color='gray', linewidth=0.3, alpha=0.3)
        max_range = max(np.abs(cos_proj).max(), np.abs(sin_proj).max())
        ax.set_xlim(-max_range, max_range) # This looks like a mistake in logic in the image (xlim usually for scatter), 
                                         # but keeping consistency: Image 2 line 242 uses max_range for xlim/ylim
        # Actually, for the waveform plot (vs index), xlim should be (0, P). 
        # The code in Image 2 lines 241-250 seems to refer to a SCATTER plot setup, 
        # but the comment says "Column 1: Fourier projection vs token index".
        # Let's trust the scatter setup in Image 2 lines 241+ which sets aspect 'equal'.
        # This suggests Column 1 might actually be the CIRCLE plot in this version?
        # Let's adjust to match Image 2 code exactly:
        ax.clear() # Reset to apply Image 2 logic strictly
        
        ax.axvline(x=0, color='gray', linewidth=0.3, alpha=0.3)
        max_range = max(np.abs(cos_proj).max(), np.abs(sin_proj).max())
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if row_idx == 0:
            ax.set_title(f'Freq {k} (Circle)', color='white', fontsize=10, pad=5)
            
        # Plot the circle (scatter)
        ax.scatter(cos_proj, sin_proj, c=np.arange(P), cmap='hsv', s=10, alpha=0.8)

        # Columns 3-7: Cross-frequency comparisons
        other_freqs = [f for f in top_freqs if f != k][:5]
        for col_idx, k2 in enumerate(other_freqs):
            ax = fig.add_subplot(gs[row_idx, col_idx + 2], facecolor='#1a1a1a')
            
            # Plot cos of freq k vs cos of freq k2
            x_vals = fourier_data[k]['cos']
            y_vals = fourier_data[k2]['cos']
            
            ax.scatter(x_vals, y_vals, c=np.arange(P), cmap='hsv', s=6, alpha=0.9)
            ax.axhline(y=0, color='gray', linewidth=0.3, alpha=0.3)
            ax.axvline(x=0, color='gray', linewidth=0.3, alpha=0.3)
            
            max_range = max(np.abs(x_vals).max(), np.abs(y_vals).max())
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Bottom row: Accuracy curve
    ax_acc = fig.add_subplot(gs[8, :], facecolor='black')
    
    # Load accuracy data from CSV
    epochs_list, train_acc_list, test_acc_list = [], [], []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ep = int(row['Epoch'])
                    if ep <= epoch: # Only show up to current epoch
                        epochs_list.append(ep)
                        train_acc_list.append(float(row['Train_Acc']))
                        test_acc_list.append(float(row['Test_Acc']))
                except (ValueError, KeyError):
                    pass
    
    if epochs_list:
        ax_acc.plot(epochs_list, train_acc_list, color='cyan', linewidth=2, label='Train Acc')
        ax_acc.plot(epochs_list, test_acc_list, color='orange', linewidth=2, label='Test Acc')
        max_epoch = max(epoch, max(epochs_list)) if epochs_list else epoch
        ax_acc.set_xlim(0, max(max_epoch * 1.05, 1))
    else:
        ax_acc.text(0.5, 0.5, 'No accuracy data yet', color='white', 
                   ha='center', va='center', transform=ax_acc.transAxes)
        
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.set_ylabel('ACCURACY', color='white', fontsize=11)
    ax_acc.tick_params(colors='white', labelsize=9)
    ax_acc.legend(loc='center right', facecolor='black', edgecolor='white', 
                 labelcolor='white', fontsize=10)
    for spine in ax_acc.spines.values():
        spine.set_color('gray')
        spine.set_linewidth(0.5)
        
    plt.savefig(viz_dir / 'grokking_dashboard.png', dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()

def generate_all_visualizations(model, epoch, test_x, test_y, P=113):
    """Generate ALL visualizations including the dashboard"""
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
        # traceback.print_exc()

    try:
        visualize_grokking_dashboard(model, epoch, viz_dir, P)
        print(f"    ✓ Grokking Dashboard")
    except Exception as e:
        print(f"    ✗ Grokking Dashboard failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"  Saved to {viz_dir}/")
    return viz_dir