import torch
import sys
from pathlib import Path
from visualize import visualize_grokking_dashboard, create_viz_dir

# Import model architecture (assuming it's in train.py)
sys.path.insert(0, str(Path(__file__).parent))
from train import ModularAdditionTransformer

# Key epochs to regenerate
epochs_to_regenerate = [0, 200, 1000, 4000, 8000, 9400, 10000, 14000, 20000, 21800]

P = 113

for epoch in epochs_to_regenerate:
    checkpoint_path = Path(f"checkpoints/model_epoch_{epoch:05d}.pt")

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        continue

    print(f"\n📊 Regenerating dashboard for epoch {epoch}...")

    # Load model
    model = ModularAdditionTransformer(vocab_size=P+1, d_model=128, nhead=4,
                                      num_layers=2, dim_feedforward=512)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create viz directory
    viz_dir = create_viz_dir(epoch)

    # Regenerate dashboard
    try:
        visualize_grokking_dashboard(model, epoch, viz_dir, P=P)
        print(f"✅ Dashboard generated: {viz_dir}/grokking_dashboard.png")
    except Exception as e:
        print(f"❌ Failed to generate dashboard: {e}")
        import traceback
        traceback.print_exc()

print("\n🎉 Done!")
