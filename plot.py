import pandas as pd
import matplotlib.pyplot as plt

# 1. Read the log file
LOG_FILE = "grokking_log.csv"

try:
    df = pd.read_csv(LOG_FILE)
except FileNotFoundError:
    print(f"Error: Could not find {LOG_FILE}. Run the training script first!")
    exit()

# 2. Setup the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# 3. Plot Accuracy (Left Y-Axis)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy', color='black', fontsize=12)
ax1.plot(df['Epoch'], df['Train_Acc'], label='Train Accuracy', color='blue', linestyle='--')
ax1.plot(df['Epoch'], df['Test_Acc'], label='Test Accuracy', color='green', linewidth=2)
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim(-0.05, 1.05)

# 4. Plot Loss (Right Y-Axis)
ax2 = ax1.twinx()
ax2.set_ylabel('Test Loss', color='red', fontsize=12)
ax2.plot(df['Epoch'], df['Test_Loss'], label='Test Loss', color='red', alpha=0.3)
ax2.tick_params(axis='y', labelcolor='red')

# 5. Styling
plt.title('Grokking Phase Transition', fontsize=14)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Show the plot
plt.show()