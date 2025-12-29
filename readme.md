# Grokking Modular Addition

![Grokking Phenomenon](grok.png)

## Setup

```bash
# 1. Create a new environment named 'grokking' with Python 3.10
conda create -n grokking python=3.10 -y

# 2. Activate the environment
conda activate grokking

# 3. Install the requirements
pip install -r requirements.txt
```

## Grokking Timeline

This demonstrates the "grokking" phenomenon where a neural network suddenly learns to generalize after initially just memorizing the training data.

### Epoch 0 - Random Initialization

**Embedding Waves:**
![Epoch 0 - Embedding Waves](visualizations/epoch_00000/embedding_waves.png)

**Grokking Dashboard:**
![Epoch 0 - Dashboard](visualizations/epoch_00000/grokking_dashboard.png)

At initialization, embeddings are random with no structure. All frequencies show scattered, noisy patterns.

---

### Epoch 200 - Early Training

**Embedding Waves:**
![Epoch 200 - Embedding Waves](visualizations/epoch_00200/embedding_waves.png)

**Grokking Dashboard:**
![Epoch 200 - Dashboard](visualizations/epoch_00200/grokking_dashboard.png)

Still mostly random, model begins memorizing training examples.

---

### Epoch 1,000 - Memorization Phase

**Embedding Waves:**
![Epoch 1000 - Embedding Waves](visualizations/epoch_01000/embedding_waves.png)

**Grokking Dashboard:**
![Epoch 1000 - Dashboard](visualizations/epoch_01000/grokking_dashboard.png)

Training accuracy high, test accuracy still low - classic overfitting/memorization.

---

### Epoch 4,000 - Mid Training

**Embedding Waves:**
![Epoch 4000 - Embedding Waves](visualizations/epoch_04000/embedding_waves.png)

**Grokking Dashboard:**
![Epoch 4000 - Dashboard](visualizations/epoch_04000/grokking_dashboard.png)

---

### Epoch 8,000 - Before Grokking

**Embedding Waves:**
![Epoch 8000 - Embedding Waves](visualizations/epoch_08000/embedding_waves.png)

**Grokking Dashboard:**
![Epoch 8000 - Dashboard](visualizations/epoch_08000/grokking_dashboard.png)

---

### Epoch 10,000 - Grokking Begins

**Embedding Waves:**
![Epoch 10000 - Embedding Waves](visualizations/epoch_10000/embedding_waves.png)

**Grokking Dashboard:**
![Epoch 10000 - Dashboard](visualizations/epoch_10000/grokking_dashboard.png)

Test accuracy begins to rise rapidly - grokking is happening!

---

### Epoch 14,000 - During Grokking

**Embedding Waves:**
![Epoch 14000 - Embedding Waves](visualizations/epoch_14000/embedding_waves.png)

**Grokking Dashboard:**
![Epoch 14000 - Dashboard](visualizations/epoch_14000/grokking_dashboard.png)

Circular patterns emerge in top row, showing the model discovering Fourier structure.

---

### Epoch 20,000 - Near Complete

**Embedding Waves:**
![Epoch 20000 - Embedding Waves](visualizations/epoch_20000/embedding_waves.png)

**Grokking Dashboard:**
![Epoch 20000 - Dashboard](visualizations/epoch_20000/grokking_dashboard.png)

---

### Epoch 21,800 - Fully Grokked

**Embedding Waves:**
![Epoch 21800 - Embedding Waves](visualizations/epoch_21800/embedding_waves.png)

**Grokking Dashboard:**
![Epoch 21800 - Dashboard](visualizations/epoch_21800/grokking_dashboard.png)

Perfect generalization achieved! Notice:
- **Top row (circles):** Clean circular patterns showing the model learned modular arithmetic structure
- **Middle/Bottom rows (waves):** Values cluster tightly around 0, meaning the model eliminated unnecessary frequencies
- **Bottom accuracy plot:** Test accuracy reaches ~100%, matching training accuracy

## Key Observations

1. **Random → Structured**: Embeddings transform from random noise to organized Fourier patterns
2. **Frequency Selection**: Model automatically discovers which frequencies are needed for mod 113 arithmetic
3. **Sin/Cos ≈ 0**: For unhelpful frequencies, values collapse to zero (model "turns off" those patterns)
4. **Circular Topology**: Top row shows emergence of circular structure, fundamental to modular arithmetic
5. **Delayed Generalization**: Training accuracy rises quickly (memorization), test accuracy rises much later (understanding)
