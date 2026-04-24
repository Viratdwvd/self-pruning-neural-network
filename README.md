# 📊 Gate Value Distribution Analysis for Model Pruning

## 🔍 Overview

This project analyzes the distribution of gate values produced by a neural network trained with **L1 regularization (λ = 1e-06)**. The primary goal is to study **sparsity behavior** and evaluate how effectively the model can be **pruned using a threshold-based approach**.

---

## 🎯 Objectives

* Understand how L1 regularization induces sparsity
* Visualize gate value distribution using histogram
* Identify an effective pruning threshold
* Analyze model compression potential

---

## 🧠 Key Concepts

### ✔️ Gate Values

* Output of a **sigmoid activation function**
* Range: **0 to 1**
* Interpretation:

  * **Near 0 → less important (can prune)**
  * **Near 1 → important (retain)**

---

### ✔️ L1 Regularization

* Adds penalty on absolute weights:

  `Loss = Original Loss + λ Σ|w|`

* Effect:

  * Pushes values toward **zero**
  * Creates **sparse models**

---

### ✔️ Pruning

* Removes parameters with low importance
* Threshold used in this project: **0.01**

---

## ⚙️ Methodology

1. Train model with **L1 regularization (λ = 1e-06)**
2. Extract gate values from trained model
3. Plot histogram of gate value distribution
4. Apply pruning threshold (0.01)
5. Analyze sparsity and pruning potential

---

## 📈 Output Analysis

* The distribution is **heavily skewed toward 0**
* Majority of gate values lie in the range:

  * **0 to 0.05**
* Very few values are close to **1**

### 🔑 Key Insight

L1 regularization successfully forces most parameters to become **insignificant (near zero)**, making them ideal candidates for pruning.

---

## ✂️ Pruning Interpretation

* **Threshold = 0.01**

  * Values below → pruned
  * Values above → retained

👉 This results in:

* High parameter reduction
* Efficient model compression
* Faster inference

---

## 📊 Results Summary

| Metric             | Observation               |
| ------------------ | ------------------------- |
| Regularization     | L1 (λ = 1e-06)            |
| Distribution Shape | Highly skewed toward zero |
| Threshold          | 0.01                      |
| Sparsity           | High                      |
| Pruning Potential  | Significant               |
![Graph](lambda_tradeoff%20(2).png)
Interpretation:

As λ increases, sparsity rises (0% → ~31%), while accuracy drops slightly (~57–58%).
The slope is shallow → pruning pressure is weak in this phase.
The model begins to identify redundant weights, but most parameters remain active.
Conclusion: L1 alone (without control) initiates pruning but doesn’t enforce strong sparsity.
![Graph](gate_distributions%20(2).png)
Interpretation:

λ = 1e-07: Gates cluster near 1 → no pruning (all connections active).
λ = 5e-07: Slight spread; a few values cross threshold → minimal pruning (~2–3%).
λ = 2e-06: Emerging spike near 0 plus a cluster near 1 → partial separation, ~31% sparsity.
Many gates still lie in the mid-range (0.1–0.8) → uncertain importance, leading to inefficient pruning.
Conclusion: The model starts separating important vs. unimportant weights, but not decisively.
![Graph](best_model_gate_distribution.png)
Interpretation:

A large spike at ~0 indicates a high proportion of pruned weights.
Very few values in the mid-range; remaining gates cluster well above the threshold (0.01).
This shows clear, binary-like behavior:
Near 0 → remove
Above threshold → keep
Implies high-quality pruning: decisive gate learning, minimal ambiguity, and strong sparsity with controlled accuracy loss.
Conclusion: With refinement (annealing + warm-up), the model achieves clean separation and effective compression.
---

## ✅ Conclusion

L1 regularization effectively induces sparsity in gate values, as seen from the distribution. A large number of parameters fall below the pruning threshold, enabling significant model compression without major performance degradation. This validates that **gate-based pruning combined with L1 regularization is an efficient optimization strategy**.

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install numpy matplotlib
```

### 2. Run Notebook

```bash
jupyter notebook
```

Open the `.ipynb` file and execute all cells.

---

## 📂 Project Structure

```
├── notebook.ipynb        # Implementation
├── output_graph.png      # Gate distribution visualization
├── README.md             # Project documentation
```

---

## 🔧 Requirements

* Python 3.x Google Collab
* NumPy
* Matplotlib
* (Optional) PyTorch / TensorFlow
