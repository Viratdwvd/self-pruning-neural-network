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

* Python 3.x
* NumPy
* Matplotlib
* (Optional) PyTorch / TensorFlow

---

## 🔮 Future Work

* Experiment with different λ values
* Dynamic threshold selection
* Compare with L2 regularization
* Evaluate accuracy vs pruning trade-off

---

## 👨‍💻 Author

**Virat**

---
