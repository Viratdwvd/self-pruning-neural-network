# Case study – AI Engineer  
Submitted by Virat Dwivedi 22MIA1101 vrtdwvd@gmail.com
# Self-Pruning Neural Network

This project implements a neural network that can prune its own weights during training. Instead of removing weights after training, the model learns which connections are important and which are not while it is being trained.

---

## Idea

Each weight in the network is associated with a learnable **gate value**.

* Gate values are passed through a sigmoid → so they stay between 0 and 1
* The effective weight used by the model becomes:

```
effective_weight = weight × sigmoid(gate_score)
```

If the gate value becomes very small (close to 0), that weight effectively stops contributing — meaning it is pruned.

---

## Loss Function

The model is trained using a combination of classification loss and sparsity penalty:

```
Total Loss = CrossEntropyLoss + λ × sum(sigmoid(gate_scores))
```

The second term (L1 penalty) pushes gate values toward zero, which encourages the network to remove unnecessary weights.

---

## Why L1 Encourages Sparsity

L1 regularization penalizes the sum of gate values. Since all gates lie between 0 and 1, minimizing this term pushes many of them toward zero.

* Important weights receive strong gradients from classification loss → gates stay high
* Unimportant weights receive weak gradients → L1 pushes their gates toward zero

This naturally separates weights into:

* active (important)
* pruned (unimportant)

---

## Results

The model was trained on CIFAR-10 using three different values of λ:

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
| ---------- | ----------------- | ------------ |
| 1e-7       | 58.6              | 0.0          |
| 5e-7       | 57.9              | 2.8          |
| 2e-6       | 57.1              | 31.6         |

Sparsity is calculated as the percentage of weights whose gate value is below **1e-2**.

---

## Observations

* At very low λ (1e-7), there is no pruning — the network behaves like a standard model
* As λ increases, more weights are pruned
* Even after removing ~31% of weights, the accuracy drops only slightly

This shows that the model contains redundant parameters and can simplify itself without a large loss in performance.

---

## Best Trade-off

Among the tested values, **λ = 2e-6** provides the best balance.

It achieves:

* noticeable sparsity (31.6%)
* only a small drop in accuracy

---

## Gate Distribution

The distribution of gate values helps visualize pruning:

* For low λ → most gates are close to 1 (no pruning)
* For higher λ → a clear spike appears near 0

This spike represents pruned weights, while the remaining values correspond to important active connections.

---

## Model Architecture

```
Input (32×32×3)
 → Flatten
 → PrunableLinear(3072 → 1024)
 → ReLU + BatchNorm + Dropout
 → PrunableLinear(1024 → 512)
 → ReLU + BatchNorm + Dropout
 → PrunableLinear(512 → 256)
 → ReLU + BatchNorm
 → PrunableLinear(256 → 10)
```

All layers use the custom `PrunableLinear` module with learnable gate parameters.

---

## Project Structure

```
worked2_31_5.ipynb        # full implementation and training
lambda_tradeoff.png       # accuracy vs sparsity plot
gate_distributions.png    # gate value distributions
REPORT.md / README.md     # explanation and results
```

---

## How to Run

Install dependencies:

```
pip install torch torchvision matplotlib numpy
```

Then run the notebook or script to reproduce results.

---

## Conclusion

This experiment shows that adding learnable gates with L1 regularization allows a neural network to prune itself during training.

The λ parameter directly controls the trade-off:

* lower λ → better accuracy, less pruning
* higher λ → more pruning, slight accuracy drop

Overall, the model is able to remove unnecessary weights while maintaining reasonable performance, which is the goal of self-pruning.
