## Key Observations

- Increasing λ leads to higher sparsity but lower accuracy, demonstrating a clear trade-off.

- The increase in sparsity is non-linear:
  - A large jump occurs between λ = 1e-8 and λ = 1e-4.
  - Further increase to λ = 0.1 yields only marginal gains.

- The model is able to retain reasonable performance even after pruning ~85% of connections, showing significant redundancy in the network.

- Beyond a certain point, increasing λ results in diminishing returns in sparsity but a sharp decline in accuracy.

## Why L1 Penalty on Sigmoid Gates Encourages Sparsity

In this model, each weight has an associated learnable gate parameter. These gate scores are passed through a sigmoid function to produce values between 0 and 1, which scale the corresponding weights. Applying an L1 penalty (sum of absolute values) on these gate values encourages sparsity because L1 regularization drives values toward zero.

Since the gates determine whether a connection contributes to the output, minimizing the L1 norm penalizes the model for keeping too many connections active. As a result, many gate values move closer to zero, reducing the effect of their corresponding weights. This leads to a sparse network where only the most important connections remain active.

---

## Results Summary

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|------------|------------------|--------------------|
| 1e-8       | 54.56            | 8.57               |
| 1e-4       | 50.18            | 84.97              |
| 0.1        | 36.85            | 88.01              |

---

## Analysis of Sparsity vs Accuracy Trade-off

The results show a clear trade-off between sparsity and accuracy:

- **λ = 1e-8**:
  - Sparsity is low (8.57%), so most connections remain active.
  - The model behaves like a dense network and achieves the highest accuracy.

- **λ = 1e-4**:
  - Sparsity increases significantly to 84.97%, while accuracy remains reasonably high (50.18%).
  - The model removes many less important connections while preserving performance.

- **λ = 0.1**:
  - Sparsity increases slightly to 88.01%, but accuracy drops to 36.85%.
  - Excessive regularization removes important connections, leading to underfitting.

A moderate value of λ provides the best balance between sparsity and accuracy.

---

## Gate Distribution Plot (Best Model)

The plot for the best-performing model (λ = 1e-4) shows:

- A large concentration of values near 0, indicating many pruned connections.
- A separate group of values away from 0, representing active and important connections.

This distribution shows that the model distinguishes between useful and less useful weights.

*(Insert plot here)*

---

## Conclusion

The model is able to learn a sparse structure by penalizing active connections during training. Increasing λ increases sparsity but reduces accuracy. A moderate value achieves a balance between model size and performance.