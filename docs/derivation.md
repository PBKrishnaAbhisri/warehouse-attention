# Mathematical derivation: scaled dot-product self-attention

## 1. Setting and shapes
One example is a window of T = 24 hours. Each hour is a row of d_in = 5 numbers (normalised orders,
sin/cos of hour, sin/cos of weekday). B windows are processed together. In the toy task T = 9 and d_in = 26.

| Tensor | Shape | Meaning |
|---|---|---|
| X | (B, 24, 5) | input window |
| W_Q, W_K | (5, d_k) | learnable projections (d_k = 16) |
| W_V | (5, d_v) | learnable projection (d_v = 16) |
| Q = X W_Q, K = X W_K | (B, 24, d_k) | queries and keys |
| V = X W_V | (B, 24, d_v) | values |
| S = Q K^T | (B, 24, 24) | scores, S[i][j] = q_i . k_j |
| A = softmax(S / sqrt(d_k)) | (B, 24, 24) | attention weights, each row sums to 1 |
| Y = A V | (B, 24, d_v) | new summary of each hour |

## 2. Query, key, value: why three projections
Each hour i has three roles. The query q_i says what kind of hour it is looking for. The key k_j says
what kind of hour j is, so it is used for matching. The value v_j is the information hour j passes on.
Separate matrices W_Q, W_K, W_V let the model learn each role. With a single shared projection the
score q_i . q_j would be symmetric and the vector used for matching would also be the one passed on.

## 3. Dot-product similarity
q . k = |q| |k| cos(angle). It is large when the two vectors point the same way, so it measures how well
hour j's key matches hour i's query. Computing it for all pairs is one matrix product, S = Q K^T,
which gives a T x T table whose row i holds the scores of hour i against every hour.

## 4. Why divide by sqrt(d_k)
Assume the entries of q and k are independent, with mean 0 and variance 1. Then each product q_m k_m
has mean 0 and variance E[q_m^2] E[k_m^2] = 1. The dot product is a sum of d_k such terms, so its variance
is d_k and its standard deviation is sqrt(d_k). Dividing by sqrt(d_k) brings the variance back to 1,
whatever d_k is. Example: d_k = 64 gives a standard deviation of 8, and a row like [8, 0, 0] gives weights
[0.9993, 0.0003, 0.0003], almost one-hot. This matters because the softmax derivative is
da_i/ds_j = a_i (delta_ij - a_j). If one weight is close to 1 and the others close to 0, every entry is
close to 0, so almost no gradient flows back through the softmax and learning stalls.
Measured: with unit-size initialisation and d_k = 64 the unscaled model started with attention entropy 0.27
(scaled: 1.65) and needed 225-350 steps to reach 90% accuracy (scaled: 50). Caveat: the unit-variance
assumption did not hold for the default initialisation (std 1/sqrt(d_in) with one-hot inputs gives score
standard deviations of only 0.1-0.8), and there the unscaled model learned slightly faster.

## 5. Softmax and numerical stability
a_j = exp(s_j) / sum_k exp(s_k). Every a_j is positive and the row sums to 1, so it is a set of
percentages. Adding the same constant c to every score does not change the result, because exp(c)
cancels between top and bottom: softmax(s - c) = softmax(s). The stable version uses c = max(s), so every
exponent is at most 0 and nothing overflows. Without it, exp(1000) = inf and inf / inf = NaN
(tested: scores [1000, 1001, 999] give NaN naively and valid weights with the stable version).

## 6. Weighted aggregation
y_i = sum_j a_ij v_j. Since a_ij >= 0 and sum_j a_ij = 1, each output row is a weighted average of the
value rows. In matrix form Y = A V, with A (24 x 24) times V (24 x d_v).

## 7. Order information
If the rows of X are permuted, the rows of Y are permuted the same way. Attention alone does not know
which hour is which, so the time features (sin/cos of hour and weekday) are part of every row of X.

## 8. Gradients
Let L be the loss and dY = dL/dY. By the chain rule: dV = A^T dY; dA = dY V^T; for each row of the
softmax, dS'_ij = A_ij (dA_ij - sum_k dA_ik A_ik); dS = dS' / sqrt(d_k); dQ = dS K; dK = dS^T Q;
dW_Q = X^T dQ, dW_K = X^T dK, dW_V = X^T dV. I did not implement this backward pass by hand (it is
optional in the PRD). I used PyTorch autograd and checked it against finite differences:
(L(w + eps) - L(w - eps)) / (2 eps), in float64 with eps = 1e-6. The largest relative difference was
2.7e-10, which is the size expected from rounding error (about 1e-16 / eps) plus the O(eps^2)
approximation error.