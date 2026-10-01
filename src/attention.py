"""First-principles scaled dot-product self-attention.
Only raw tensor ops (matmul, exp, sum, max). No nn.MultiheadAttention,
no F.scaled_dot_product_attention in the implementation itself.

ANALOGY: a meeting of past hours
--------------------------------
Think of each past hour (8 AM, 9 AM, 10 AM, ...) as a person in a meeting.
Each person has three things:
    Query (Q): a question card  - "what kind of hour am I looking for?"
    Key   (K): a name badge     - "what kind of hour am I?"
    Value (V): a report         - "the information I can pass on"
Each person compares their question with everyone's badge, turns the scores
into percentages, and writes a new summary = percentage-weighted mix of reports.
We then use the LAST hour's summary (it has listened to everyone) to predict.

WORKED EXAMPLE (3 hours, 2 features, d_k = 2)
---------------------------------------------
X (one row per hour)      = [[1,0], [0,1], [1,1]]         # 8 AM, 9 AM, 10 AM
W_Q = [[1,0],[0,1]]   W_K = [[1,0],[1,1]]   W_V = [[2,0],[0,3]]

Q = X @ W_Q = [[1,0], [0,1], [1,1]]
K = X @ W_K = [[1,0], [1,1], [2,1]]
V = X @ W_V = [[2,0], [0,3], [2,3]]

S = Q @ K.T  (entry i,j = question of hour i . badge of hour j)
          8AM  9AM  10AM
    8AM [  1    1    2 ]
    9AM [  0    1    1 ]
    10AM[  1    2    3 ]

S' = S / sqrt(d_k) = S / 1.414.  The 10 AM row becomes [0.707, 1.414, 2.121]

softmax of the 10 AM row:
    subtract max (2.121)      -> [-1.414, -0.707, 0]
    exp                       -> [0.243, 0.493, 1.000]
    divide by sum (1.736)     -> [0.140, 0.284, 0.576]   (this row of A sums to 1)
  So 10 AM listens 14% to 8 AM, 28% to 9 AM, 58% to itself.
  (Without the scaling the row would be [0.090, 0.245, 0.665]: sharper.)

Y (10 AM row) = 0.140*[2,0] + 0.284*[0,3] + 0.576*[2,3] = [1.432, 2.580]
  This summary goes to the prediction head, which turns it into one number
  (the predicted orders for the next hour).
"""
import math
import torch
import torch.nn as nn


def naive_softmax(scores, dim=-1):
    """Textbook softmax. Overflows for large logits (kept for the stability demo)."""
    e = torch.exp(scores)                       # make everything positive, boost big scores
    return e / e.sum(dim=dim, keepdim=True)     # divide by row total -> each row sums to 1


def stable_softmax(scores, dim=-1):
    """softmax(s) = softmax(s - max(s)). Same result, but exp() only sees values <= 0."""
    # Subtracting the row max does not change the answer (the common factor cancels
    # between top and bottom) but stops exp(1000) = inf -> inf/inf = nan.
    shifted = scores - scores.max(dim=dim, keepdim=True).values
    e = torch.exp(shifted)
    return e / e.sum(dim=dim, keepdim=True)



def attention(X, W_Q, W_K, W_V, scale=True, stable=True):
    """X: (B, T, d_in). W_Q, W_K: (d_in, d_k). W_V: (d_in, d_v).
    Returns Y: (B, T, d_v) and attention weights A: (B, T, T).

    B = how many windows are processed at once, T = 24 hours in a window,
    d_in = 5 features per hour, d_k = size of question/badge, d_v = size of report.
    """
    d_k = W_K.shape[1]                            # needed for the scaling below

    # Step 1: every hour writes a question, a badge and a report.
    Q = X @ W_Q                                   # (B, T, d_k) each hour's question
    K = X @ W_K                                   # (B, T, d_k) each hour's badge
    V = X @ W_V                                   # (B, T, d_v) each hour's report

    # Step 2: score every question against every badge (dot product = how well they match).
    # K.transpose(-2, -1) swaps the last two dims so the shapes line up:
    # (B,T,d_k) @ (B,d_k,T) -> (B,T,T). Row i = how hour i rates every hour j.
    S = Q @ K.transpose(-2, -1)                   # (B, T, T)

    # Step 3: scale. A dot product of d_k terms has std ~ sqrt(d_k), so dividing by
    # sqrt(d_k) keeps the scores around size 1 and stops softmax from saturating.
    S_prime = S / math.sqrt(d_k) if scale else S  # (B, T, T)  (scale flag = ablation switch)

    # Step 4: turn each row of scores into percentages that sum to 1.
    softmax_fn = stable_softmax if stable else naive_softmax
    A = softmax_fn(S_prime, dim=-1)               # (B, T, T) row i = who hour i listens to

    # Step 5: each hour's new summary = percentage-weighted mix of all the reports.
    Y = A @ V                                     # (B, T, d_v)
    return Y, A                                   # A is returned so we can plot it later


class SelfAttention(nn.Module):
    """Holds the three projection matrices as learnable parameters."""

    def __init__(self, d_in, d_k, d_v=None, scale=True, init_std=None):
        super().__init__()
        d_v = d_v or d_k                          # default: report size = question size
        self.scale = scale
        # Start with small random numbers so Q, K, V have about unit variance.
        # Training will adjust these so questions and badges match useful hours.
        std = init_std if init_std is not None else 1.0 / math.sqrt(d_in)
        self.W_Q = nn.Parameter(torch.randn(d_in, d_k) * std)
        self.W_K = nn.Parameter(torch.randn(d_in, d_k) * std)
        self.W_V = nn.Parameter(torch.randn(d_in, d_v) * std)

    def forward(self, X):
        return attention(X, self.W_Q, self.W_K, self.W_V, scale=self.scale)