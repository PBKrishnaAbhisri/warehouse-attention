import torch
from experiments.toy_task import train, make_batch, N_PAIRS

model, _ = train()
gen = torch.Generator().manual_seed(123)
X, y = make_batch(1, gen)
with torch.no_grad():
    logits, A = model(X)

pairs = [(int(X[0, i, :16].argmax()), int(X[0, i, 16:].argmax())) for i in range(N_PAIRS)]
qkey = int(X[0, N_PAIRS, :16].argmax())
print("pairs (name, number):", pairs)
print("question name:", qkey, "| correct number:", int(y[0]), "| model says:", int(logits.argmax()))
print("matching entry is row:", [i for i, (k, _) in enumerate(pairs) if k == qkey][0])
print("question row's percentages over rows 0-7 and itself (row 8):")
print([round(float(w), 3) for w in A[0, -1]])