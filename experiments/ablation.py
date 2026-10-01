"""Ablation: scaled vs unscaled attention on the recall task, two init conditions."""
import json
import statistics
from experiments.toy_task import train, evaluate

D_KS = [4, 16, 64]
INITS = {"A_small_init": None, "B_unit_init": 1.0}
SEEDS = [0, 1, 2]


def steps_to_90(hist):
    for h in hist:
        if h["acc"] >= 0.9:
            return h["step"]
    return None  # never reached


results = []
for init_name, init_std in INITS.items():
    for d_k in D_KS:
        for scale in (True, False):
            for seed in SEEDS:
                model, hist = train(d_k=d_k, scale=scale, init_std=init_std,
                                    steps=1500, seed=seed, log_every=25)
                results.append(dict(init=init_name, d_k=d_k, scaled=scale, seed=seed,
                                    test_acc=evaluate(model), final_loss=hist[-1]["loss"],
                                    steps_to_90=steps_to_90(hist),
                                    entropy_start=hist[0]["entropy"],
                                    entropy_end=hist[-1]["entropy"]))
                print(results[-1], flush=True)

with open("results/ablation.json", "w") as f:
    json.dump(results, f, indent=2)

print("\ninit          d_k scaled  test_acc  steps_to_90 (per seed)   ent_start  ent_end")
for init_name in INITS:
    for d_k in D_KS:
        for scale in (True, False):
            rs = [r for r in results
                  if r["init"] == init_name and r["d_k"] == d_k and r["scaled"] == scale]
            acc = statistics.mean(r["test_acc"] for r in rs)
            st = [r["steps_to_90"] for r in rs]
            es = statistics.mean(r["entropy_start"] for r in rs)
            ee = statistics.mean(r["entropy_end"] for r in rs)
            print(f"{init_name:13s} {d_k:3d} {str(scale):6s} {acc:8.3f}  {str(st):24s} {es:8.3f} {ee:8.3f}")