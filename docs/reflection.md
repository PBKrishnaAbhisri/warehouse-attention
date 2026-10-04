# Reflection

Process honesty: Claude wrote most of the code and documents; I ran everything, read the results and
asked for explanations until I understood the flow. Phase 0 was committed before any experiment. The
model and failure analyses are post-hoc (no prior predictions); the ablation predictions were committed
with the results.

1. What I expected: that attention would clearly beat the simple baselines, that dividing by sqrt(d_k)
   would matter, and that same-hour-yesterday would be the strongest baseline.
2. Hypotheses that held: scaling matters when scores are large (gap grows with d_k under unit init);
   uniform attention is much worse than learned attention; longer training helps a little; all models
   degrade under shift and the worst errors are at spike hours.
3. Hypotheses that were wrong: same-hour-yesterday is not the best baseline (it fails on Mon, Sat, Sun);
   ridge would be worse than attention (it matched it); attention would stay within 0.3 MAE of ridge
   under larger spikes (it was 0.57 worse); attention locks onto the spike (the weight was below usual).
4. What surprised me: ridge matched attention; attention weights were flat and the same on all weekdays;
   in the toy task the query row split its attention about 50/50 with itself; with the default init
   the scores were tiny, so Phase 0's assumption did not hold.
5. Hardest problem: understanding the shapes and flow of attention well enough to explain it. I needed
   the worked 3-hour example several times. On the practical side, the venv activation and GitHub login
   took time.
6. Failure investigated: after a large spike, attention keeps reacting to it for up to 24 hours,
   over-predicts, and ends worse than last value.
7. What I understand better: why softmax saturates and how scaling prevents it; what a gradient check
   proves; why baselines and a validation split matter; and that attention helps when the useful
   position depends on the content, as in the toy task, and not when it is fixed, as in my generator.
8. What remains uncertain: the cause of the spike echo (clipping inputs is untested); whether attention
   would win on a generator with content-dependent structure; only 3 seeds; Adam may hide part of the
   saturation effect; both models under-predict by about 3 on normal windows.
9. Where AI helped: wrote the code, tests and experiment scripts, explained the maths with analogies and
   examples, and noticed before the ablation that Phase 0's init assumption did not hold.
10. Where I corrected or rejected AI output: its baseline estimates, its ridge prediction and its
   "attention locks onto the spike" idea were wrong and were corrected by experiments. I also moved a
   misplaced test out of src/attention.py and chose not to add one extra test.
11. With one more day: clip the inputs to test the spike echo; check ridge error by weekday; use more
   seeds with confidence intervals; build a generator where the useful hour depends on events; compare
   SGD with Adam in the ablation.