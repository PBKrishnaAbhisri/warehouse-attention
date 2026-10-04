# AI assistance log

Tool: Claude (chat) for every entry. Rule I followed: nothing counts as verified until a test or
experiment confirmed it.

Who did what: Claude wrote almost all the code and documents in chat. I created the files, pasted
the code, ran every command, sent the outputs back, asked for explanations until I understood the
flow, fixed my own setup and git problems, and chose the next step. I did not write the core
attention code myself, so I relied on the explanations and the checks below.

## 1. Planning
- Task: explain the PRD in simple words, choose tools, plan the steps.
- Verification: I checked the plan against the PRD's required items.
- Understanding: attention from scratch, proof it is correct, use it on warehouse demand, compare
  it fairly with simple baselines.

## 2. Setup
- Task: venv, installs, folders. I noticed the activation line was commented out and asked.
  Later I ran python on the activate script by mistake; the right command is .venv\Scripts\Activate.ps1.
- Verification: torch 2.14.1+cpu, numpy 2.2.6, Python 3.10.0; requirements.txt saved.
- Understanding: a venv keeps this project's packages separate.

## 3. Phase 0 design
- Task: design, numbers and hypotheses H1-H3 before any code. I committed it first (105d381).
- Later finding: the baseline estimates were wrong (same-hour-yesterday guessed 6-9, measured 12-13).
  I did not edit Phase 0 afterwards.
- Understanding: Phase 0 is a prediction on record, so wrong guesses get reported.

## 4. Attention (src/attention.py) and tests
- Task: Q/K/V, scaling, stable softmax, A times V, plus 8 tests (8 passed).
- Verification: the PyTorch reference check is only inside a test. Claude checked the 3-hour example
  with its own script; I did not add it as a test.
- Issue found: I left a test inside src/attention.py and moved it. My test file name had a typo.
- Understanding: each hour makes a question (Q), badge (K) and report (V). Scores turn into
  percentages with softmax, and the output is a weighted mix of the reports.

## 5. Worked example
- Task: explain with an analogy and a 3-hour example, then use it as code comments.
- Verification: I followed the example against the code line by line; I did not run the extra check.
- Understanding: I can follow the flow of the example.

## 6. Gradient check
- Result: relative difference about 1e-10 (float64); the gradcheck test passed. I did not run the
  optional float32 comparison.
- Understanding: a gradient is the slope training follows. Finite differences measure it by nudging
  a number, so they are an independent check.

## 7. Toy task (associative recall)
- Result: 100% test accuracy (chance 10%). In my example the query gave 50.6% attention to the
  matching entry and 45.2% to itself.
- Claude expected sharper attention; the measured entropy was about 0.87.
- Understanding: attention can learn to find the right position.

## 8. Scaled vs unscaled ablation
- Claude noticed beforehand that Phase 0's unit-variance assumption fails for the default init, so
  we ran two init conditions.
- Result: with the default init, unscaled learned slightly faster. With unit init and d_k = 64,
  unscaled needed 225-350 steps to reach 90% (scaled: 50) and got 94.6% mean accuracy (scaled: 100%).
- Honesty: the predictions file and the results were committed together, so git does not show which
  came first.
- Understanding: big scores make softmax nearly one-hot, where its slope is near zero and learning
  stalls. Dividing by sqrt(d_k) keeps scores near size 1.

## 9. Data generator, baselines, tests
- Verification: tests are in tests/test_data_gen.py. My pasted pytest output covered the 8 attention
  tests only. My baseline numbers matched Claude's own run.
- Understanding: the data is trend + daily + weekly + events + noise, split by time so the future
  does not leak into training.

## 10. Warehouse attention model
- Result: attention 6.14 MAE vs last value 7.37 (official test).
- Honesty: I ran the model before writing predictions, so docs/model_predictions.md is post-hoc.
- Understanding: the model uses the last hour's summary plus the next hour's time features.

## 11. Controls (ridge, uniform attention, 200 epochs)
- Claude's draft prediction said ridge could not handle weekday effects. It was wrong: the data is
  additive and ridge matched attention (6.04 vs 5.98).
- Understanding: here the useful positions are fixed, so fixed weights work. Attention helps when
  the useful position depends on the content, like in the toy task.

## 12. Distribution shift
- Result: both models degrade alike under noise (about +61%). Under spikes x2, attention is 0.57 MAE
  worse than ridge (7.77 vs 7.20). Claude's predictions 2 and 3 were wrong (docs/shift_predictions.md).
- Understanding: noise hurts every model; big input spikes hurt attention more.

## 13. Failure investigation (docs/failure_analysis.md)
- Result: 7-24 hours after a spike, attention's prediction still moves by 10-14 orders even though
  the spike's true effect is zero. Ridge does it less.
- Claude first thought attention locks onto the spike, but the weight there was below usual. The
  cause is not confirmed. I wrote no prediction before this run.

## Where I corrected, questioned or rejected AI output
- I noticed the commented-out activation line and asked about it.
- I chose not to add the hand-example test and asked for explanations and comments instead.
- I moved a misplaced test out of src/attention.py.
- Several times I asked Claude to restart from the problem statement until I understood.
- Experiments showed these AI claims were wrong: the baseline estimates, the init assumption, the
  ridge reasoning, and the idea that attention locks onto the spike.