# AI assistance log

Tool: Claude (chat), used for every entry below.
Rule I followed: nothing counts as verified until a test or experiment confirmed it.

Who did what: Claude wrote nearly all of the code and documents in this project, in chat.
My part: I created the folders and files, copied the code in, ran every command, pasted the
outputs back to Claude, asked for explanations (analogies and worked examples) until I understood
the flow, fixed my own setup and git problems, and decided what to do next. I did not write the
core attention code myself, so I leaned on the explanations and the checks below.

## 1. Reading the PRD and planning
- Task: explain the PRD in simple words, choose the tools, plan the steps.
- AI output: a summary of the PRD, the deliverable list, and the stack (Python, PyTorch with raw
  tensor operations, NumPy, pytest, matplotlib).
- Verification: I checked the plan against the PRD's list of required items.
- Understanding: the project is one experiment: build attention from scratch, prove it is correct,
  use it to predict warehouse demand, and compare it honestly with simple baselines.

## 2. Environment setup
- Task: virtual environment, installs, folders. Claude's first command block had the activation
  step as a comment, which I noticed and asked about. Later I also ran `python` on the activate
  script by mistake; the right command is `.venv\Scripts\Activate.ps1`.
- Verification: the import check printed torch 2.14.1+cpu and numpy 2.2.6; Python 3.10.0;
  requirements.txt was saved with pip freeze.
- Understanding: a venv keeps this project's packages separate, and requirements.txt lets a
  reviewer recreate them.

## 3. Phase 0 design (docs/phase0_design.md)
- Task: write the design, the generator numbers and hypotheses H1-H3 before any code.
- AI output: the full draft. I saved it as my Phase 0 document and committed it before running any
  experiment (commit 105d381).
- Later finding: the draft's baseline estimates were wrong. It predicted same-hour-yesterday at
  6-9 MAE; it measured 12-13, because yesterday is a different weekly level on Mon, Sat and Sun.
  I did not edit Phase 0 afterwards.
- Understanding: Phase 0 is a prediction on the record, so a wrong guess is reported, not hidden.

## 4. First-principles attention (src/attention.py, tests/test_attention.py)
- Task: Q/K/V, scaling, stable softmax, A times V, plus tests.
- AI output: the file and 8 tests. The comparison with PyTorch's built-in attention appears only
  inside a test, never in src/.
- Verification: pytest showed 8 passed. The 3-hour worked example was checked by Claude with its
  own numpy script; I did not add it as a test in my repo.
- Issue found: I had left a test function inside src/attention.py. I moved it to the tests file.
  My test file name also had a typo (test_attenttion.py).
- Understanding: each hour makes a question (Q), a badge (K) and a report (V); the scores Q.K are
  turned into percentages by softmax; the output is a percentage-weighted mix of the reports.

## 5. Worked example and comments
- Task: explain the computation with an analogy and numbers, then use it as the file's comments.
- AI output: a 3-hour, 2-feature example worked by hand (scores, scaling, softmax, mixing).
- Verification: I put the example in the docstring of attention.py and followed the steps
  against the code line by line. I did not run the separate check command.
- Understanding: I can follow the flow from the worked example.

## 6. Gradient verification (experiments/gradient_check.py, gradcheck test)
- Task: compare autograd gradients with finite differences.
- Verification: the output showed a relative difference of about 1e-10 for W_Q, W_K and W_V
  (float64, eps 1e-6), and the gradcheck test passed. I did not run the optional float32 comparison.
- Understanding: a gradient is the slope training follows; finite differences measure it by
  nudging a number up and down, so it is an independent check. The leftover 1e-10 is measurement
  and rounding error.

## 7. Toy task: associative recall (experiments/toy_task.py)
- Task: a small problem where attention must find the matching entry.
- Result: 100% test accuracy (chance is 10%). In the example I looked at, the query row gave
  50.6% weight to the matching entry and 45.2% to itself.
- Where Claude's expectation was off: it expected sharper attention; the measured entropy was
  about 0.87, and the 50/50 split with the query row was the likely reason.
- Understanding: this shows attention can learn to find the right position; it is the contrast
  case for the warehouse data, where the useful positions are fixed.

## 8. Scaled vs unscaled ablation
- Task: compare dividing by sqrt(d_k) with not dividing, on the toy task.
- AI contribution: before running, Claude checked with a numpy copy of my data that the Phase 0
  assumption (unit-variance Q and K) does not hold for the default init, so we ran two init
  conditions.
- Result: with the default init the unscaled version learned slightly faster. With unit-size init
  and d_k = 64, unscaled needed 225-350 steps to reach 90% against 50 for scaled, and reached
  94.6% mean test accuracy (scaled reached 100%).
- Honesty: the predictions file and the results were committed together, so git does not show
  which came first.
- Understanding: large scores make softmax nearly one-hot, where its slope is close to zero, so
  learning stalls; dividing by sqrt(d_k) keeps scores around size 1.

## 9. Data generator, baselines, tests
- Task: generate the synthetic warehouse data, cut it into 24-hour windows, add three baselines.
- Verification: tests were added in tests/test_data_gen.py (determinism, noise level, window
  boundaries); the pytest output I pasted in chat covered the 8 attention tests only. The baseline
  numbers from my runs matched the numbers Claude got in its own environment. An error-by-weekday
  check confirmed why same-hour-yesterday fails on Mon, Sat and Sun.
- Understanding: the data is trend + daily + weekly + events + noise, and the split is by time so
  the future never leaks into training.

## 10. Warehouse attention model
- Task: attention plus a linear head, a training loop, comparison with the baselines.
- Result: attention 6.14 MAE against last value 7.37 on the official test.
- Honesty: I ran the model before writing predictions, so docs/model_predictions.md is post-hoc and
  says so.
- Understanding: the model uses the last hour's summary, after it has looked at all 24 hours, plus
  the next hour's time features.

## 11. Controls (ridge, uniform attention, 200 epochs)
- Task: test whether attention itself adds value.
- Where Claude's reasoning was wrong: its draft prediction said ridge could not capture weekday
  effects. The generator is additive, so ridge matched attention (6.04 against 5.98).
- Understanding: on this data the useful positions are always the same, so fixed weights work;
  attention helps when the useful position depends on the content, as in the toy task.

## 12. Distribution shift
- Task: train once on the original data, test with noise x2 and with spikes x2.
- Result: attention and ridge behave alike under noise (about +61% MAE), but attention is 0.57 MAE
  worse under larger spikes (7.77 against 7.20).
- Claude's predictions 2 and 3 were wrong; they are scored in docs/shift_predictions.md.
- Understanding: noise raises everyone's error, while big input spikes hurt attention more than the
  plain linear model.

## 13. Failure investigation (experiments/failure.py, docs/failure_analysis.md)
- Task: investigate why attention does worse after large spikes.
- Result: after a spike, the attention prediction still moves by 10-14 orders when the spike is
  removed from the inputs, even 7-24 hours later when the spike's true effect is zero. Ridge does
  the same, less.
- Where Claude's hypothesis was only partly right: it first suspected attention locks onto the
  spike, but the weight on the spike's position was below usual (0.64-0.93x). The cause is not
  confirmed; clipping the inputs is the untested next check.
- Honesty: I wrote no prediction before this run, so the analysis is post-hoc and says so.

## Where I corrected, questioned or rejected AI output
- I noticed the commented-out activation line in the setup commands and asked about it.
- I chose not to add the hand-example test and asked for explanations and code comments instead.
- I moved a misplaced test function out of src/attention.py.
- I asked Claude to restart its explanations from the problem statement several times when I was
  lost, until I understood the flow.
- Wrong AI claims that experiments exposed: the baseline estimates (entry 3), the unit-variance
  init assumption (entry 8), the ridge reasoning (entry 11), the attention-locks-on-spike idea
  (entry 13).