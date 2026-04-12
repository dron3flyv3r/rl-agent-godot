
Phase 1 — Baseline: plain BC (control)
This gives you a reference point to compare DAgger against.

Open demo/10 ImitationMazeDemo/ImitationMazeDemo.tscn
In the RL Imitation dock → Record tab: set mode to Script, output name maze_seed, start recording. Let it run for ~500–1000 frames then stop. This generates your seed dataset via the scripted heuristic.
Switch to the Train tab, select BC, pick the maze_seed_*.rldem file, use default hyperparameters (20 epochs, lr=3e-4), hit Train.
Once done, use Run Inference with the resulting checkpoint and watch the agent. Note roughly what % of episodes it solves — the gap-navigation is where it's likely to fail (it never saw the agent stuck near the wall in the seed data).
Phase 2 — Manual DAgger (one round, validate the mechanism)
Before running Auto DAgger, confirm the per-round flow works correctly.

Switch algorithm to Manual DAgger. Set:
Checkpoint: the BC checkpoint from step 3
Seed dataset: maze_seed_*.rldem
Mixing Beta: 0.5 (round 1 → effective = 0.5¹ = 50% expert-driven)
Add Frames: 512
Click Run DAgger Round. The game will launch — watch the agent move. With β=0.5 you should see the agent alternate between smooth expert paths and the learner's more erratic exploration.
After it finishes, open the generated .rldem in the Dataset Info tab and check frame count = seed + 512.
Check the Godot output panel for the log line: [DAggerBootstrap] Beta mixing: base=0.500 round=1 effective=0.500 (~50% expert-driven steps). Confirms the right values were passed through.
Also check the .status file (it's a JSON next to the .rldem in RL-Agent-Demos/) — it should contain ExpertDrivenRate close to 0.5.
Now retrain BC on the new aggregated dataset. Run inference again — success rate should improve, especially for starts on the side of the wall far from the gap.
Phase 3 — Auto DAgger (test the full loop and beta decay)
Reset to the original seed dataset. Set:
Algorithm: Auto DAgger
Mixing Beta: 0.5
Rounds: 4
Add Frames: 512 per round
Epochs: 20
Hit Run Auto DAgger and let it complete all 4 rounds.
After each round completes you should see the status label update, and the Godot output should print the effective beta decaying:
Round 1: effective=0.500 (~50% expert)
Round 2: effective=0.250 (~25% expert)
Round 3: effective=0.125 (~13% expert)
Round 4: effective=0.063 (~6% expert)
Run inference on the final checkpoint. Success rate should be noticeably higher than the plain BC baseline.
Phase 4 — Validate the beta decay effect is real
This is the most important test to confirm the implementation actually helps.

Run the full Auto DAgger experiment twice more with:
beta=1.0 (always expert — equivalent to collecting more BC demos, no decay)
beta=0.0 (always learner — pure exploration labelling)
Compare final checkpoint quality across the three runs (0.0, 0.5, 1.0). You should find:
beta=1.0 performs similarly to plain BC with more data (expert always drives, so no new hard states are covered)
beta=0.0 may underperform early (learner is terrible in round 1, so collected states are garbage), but could catch up in later rounds once BC improves
beta=0.5 should give the best balance — covers hard states the learner visits while still being guided enough to reach them
The maze is a particularly good test for this because the "hard state" (approaching the wall from the wrong side) is naturally rare in expert demonstrations but common in learner rollouts.