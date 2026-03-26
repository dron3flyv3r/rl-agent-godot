# Improvements 2

This document lists additional improvement opportunities found by reviewing `addons/rl-agent-plugin` with emphasis on user experience, reliability, reproducibility, and day-to-day workflow quality.

This file intentionally focuses on newly identified opportunities that are not already covered in `IMPROVEMENTS.md`.

---
## 1. Reproducibility & Experiment Management

### A. Add explicit deterministic seed control

Current finding:
- Training and network initialization currently rely on randomized RNG initialization paths.
- There is no first-class inspector-level seed setting to reproduce the same run behavior.
- Distributed workers also do not expose a clear seed strategy for deterministic-but-diverse rollouts.

How to fix it:
1. Add `RandomSeed` to run-level config (for example on `RLRunConfig`).
2. Route this seed into trainer sampling RNGs, network initialization, self-play sampling, and curriculum randomness.
3. Add deterministic per-worker seed derivation (`baseSeed + workerId * prime`) for distributed mode.
4. Store the resolved seed map in run metadata and checkpoint metadata.

### B. Persist immutable run recipes per launch

Current finding:
- Launch manifests store config resource paths, but not immutable snapshots of resolved values.
- If a resource is edited later, it is hard to reconstruct exactly what was used for an old run.
- Comparison and reproducibility become fragile when settings drift over time.

How to fix it:
1. Save a `run_recipe.json` in each run folder containing resolved training, run, curriculum, self-play, and distributed settings.
2. Include plugin version, engine version, and scene path hash.
3. Add a one-click `Re-run with same recipe` action in the dashboard.
4. Keep path-based fields for convenience, but treat recipe snapshots as the source of truth for historical runs.

### C. Support true training resume (not only weight warm-start)

Current finding:
- Checkpoints primarily capture model state and metadata.
- Full optimizer/replay/trainer progression state is not exposed as a first-class resume contract.
- Long-running training sessions cannot be resumed with full continuity after interruption.

How to fix it:
1. Extend checkpoint payloads with optimizer moments and trainer update state.
2. For SAC, include replay-buffer persistence options (full or sampled snapshot).
3. Add `Resume From Checkpoint` launch flow in editor toolbar and setup dock.
4. Validate compatibility before resume and show explicit incompatibility reasons.

---
## 2. Reliability & Safety Nets

### A. Add numeric health guards with auto-pause and rollback

Current finding:
- Metrics capture useful values, but there is no centralized health monitor for NaN/Inf or divergence spikes.
- Runs can continue after numerical instability without immediate guided recovery.
- Users often discover failures late through flatlined charts or invalid checkpoints.

How to fix it:
1. Add health checks for NaN/Inf in observations, actions, losses, and weights.
2. Add configurable guardrails (max loss, min entropy floor, gradient explosion threshold).
3. Auto-pause on guardrail violations and surface a clear dashboard incident card.
4. Offer one-click rollback to last known healthy checkpoint.

### B. Add atomic checkpoint writes with integrity validation

Current finding:
- Checkpoint writes are functional, but corruption protection and integrity proofing are limited.
- Interrupted writes can leave incomplete or ambiguous artifacts.
- Load errors are surfaced, but there is no checksum-backed validation workflow.

How to fix it:
1. Write checkpoints to temporary files and atomically rename on success.
2. Generate checksums (for example SHA-256) and store them in sidecar metadata.
3. Validate checksums during load and dashboard listing.
4. Add `repair` and `skip corrupt file` behaviors in checkpoint discovery.

### C. Add pre-launch training diagnostics beyond structural validation

Current finding:
- Structural validation is strong, but runtime-behavior diagnostics are still mostly post-launch.
- Users can start runs with extreme reward scales or unstable observation magnitudes.
- Failures can happen after significant wasted training time.

How to fix it:
1. Add a short pre-launch probe pass (few episodes, no training) to profile reward and observation ranges.
2. Flag suspicious scales with concrete remediation hints.
3. Provide confidence score (`ready`, `risky`, `blocked`) in setup dock.
4. Save probe reports to run folders for later debugging.

---
## 3. Observability & Analysis

### A. Add high-frequency telemetry channels for step-time profiling

Current finding:
- Dashboard currently relies on periodic metric polling and episode-level summaries.
- Detailed frame-level timing (collect, infer, train, IO) is not first-class in user-facing charts.
- Performance bottlenecks are difficult to localize.

How to fix it:
1. Add structured telemetry stream for per-step timings and queue pressure.
2. Track phase timings: observation, action sampling, physics, trainer update, checkpoint IO.
3. Render timeline and percentile charts in dashboard.
4. Keep JSONL export fallback for offline analysis.

### B. Add policy behavior analytics (action histograms and saturation)

Current finding:
- Current charts emphasize reward and losses, but not action utilization patterns.
- It is hard to detect collapsed policies that overuse one action or saturate continuous outputs.
- Users need behavior-level evidence to diagnose learning stalls.

How to fix it:
1. Log discrete action histograms per policy group.
2. Log continuous action mean/std and saturation rate near min/max bounds.
3. Show trends in dashboard and quick warnings for persistent collapse.
4. Correlate behavior metrics with reward and entropy curves.

### C. Add gradient and update quality diagnostics

Current finding:
- Core optimization metrics are present, but update-quality visibility is still limited.
- Users cannot directly inspect gradient norms, clipping ratios, and update magnitudes over time.
- Tuning stability-related hyperparameters requires guesswork.

How to fix it:
1. Record gradient norm stats (mean, max, clipped ratio) per update.
2. Add explained variance and value-target error percentiles where applicable.
3. Show diagnostics panel with threshold highlighting in dashboard.
4. Include diagnostics in exported run reports.

---
## 4. Distributed Training Operations

### A. Add a dedicated worker health panel in dashboard

Current finding:
- Distributed mode has runtime logging and recovery hooks, but worker-level visibility in dashboard is limited.
- Users cannot quickly inspect per-worker liveness, throughput, lag, or reconnect history.
- Diagnosing distributed slowdowns requires log inspection.

How to fix it:
1. Add `Workers` panel with worker id, status, last heartbeat, and rollout throughput.
2. Show per-worker contribution and staleness indicators.
3. Surface reconnect/relaunch events in a visible event feed.
4. Add quick actions (mute worker, restart worker, isolate worker for diagnostics).

### B. Add adaptive rollout backpressure and staleness controls

Current finding:
- Async rollout policies currently focus on `Pause` vs `Cap` behavior.
- Users have limited control over staleness budgets and fairness across worker contributions.
- Aggressive worker throughput can degrade update quality when not governed adaptively.

How to fix it:
1. Add bounded staleness policy with configurable max age/updates.
2. Introduce weighted sampling by recency and worker reliability.
3. Expose queue pressure and discard reasons in telemetry.
4. Provide sensible presets for local and high-worker-count setups.

### C. Add distributed setup assistant and diagnostics checklist

Current finding:
- Distributed configuration is powerful but still easy to misconfigure for new users.
- Common issues (ports, executable path, headless flags) are mostly diagnosed after launch.
- Setup friction is high for first-time distributed users.

How to fix it:
1. Add `Distributed Setup Wizard` with connectivity pre-checks.
2. Validate executable path, port availability, and argument compatibility before launch.
3. Generate copyable run commands for manual worker startup.
4. Persist a diagnostics report in run metadata.

---
## 5. Editor Workflow & Onboarding

### A. Make validation errors clickable and node-targeted

Current finding:
- Validation summaries are informative, but error navigation is still mostly text-driven.
- Users must manually locate offending nodes/resources in larger scenes.
- Fix loops are slower than necessary.

How to fix it:
1. Attach node/resource references to each validation issue.
2. Make issue rows clickable to focus the scene tree selection and inspector.
3. Add `Fix next issue` workflow in setup dock.
4. Support severity groups (`error`, `warning`, `hint`) with quick filtering.

### B. Add guided scene setup wizard for first-time users

Current finding:
- Getting started is documented, but onboarding still requires manual multi-step setup.
- New users can make structural mistakes before first successful run.
- Repetitive setup cost appears in every fresh project.

How to fix it:
1. Add `Create RL Scene` wizard that scaffolds academy, default configs, and sample agent scripts.
2. Offer 2D/3D templates and algorithm presets.
3. Generate a minimal runnable scene with one-click quick test.
4. Provide contextual hints in inspector for each generated asset.

### C. Add reusable preset library for common task archetypes

Current finding:
- Users tune from scratch even for repeated patterns (reach target, locomotion, tagging).
- Existing demos help, but there is no formal preset system in editor UX.
- Consistency between projects is hard to maintain.

How to fix it:
1. Add preset assets for network, algorithm, and reward template bundles.
2. Provide preset packs by objective type and difficulty.
3. Include provenance tags (origin, recommended observation scale, expected episode length).
4. Let users save and share custom presets between projects.

---
## 6. Inference & Deployment Experience

### A. Add batched inference API for runtime usage

Current finding:
- Inference helpers focus on single-observation prediction flow.
- Multi-agent runtime inference from gameplay code can incur avoidable per-call overhead.
- Large scenes would benefit from vectorized prediction calls.

How to fix it:
1. Extend inference runner with `PredictBatch(float[][] observations)`.
2. Validate batch shape once and execute a batched forward path.
3. Return structured per-agent decisions with index mapping.
4. Add performance benchmarks in docs and dashboard profiler.

### B. Add hot-swap model loading with fallback behavior

Current finding:
- Inference loading is robust, but runtime hot-swapping and fallback paths are limited.
- Failed model updates can force scene interruption or manual recovery.
- User-facing deployment workflows need safer live updates.

How to fix it:
1. Add `TrySwapModel` API with compatibility checks before activation.
2. Keep last-known-good policy active if new model load fails.
3. Emit clear swap result events for game code and editor logs.
4. Add optional staged rollout (`N% agents on new model`) for live validation.

### C. Add inference profiling and trace capture

Current finding:
- It is hard to measure inference latency and action decisions over time in live scenes.
- Debugging deployment regressions requires manual logging.
- There is no structured action trace export flow.

How to fix it:
1. Add inference timing counters (avg, p95, max) per policy group.
2. Add optional action trace capture (observation hash, action, timestamp).
3. Surface traces in dashboard and allow export.
4. Provide replay tooling for deterministic postmortems when seeds are fixed.

### D. Add explicit deterministic inference mode

Current finding:
- Inference behavior can still vary by runtime conditions and model-loading paths.
- There is no single, inspector-visible toggle that guarantees deterministic decision output for the same observation input.
- Teams need deterministic playback for QA, regression testing, and multiplayer lockstep scenarios.

How to fix it:
1. Add `DeterministicInference` option at academy or policy-group level.
2. Force deterministic action selection paths for both discrete and continuous policies.
3. Disable stochastic branches and document exact deterministic semantics per algorithm.
4. Surface deterministic mode status in runtime overlay and dashboard metadata.
5. Validate deterministic compatibility at load time and report clear fallback/warning behavior.

---
## 7. Run Lifecycle & Collaboration

### A. Add side-by-side run comparison mode in dashboard

Current finding:
- Dashboard interaction is currently centered around one selected run at a time.
- Cross-run comparisons require manual switching and mental diffing.
- Team iteration speed suffers when comparing hyperparameter experiments.

How to fix it:
1. Add multi-run selection and overlay charts.
2. Provide automatic best-run ranking by configurable objective.
3. Add confidence bands or moving-window summaries for fair comparison.
4. Support compare presets and saved dashboards.

### B. Add run tagging, notes, and experiment metadata

Current finding:
- Run organization relies on IDs/prefixes and optional display names.
- Rich experiment context (goal, changes, operator notes) is not first-class.
- Collaborative workflows lose important decision history.

How to fix it:
1. Add editable run tags and markdown notes in dashboard.
2. Persist metadata in per-run manifest files.
3. Add filtering and grouping by tags, algorithm, and scene.
4. Include metadata in exports and model bundles.

### C. Add retention and archival policy controls

Current finding:
- Frequent checkpoints and runs can accumulate quickly.
- Cleanup and archival strategy is currently manual.
- Storage pressure can silently degrade iteration experience.

How to fix it:
1. Add retention rules (keep best N, keep every Kth, keep recent M days).
2. Add archive actions (compress run, move to archive path, export summary).
3. Show storage usage in dashboard.
4. Add safe cleanup preview before deletion.

---
## Priority Order

Suggested implementation order for the first five items:
1. Add explicit deterministic seed control and per-worker seed derivation.
2. Add numeric health guards with auto-pause and rollback.
3. Persist immutable run recipes and one-click rerun from dashboard.
4. Add clickable validation issues that jump to offending nodes/resources.
5. Add worker health panel and distributed diagnostics in dashboard.

| Implemented | Priority | Item |
| --- | --- | --- |
| No | 1 (high) | Deterministic seed control for reproducible runs |
| No | 1 (high) | Numeric health guards (NaN/Inf/divergence) with rollback |
| No | 1 (high) | Immutable run recipe snapshots + one-click rerun |
| No | 2 (high) | Clickable validation errors with direct node/resource navigation |
| No | 2 (high) | Worker health and throughput panel for distributed mode |
| No | 2 (high) | True resume training with optimizer/trainer state |
| No | 3 (medium) | High-frequency per-step telemetry and profiling timeline |
| No | 3 (medium) | Action utilization and continuous saturation analytics |
| No | 3 (medium) | Gradient/update diagnostics in dashboard |
| No | 3 (medium) | Adaptive rollout backpressure and staleness controls |
| No | 4 (nice to have) | Guided RL scene setup wizard |
| No | 4 (nice to have) | Reusable preset library for common task archetypes |
| No | 4 (nice to have) | Batched inference API for runtime integrations |
| No | 4 (nice to have) | Deterministic inference mode toggle for QA/replay/lockstep |
| No | 4 (nice to have) | Model hot-swap with fallback and staged rollout |
| No | 5 (advanced) | Multi-run side-by-side comparison and ranking |
| No | 5 (advanced) | Run tags/notes/metadata for collaboration |
| No | 5 (advanced) | Retention and archival policy controls |
