# Improvements

This document combines the proposed improvements with findings from the current codebase review of `RLAgent2D`, `RLAcademy`, `TrainingBootstrap`, the editor tooling, the trainer implementations, and checkpoint/inference flow.

Implemented status in the priority table reflects the current repository state, not the desired end state.

---
## 1. Agent API — Cleaner & Safer

### A. Remove the dual step/reward API

Current finding:
- `RLAgent2D` still keeps both `OnStep()` and the legacy `ComputeReward()` / `IsEpisodeDone()` fallback path.
- `GetObservation()` also remains as a compatibility wrapper around `CollectObservations()`.
- This keeps old and new patterns alive at the same time and makes the API harder to learn.

How to fix it:
1. Make `CollectObservations(ObservationBuffer buffer)`, `OnStep()`, and `OnEpisodeBegin()` the only public agent lifecycle hooks.
2. Remove `ComputeReward()` and `IsEpisodeDone()` entirely instead of keeping a fallback path.
3. Remove `GetObservation()` as a public customization point and keep observation collection routed through `CollectObservations(...)`.
4. Update runtime/editor callers so they only use the final API shape.
5. Update the demo agents, templates, and docs so new users only see the single supported path.

### B. Replace reflection-based action binding with explicit action registration

Current finding:
- Actions are currently defined through `[DiscreteAction]` / `[ContinuousAction]` attributes and resolved through reflection in `RLActionBinding`.
- Mixed action spaces are not represented cleanly.
- The current flattened discrete action model hides structure and gives no compile-time guarantees.
- Editor validation also depends on the same reflection path, so the fragility affects both runtime and tooling.

Better target API:

```csharp
protected override void DefineActions(ActionSpace actions)
{
    actions.AddDiscrete("Movement", "Idle", "Up", "Down", "Left", "Right");
    actions.AddContinuous("Steering", 2, min: -1f, max: 1f);
}

protected override void OnActionsReceived(ActionBuffer actions)
{
    int move = actions.GetDiscrete("Movement");
    float[] steer = actions.GetContinuous("Steering");
}
```

How to fix it:
1. Introduce `ActionSpaceBuilder`, `ActionSpec`, and `ActionBuffer` runtime types.
2. Add `DefineActions(ActionSpaceBuilder builder)` to `RLAgent2D`.
3. Add `OnActionsReceived(ActionBuffer actions)` and route all action application through it.
4. Remove `RLActionBinding`, `RLActionAttributes`, and reflection-based action discovery after migration.
5. Make editor validation read action definitions from the explicit spec instead of from reflected attributes.
6. Store named action metadata in manifests/checkpoints so training and inference agree on the exact schema.

### C. Typed `ObservationBuffer` with reusable sensors and automatic size validation

Current finding:
- `ObservationBuffer` is just a growable `List<float>` with helper `Add(...)` methods.
- There is no built-in size contract, no reusable sensor abstraction, and no debug validation that the same agent emits a stable observation size every episode.
- Editor validation does not inspect observation size at all; bootstrap discovers it late by calling `GetObservation()` on the first train agent.

How to fix it:
1. Add `IObservationSensor` with `int Size { get; }` and `void Write(ObservationBuffer buffer)`.
2. Extend the buffer with `AddSensor(...)`, named segments, and optional debug labels.
3. Cache the first valid observation size for each policy group and assert that later observations match.
4. Expose a scene-level "Infer Observation Size" workflow in `RLAcademy`.
5. Add reusable built-in sensors such as velocity, raycasts, relative target position, and normalized transform data.

### D. Cleaner reward API with named reward components

Current finding:
- Rewards are currently accumulated as anonymous floats through `AddReward(float)` and `SetReward(float)`.
- The dashboard can only show total episode reward, not why the reward changed.

Better target API:

```csharp
AddReward(0.10f, "distance_progress");
AddReward(-0.01f, "step_penalty");
```

How to fix it:
1. Add overloads for `AddReward(float amount, string tag)` and `SetReward(float amount, string tag)`.
2. Track per-step and per-episode reward breakdowns inside agent runtime state.
3. Extend metrics writing to emit reward-component totals.
4. Surface reward components in the dashboard and future in-editor debug overlays.

### E. Remove duplicated agent configuration surfaces

Current finding:
- `RLAgent2D` exposes inline `ControlMode`, `PolicyGroup`, and `InferenceCheckpointPath`.
- `RLAgentConfig` exposes the same settings again as an optional resource.
- This creates two sources of truth and makes it harder to understand what is actually active.

How to fix it:
1. Decide on one configuration model: either inline exports only or resource-backed config only.
2. If resource-backed wins, make the agent inspector show the resolved values clearly.
3. Remove the duplicate inline storage fields after migration.
4. Update validation to report the final resolved source for each setting.

---
## 2. Configuration — Eliminate Mismatch Bugs

### A. Replace `RLTrainerConfig` + `RLNetworkConfig` with one `RLTrainingConfig`

Current finding:
- `RLAcademy` currently references two separate resources.
- The actual agent observation/action sizes are not stored in either config and are inferred later.
- Inference also depends on `RLNetworkConfig` matching the training-time network shape, but checkpoints do not fully describe that dependency.

Better target API:

```csharp
[GlobalClass]
public partial class RLTrainingConfig : Resource
{
    [ExportGroup("Algorithm")]
    [Export] public RLAlgorithm Algorithm { get; set; } = RLAlgorithm.PPO;

    [ExportGroup("Network")]
    [Export] public int[] HiddenSizes { get; set; } = [128, 128];

    [ExportGroup("PPO")]
    [Export] public float LearningRate { get; set; } = 3e-4f;

    [ExportGroup("SAC")]
    [Export] public float SacTau { get; set; } = 0.005f;
}
```

How to fix it:
1. Create `RLTrainingConfig` as the single exported resource for algorithm, optimizer, and network settings.
2. Move all trainer/network settings into it and group the inspector fields by algorithm.
3. Add a migration path that can import existing `RLTrainerConfig` and `RLNetworkConfig` resources.
4. Update `RLAcademy`, `TrainingBootstrap`, trainers, and editor validation to consume only the unified resource.

### B. Auto-infer observation and action sizes in `RLAcademy`

Current finding:
- Observation size is discovered by executing observation collection on a live scene instance during bootstrap.
- Action size is inferred from agent reflection metadata.
- Failure happens late, usually when training starts.

Better target API:

```csharp
[Export] public int ObservationSize { get; private set; }
[Export] public int DiscreteActionCount { get; private set; }
[Export] public int ContinuousActionDims { get; private set; }
```

How to fix it:
1. Add an inspector action on `RLAcademy`: "Infer Sizes from Scene".
2. Scan all train agents, call the canonical observation/action definition path once, and compute group-level sizes.
3. Persist the inferred sizes as read-only exported fields on the academy or policy-group resource.
4. Show mismatches immediately when agents in the same group disagree.

### C. Validate at manifest-write time and show all errors in-editor

Current finding:
- Validation exists in the editor, but it does not inspect observation lengths.
- Bootstrap still performs critical runtime-only validation and can terminate training after launch.
- Current editor summaries are text-only and do not expose clickable offending nodes.

How to fix it:
1. Move all scene contract validation into the editor-side validation pipeline before launch.
2. Make manifest creation fail if any policy group is invalid.
3. Store node paths for each validation error.
4. Add clickable inspector/dashboard links that select the offending node in the editor.
5. Keep bootstrap validation as a final safety net, but treat editor validation as the primary gate.

### D. Make editor validation refresh when properties change

Current finding:
- The current plugin refreshes validation when the edited scene path changes, not when relevant node properties change.
- This means the status panel can go stale while the scene is being edited.

How to fix it:
1. Subscribe to editor scene/property change notifications.
2. Re-run lightweight validation when `RLAcademy`, `RLAgent2D`, or related config resources change.
3. Debounce validation to avoid running on every keystroke.
4. Keep the dock state, toolbar state, and tooltip error in sync.

---
## 3. Multi-Agent & Policy Grouping — Make It Explicit

### A. Replace string-based `PolicyGroup` with a typed resource

Current finding:
- Policy sharing is currently driven by a plain string on the agent.
- Training bootstrap falls back to scene-relative node paths when the string is empty.
- Editor validation uses a different fallback (`__agent__{node.Name}`), so validation and runtime do not always group agents the same way.

Better target API:

```csharp
[GlobalClass]
public partial class RLPolicyGroupConfig : Resource
{
    [Export] public string GroupId { get; set; }
    [Export] public RLTrainingConfig TrainingConfig { get; set; }
    [Export] public RLAgentControlMode Mode { get; set; }
    [Export] public string InferenceCheckpointPath { get; set; }
}
```

How to fix it:
1. Create `RLPolicyGroupConfig` as the explicit shared-policy resource.
2. Let multiple agents reference the same resource instance to share one trainer/inference policy.
3. Remove string matching and fallback group synthesis from both validation and bootstrap.
4. Make the editor list every agent using each policy-group resource.
5. Keep a stable `GroupId` only for run metadata and checkpoint naming, not as the primary binding mechanism.

### B. Add built-in self-play and opponent sampling

Current finding:
- The tag demo currently handles standalone control and mode switching manually in scene logic.
- There is no first-class self-play checkpoint bank or historical opponent sampling flow.

Better target API:

```csharp
[Export] public bool SelfPlay { get; set; }
[Export] public float HistoricalOpponentRate { get; set; } = 0.5f;
```

How to fix it:
1. Extend `RLPolicyGroupConfig` with self-play settings.
2. Save periodic frozen checkpoints into a per-group opponent pool.
3. Sample opponents from latest or historical snapshots based on the configured rate.
4. Expose the current matchup selection in metrics and the dashboard.

### C. Make policy-group membership visible in the editor

Current finding:
- The editor can summarize groups, but the grouping source is not explicit in the inspector.
- It is still too easy to miss which agents actually share weights.

How to fix it:
1. Show a policy-group summary panel on `RLAcademy`.
2. List each group, the assigned config, the number of agents, and the concrete node paths.
3. Show warnings when a group contains inconsistent action or observation definitions.

---
## 4. Training Architecture — Batch & Gradient Quality

### A. Finish true batched forward passes

Current finding:
- `BatchSize` already clones multiple scene instances.
- Action sampling and value estimation are still executed one agent at a time.
- This is environment duplication, not true neural-network batching.

How to fix it:
1. Add batched inference/update methods to the trainer/network layer.
2. Collect all observations for a policy group into one contiguous batch tensor per decision step.
3. Run one forward pass per group instead of one pass per agent.
4. Return batched decisions and scatter them back to the agents.
5. Keep the current multi-scene environment cloning, but use it as input to a batched trainer.

### B. Fix PPO implementation gaps

Current finding:
- GAE is already computed explicitly before epochs, which is good.
- PPO still trains sample-by-sample with no shuffling and no minibatches.
- There is no global gradient clipping.
- There is no clipped value loss.
- `clip_fraction` is not logged.
- Reported policy/value metrics are only rough proxies and entropy is currently not accumulated meaningfully.

How to fix it:
1. Shuffle rollout samples before each epoch.
2. Train with minibatches instead of per-sample full passes.
3. Add global gradient norm clipping, default `0.5`.
4. Add value loss clipping as a separate option.
5. Log `clip_fraction`, explained variance, and better entropy/value metrics.
6. Keep GAE precomputation, but move the full PPO update loop closer to standard PPO math.

### C. Add first-class hyperparameter schedules

Better target API:

```csharp
[Export] public LRSchedule LearningRateSchedule { get; set; }
```

How to fix it:
1. Introduce schedule resources for constant, linear decay, and cosine annealing.
2. Evaluate schedules per update step.
3. Apply the same pattern to entropy coefficient and other annealed hyperparameters later.

### D. Add a neural-network graph builder instead of array-only layer definitions

Current finding:
- `RLNetworkConfig` currently uses `int[] HiddenLayerSizes` as the main network definition.
- This is enough for a basic MLP, but it does not scale cleanly to branching heads, shared trunks, future CNN/recurrent support, or more advanced architectures.
- It also makes the network structure harder to inspect visually in the editor.

Better target API:

```csharp
[Export] public RLNetworkGraph Graph { get; set; }
```

How to fix it:
1. Introduce graph resources such as `RLNetworkGraph`, `RLNetworkNode`, and `RLNetworkEdge`.
2. Add an editor graph builder that can define trunks, heads, merges, and output nodes visually.
3. Support a simple built-in MLP template so users can still create common policies quickly.
4. Compile the graph resource into the runtime trainer/inference network representation at load time.
5. Validate graph shape compatibility against observation and action specs before training starts.
6. Keep import support for old `HiddenLayerSizes` configs so existing scenes continue to load.

### E. Add optional reward normalization

Current finding:
- Reward shaping currently depends entirely on raw reward scale.
- There is no running normalization layer at the academy or trainer level.

How to fix it:
1. Add an academy/training-config flag for reward normalization.
2. Track running mean/std of rewards or returns.
3. Apply normalized rewards consistently during training only.
4. Log both raw and normalized reward values for debugging.

### F. Tighten trainer bookkeeping and metrics

Current finding:
- Periodic checkpoints are created with incomplete bookkeeping data during updates.
- Some trainer metrics are placeholders rather than directly meaningful optimization metrics.

How to fix it:
1. Pass real run/update metadata into every checkpoint save path.
2. Make `TrainerUpdateStats` carry all logged metrics explicitly.
3. Ensure the dashboard only renders metrics that are mathematically defined and actually measured.

---
## 5. Inference — Full Feature Parity

### A. Implement continuous action inference

Current finding:
- `RLAcademy` currently warns that continuous actions are not supported for inference yet.
- Inference only drives discrete greedy action selection.

How to fix it:
1. Add continuous-action model loading and prediction.
2. Add deterministic continuous inference behavior for evaluation mode.
3. Route predicted vectors through the new `ActionBuffer` path.
4. Validate continuous action dimensions against checkpoint metadata before activating inference.

### B. Make inference algorithm-aware

Current finding:
- Inference currently instantiates `PolicyValueNetwork` directly.
- That is tightly coupled to the PPO-style discrete architecture.
- SAC checkpoints and continuous policies are not first-class citizens in the inference path.
- Current checkpoint loading also relies on the academy's `RLNetworkConfig` matching training-time architecture.

How to fix it:
1. Introduce an `IInferencePolicy` or `IPolicyNetwork` abstraction.
2. Store algorithm and architecture metadata in checkpoints.
3. Load the correct inference backend based on checkpoint metadata instead of editor config guesses.
4. Reject incompatible checkpoints before running the scene.

### C. Add in-editor inference mode

How to fix it:
1. Add an academy/editor command to run the loaded model inside the currently edited scene.
2. Avoid a full play-mode scene switch for quick sanity checks.
3. Show current control mode, loaded checkpoint, and active policy group in the editor UI.

### D. Add `RLInferenceRunner` for code-driven inference

Better target API:

```csharp
var runner = new RLInferenceRunner("res://runs/my_run/checkpoint_1000.rlmodel");
float[] obs = GetMyObservations();
var action = runner.Predict(obs);
```

How to fix it:
1. Wrap checkpoint loading, model selection, and action prediction in a reusable runtime helper.
2. Support both discrete and continuous outputs.
3. Add explicit validation errors for observation/action mismatches.

---
## 6. Checkpoints — Robustness & Discoverability

### A. Store full training context in every checkpoint

Current finding:
- `RLCheckpoint` currently stores only run/update counters, raw weights, and layer shapes.
- It does not store algorithm kind, action schema, observation size contract, network config, or hyperparameters.
- This weak metadata is one reason inference has to guess.

Better target format:

```json
{
  "formatVersion": 2,
  "algorithm": "PPO",
  "obsSize": 12,
  "actionSize": 4,
  "hyperparams": { "lr": 3e-4, "gamma": 0.99 },
  "weights": [...]
}
```

How to fix it:
1. Add `formatVersion`, algorithm kind, observation spec, action spec, and config metadata.
2. Include enough data to rebuild the correct inference network without consulting scene resources.
3. Add backward-compatible loaders for old checkpoints.
4. Validate checkpoint metadata before applying weights.

### B. Surface `CheckpointRegistry` in the dashboard

Current finding:
- `CheckpointRegistry` exists, but it is mostly an opaque file finder.
- The dashboard does not present a checkpoint timeline, rollback actions, or checkpoint annotations.

How to fix it:
1. Extend the dashboard with a per-run checkpoint list.
2. Show save step, episode count, reward snapshot, algorithm, and group.
3. Allow rollback/export from any checkpoint entry.
4. Group checkpoints by policy group when multi-policy runs are present.

### C. Fix checkpoint bookkeeping bugs

Current finding:
- Periodic trainer checkpoints currently do not carry enough reliable run metadata.
- Checkpoint payloads should reflect the real run id, group id, and update count consistently.

How to fix it:
1. Separate `RunId` from `GroupId` in the checkpoint payload.
2. Always pass the real update count when saving checkpoints during training.
3. Add regression tests around checkpoint save/load round-trips.

---
## 7. Editor DX — Faster Iteration

### A. Show live scene validation in the inspector and dock

Current finding:
- The dock already shows validation summaries.
- The indicator is useful, but it is not yet a fully live scene-health system.

How to fix it:
1. Add a compact green/red/yellow status indicator on `RLAcademy`.
2. Mirror the same state in the dock and toolbar.
3. Recompute validation after relevant edits, not just scene switches.

### B. Replace disk polling with named pipes or `MemoryMappedFile`

Current finding:
- The dashboard currently polls metrics and status files every 2 seconds.
- This adds latency and unnecessary file I/O.

How to fix it:
1. Introduce a lightweight IPC channel between the training process and the editor.
2. Keep file-based logging as a fallback/export path.
3. Switch the dashboard to push-driven updates when IPC is available.

### C. Add a "Quick Test" mode

How to fix it:
1. Add a launch mode that forces `BatchSize = 1` and stops after a small episode count.
2. Show a short summary report at the end of the run.
3. Keep this separate from full training so it is safe for iteration.

### D. Disable invalid launches and show the first error directly

Current finding:
- The current plugin blocks launch after validation, but the toolbar button is not disabled ahead of time.
- The first validation error is not surfaced as the button tooltip.

How to fix it:
1. Disable `Start Training` whenever validation is invalid.
2. Set the tooltip to the first blocking error.
3. Re-enable automatically when the scene becomes valid again.

### E. Add an observation/reward spy overlay

How to fix it:
1. Add a debug overlay for human-controlled or editor-test agents.
2. Show current observation values, named reward components, and chosen actions.
3. Allow pinning a specific agent in multi-agent scenes.

---
## 8. Advanced Features That the Architecture Should Enable

### A. Curriculum learning hook

Better target API:

```csharp
public virtual void OnTrainingProgress(float progress) { }
```

How to fix it:
1. Add periodic training-progress callbacks on `RLAcademy` or policy groups.
2. Let scenes adapt difficulty over time in a controlled way.
3. Log curriculum phase changes in run metrics.

### B. Custom network architectures through an interface

Better target API:

```csharp
public interface IPolicyNetwork
{
    PolicyDecision SampleAction(float[] obs);
    float EstimateValue(float[] obs);
    void ApplyGradient(TrainingSample sample);
    void Save(string path);
    void Load(string path);
}
```

How to fix it:
1. Introduce a network interface for training and inference.
2. Move `PolicyValueNetwork` and SAC network implementations behind the interface.
3. Keep the built-in MLP path as the default implementation.
4. Document extension points for recurrent, convolutional, or attention-based policies.

### C. Support multiple observation streams

Better target API:

```csharp
buffer.AddVector("position", Position);
buffer.AddImage("camera", GetCameraPixels(), 64, 64, 3);
```

How to fix it:
1. Extend observation collection to support named streams and typed modalities.
2. Add vector/image schemas to the observation spec.
3. Let the training backend choose the correct encoder stack per stream.

### D. Add evaluation rollouts

Current finding:
- Training metrics currently mix learning-time behavior and policy-quality signals.
- There is no built-in greedy evaluation pass.

How to fix it:
1. Run periodic evaluation episodes with exploration disabled.
2. Log evaluation reward separately from training reward.
3. Surface both curves in the dashboard.

---
## Priority Order

Suggested implementation order for the first five items:
1. Replace attribute/reflection actions with `DefineActions` / `OnActionsReceived`.
2. Remove the dual lifecycle API so `OnStep()` is the only supported path.
3. Auto-infer observation/action sizes in `RLAcademy` and validate before launch.
4. Implement continuous action inference and make inference algorithm-aware.
5. Store full checkpoint metadata so inference, validation, and exports can trust the file.

| Implemented | Priority | Item |
| --- | --- | --- |
| Yes | 1 (blocking) | Replace attribute/reflection actions with `DefineActions` / `OnActionsReceived` |
| Yes | 1 (blocking) | Remove the dual lifecycle API so `OnStep()` is the only supported path |
| Yes | 1 (blocking) | Auto-infer observation/action sizes in `RLAcademy` and validate before launch |
| Yes | 1 (blocking) | Implement continuous action inference |
| Yes | 1 (blocking) | Make inference algorithm-aware instead of hardwiring `PolicyValueNetwork` |
| Yes | 1 (blocking) | Store full checkpoint metadata so inference and validation can trust the file |
| Yes | 2 (high) | Replace string `PolicyGroup` with `RLPolicyGroupConfig` |
| Yes | 2 (high) | Add typed `ObservationBuffer`, sensors, and observation-size validation |
| Yes | 2 (high) | Disable invalid training launches and keep validation live while editing |
| Yes | 2 (high) | Fix checkpoint bookkeeping so run id / update count are always correct |
| Yes | 3 (medium) | Finish true batched forward passes on top of the existing multi-scene `BatchSize` support |
| Yes | 3 (medium) | PPO correctness fixes: shuffle, minibatches, gradient clip, value clip, clip fraction |
| Yes | 3 (medium) | Replace array-only network configs with a neural-network graph builder |
| Yes | 3 (medium) | Named reward components with metrics breakdown |
| Yes | 3 (medium) | Observation/reward spy overlay |
| Yes | 3 (medium) | Built-in self-play and historical opponent sampling (+ PFSP, Elo tracking, PolicyPool) |
| Yes | 3 (medium) | Surface checkpoint history in the dashboard on top of the existing registry/export work |
| Yes | 4 (nice to have) | Hyperparameter schedules |
| Yes | 4 (nice to have) | `RLInferenceRunner` helper |
| Yes | 4 (nice to have) | In-editor quick inference mode |
| No | 4 (nice to have) | Quick Test mode |
| Yes | 4 (nice to have) | Remove duplicated inline agent config vs `RLAgentConfig` resource state |
| No | 5 (advanced) | Curriculum learning hook |
| Yes | 5 (advanced) | `IPolicyNetwork` interface for custom architectures |
| No | 5 (advanced) | Multi-observation / heterogeneous input support |
| No | 5 (advanced) | Evaluation rollouts |
