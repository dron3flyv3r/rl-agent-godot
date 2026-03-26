# Implementation Plan: Inference Overhaul

Covers three tightly coupled improvements:
1. Implement continuous action inference
2. Make inference algorithm-aware instead of hardwiring `PolicyValueNetwork`
3. Store full checkpoint metadata so inference and validation can trust the file

These three are treated as one unit because each depends on the others:
- Continuous inference requires knowing which network to load, which requires checkpoint metadata.
- Algorithm-aware loading requires the checkpoint to self-describe its algorithm and action schema.
- Full metadata is most valuable when inference can actually use it to pick the right backend.

---

## Current State

| Problem | Location |
|---|---|
| `RLAcademy.TryInitializeInference()` skips agents with continuous actions with a warning | `RLAcademy.cs:226-230` |
| `PolicyValueNetwork.SelectGreedyAction()` is argmax-only — no continuous path | `PolicyValueNetwork.cs:148-163` |
| `RLAcademy` creates `PolicyValueNetwork` directly for every inference agent | `RLAcademy.cs:277-289` |
| `RLCheckpoint` carries no algorithm tag, no action schema, no hyperparameter metadata | `RLCheckpoint.cs:8-15` |
| `RLModelLoader` reads binary format version 1 with no extended metadata section | `RLModelLoader.cs:67` |
| Inference reads `obsSize` and `actionCount` by indexing raw `LayerShapeBuffer` offsets | `RLAcademy.cs:267-269` |
| `SacTrainer` has no checkpoint path that feeds inference; only `PpoTrainer`'s network is loadable | `SacTrainer.cs` |

---

## Step 1 — Extend `RLCheckpoint` with full training metadata

**Goal**: The checkpoint file is the single source of truth. No field should require consulting scene resources.

### 1.1 Add metadata fields to `RLCheckpoint`

File: `addons/rl-agent-plugin/Resources/RLCheckpoint.cs`

Add the following properties alongside the existing ones:

```csharp
// Format version — bump to 2 for the new schema
public int FormatVersion { get; set; } = 2;

// Algorithm that produced this checkpoint
public string Algorithm { get; set; } = "PPO"; // "PPO" | "SAC"

// Observation and action dimensions explicitly stored
public int ObservationSize { get; set; }
public int DiscreteActionCount { get; set; }
public int ContinuousActionDimensions { get; set; }

// Named discrete action labels per group, e.g. {"Movement": ["Idle","Up","Down","Left","Right"]}
public Dictionary<string, string[]> DiscreteActionLabels { get; set; } = new();

// Named continuous action ranges per group, e.g. {"Steering": {Dims:2, Min:-1, Max:1}}
public Dictionary<string, (int Dims, float Min, float Max)> ContinuousActionRanges { get; set; } = new();

// Snapshot of key training hyperparameters (informational, not used to reconstruct network)
public Dictionary<string, float> Hyperparams { get; set; } = new();

// Hidden layer sizes used to build the network
public int[] HiddenLayerSizes { get; set; } = Array.Empty<int>();
```

### 1.2 Update `RLCheckpoint.SaveToFile()`

File: `addons/rl-agent-plugin/Resources/RLCheckpoint.cs:21-62`

Write all new fields into the JSON dict under a `"meta"` key so the format is clearly versioned:

```json
{
  "format_version": 2,
  "run_id": "...",
  "total_steps": 12000,
  "episode_count": 340,
  "update_count": 120,
  "meta": {
    "algorithm": "PPO",
    "obs_size": 12,
    "discrete_action_count": 5,
    "continuous_action_dims": 0,
    "hidden_layer_sizes": [128, 128],
    "hyperparams": { "lr": 0.0003, "gamma": 0.99 },
    "discrete_action_labels": { "Movement": ["Idle","Up","Down","Left","Right"] },
    "continuous_action_ranges": {}
  },
  "shapes": [...],
  "weights": [...]
}
```

### 1.3 Update `RLCheckpoint.LoadFromFile()`

File: `addons/rl-agent-plugin/Resources/RLCheckpoint.cs:67-115`

- Read `format_version`. If it is `1` (old format), fill in defaults and derive `ObservationSize` / `DiscreteActionCount` from `LayerShapeBuffer` as before (backward-compat path).
- If `format_version >= 2`, read the `"meta"` block and populate all new fields.

### 1.4 Update `RLModelExporter.Export()` and `RLModelLoader.LoadFromFile()`

Files: `RLModelExporter.cs:40-100`, `RLModelLoader.cs:26-101`

Binary `.rlmodel` format bump to version 2:
- After the existing per-layer weight data, append a metadata JSON blob (length-prefixed string) so the binary format is self-describing.
- `RLModelLoader` reads version byte. Version 1 → parse weights only, synthesize defaults. Version 2 → parse weights then read metadata blob into `RLCheckpoint`.

### 1.5 Wire metadata into trainer checkpoint creation

Files: `PpoTrainer.cs`, `SacTrainer.cs`, `PolicyValueNetwork.cs:111-132`

When `CreateCheckpoint(...)` or `SaveCheckpoint(...)` is called, pass through the algorithm name, action schema, hidden sizes, and current hyperparameters.

- `PpoTrainer.CreateCheckpoint()` → sets `Algorithm = "PPO"`, fills `DiscreteActionCount`, `HiddenLayerSizes`, and `Hyperparams` from `RLTrainerConfig` / `RLNetworkConfig`.
- `SacTrainer.CreateCheckpoint()` → sets `Algorithm = "SAC"`, fills `ContinuousActionDimensions` or `DiscreteActionCount` depending on `_isContinuous`.
- Both trainers receive the `ActionSpaceBuilder` from the agent at startup so they can copy `DiscreteActionLabels` and `ContinuousActionRanges`.

**Where to get the `ActionSpaceBuilder`**: `TrainingBootstrap` already builds policy groups from agents at startup (`TrainingBootstrap.cs:232`). Pass each group's first agent's `ActionSpaceBuilder` into the corresponding trainer constructor.

---

## Step 2 — Introduce `IInferencePolicy` abstraction

**Goal**: Decouple `RLAcademy` from `PolicyValueNetwork` so that the correct backend is selected based on checkpoint metadata.

### 2.1 Define the interface

New file: `addons/rl-agent-plugin/Runtime/IInferencePolicy.cs`

```csharp
public interface IInferencePolicy
{
    void LoadCheckpoint(RLCheckpoint checkpoint);
    // Returns a PolicyDecision with either DiscreteAction or ContinuousActions populated
    PolicyDecision Predict(float[] observation);
}
```

`PolicyDecision` already exists in the codebase and holds both discrete and continuous fields, so no new type is needed.

### 2.2 Implement `PpoInferencePolicy`

New file: `addons/rl-agent-plugin/Runtime/PpoInferencePolicy.cs`

Wraps the existing `PolicyValueNetwork`:

```csharp
public class PpoInferencePolicy : IInferencePolicy
{
    private readonly PolicyValueNetwork _network;

    public PpoInferencePolicy(int obsSize, int actionCount, RLNetworkConfig config)
    {
        _network = new PolicyValueNetwork(obsSize, actionCount, config);
    }

    public void LoadCheckpoint(RLCheckpoint checkpoint) =>
        _network.LoadCheckpoint(checkpoint);

    public PolicyDecision Predict(float[] observation)
    {
        int action = _network.SelectGreedyAction(observation);
        return new PolicyDecision { DiscreteAction = action };
    }
}
```

This is a thin wrapper — no logic change to `PolicyValueNetwork`.

### 2.3 Implement `SacInferencePolicy` for continuous actions

New file: `addons/rl-agent-plugin/Runtime/SacInferencePolicy.cs`

`SacNetwork` already exists and is used by `SacTrainer`. Re-use it for inference:

```csharp
public class SacInferencePolicy : IInferencePolicy
{
    private readonly SacNetwork _network;
    private readonly bool _isContinuous;

    public SacInferencePolicy(int obsSize, int actionDims, bool isContinuous, RLNetworkConfig config)
    {
        _network = new SacNetwork(obsSize, actionDims, isContinuous, config);
        _isContinuous = isContinuous;
    }

    public void LoadCheckpoint(RLCheckpoint checkpoint) =>
        _network.LoadActorCheckpoint(checkpoint); // loads actor weights only

    public PolicyDecision Predict(float[] observation)
    {
        if (_isContinuous)
        {
            // Deterministic: use tanh(mean) without noise
            float[] actions = _network.DeterministicContinuousAction(observation);
            return new PolicyDecision { ContinuousActions = actions };
        }
        else
        {
            // Greedy discrete: argmax over actor logits
            int action = _network.GreedyDiscreteAction(observation);
            return new PolicyDecision { DiscreteAction = action };
        }
    }
}
```

**New methods needed on `SacNetwork`**:
- `DeterministicContinuousAction(float[] obs)` — forward pass through actor, apply tanh to mean (no sampled noise). This is the standard evaluation-mode behaviour for SAC continuous policies.
- `GreedyDiscreteAction(float[] obs)` — argmax over actor logits for discrete SAC.
- `LoadActorCheckpoint(RLCheckpoint checkpoint)` — load only actor weights (SAC checkpoints bundle actor + Q-networks; inference only needs the actor).

### 2.4 Add `InferencePolicyFactory`

New file: `addons/rl-agent-plugin/Runtime/InferencePolicyFactory.cs`

```csharp
public static class InferencePolicyFactory
{
    public static IInferencePolicy Create(RLCheckpoint checkpoint, RLNetworkConfig networkConfig)
    {
        return checkpoint.Algorithm switch
        {
            "SAC" => new SacInferencePolicy(
                checkpoint.ObservationSize,
                checkpoint.ContinuousActionDimensions > 0
                    ? checkpoint.ContinuousActionDimensions
                    : checkpoint.DiscreteActionCount,
                checkpoint.ContinuousActionDimensions > 0,
                networkConfig),
            _ => new PpoInferencePolicy(
                checkpoint.ObservationSize,
                checkpoint.DiscreteActionCount,
                networkConfig)
        };
    }
}
```

For old format-version-1 checkpoints, `Algorithm` defaults to `"PPO"` and `DiscreteActionCount` is derived from layer shapes — preserving full backward compatibility.

---

## Step 3 — Refactor `RLAcademy` inference to use the factory

**Goal**: Remove all hardwired `PolicyValueNetwork` references from `RLAcademy` and remove the continuous-action exclusion.

### 3.1 Replace `_agentInferenceNetworks` dictionary type

File: `RLAcademy.cs:63-66`

```csharp
// Before
private readonly Dictionary<RLAgent2D, PolicyValueNetwork> _agentInferenceNetworks = new();

// After
private readonly Dictionary<RLAgent2D, IInferencePolicy> _agentInferencePolicies = new();
```

### 3.2 Refactor `TryInitializeInference()`

File: `RLAcademy.cs:192-296`

Key changes:

1. **Remove the continuous-action skip block** (`lines 226-230`). It is no longer needed.

2. **After loading the checkpoint**, validate dimensions using the explicit metadata fields instead of indexing `LayerShapeBuffer` offsets:
   ```csharp
   // Before (fragile index arithmetic)
   var obsSize     = checkpoint.LayerShapeBuffer[0];
   var actionCount = checkpoint.LayerShapeBuffer[(layerCount - 2) * 3 + 1];

   // After (explicit metadata)
   int obsSize    = checkpoint.ObservationSize;
   int actionCount = checkpoint.DiscreteActionCount > 0
       ? checkpoint.DiscreteActionCount
       : checkpoint.ContinuousActionDimensions;
   ```

3. **Replace direct `PolicyValueNetwork` construction** with the factory:
   ```csharp
   // Before
   var network = new PolicyValueNetwork(obsSize, actionCount, networkConfig);
   network.LoadCheckpoint(checkpoint);
   _agentInferenceNetworks[agent] = network;

   // After
   var policy = InferencePolicyFactory.Create(checkpoint, networkConfig);
   policy.LoadCheckpoint(checkpoint);
   _agentInferencePolicies[agent] = policy;
   ```

4. **Add a dimension-mismatch guard** using the new explicit fields:
   ```csharp
   int agentObsSize = agent.GetExpectedObservationSize(); // see note below
   if (obsSize != agentObsSize)
   {
       GD.PrintErr($"Checkpoint obs size {obsSize} != agent obs size {agentObsSize} for '{agent.Name}'");
       continue;
   }
   ```
   Note: `GetExpectedObservationSize()` calls `CollectObservationArray()` once on the agent to measure the live size. This is consistent with how `TrainingBootstrap` currently discovers sizes.

### 3.3 Refactor `_PhysicsProcess()` inference block

File: `RLAcademy.cs:91-133`

```csharp
// Before
agent.ApplyAction(network.SelectGreedyAction(observation));

// After
var decision = policy.Predict(observation);
if (decision.DiscreteAction >= 0)
    agent.ApplyAction(decision.DiscreteAction);
else if (decision.ContinuousActions?.Length > 0)
    agent.ApplyAction(decision.ContinuousActions);
```

Action repeat stays the same — the pending action fields on the agent already handle both discrete and continuous (same pattern as `TrainingBootstrap.ApplyDecision()`).

---

## Step 4 — Extend `SacNetwork` to support deterministic inference

File: `addons/rl-agent-plugin/Runtime/SacNetwork.cs`

### 4.1 Add `DeterministicContinuousAction(float[] obs)`

During training, `SacNetwork.SampleContinuousAction()` adds Gaussian noise (reparameterization). For inference, skip the noise and return `tanh(mean)` directly:

```csharp
public float[] DeterministicContinuousAction(float[] obs)
{
    float[] mean = _actor.Forward(obs); // existing forward pass returns pre-tanh mean
    var result = new float[mean.Length];
    for (int i = 0; i < mean.Length; i++)
        result[i] = MathF.Tanh(mean[i]);
    return result;
}
```

### 4.2 Add `GreedyDiscreteAction(float[] obs)`

```csharp
public int GreedyDiscreteAction(float[] obs)
{
    float[] logits = _actor.Forward(obs);
    int best = 0;
    for (int i = 1; i < logits.Length; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}
```

### 4.3 Add `LoadActorCheckpoint(RLCheckpoint checkpoint)`

SAC saves the full network (actor + Q networks). For inference only the actor weights are needed. Add a method that loads only the actor's weight slice from the checkpoint buffer, identified by a convention (e.g. actor layers are stored first in the weight buffer, as they are during `SacNetwork.CreateCheckpoint()`).

If the checkpoint format is not already split by component, the simplest change is to save actor weights into a separate checkpoint key (`"actor_weights"`, `"actor_shapes"`) when `Algorithm == "SAC"` and read from that key in `LoadActorCheckpoint()`. Update `SacTrainer.CreateCheckpoint()` accordingly.

---

## Step 5 — Validation improvements

### 5.1 Editor validation: check algorithm/checkpoint compatibility

File: `addons/rl-agent-plugin/Editor/RLAgentPluginEditor.cs` (or wherever scene validation runs)

When an agent has `ControlMode == Inference` and a `InferenceCheckpointPath` is set, load the checkpoint header and check:
- `ObservationSize` matches the agent's declared/inferred observation size.
- If `Algorithm == "SAC"` and the agent has discrete actions in its `ActionSpaceBuilder`, warn that the checkpoint was trained with SAC discrete — confirm this is intentional.
- If `Algorithm == "PPO"` but `ContinuousActionDimensions > 0`, report an error (PPO can't produce continuous actions).

These checks run in the editor before launch. Surfacing them here means the user knows about mismatches before clicking Start.

### 5.2 Runtime pre-flight in `TryInitializeInference()`

Keep runtime validation as a final safety net. With explicit checkpoint metadata, the checks are now:

```csharp
if (checkpoint.Algorithm == "PPO" && agent.HasContinuousActions())
    GD.PrintErr($"Checkpoint is PPO (discrete-only) but agent '{agent.Name}' expects continuous actions.");

if (checkpoint.ContinuousActionDimensions > 0
    && checkpoint.ContinuousActionDimensions != agent.ActionSpaceBuilder.TotalContinuousDimensions)
    GD.PrintErr($"Continuous dims mismatch: checkpoint={checkpoint.ContinuousActionDimensions}, agent={...}");
```

---

## Step 6 — Update `RLModelExporter` and binary format

File: `addons/rl-agent-plugin/Runtime/RLModelExporter.cs:40-100`

The `.rlmodel` binary export is what end-users ship. It must also be self-describing.

1. Bump the version byte to `2`.
2. After writing all layer data (as in version 1), append a metadata section:
   - Write a 4-byte int: length of the following JSON string in UTF-8 bytes.
   - Write the JSON string containing all `RLCheckpoint` metadata fields (excluding the raw weight arrays, which are already encoded as binary).
3. `RLModelLoader` reads the version byte. For version `1`, return a checkpoint with synthesised defaults. For version `2`, parse the metadata section and populate all fields.

This gives `.rlmodel` files full self-description without breaking existing exported models.

---

## Implementation Order

Work through steps in the following sequence so each change is testable in isolation:

1. **Step 1** — Extend `RLCheckpoint` schema and save/load. Run a training session and verify the JSON checkpoint contains the new `"meta"` block. Check that old checkpoints still load (format_version == 1 fallback).

2. **Step 4** — Add `DeterministicContinuousAction`, `GreedyDiscreteAction`, and `LoadActorCheckpoint` to `SacNetwork`. Unit-testable independently of `RLAcademy`.

3. **Step 2** — Introduce `IInferencePolicy`, `PpoInferencePolicy`, `SacInferencePolicy`, and `InferencePolicyFactory`. No changes to `RLAcademy` yet — verifiable by instantiation tests.

4. **Step 3** — Refactor `RLAcademy`. Replace `PolicyValueNetwork` dict with `IInferencePolicy` dict, remove the continuous-action skip, and update `_PhysicsProcess`. Test with a discrete PPO agent first to confirm no regression, then test with a SAC continuous agent.

5. **Step 5** — Add validation. Editor-side first, runtime second. Confirm that mismatched checkpoints are caught before launch.

6. **Step 6** — Update `RLModelExporter` and bump the binary format. Export a model, verify it loads correctly in a fresh session.

---

## Files Changed Summary

| File | Change |
|---|---|
| `Resources/RLCheckpoint.cs` | Add metadata fields; update SaveToFile / LoadFromFile |
| `Runtime/PpoTrainer.cs` | Pass algorithm + schema to CreateCheckpoint |
| `Runtime/SacTrainer.cs` | Pass algorithm + schema to CreateCheckpoint; actor-only checkpoint key |
| `Runtime/PolicyValueNetwork.cs` | Pass metadata when building checkpoint in SaveCheckpoint |
| `Runtime/SacNetwork.cs` | Add DeterministicContinuousAction, GreedyDiscreteAction, LoadActorCheckpoint |
| `Runtime/IInferencePolicy.cs` | **New** — interface |
| `Runtime/PpoInferencePolicy.cs` | **New** — wraps PolicyValueNetwork |
| `Runtime/SacInferencePolicy.cs` | **New** — wraps SacNetwork |
| `Runtime/InferencePolicyFactory.cs` | **New** — creates correct backend from checkpoint metadata |
| `Runtime/RLAcademy.cs` | Replace PolicyValueNetwork dict; remove continuous skip; use factory; update physics loop |
| `Runtime/RLModelExporter.cs` | Bump binary format to v2; write metadata section |
| `Runtime/RLModelLoader.cs` | Read v2 metadata section; v1 backward-compat fallback |
| `Scenes/TrainingBootstrap.cs` | Pass ActionSpaceBuilder to trainer constructors for schema capture |
| `Editor/RLAgentPluginEditor.cs` | Add checkpoint/agent compatibility checks to editor validation |
