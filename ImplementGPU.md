# GPU Accelerated CNN Training — Implementation Plan

## Motivation

PPO background updates with camera observations take ~9–10 seconds per update on CPU.
The bottleneck is CNN forward+backward running sample-by-sample on a single CPU core.
Goal: move CNN training to GPU via Vulkan compute, targeting <1 second per update.

## Why Vulkan (not CUDA)

- CUDA is NVIDIA-only. Vulkan works on both NVIDIA and AMD.
- Godot exposes full Vulkan compute via `RenderingDevice` — no external dependencies.
- FP32 precision is sufficient for RL training (same as current CPU path).
- Speed gap vs CUDA is negligible at our scale (small CNN, not transformer-scale).

---

## Architecture

```
IEncoder (interface)
├── CnnEncoder       ← existing C++ GDExtension (CPU, headless workers, fallback)
└── GpuCnnEncoder    ← new Vulkan compute via RenderingDevice (training master)
```

`PolicyValueNetwork` changes its `CnnEncoder?` field to `IEncoder?`. All trainers and
call sites are unaffected — they go through `PolicyValueNetwork.ApplyGradients` as before.

### Thread model

`RenderingServer.CreateLocalRenderingDevice()` returns a thread-local `RenderingDevice`.
The background training thread creates and owns its own GPU context — completely
independent of the main render/game loop. No synchronisation with gameplay needed.

---

## Shaders

Seven GLSL 4.50 compute shaders, compiled to SPIR-V at startup via
`RenderingDevice.ShaderCompileSpirVFromSource`. Stored as embedded C# string constants
in `GpuShaderSources.cs` (avoids file I/O, easier to distribute).

| File | Purpose |
|------|---------|
| `conv_forward.glsl` | Im2col inline + matmul + ReLU fused, writes `CachedCol` + `CachedPreact` for backprop |
| `conv_backward_filter.glsl` | `dL/dW = col^T @ dOut` — accumulates into `GradW` buffer |
| `conv_backward_input.glsl` | `dL/dInput` via transposed convolution |
| `relu_backward.glsl` | Gates gradient through ReLU mask using `CachedPreact` |
| `linear_forward.glsl` | Tiled 16×16 GEMM for projection layer |
| `linear_backward.glsl` | `dW` and `dInput` for the linear projection layer |
| `adam_update.glsl` | Adam step + zeroes grad buffer in one pass |

### Memory layout

- **HWC throughout** — matches the existing native `RlCnnEncoder`, no transposing on checkpoint load/save.
- **Im2col approach** — reduces convolution to GEMM, universally efficient on both NVIDIA and AMD
  without vendor-specific intrinsics. The im2col expansion (`CachedCol`) is written during forward
  and reused during the weight gradient pass.

### Workgroup sizes

| Shader | `local_size` | Dispatch over |
|--------|-------------|---------------|
| `conv_forward` | 8×8×1 | `(outH*outW, outC, N)` |
| `conv_backward_filter` | 256×1×1 | one workgroup per `(oc, kh*kw*ic)`, reduce over `N*outH*outW` |
| `conv_backward_input` | 256×1×1 | one thread per input element |
| `relu_backward` | 256×1×1 | 1D over all preact elements |
| `linear_forward` | 16×16×1 | tiled GEMM over `(N, outSize)` |
| `linear_backward` | 16×16×1 | tiled GEMM |
| `adam_update` | 256×1×1 | one thread per parameter |

---

## New Files

```
addons/rl-agent-plugin/Runtime/Networks/IEncoder.cs
addons/rl-agent-plugin/Runtime/Gpu/GpuDevice.cs
addons/rl-agent-plugin/Runtime/Gpu/GpuCnnEncoder.cs
addons/rl-agent-plugin/Runtime/Gpu/GpuShaderSources.cs
addons/rl-agent-plugin/Runtime/Gpu/Shaders/conv_forward.glsl
addons/rl-agent-plugin/Runtime/Gpu/Shaders/conv_backward_filter.glsl
addons/rl-agent-plugin/Runtime/Gpu/Shaders/conv_backward_input.glsl
addons/rl-agent-plugin/Runtime/Gpu/Shaders/relu_backward.glsl
addons/rl-agent-plugin/Runtime/Gpu/Shaders/linear_forward.glsl
addons/rl-agent-plugin/Runtime/Gpu/Shaders/linear_backward.glsl
addons/rl-agent-plugin/Runtime/Gpu/Shaders/adam_update.glsl
```

## Modified Files

```
addons/rl-agent-plugin/Runtime/Networks/CnnEncoder.cs          — implement IEncoder
addons/rl-agent-plugin/Runtime/Networks/PolicyValueNetwork.cs  — IEncoder?, batched training path
```

---

## IEncoder Interface

```csharp
public interface IEncoder
{
    int OutputSize { get; }
    bool SupportsBatchedTraining { get; }

    // Single-sample path — used during rollout collection (main thread, CPU encoder only)
    float[] Forward(float[] input);

    // Per-sample gradient accumulation — CPU path (PPO/SAC sample-by-sample loop)
    ICnnGradientToken CreateGradientToken();
    float[]           AccumulateGradients(float[] outputGrad, ICnnGradientToken token);
    void              ApplyGradients(ICnnGradientToken token, float lr, float gradScale);
    float             GradNormSquared(ICnnGradientToken token);

    // Batched training path — GPU encoder implements natively; CPU encoder wraps the loop
    void   ForwardBatch(float[] inputBatch, int batchSize, float[] outputBatch);
    void   AccumulateGradientsBatch(float[] outputGradBatch, int batchSize, ICnnGradientToken token);

    // Serialization (weights round-trip to CPU for checkpoints)
    void   AppendSerialized(ICollection<float> weights, ICollection<int> shapes);
    void   LoadSerialized(IReadOnlyList<float> w, ref int wi,
                          IReadOnlyList<int> s, ref int si);

    // Weight copy (SAC target network sync)
    void   CopyWeightsTo(IEncoder other);
}
```

`ICnnGradientToken` replaces `CnnGradientBuffer`. CPU implementation wraps the
native packed gradient buffer; GPU implementation holds encoder-owned GPU buffers.

---

## GpuCnnEncoder Structure

```csharp
public sealed class GpuCnnEncoder : IEncoder, IDisposable
{
    private RenderingDevice _rd;          // thread-local, created on training thread

    private struct GpuConvLayer {
        public RID Filters, Biases;       // weight buffers
        public RID Output, PreAct;        // forward cache
        public RID ColBuf;                // im2col scratch
        public RID GradW, GradB;          // gradient accumulators
        public RID MomentW1, MomentW2;    // Adam moments for weights
        public RID MomentB1, MomentB2;    // Adam moments for biases
        // geometry
        public int N, inC, inH, inW, outC, outH, outW, kH, kW, stride;
    }

    private struct GpuLinearLayer {
        public RID Weights, Biases;
        public RID Output;
        public RID GradW, GradB;
        public RID MomentW1, MomentW2, MomentB1, MomentB2;
        public int inSize, outSize;
    }

    private GpuConvLayer[]  _convLayers;
    private GpuLinearLayer  _proj;
    private RID             _inputBuf;
    private RID[]           _inputGradBufs;   // one per layer

    // Compiled pipelines
    private RID _pipelineConvFwd, _pipelineConvBwdFilter, _pipelineConvBwdInput;
    private RID _pipelineReluBwd, _pipelineLinFwd, _pipelineLinBwd;
    private RID _pipelineAdam;

    private int _maxBatchSize;
}
```

**Initialization** (called on the training thread):
1. `_rd = RenderingServer.CreateLocalRenderingDevice()`
2. Compile all shaders → SPIR-V → pipelines
3. Allocate weight/moment/gradient buffers, upload initial weights
4. Allocate scratch buffers sized for `_maxBatchSize`

**Fallback detection:**
```csharp
public static bool IsVulkanAvailable()
{
    try { using var rd = RenderingServer.CreateLocalRenderingDevice(); return rd != null; }
    catch { return false; }
}
```

---

## Integration into PolicyValueNetwork

Add `SupportsBatchedTraining` to `IEncoder`. When true, `PolicyValueNetwork.ApplyGradients`
uses the fast path:

```
1. Extract all image-stream slices into one contiguous float[] inputBatch
2. encoder.ForwardBatch(inputBatch, N, embeddingBatch)
3. CPU trunk/head loss computation still runs sample-by-sample using the cached embeddings
4. Collect dLoss/dEmbedding for the whole mini-batch
5. encoder.AccumulateGradientsBatch(dLoss_dEmbeddingBatch, N, token)
6. encoder.ApplyGradients(token, lr, gradScale)   ← Adam on GPU once per mini-batch
```

Trunk and head layers remain on CPU for now (they are small and not the bottleneck).

---

## Implementation Phases

### Phase 1 — Interface extraction ✅ COMPLETE
- Create `IEncoder` + `ICnnGradientToken`
- `CnnEncoder : IEncoder`
- `PolicyValueNetwork`: `CnnEncoder?` → `IEncoder?`
- **Validation:** timing identical, no behavior change

### Phase 2 — GPU infrastructure ✅ COMPLETE
- `GpuDevice.cs` — device lifecycle, buffer helpers, `IsAvailable()`
- `GpuShaderSources.cs` — embedded GLSL string constants (passthrough for now)
- `GpuCnnEncoder.cs` — skeleton: `Forward` uploads input, dispatches passthrough, reads back
- **Validation:** passthrough latency < 1ms on any discrete GPU

### Phase 3 — Linear layer on GPU ✅ COMPLETE
- `linear_forward.glsl`, `linear_backward.glsl`, `adam_update.glsl`
- `GpuCnnEncoder` with zero conv layers + one linear projection
- **Validation:** output and updated weights match CPU `DenseLayer` to < 1e-4

### Phase 4 — Conv forward pass ✅ COMPLETE
- `conv_forward.glsl` (im2col inline, ReLU fused)
- **Validation:** output matches `RlCnnEncoder` CPU to < 1e-4; 5–20× forward speedup

### Phase 5 — Conv backward pass ✅ COMPLETE
- `conv_backward_filter.glsl`, `conv_backward_input.glsl`, `relu_backward.glsl`
- Single-sample GPU forward/backward/update matches the managed/native reference
- **Validation:** forward, pixel grads, flat grads, grad norm, and Adam updates all match reference

### Phase 6 — Integration into batched training path ✅ COMPLETE
- `PolicyValueNetwork.ApplyGradients` now precomputes CNN embeddings for GPU encoders in mini-batches
- Per-sample CPU trunk/head backprop is preserved, but encoder output grads are accumulated and sent back
  through the GPU encoder once per mini-batch
- PPO trainer now routes image-observation training through GPU-backed training networks while rollout
  inference stays on CPU
- Distributed async PPO now caps master-side rollout growth during background training, so visible gameplay
  can continue without inflating the next training batch
- Normal checkpoint persistence was moved off the main thread to avoid every-10th-update hitching from ZIP writes
- **Validation:** `demo/06 BallTracker` now shows stable batched GPU training with `PPO.BackgroundUpdate`
  around ~0.5s on the target machine, well below the original CPU path

### Phase 7 — Serialization + fallback polish ✅ COMPLETE
- `AppendSerialized` / `LoadSerialized` — GPU buffer readback writes/reads weights via `DownloadBuffer`/`UploadBuffer`
- `CopyWeightsTo` — GPU-to-GPU (and CPU→GPU) weight copy via serialize round-trip; fast native path retained for CPU→CPU
- Headless worker fallback: `CreateImageEncoder` falls back to `CnnEncoder` with a warning when `GpuDevice.IsAvailable()` returns false; workers always use `preferGpuImageEncoders: false`
- Polished CPU native wrapper: `CnnEncoder.CopyWeightsTo` now supports any `IEncoder` target; `AccumulateGradientsBatch` replaced silent misuse with a clear `NotSupportedException` (callers must check `SupportsBatchedTraining`)
- **Validation:** GPU master saves checkpoint via `GpuCnnEncoder.AppendSerialized`; CPU worker loads via `CnnEncoder.LoadSerialized` — same shape descriptor format, round-trip verified

### Phase 8 — Performance polish ✅ COMPLETE
- **Batched compute lists** — `ForwardBatch`, `AccumulateGradientsBatch`, and `ApplyGradients` now each open a
  single `ComputeList`, record all their dispatches with `AddBarrier` between data-dependent stages, and call
  `EndSubmitAndSync` once. For a 2-layer CNN this collapses 3/7/6 Submit+Sync stalls per mini-batch step to 1 each.
- **GPU norm reduction** — `GradNormSquared` replaced N large CPU readbacks with a `NormSquaredAccumulate`
  compute shader: one workgroup per gradient buffer accumulates into a shared 1-float buffer, then only 1 float
  is read back. New shader added to `GpuShaderSources.cs`.
- **Dead code removed** — `DispatchAdamDense` and `DispatchAdamNative` (single-shot wrappers) deleted; all
  callers now use the list-based `AdamDenseToList` / `AdamNativeToList` helpers.
- **Remaining opportunity** — trunk/head CPU↔GPU round-trip (embedding readback after `ForwardBatch`) is the
  next bottleneck; eliminating it requires moving trunk/head layers to GPU (out of scope for this phase).
- **Validation:** run BallTracker with `RLProfiler` and compare `GpuCnn.ForwardBatch.Dispatch`,
  `GpuCnn.BackwardBatch.Dispatch`, `GpuCnn.ApplyGradients`, and `GpuCnn.GradNormSquared` timings against
  the old per-dispatch breakdown.

---

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| PPO update time | ~9 seconds | < 1 second |
| GPU vendor support | N/A | NVIDIA + AMD (Vulkan) |
| Headless worker | CPU C++ (unchanged) | CPU C++ (unchanged) |
| Checkpoint format | unchanged | unchanged |
