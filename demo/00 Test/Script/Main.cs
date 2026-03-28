using Godot;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;
using RlAgentPlugin.Runtime;

public partial class Main : Node
{
    public override void _Ready()
    {
        GD.Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        GD.Print("[GPU Test] Starting Phase 5 validation");
        GD.Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Run on a background thread — this is how the real training path will work.
        Task.Run(RunTests);
    }

    private static void RunTests()
    {
        // ── Test 1: Vulkan availability ───────────────────────────────────────
        GD.Print("\n[Test 1] Vulkan availability check");
        var available = GpuDevice.IsAvailable();
        GD.Print($"  GpuDevice.IsAvailable() = {available}");
        if (!available)
        {
            GD.PushError("[GPU Test] Vulkan not available — cannot continue. " +
                         "Make sure you are running with a Vulkan-capable GPU and display server.");
            return;
        }
        GD.Print("  PASS");

        // ── Test 2: Device creation on background thread ──────────────────────
        GD.Print("\n[Test 2] Device creation on background thread");
        try
        {
            using var gpu = new GpuDevice();
            GD.Print("  RenderingDevice created successfully");
            GD.Print("  PASS");
        }
        catch (Exception ex)
        {
            GD.PushError($"[GPU Test] Device creation failed: {ex.Message}");
            return;
        }

        // ── Test 3: Buffer upload → passthrough dispatch → readback ──────────
        GD.Print("\n[Test 3] Buffer round-trip (upload → passthrough → readback)");
        RunPassthroughTest(size: 16,   label: "small  (16 floats)");
        RunPassthroughTest(size: 256,  label: "medium (256 floats)");
        RunPassthroughTest(size: 4096, label: "large  (4096 floats — 64×64 grayscale)");

        // ── Test 4: GpuCnnEncoder linear projection vs CPU DenseLayer ─────────
        GD.Print("\n[Test 4] GpuCnnEncoder linear projection vs CPU DenseLayer");
        RunLinearProjectionComparison();

        // ── Test 5: Conv forward vs native CPU encoder ────────────────────────
        GD.Print("\n[Test 5] GpuCnnEncoder conv forward vs native CnnEncoder");
        RunConvForwardComparison();

        // ── Test 6: Conv training parity vs managed reference encoder ─────────
        GD.Print("\n[Test 6] GpuCnnEncoder conv training vs managed reference");
        RunConvTrainingComparison();

        // ── Test 7: Latency benchmark ─────────────────────────────────────────
        GD.Print("\n[Test 7] Passthrough latency benchmark (4096 floats, 100 calls)");
        RunLatencyBenchmark(floatCount: 4096, iterations: 100);

        GD.Print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        GD.Print("[GPU Test] Phase 5 validation complete");
        GD.Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static void RunPassthroughTest(int size, string label)
    {
        try
        {
            using var gpu = new GpuDevice();

            // Build known input: sequential values 0.0, 1.0, 2.0, ...
            var input = new float[size];
            for (var i = 0; i < size; i++) input[i] = i;

            // Compile shader and pipeline.
            var shader   = gpu.CompileComputeShader(GpuShaderSources.Passthrough);
            var pipeline = gpu.Rd.ComputePipelineCreate(shader);

            // Allocate buffers.
            var inputBuf  = gpu.CreateBuffer(size, input);
            var outputBuf = gpu.CreateBuffer(size);

            // Build uniform set and dispatch.
            var uniforms = new Godot.Collections.Array<RDUniform>();
            var u0 = new RDUniform { UniformType = RenderingDevice.UniformType.StorageBuffer, Binding = 0 };
            u0.AddId(inputBuf);
            uniforms.Add(u0);
            var u1 = new RDUniform { UniformType = RenderingDevice.UniformType.StorageBuffer, Binding = 1 };
            u1.AddId(outputBuf);
            uniforms.Add(u1);
            var uniformSet = gpu.Rd.UniformSetCreate(uniforms, shader, 0);

            var pc = GpuDevice.PushConstant((uint)size);

            var list = gpu.Rd.ComputeListBegin();
            gpu.Rd.ComputeListBindComputePipeline(list, pipeline);
            gpu.Rd.ComputeListBindUniformSet(list, uniformSet, 0);
            gpu.Rd.ComputeListSetPushConstant(list, pc, (uint)pc.Length);
            gpu.Rd.ComputeListDispatch(list, xGroups: ((uint)size + 255u) / 256u, yGroups: 1, zGroups: 1);
            gpu.Rd.ComputeListEnd();
            gpu.SubmitAndSync();

            // Read back and verify.
            var output = gpu.DownloadBuffer(outputBuf, size);

            var maxError = 0f;
            var firstBadIdx = -1;
            for (var i = 0; i < size; i++)
            {
                var err = MathF.Abs(output[i] - input[i]);
                if (err > maxError) { maxError = err; firstBadIdx = i; }
            }

            gpu.Rd.FreeRid(uniformSet);
            gpu.Rd.FreeRid(outputBuf);
            gpu.Rd.FreeRid(inputBuf);
            gpu.Rd.FreeRid(pipeline);
            gpu.Rd.FreeRid(shader);

            if (maxError < 1e-6f)
            {
                GD.Print($"  {label}  →  max_error={maxError:E2}  PASS");
            }
            else
            {
                GD.PushError($"  {label}  →  FAIL  max_error={maxError:E2} at index {firstBadIdx} " +
                             $"(expected {input[firstBadIdx]}, got {output[firstBadIdx]})");
            }
        }
        catch (Exception ex)
        {
            GD.PushError($"  {label}  →  EXCEPTION: {ex.Message}");
        }
    }

    private static void RunLinearProjectionComparison()
    {
        try
        {
            const int W = 4, H = 4, C = 1;
            const int InputSize = W * H * C;
            const int OutputSize = 8;

            var def = new RLCnnEncoderDef
            {
                FilterCounts = Array.Empty<int>(),
                KernelSizes  = Array.Empty<int>(),
                Strides      = Array.Empty<int>(),
                OutputSize   = OutputSize,
            };

            using var encoder = new GpuCnnEncoder(W, H, C, def);
            var dense = new DenseLayer(InputSize, OutputSize, null, RLOptimizerKind.Adam);

            var layerWeights = new float[InputSize * OutputSize + OutputSize];
            for (var i = 0; i < layerWeights.Length; i++)
                layerWeights[i] = ((i % 11) - 5) * 0.03125f;

            var gpuShapes = new[] { 0, InputSize, OutputSize };
            var denseShapes = new[] { (int)RLLayerKind.Dense, InputSize, OutputSize, 0 };
            var wi = 0;
            var si = 0;
            encoder.LoadSerialized(layerWeights, ref wi, gpuShapes, ref si);
            wi = 0;
            si = 0;
            dense.LoadSerialized(layerWeights, ref wi, denseShapes, ref si);

            var inputA = new float[InputSize];
            var inputB = new float[InputSize];
            for (var i = 0; i < InputSize; i++)
            {
                inputA[i] = ((i % 7) - 3) * 0.2f;
                inputB[i] = ((i % 5) - 2) * 0.15f;
            }

            var outputGradA = new float[OutputSize];
            var outputGradB = new float[OutputSize];
            for (var i = 0; i < OutputSize; i++)
            {
                outputGradA[i] = ((i % 3) - 1) * 0.25f;
                outputGradB[i] = ((i % 4) - 1.5f) * 0.2f;
            }

            var gpuToken = encoder.CreateGradientToken();
            var cpuBuffer = dense.CreateGradientBuffer();

            var gpuOutA = encoder.Forward(inputA);
            var cpuOutA = dense.Forward(inputA);
            var gpuInGradA = encoder.AccumulateGradients(outputGradA, gpuToken);
            var cpuInGradA = dense.AccumulateGradients(outputGradA, cpuBuffer);

            var gpuOutB = encoder.Forward(inputB);
            var cpuOutB = dense.Forward(inputB);
            var gpuInGradB = encoder.AccumulateGradients(outputGradB, gpuToken);
            var cpuInGradB = dense.AccumulateGradients(outputGradB, cpuBuffer);

            var gpuGradNorm = encoder.GradNormSquared(gpuToken);
            var cpuGradNorm = cpuBuffer.SumSquares();

            const float LearningRate = 0.001f;
            const float GradScale = 0.5f;
            encoder.ApplyGradients(gpuToken, LearningRate, GradScale);
            dense.ApplyGradients(cpuBuffer, LearningRate, GradScale);

            var gpuWeights = new List<float>();
            var gpuSerializedShapes = new List<int>();
            encoder.AppendSerialized(gpuWeights, gpuSerializedShapes);

            var cpuWeights = new List<float>();
            var cpuSerializedShapes = new List<int>();
            dense.AppendSerialized(cpuWeights, cpuSerializedShapes);

            var forwardErrA   = MaxError(gpuOutA, cpuOutA);
            var forwardErrB   = MaxError(gpuOutB, cpuOutB);
            var inputGradErrA = MaxError(gpuInGradA, cpuInGradA);
            var inputGradErrB = MaxError(gpuInGradB, cpuInGradB);
            var gradNormErr   = MathF.Abs(gpuGradNorm - cpuGradNorm);
            var weightErr     = MaxError(gpuWeights, cpuWeights);
            var shapeMatch    = ListsEqual(gpuSerializedShapes, new[] { 0, InputSize, OutputSize });

            GD.Print($"  forward A max_error     = {forwardErrA:E2}");
            GD.Print($"  forward B max_error     = {forwardErrB:E2}");
            GD.Print($"  input grad A max_error  = {inputGradErrA:E2}");
            GD.Print($"  input grad B max_error  = {inputGradErrB:E2}");
            GD.Print($"  grad norm error         = {gradNormErr:E2}");
            GD.Print($"  post-update weight err  = {weightErr:E2}");
            GD.Print($"  serialized shape match  = {shapeMatch}");

            var pass = forwardErrA < 1e-5f &&
                       forwardErrB < 1e-5f &&
                       inputGradErrA < 1e-5f &&
                       inputGradErrB < 1e-5f &&
                       gradNormErr < 1e-5f &&
                       weightErr < 1e-5f &&
                       shapeMatch;
            GD.Print(pass ? "  PASS" : "  FAIL");
        }
        catch (Exception ex)
        {
            GD.PushError($"  Linear projection comparison EXCEPTION: {ex.Message}");
        }
    }

    private static void RunConvForwardComparison()
    {
        try
        {
            const int W = 8, H = 8, C = 1;
            const int OutputSize = 12;

            var def = new RLCnnEncoderDef
            {
                FilterCounts = new[] { 4, 6 },
                KernelSizes  = new[] { 3, 3 },
                Strides      = new[] { 1, 1 },
                OutputSize   = OutputSize,
            };

            using var gpu = new GpuCnnEncoder(W, H, C, def);
            var cpu = new ManagedCnnReference(W, H, C, def);

            var cpuWeights = new List<float>();
            var shapes = new List<int>();
            cpu.AppendSerialized(cpuWeights, shapes);

            var deterministicWeights = new float[cpuWeights.Count];
            for (var i = 0; i < deterministicWeights.Length; i++)
                deterministicWeights[i] = ((i % 17) - 8) * 0.015625f;

            var wi = 0;
            var si = 0;
            cpu.LoadSerialized(deterministicWeights, ref wi, shapes, ref si);
            wi = 0;
            si = 0;
            gpu.LoadSerialized(deterministicWeights, ref wi, shapes, ref si);

            var inputA = new float[W * H * C];
            var inputB = new float[W * H * C];
            for (var i = 0; i < inputA.Length; i++)
            {
                inputA[i] = ((i % 9) - 4) * 0.125f;
                inputB[i] = ((i % 11) - 5) * 0.09375f;
            }

            var cpuOutA = cpu.Forward(inputA);
            var gpuOutA = gpu.Forward(inputA);
            var cpuOutB = cpu.Forward(inputB);
            var gpuOutB = gpu.Forward(inputB);

            var gpuRoundTripWeights = new List<float>();
            var gpuRoundTripShapes = new List<int>();
            gpu.AppendSerialized(gpuRoundTripWeights, gpuRoundTripShapes);

            var forwardErrA = MaxError(cpuOutA, gpuOutA);
            var forwardErrB = MaxError(cpuOutB, gpuOutB);
            var weightErr   = MaxError(deterministicWeights, gpuRoundTripWeights);
            var shapesMatch = ListsEqual(shapes, gpuRoundTripShapes);

            GD.Print($"  conv forward A max_error = {forwardErrA:E2}");
            GD.Print($"  conv forward B max_error = {forwardErrB:E2}");
            GD.Print($"  round-trip weight error  = {weightErr:E2}");
            GD.Print($"  serialized shape match   = {shapesMatch}");

            var pass = forwardErrA < 1e-4f &&
                       forwardErrB < 1e-4f &&
                       weightErr < 1e-7f &&
                       shapesMatch;
            GD.Print(pass ? "  PASS" : "  FAIL");
        }
        catch (Exception ex)
        {
            GD.PushError($"  Conv forward comparison EXCEPTION: {ex.Message}");
        }
    }

    private static void RunConvTrainingComparison()
    {
        try
        {
            const int W = 8, H = 8, C = 1;
            const int OutputSize = 12;

            var def = new RLCnnEncoderDef
            {
                FilterCounts = new[] { 4, 6 },
                KernelSizes  = new[] { 3, 3 },
                Strides      = new[] { 1, 1 },
                OutputSize   = OutputSize,
            };

            using var gpu = new GpuCnnEncoder(W, H, C, def);
            var cpu = new ManagedCnnReference(W, H, C, def);

            var initialWeights = new List<float>();
            var shapes = new List<int>();
            cpu.AppendSerialized(initialWeights, shapes);

            var deterministicWeights = new float[initialWeights.Count];
            for (var i = 0; i < deterministicWeights.Length; i++)
                deterministicWeights[i] = ((i % 19) - 9) * 0.01171875f;

            var wi = 0;
            var si = 0;
            cpu.LoadSerialized(deterministicWeights, ref wi, shapes, ref si);
            wi = 0;
            si = 0;
            gpu.LoadSerialized(deterministicWeights, ref wi, shapes, ref si);

            var inputA = new float[W * H * C];
            var inputB = new float[W * H * C];
            for (var i = 0; i < inputA.Length; i++)
            {
                inputA[i] = ((i % 9) - 4) * 0.125f;
                inputB[i] = ((i % 7) - 3) * 0.15625f;
            }

            var outputGradA = new float[OutputSize];
            var outputGradB = new float[OutputSize];
            for (var i = 0; i < OutputSize; i++)
            {
                outputGradA[i] = ((i % 5) - 2) * 0.2f;
                outputGradB[i] = ((i % 4) - 1.5f) * 0.175f;
            }

            var cpuToken = cpu.CreateGradientBuffer();
            var gpuToken = gpu.CreateGradientToken();

            var cpuOutA = cpu.Forward(inputA);
            var gpuOutA = gpu.Forward(inputA);
            var cpuInGradA = cpu.AccumulateGradients(outputGradA, cpuToken);
            var gpuInGradA = gpu.AccumulateGradients(outputGradA, gpuToken);

            var cpuOutB = cpu.Forward(inputB);
            var gpuOutB = gpu.Forward(inputB);
            var cpuInGradB = cpu.AccumulateGradients(outputGradB, cpuToken);
            var gpuInGradB = gpu.AccumulateGradients(outputGradB, gpuToken);

            var cpuGradNorm = cpu.GradNormSquared(cpuToken);
            var gpuGradNorm = gpu.GradNormSquared(gpuToken);
            var cpuFlatGrad = cpu.GetFlatGradients(cpuToken);
            var gpuFlatGrad = gpu.DebugReadGradientBuffer();
            var canCompareFlatGrad = cpuFlatGrad.Length == gpuFlatGrad.Length && cpuFlatGrad.Length > 0;
            var cpuFlatGradNorm = canCompareFlatGrad ? SumSquares(cpuFlatGrad) : 0f;
            var gpuFlatGradNorm = canCompareFlatGrad ? SumSquares(gpuFlatGrad) : 0f;

            const float LearningRate = 0.001f;
            const float GradScale = 0.5f;
            cpu.ApplyGradients(cpuToken, LearningRate, GradScale);
            gpu.ApplyGradients(gpuToken, LearningRate, GradScale);

            var cpuWeights = new List<float>();
            var cpuShapes = new List<int>();
            cpu.AppendSerialized(cpuWeights, cpuShapes);

            var gpuWeights = new List<float>();
            var gpuShapes = new List<int>();
            gpu.AppendSerialized(gpuWeights, gpuShapes);

            var forwardErrA   = MaxError(cpuOutA, gpuOutA);
            var forwardErrB   = MaxError(cpuOutB, gpuOutB);
            var inputGradErrA = MaxError(cpuInGradA, gpuInGradA);
            var inputGradErrB = MaxError(cpuInGradB, gpuInGradB);
            var gradNormErr   = MathF.Abs(cpuGradNorm - gpuGradNorm);
            var flatGradErr   = canCompareFlatGrad ? MaxError(cpuFlatGrad, gpuFlatGrad) : 0f;
            var weightErr     = MaxError(cpuWeights, gpuWeights);
            var shapesMatch   = ListsEqual(cpuShapes, gpuShapes);
            var sectionErrors = canCompareFlatGrad
                ? ComputeCnnGradientSectionErrors(cpuFlatGrad, gpuFlatGrad, shapes)
                : new List<string> { "flat grad section compare unavailable" };

            GD.Print($"  conv forward A max_error = {forwardErrA:E2}");
            GD.Print($"  conv forward B max_error = {forwardErrB:E2}");
            GD.Print($"  pixel grad A max_error   = {inputGradErrA:E2}");
            GD.Print($"  pixel grad B max_error   = {inputGradErrB:E2}");
            GD.Print($"  grad norm error          = {gradNormErr:E2}");
            GD.Print(canCompareFlatGrad
                ? $"  flat grad max_error      = {flatGradErr:E2}"
                : "  flat grad max_error      = unavailable");
            GD.Print(canCompareFlatGrad
                ? $"  cpu flat grad norm       = {cpuFlatGradNorm:E2}"
                : "  cpu flat grad norm       = unavailable");
            GD.Print(canCompareFlatGrad
                ? $"  gpu flat grad norm       = {gpuFlatGradNorm:E2}"
                : "  gpu flat grad norm       = unavailable");
            foreach (var line in sectionErrors)
                GD.Print($"  {line}");
            GD.Print($"  post-update weight err   = {weightErr:E2}");
            GD.Print($"  serialized shape match   = {shapesMatch}");

            var pass = forwardErrA < 1e-4f &&
                       forwardErrB < 1e-4f &&
                       inputGradErrA < 1e-4f &&
                       inputGradErrB < 1e-4f &&
                       gradNormErr < 1e-4f &&
                       weightErr < 1e-4f &&
                       shapesMatch;
            GD.Print(pass ? "  PASS" : "  FAIL");
        }
        catch (Exception ex)
        {
            GD.PushError($"  Conv training comparison EXCEPTION: {ex.Message}");
        }
    }

    private static float MaxError(IReadOnlyList<float> a, IReadOnlyList<float> b)
    {
        var maxError = 0f;
        for (var i = 0; i < a.Count; i++)
            maxError = MathF.Max(maxError, MathF.Abs(a[i] - b[i]));
        return maxError;
    }

    private static List<string> ComputeCnnGradientSectionErrors(
        IReadOnlyList<float> cpuGrad,
        IReadOnlyList<float> gpuGrad,
        IReadOnlyList<int> shapes)
    {
        var lines = new List<string>();
        var offset = 0;
        var nConv = shapes[0];
        for (var c = 0; c < nConv; c++)
        {
            var outC = shapes[1 + c * 5 + 0];
            var kH   = shapes[1 + c * 5 + 1];
            var kW   = shapes[1 + c * 5 + 2];
            var inC  = shapes[1 + c * 5 + 3];
            var filterCount = outC * kH * kW * inC;

            lines.Add($"conv{c} filter grad err   = {SliceMaxError(cpuGrad, gpuGrad, offset, filterCount):E2}");
            offset += filterCount;
            lines.Add($"conv{c} bias grad err     = {SliceMaxError(cpuGrad, gpuGrad, offset, outC):E2}");
            offset += outC;
        }

        var projIn = shapes[1 + nConv * 5 + 0];
        var projOut = shapes[1 + nConv * 5 + 1];
        var projWeightCount = projIn * projOut;
        lines.Add($"proj weight grad err    = {SliceMaxError(cpuGrad, gpuGrad, offset, projWeightCount):E2}");
        offset += projWeightCount;
        lines.Add($"proj bias grad err      = {SliceMaxError(cpuGrad, gpuGrad, offset, projOut):E2}");
        return lines;
    }

    private static float SliceMaxError(
        IReadOnlyList<float> a,
        IReadOnlyList<float> b,
        int offset,
        int count)
    {
        var maxError = 0f;
        for (var i = 0; i < count; i++)
            maxError = MathF.Max(maxError, MathF.Abs(a[offset + i] - b[offset + i]));
        return maxError;
    }

    private static float SumSquares(IReadOnlyList<float> values)
    {
        var sum = 0f;
        for (var i = 0; i < values.Count; i++)
            sum += values[i] * values[i];
        return sum;
    }

    private static bool ListsEqual(IReadOnlyList<int> a, IReadOnlyList<int> b)
    {
        if (a.Count != b.Count) return false;
        for (var i = 0; i < a.Count; i++)
            if (a[i] != b[i]) return false;
        return true;
    }

    private static void RunLatencyBenchmark(int floatCount, int iterations)
    {
        try
        {
            using var gpu = new GpuDevice();
            var shader   = gpu.CompileComputeShader(GpuShaderSources.Passthrough);
            var pipeline = gpu.Rd.ComputePipelineCreate(shader);
            var inputBuf  = gpu.CreateBuffer(floatCount);
            var outputBuf = gpu.CreateBuffer(floatCount);

            var uniforms = new Godot.Collections.Array<RDUniform>();
            var u0 = new RDUniform { UniformType = RenderingDevice.UniformType.StorageBuffer, Binding = 0 };
            u0.AddId(inputBuf);
            uniforms.Add(u0);
            var u1 = new RDUniform { UniformType = RenderingDevice.UniformType.StorageBuffer, Binding = 1 };
            u1.AddId(outputBuf);
            uniforms.Add(u1);
            var uniformSet = gpu.Rd.UniformSetCreate(uniforms, shader, 0);

            var pc = GpuDevice.PushConstant((uint)floatCount);

            // Warmup.
            for (var i = 0; i < 5; i++)
            {
                var wl = gpu.Rd.ComputeListBegin();
                gpu.Rd.ComputeListBindComputePipeline(wl, pipeline);
                gpu.Rd.ComputeListBindUniformSet(wl, uniformSet, 0);
                gpu.Rd.ComputeListSetPushConstant(wl, pc, (uint)pc.Length);
                gpu.Rd.ComputeListDispatch(wl, xGroups: ((uint)floatCount + 255u) / 256u, yGroups: 1, zGroups: 1);
                gpu.Rd.ComputeListEnd();
                gpu.SubmitAndSync();
            }

            // Timed runs.
            var sw = Stopwatch.StartNew();
            for (var i = 0; i < iterations; i++)
            {
                var list = gpu.Rd.ComputeListBegin();
                gpu.Rd.ComputeListBindComputePipeline(list, pipeline);
                gpu.Rd.ComputeListBindUniformSet(list, uniformSet, 0);
                gpu.Rd.ComputeListSetPushConstant(list, pc, (uint)pc.Length);
                gpu.Rd.ComputeListDispatch(list, xGroups: ((uint)floatCount + 255u) / 256u, yGroups: 1, zGroups: 1);
                gpu.Rd.ComputeListEnd();
                gpu.SubmitAndSync();
            }
            sw.Stop();

            var avgMs = sw.Elapsed.TotalMilliseconds / iterations;
            GD.Print($"  {iterations} iterations, {floatCount} floats each");
            GD.Print($"  avg = {avgMs:F3}ms  total = {sw.Elapsed.TotalMilliseconds:F1}ms");
            GD.Print(avgMs < 1.0 ? "  PASS (< 1ms target)" : $"  NOTE: avg {avgMs:F3}ms > 1ms target (acceptable on integrated GPU)");

            gpu.Rd.FreeRid(uniformSet);
            gpu.Rd.FreeRid(outputBuf);
            gpu.Rd.FreeRid(inputBuf);
            gpu.Rd.FreeRid(pipeline);
            gpu.Rd.FreeRid(shader);
        }
        catch (Exception ex)
        {
            GD.PushError($"  Latency benchmark EXCEPTION: {ex.Message}");
        }
    }
}
