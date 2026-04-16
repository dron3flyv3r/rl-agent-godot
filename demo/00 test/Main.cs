using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

public partial class Main : Node
{
    private readonly List<string> _lines = new();
    private int _passCount;
    private int _failCount;

    public override void _Ready()
    {
        RunValidationSuite();
    }

    private void RunValidationSuite()
    {
        var startedAt = DateTime.UtcNow;
        LogHeader(startedAt);

        TestClassAvailability();
        TestWizardDefaultsHelpers();
        TestDenseLayer();
        TestAdamWLayer();
        TestRMSPropLayer();
        TestLayerNormLayer();
        TestLstmLayer();
        TestGruLayer();
        TestCnnEncoder();
        TestAcademyExtensibility();

        LogSummary(startedAt, DateTime.UtcNow);
        PrintReport();
    }

    private void TestClassAvailability()
    {
        var classes = new[] { "RlDenseLayer", "RlLayerNormLayer", "RlLstmLayer", "RlGruLayer", "RlCnnEncoder" };
        foreach (var className in classes)
        {
            RunTest($"Class availability: {className}", () =>
            {
                var obj = ClassDB.Instantiate(className).AsGodotObject();
                if (obj is null)
                    throw new InvalidOperationException($"ClassDB.Instantiate returned null for {className}.");
                ReleaseGodotObject(obj);
                return "instantiation ok";
            });
        }
    }

    private void TestWizardDefaultsHelpers()
    {
        RunTest("Wizard sanitize identifier", () =>
        {
            var sanitized = Runtime.RLSetupWizardDefaults.SanitizeIdentifier("Boss Agent#1");
            Ensure(sanitized == "bossagent1", $"Expected 'bossagent1', got '{sanitized}'.");
            return sanitized;
        });

        RunTest("Wizard unique identifier suffix", () =>
        {
            var unique = Runtime.RLSetupWizardDefaults.MakeUniqueIdentifier(
                "Agent",
                new[] { "agent", "agent_2", "runner" });
            Ensure(unique == "agent_3", $"Expected 'agent_3', got '{unique}'.");
            return unique;
        });

        RunTest("Wizard unique file name suffix", () =>
        {
            var unique = Runtime.RLSetupWizardDefaults.MakeUniqueFileName(
                "TagDemo.runner.policy.tres",
                new[] { "TagDemo.runner.policy.tres", "TagDemo.runner.policy_2.tres" });
            Ensure(unique == "TagDemo.runner.policy_3.tres", $"Expected 'TagDemo.runner.policy_3.tres', got '{unique}'.");
            return unique;
        });
    }

    private void TestDenseLayer()
    {
        GodotObject? dense = null;
        GodotObject? denseCopy = null;
        GodotObject? denseSource = null;

        try
        {
            dense = RequireInstance("RlDenseLayer");
            dense.Call("initialize", 4, 3, 1, 0);

            var denseInputA = new float[] { 0.2f, -0.1f, 0.5f, 1.0f };
            var denseInputB = new float[] { 0.4f, 0.1f, -0.3f, 0.7f };
            var denseInputC = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };

            RunTest("Dense forward", () =>
            {
                var output = RequireFloatArray(dense.Call("forward", denseInputA));
                Ensure(output.Length == 3, "Expected output length 3.");
                EnsureAllFinite(output, "Dense forward output");
                return $"len={output.Length} sample={FormatArray(output, 3)}";
            });

            RunTest("Dense forward determinism", () =>
            {
                var first = RequireFloatArray(dense.Call("forward", denseInputA));
                var second = RequireFloatArray(dense.Call("forward", denseInputA));
                EnsureArraysClose(first, second, 1e-6f, "Dense deterministic forward");
                return "same_input_same_output=true";
            });

            RunTest("Dense forward_batch shape", () =>
            {
                var flatInput = Concat(denseInputA, denseInputB);
                var flatOutput = RequireFloatArray(dense.Call("forward_batch", flatInput, 2));
                Ensure(flatOutput.Length == 6, "Expected batch output length 6.");
                EnsureAllFinite(flatOutput, "Dense forward_batch output");
                return $"batch_out_len={flatOutput.Length}";
            });

            RunTest("Dense forward_batch parity", () =>
            {
                var flatInput = Concat(denseInputA, denseInputB);
                var batchOut = RequireFloatArray(dense.Call("forward_batch", flatInput, 2));
                var outA = RequireFloatArray(dense.Call("forward", denseInputA));
                var outB = RequireFloatArray(dense.Call("forward", denseInputB));
                EnsureSliceClose(batchOut, 0, outA, 1e-5f, "Dense batch parity sample A");
                EnsureSliceClose(batchOut, 3, outB, 1e-5f, "Dense batch parity sample B");
                return "forward_batch_matches_single=true";
            });

            RunTest("Dense compute_input_grad", () =>
            {
                _ = RequireFloatArray(dense.Call("forward", denseInputB));
                var inputGrad = RequireFloatArray(dense.Call("compute_input_grad", new float[] { 0.3f, -0.2f, 0.1f }));
                Ensure(inputGrad.Length == 4, "Expected input gradient length 4.");
                EnsureAllFinite(inputGrad, "Dense compute_input_grad output");
                return $"input_grad_len={inputGrad.Length}";
            });

            RunTest("Dense finite-difference input gradient", () =>
            {
                var outputGrad = new float[] { 0.3f, -0.2f, 0.1f };
                _ = RequireFloatArray(dense.Call("forward", denseInputB));
                var analytic = RequireFloatArray(dense.Call("compute_input_grad", outputGrad));
                var numeric = FiniteDifferenceInputGradient(dense, denseInputB, outputGrad, 1e-3f);
                var maxDiff = MaxAbsDiff(analytic, numeric);
                Ensure(maxDiff <= 0.01f, "Dense finite-difference gradient mismatch is too large.");
                return $"max_abs_diff={maxDiff.ToString("0.######", CultureInfo.InvariantCulture)}";
            });

            RunTest("Dense accumulate buffer nonzero", () =>
            {
                _ = RequireFloatArray(dense.Call("forward", denseInputB));
                Variant gradBuffer = dense.Call("create_gradient_buffer");
                var inputGrad = AccumulateWithBuffer(dense, new float[] { 1.0f, -0.3f, 0.2f }, ref gradBuffer);
                var (bufLen, bufSumAbs, bufMaxAbs) = GetBufferStats(gradBuffer);
                var gradNormSq = (float)dense.Call("grad_norm_squared", gradBuffer);

                Ensure(inputGrad.Length == 4, "Expected input gradient length 4.");
                EnsureAllFinite(inputGrad, "Dense input gradient");
                Ensure(bufLen > 0, "Dense gradient buffer is empty after accumulation.");
                Ensure(bufSumAbs > 0f, "Dense gradient buffer content is all zeros after accumulation.");
                EnsurePositiveFinite(gradNormSq, "Dense gradient norm squared");
                return $"input_grad_len={inputGrad.Length} grad_norm_sq={gradNormSq:F6} buf_len={bufLen} buf_sum_abs={bufSumAbs.ToString("0.######", CultureInfo.InvariantCulture)} buf_max_abs={bufMaxAbs.ToString("0.######", CultureInfo.InvariantCulture)}";
            });

            RunTest("Dense backward changes output", () =>
            {
                var before = RequireFloatArray(dense.Call("forward", denseInputC));
                _ = RequireFloatArray(dense.Call("forward", denseInputB));
                _ = RequireFloatArray(dense.Call("backward", new float[] { 0.8f, -0.4f, 0.3f }, 0.002f, 1.0f));
                var after = RequireFloatArray(dense.Call("forward", denseInputC));
                var delta = MaxAbsDiff(before, after);
                Ensure(delta > 1e-7f, "Expected output to change after backward update.");
                return $"max_abs_delta={delta.ToString("0.######", CultureInfo.InvariantCulture)}";
            });

            RunTest("Dense serialization/copy/update", () =>
            {
                var weights = dense.Call("get_weights");
                var shapes = dense.Call("get_shapes");

                denseCopy = RequireInstance("RlDenseLayer");
                denseCopy.Call("initialize", 4, 3, 1, 0);
                denseCopy.Call("set_weights", weights, shapes);

                denseCopy.Call("copy_weights_from", dense);
                denseCopy.Call("soft_update_from", dense, 0.5f);

                var copiedOut = RequireFloatArray(denseCopy.Call("forward", new float[] { 0.1f, 0.2f, 0.3f, 0.4f }));
                Ensure(copiedOut.Length == 3, "Expected copied dense output length 3.");
                EnsureAllFinite(copiedOut, "Copied dense output");
                return $"weights_restored=ok copied_output={FormatArray(copiedOut, 3)}";
            });

            RunTest("Dense serialization preserves output", () =>
            {
                var baseline = RequireFloatArray(dense.Call("forward", denseInputA));
                var weights = dense.Call("get_weights");
                var shapes = dense.Call("get_shapes");
                dense.Call("set_weights", weights, shapes);
                var afterSet = RequireFloatArray(dense.Call("forward", denseInputA));
                EnsureArraysClose(baseline, afterSet, 1e-6f, "Dense serialization roundtrip");
                return "roundtrip_output_match=true";
            });

            RunTest("Dense shape descriptor", () =>
            {
                var shapes = RequireIntArray(dense.Call("get_shapes"));
                Ensure(shapes.Length == 4, "Expected 4 shape entries for dense layer.");
                Ensure(shapes[0] == 0, "Expected RLLayerKind.Dense (0). ");
                Ensure(shapes[1] == 4 && shapes[2] == 3, "Expected shape [4,3] for dense layer.");
                Ensure(shapes[3] == 1, "Expected activation code 1 for Tanh.");
                return $"shapes={FormatIntArray(shapes)}";
            });

            RunTest("Dense copy_weights_from parity", () =>
            {
                denseSource = RequireInstance("RlDenseLayer");
                denseSource.Call("initialize", 4, 3, 1, 0);
                dense.Call("copy_weights_from", denseSource);

                var sourceOut = RequireFloatArray(denseSource.Call("forward", denseInputA));
                var targetOut = RequireFloatArray(dense.Call("forward", denseInputA));
                EnsureArraysClose(sourceOut, targetOut, 1e-6f, "Dense copy_weights_from parity");
                return "copy_parity=true";
            });

            RunTest("Dense soft_update tau boundaries", () =>
            {
                denseSource ??= RequireInstance("RlDenseLayer");
                denseSource.Call("initialize", 4, 3, 1, 0);

                var before = RequireFloatArray(dense.Call("forward", denseInputA));
                dense.Call("soft_update_from", denseSource, 0.0f);
                var afterTau0 = RequireFloatArray(dense.Call("forward", denseInputA));
                EnsureArraysClose(before, afterTau0, 1e-6f, "Dense tau=0 should keep target unchanged");

                dense.Call("soft_update_from", denseSource, 1.0f);
                var sourceOut = RequireFloatArray(denseSource.Call("forward", denseInputA));
                var afterTau1 = RequireFloatArray(dense.Call("forward", denseInputA));
                EnsureArraysClose(sourceOut, afterTau1, 1e-6f, "Dense tau=1 should match source");
                return "tau0_keep=true tau1_match=true";
            });
        }
        finally
        {
            ReleaseGodotObject(dense);
            ReleaseGodotObject(denseCopy);
            ReleaseGodotObject(denseSource);
        }
    }

    private void TestAdamWLayer()
    {
        GodotObject? adamw = null;
        GodotObject? adam  = null;

        try
        {
            var inputA  = new float[] { 0.2f, -0.1f, 0.5f, 1.0f };
            var outGrad = new float[] { 0.8f, -0.4f, 0.3f };

            RunTest("AdamW backward changes output", () =>
            {
                adamw = RequireInstance("RlDenseLayer");
                adamw.Call("initialize", 4, 3, 1, 2);  // optimizer=2 = AdamW
                adamw.Call("set_weight_decay", 0.01f);

                var before = RequireFloatArray(adamw.Call("forward", inputA));
                _ = RequireFloatArray(adamw.Call("forward", inputA));
                _ = RequireFloatArray(adamw.Call("backward", outGrad, 0.002f, 1.0f));
                var after = RequireFloatArray(adamw.Call("forward", inputA));
                var delta = MaxAbsDiff(before, after);
                Ensure(delta > 1e-7f, "Expected output to change after AdamW backward.");
                return $"max_abs_delta={delta.ToString("0.######", CultureInfo.InvariantCulture)}";
            });

            RunTest("AdamW apply_gradients changes output", () =>
            {
                adamw ??= RequireInstance("RlDenseLayer");
                adamw.Call("initialize", 4, 3, 1, 2);
                adamw.Call("set_weight_decay", 0.01f);

                var before     = RequireFloatArray(adamw.Call("forward", inputA));
                Variant gradBuf = adamw.Call("create_gradient_buffer");
                _ = AccumulateWithBuffer(adamw, outGrad, ref gradBuf);
                adamw.Call("apply_gradients", gradBuf, 0.002f, 1.0f);
                var after = RequireFloatArray(adamw.Call("forward", inputA));
                var delta = MaxAbsDiff(before, after);
                Ensure(delta > 1e-7f, "Expected output to change after AdamW apply_gradients.");
                return $"max_abs_delta={delta.ToString("0.######", CultureInfo.InvariantCulture)}";
            });

            RunTest("AdamW weight decay reduces weight norm vs Adam", () =>
            {
                // Two identical layers (no activation for clean norm comparison).
                adam  = RequireInstance("RlDenseLayer");
                adamw = RequireInstance("RlDenseLayer");
                adam.Call("initialize",  4, 3, 0, 0);  // Adam,  no activation
                adamw.Call("initialize", 4, 3, 0, 2);  // AdamW, no activation
                adamw.Call("set_weight_decay", 0.1f);

                // Copy Adam's initial weights into AdamW so both start identical.
                var weights = adam.Call("get_weights");
                var shapes  = adam.Call("get_shapes");
                adamw.Call("set_weights", weights, shapes);

                // Run the same gradient steps on both.
                var stepInput   = new float[] { 0.5f, -0.3f, 0.4f, 0.8f };
                var stepOutGrad = new float[] { 0.5f, -0.3f, 0.4f };
                for (var i = 0; i < 30; i++)
                {
                    _ = RequireFloatArray(adam.Call("forward",  stepInput));
                    _ = RequireFloatArray(adamw.Call("forward", stepInput));
                    _ = RequireFloatArray(adam.Call("backward",  stepOutGrad, 0.01f, 1.0f));
                    _ = RequireFloatArray(adamw.Call("backward", stepOutGrad, 0.01f, 1.0f));
                }

                var wAdam  = RequireFloatArray(adam.Call("get_weights"));
                var wAdamW = RequireFloatArray(adamw.Call("get_weights"));
                var normAdam  = L2Norm(wAdam);
                var normAdamW = L2Norm(wAdamW);
                Ensure(normAdamW < normAdam,
                    $"Expected AdamW weight norm ({normAdamW.ToString("0.####", CultureInfo.InvariantCulture)}) " +
                    $"< Adam weight norm ({normAdam.ToString("0.####", CultureInfo.InvariantCulture)}) with wd=0.1.");
                return $"norm_adam={normAdam.ToString("0.####", CultureInfo.InvariantCulture)} norm_adamw={normAdamW.ToString("0.####", CultureInfo.InvariantCulture)}";
            });

            RunTest("AdamW biases not decayed (match Adam biases within threshold)", () =>
            {
                // With wd applied only to weights, biases should evolve the same as standard Adam.
                adam  = RequireInstance("RlDenseLayer");
                adamw = RequireInstance("RlDenseLayer");
                adam.Call("initialize",  4, 3, 0, 0);
                adamw.Call("initialize", 4, 3, 0, 2);
                adamw.Call("set_weight_decay", 0.5f);   // large wd to amplify any accidental bias decay

                var weights = adam.Call("get_weights");
                var shapes  = adam.Call("get_shapes");
                adamw.Call("set_weights", weights, shapes);

                // One update step with an all-ones input so bias grad == outGrad.
                var onesInput   = new float[] { 1f, 1f, 1f, 1f };
                var stepOutGrad = new float[] { 0.5f, -0.3f, 0.4f };
                _ = RequireFloatArray(adam.Call("forward",  onesInput));
                _ = RequireFloatArray(adamw.Call("forward", onesInput));
                _ = RequireFloatArray(adam.Call("backward",  stepOutGrad, 0.01f, 1.0f));
                _ = RequireFloatArray(adamw.Call("backward", stepOutGrad, 0.01f, 1.0f));

                // Bias values live at the END of get_weights() — last 3 floats for a 4→3 layer.
                var wAdam  = RequireFloatArray(adam.Call("get_weights"));
                var wAdamW = RequireFloatArray(adamw.Call("get_weights"));
                var bAdam  = new float[] { wAdam[wAdam.Length - 3],  wAdam[wAdam.Length - 2],  wAdam[wAdam.Length - 1]  };
                var bAdamW = new float[] { wAdamW[wAdamW.Length - 3], wAdamW[wAdamW.Length - 2], wAdamW[wAdamW.Length - 1] };
                var maxBiasDiff = MaxAbsDiff(bAdam, bAdamW);
                Ensure(maxBiasDiff < 1e-6f,
                    $"Biases differ between Adam and AdamW (max_diff={maxBiasDiff.ToString("0.######", CultureInfo.InvariantCulture)}); biases should not be decayed.");
                return $"max_bias_diff={maxBiasDiff.ToString("0.######", CultureInfo.InvariantCulture)}";
            });
        }
        finally
        {
            ReleaseGodotObject(adamw);
            ReleaseGodotObject(adam);
        }
    }

    private void TestRMSPropLayer()
    {
        GodotObject? rmsprop = null;
        GodotObject? adam    = null;

        try
        {
            var inputA  = new float[] { 0.2f, -0.1f, 0.5f, 1.0f };
            var outGrad = new float[] { 0.8f, -0.4f, 0.3f };

            RunTest("RMSProp backward changes output", () =>
            {
                rmsprop = RequireInstance("RlDenseLayer");
                rmsprop.Call("initialize", 4, 3, 1, 3);  // optimizer=3 = RMSProp

                var before = RequireFloatArray(rmsprop.Call("forward", inputA));
                _ = RequireFloatArray(rmsprop.Call("forward", inputA));
                _ = RequireFloatArray(rmsprop.Call("backward", outGrad, 0.002f, 1.0f));
                var after = RequireFloatArray(rmsprop.Call("forward", inputA));
                var delta = MaxAbsDiff(before, after);
                Ensure(delta > 1e-7f, "Expected output to change after RMSProp backward.");
                return $"max_abs_delta={delta.ToString("0.######", CultureInfo.InvariantCulture)}";
            });

            RunTest("RMSProp apply_gradients changes output", () =>
            {
                rmsprop ??= RequireInstance("RlDenseLayer");
                rmsprop.Call("initialize", 4, 3, 1, 3);

                var before      = RequireFloatArray(rmsprop.Call("forward", inputA));
                Variant gradBuf = rmsprop.Call("create_gradient_buffer");
                _ = AccumulateWithBuffer(rmsprop, outGrad, ref gradBuf);
                rmsprop.Call("apply_gradients", gradBuf, 0.002f, 1.0f);
                var after = RequireFloatArray(rmsprop.Call("forward", inputA));
                var delta = MaxAbsDiff(before, after);
                Ensure(delta > 1e-7f, "Expected output to change after RMSProp apply_gradients.");
                return $"max_abs_delta={delta.ToString("0.######", CultureInfo.InvariantCulture)}";
            });

            RunTest("RMSProp variance accumulation reduces effective step size", () =>
            {
                // With a fixed gradient, RMSProp step size shrinks as variance accumulates.
                // Early step should be larger than a later step taken with the same gradient.
                rmsprop = RequireInstance("RlDenseLayer");
                rmsprop.Call("initialize", 4, 3, 0, 3);  // no activation for clean measurement

                var stepInput   = new float[] { 1f, 0f, 0f, 0f };
                var stepOutGrad = new float[] { 1f, 0f, 0f };

                _ = RequireFloatArray(rmsprop.Call("forward", stepInput));
                var w0 = RequireFloatArray(rmsprop.Call("get_weights"));
                _ = RequireFloatArray(rmsprop.Call("backward", stepOutGrad, 0.1f, 1.0f));
                var w1 = RequireFloatArray(rmsprop.Call("get_weights"));
                var step1 = Mathf.Abs(w1[0] - w0[0]);

                // Run 50 more identical steps to build up variance.
                for (var i = 0; i < 50; i++)
                {
                    _ = RequireFloatArray(rmsprop.Call("forward", stepInput));
                    _ = RequireFloatArray(rmsprop.Call("backward", stepOutGrad, 0.1f, 1.0f));
                }

                var wN = RequireFloatArray(rmsprop.Call("get_weights"));
                _ = RequireFloatArray(rmsprop.Call("forward", stepInput));
                var wN1 = RequireFloatArray(rmsprop.Call("get_weights"));
                _ = RequireFloatArray(rmsprop.Call("backward", stepOutGrad, 0.1f, 1.0f));
                var wN2 = RequireFloatArray(rmsprop.Call("get_weights"));
                var stepN = Mathf.Abs(wN2[0] - wN1[0]);

                Ensure(stepN < step1,
                    $"Expected later RMSProp step ({stepN.ToString("0.######", CultureInfo.InvariantCulture)}) " +
                    $"< first step ({step1.ToString("0.######", CultureInfo.InvariantCulture)}) as variance accumulates.");
                return $"step1={step1.ToString("0.######", CultureInfo.InvariantCulture)} stepN={stepN.ToString("0.######", CultureInfo.InvariantCulture)}";
            });

            RunTest("RMSProp vs Adam produce different weight updates", () =>
            {
                adam    = RequireInstance("RlDenseLayer");
                rmsprop = RequireInstance("RlDenseLayer");
                adam.Call("initialize",    4, 3, 0, 0);  // Adam
                rmsprop.Call("initialize", 4, 3, 0, 3);  // RMSProp

                // Start both from the same weights.
                var weights = adam.Call("get_weights");
                var shapes  = adam.Call("get_shapes");
                rmsprop.Call("set_weights", weights, shapes);

                var stepInput   = new float[] { 0.5f, -0.3f, 0.4f, 0.8f };
                var stepOutGrad = new float[] { 0.5f, -0.3f, 0.4f };
                for (var i = 0; i < 10; i++)
                {
                    _ = RequireFloatArray(adam.Call("forward",    stepInput));
                    _ = RequireFloatArray(rmsprop.Call("forward", stepInput));
                    _ = RequireFloatArray(adam.Call("backward",    stepOutGrad, 0.01f, 1.0f));
                    _ = RequireFloatArray(rmsprop.Call("backward", stepOutGrad, 0.01f, 1.0f));
                }

                var wAdam    = RequireFloatArray(adam.Call("get_weights"));
                var wRmsProp = RequireFloatArray(rmsprop.Call("get_weights"));
                var diff = MaxAbsDiff(wAdam, wRmsProp);
                Ensure(diff > 1e-5f, $"Expected Adam and RMSProp to diverge after 10 steps (max_diff={diff.ToString("0.######", CultureInfo.InvariantCulture)}).");
                return $"max_weight_diff={diff.ToString("0.######", CultureInfo.InvariantCulture)}";
            });
        }
        finally
        {
            ReleaseGodotObject(rmsprop);
            ReleaseGodotObject(adam);
        }
    }

    private void TestLayerNormLayer()
    {
        GodotObject? ln = null;
        GodotObject? lnSource = null;

        try
        {
            ln = RequireInstance("RlLayerNormLayer");
            ln.Call("initialize", 4);

            var lnInputA = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var lnInputB = new float[] { 0.8f, -0.4f, 0.1f, 0.2f };
            var lnInputC = new float[] { -1.2f, 0.3f, 0.7f, 2.1f };

            RunTest("LayerNorm forward", () =>
            {
                var output = RequireFloatArray(ln.Call("forward", lnInputA));
                Ensure(output.Length == 4, "Expected output length 4.");
                EnsureAllFinite(output, "LayerNorm forward output");
                return $"len={output.Length} sample={FormatArray(output, 4)}";
            });

            RunTest("LayerNorm forward constant input", () =>
            {
                var output = RequireFloatArray(ln.Call("forward", new float[] { 5f, 5f, 5f, 5f }));
                Ensure(output.Length == 4, "Expected output length 4.");
                EnsureAllFinite(output, "LayerNorm constant forward output");
                Ensure(MaxAbs(output) < 1e-4f, "Expected near-zero output for constant input with default gamma/beta.");
                return $"max_abs={MaxAbs(output).ToString("0.######", CultureInfo.InvariantCulture)}";
            });

            RunTest("LayerNorm forward determinism", () =>
            {
                var first = RequireFloatArray(ln.Call("forward", lnInputA));
                var second = RequireFloatArray(ln.Call("forward", lnInputA));
                EnsureArraysClose(first, second, 1e-6f, "LayerNorm deterministic forward");
                return "same_input_same_output=true";
            });

            RunTest("LayerNorm forward_batch parity", () =>
            {
                var flatInput = Concat(lnInputA, lnInputB);
                var batchOut = RequireFloatArray(ln.Call("forward_batch", flatInput, 2));
                Ensure(batchOut.Length == 8, "Expected LayerNorm batch output length 8.");
                var outA = RequireFloatArray(ln.Call("forward", lnInputA));
                var outB = RequireFloatArray(ln.Call("forward", lnInputB));
                EnsureSliceClose(batchOut, 0, outA, 1e-5f, "LayerNorm batch parity sample A");
                EnsureSliceClose(batchOut, 4, outB, 1e-5f, "LayerNorm batch parity sample B");
                return "forward_batch_matches_single=true";
            });

            RunTest("LayerNorm compute_input_grad", () =>
            {
                _ = RequireFloatArray(ln.Call("forward", lnInputB));
                var inputGrad = RequireFloatArray(ln.Call("compute_input_grad", new float[] { 0.2f, -0.1f, 0.4f, -0.3f }));
                Ensure(inputGrad.Length == 4, "Expected input gradient length 4.");
                EnsureAllFinite(inputGrad, "LayerNorm compute_input_grad output");
                return $"input_grad_len={inputGrad.Length}";
            });

            RunTest("LayerNorm finite-difference input gradient", () =>
            {
                var outputGrad = new float[] { 0.2f, -0.1f, 0.4f, -0.3f };
                _ = RequireFloatArray(ln.Call("forward", lnInputB));
                var analytic = RequireFloatArray(ln.Call("compute_input_grad", outputGrad));
                var numeric = FiniteDifferenceInputGradient(ln, lnInputB, outputGrad, 1e-3f);
                var maxDiff = MaxAbsDiff(analytic, numeric);
                Ensure(maxDiff <= 0.02f, "LayerNorm finite-difference gradient mismatch is too large.");
                return $"max_abs_diff={maxDiff.ToString("0.######", CultureInfo.InvariantCulture)}";
            });

            RunTest("LayerNorm accumulate buffer nonzero", () =>
            {
                _ = RequireFloatArray(ln.Call("forward", lnInputB));
                Variant gradBuffer = ln.Call("create_gradient_buffer");
                var inputGrad = AccumulateWithBuffer(ln, new float[] { 0.5f, -0.5f, 0.25f, -0.25f }, ref gradBuffer);
                var (bufLen, bufSumAbs, bufMaxAbs) = GetBufferStats(gradBuffer);
                var gradNormSq = (float)ln.Call("grad_norm_squared", gradBuffer);

                Ensure(inputGrad.Length == 4, "Expected input gradient length 4.");
                EnsureAllFinite(inputGrad, "LayerNorm input gradient");
                Ensure(bufLen > 0, "LayerNorm gradient buffer is empty after accumulation.");
                Ensure(bufSumAbs > 0f, "LayerNorm gradient buffer content is all zeros after accumulation.");
                EnsurePositiveFinite(gradNormSq, "LayerNorm gradient norm squared");
                return $"input_grad_len={inputGrad.Length} grad_norm_sq={gradNormSq:F6} buf_len={bufLen} buf_sum_abs={bufSumAbs.ToString("0.######", CultureInfo.InvariantCulture)} buf_max_abs={bufMaxAbs.ToString("0.######", CultureInfo.InvariantCulture)}";
            });

            RunTest("LayerNorm backward changes output", () =>
            {
                var before = RequireFloatArray(ln.Call("forward", lnInputC));
                _ = RequireFloatArray(ln.Call("forward", lnInputB));
                _ = RequireFloatArray(ln.Call("backward", new float[] { 0.4f, -0.4f, 0.2f, -0.2f }, 0.002f, 1.0f));
                var after = RequireFloatArray(ln.Call("forward", lnInputC));
                var delta = MaxAbsDiff(before, after);
                Ensure(delta > 1e-7f, "Expected LayerNorm output to change after backward update.");
                return $"max_abs_delta={delta.ToString("0.######", CultureInfo.InvariantCulture)}";
            });

            RunTest("LayerNorm serialization preserves output", () =>
            {
                var baseline = RequireFloatArray(ln.Call("forward", lnInputA));
                var weights = ln.Call("get_weights");
                var shapes = ln.Call("get_shapes");
                ln.Call("set_weights", weights, shapes);
                var afterSet = RequireFloatArray(ln.Call("forward", lnInputA));
                EnsureArraysClose(baseline, afterSet, 1e-6f, "LayerNorm serialization roundtrip");
                return "roundtrip_output_match=true";
            });

            RunTest("LayerNorm shape descriptor", () =>
            {
                var shapes = RequireIntArray(ln.Call("get_shapes"));
                Ensure(shapes.Length == 2, "Expected 2 shape entries for LayerNorm.");
                Ensure(shapes[0] == 2, "Expected RLLayerKind.LayerNorm (2). ");
                Ensure(shapes[1] == 4, "Expected size 4 for LayerNorm descriptor.");
                return $"shapes={FormatIntArray(shapes)}";
            });

            RunTest("LayerNorm copy_weights_from parity", () =>
            {
                lnSource = RequireInstance("RlLayerNormLayer");
                lnSource.Call("initialize", 4);
                ln.Call("copy_weights_from", lnSource);

                var sourceOut = RequireFloatArray(lnSource.Call("forward", lnInputA));
                var targetOut = RequireFloatArray(ln.Call("forward", lnInputA));
                EnsureArraysClose(sourceOut, targetOut, 1e-6f, "LayerNorm copy_weights_from parity");
                return "copy_parity=true";
            });

            RunTest("LayerNorm soft_update tau boundaries", () =>
            {
                lnSource ??= RequireInstance("RlLayerNormLayer");
                lnSource.Call("initialize", 4);

                var before = RequireFloatArray(ln.Call("forward", lnInputA));
                ln.Call("soft_update_from", lnSource, 0.0f);
                var afterTau0 = RequireFloatArray(ln.Call("forward", lnInputA));
                EnsureArraysClose(before, afterTau0, 1e-6f, "LayerNorm tau=0 should keep target unchanged");

                ln.Call("soft_update_from", lnSource, 1.0f);
                var sourceOut = RequireFloatArray(lnSource.Call("forward", lnInputA));
                var afterTau1 = RequireFloatArray(ln.Call("forward", lnInputA));
                EnsureArraysClose(sourceOut, afterTau1, 1e-6f, "LayerNorm tau=1 should match source");
                return "tau0_keep=true tau1_match=true";
            });
        }
        finally
        {
            ReleaseGodotObject(ln);
            ReleaseGodotObject(lnSource);
        }
    }

    private void TestLstmLayer()
    {
        GodotObject? lstm = null;
        GodotObject? lstmCopy = null;
        GodotObject? lstmSource = null;

        try
        {
            lstm = RequireInstance("RlLstmLayer");
            lstm.Call("initialize", 3, 2, 0);

            var x0 = new float[] { 0.2f, -0.1f, 0.3f };
            var x1 = new float[] { -0.4f, 0.5f, 0.1f };
            var h0 = new float[] { 0f, 0f };
            var c0 = new float[] { 0f, 0f };
            var hSeed = new float[] { 0.15f, -0.25f };
            var cSeed = new float[] { -0.05f, 0.2f };

            RunTest("LSTM forward shape", () =>
            {
                var output = RequireFloatArray(lstm.Call("forward", x0, h0, c0));
                Ensure(output.Length == 4, "Expected LSTM forward to return [h|c] with length 4.");
                EnsureAllFinite(output, "LSTM forward output");
                return $"len={output.Length} sample={FormatArray(output, 4)}";
            });

            RunTest("LSTM forward determinism", () =>
            {
                var first = RequireFloatArray(lstm.Call("forward", x0, h0, c0));
                var second = RequireFloatArray(lstm.Call("forward", x0, h0, c0));
                EnsureArraysClose(first, second, 1e-6f, "LSTM deterministic forward");
                return "same_input_same_output=true";
            });

            RunTest("LSTM sequence parity", () =>
            {
                var step0 = RequireFloatArray(lstm.Call("forward", x0, hSeed, cSeed));
                var h1 = Slice(step0, 0, 2);
                var c1 = Slice(step0, 2, 2);
                var step1 = RequireFloatArray(lstm.Call("forward", x1, h1, c1));

                var seqOut = RequireFloatArray(lstm.Call("forward_sequence", Concat(x0, x1), 2, hSeed, cSeed));
                Ensure(seqOut.Length == 4, "Expected forward_sequence to emit T*hidden = 4 values.");
                EnsureSliceClose(seqOut, 0, h1, 1e-5f, "LSTM forward_sequence parity step 0");
                EnsureSliceClose(seqOut, 2, Slice(step1, 0, 2), 1e-5f, "LSTM forward_sequence parity step 1");
                return "forward_sequence_matches_stepwise=true";
            });

            RunTest("LSTM sequence gradients mutate buffer", () =>
            {
                _ = RequireFloatArray(lstm.Call("forward_sequence", Concat(x0, x1), 2, h0, c0));
                Variant gradBuffer = lstm.Call("create_gradient_buffer");
                var payload = RequireArray(lstm.Call(
                    "accumulate_sequence_gradients",
                    new float[] { 0.3f, -0.2f, 0.1f, 0.4f },
                    gradBuffer,
                    h0,
                    c0));
                Ensure(payload.Count == 2, "LSTM accumulate_sequence_gradients returned invalid payload.");
                var inputGrad = RequireFloatArray((Variant)payload[0]);
                gradBuffer = (Variant)RequireFloatArray((Variant)payload[1]);
                var (bufLen, bufSumAbs, bufMaxAbs) = GetBufferStats(gradBuffer);
                var gradNormSq = (float)lstm.Call("grad_norm_squared", gradBuffer);
                Ensure(inputGrad.Length == 6, "Expected T*input gradients for LSTM sequence.");
                EnsureAllFinite(inputGrad, "LSTM sequence input gradients");
                Ensure(bufLen > 0, "LSTM gradient buffer is empty after BPTT.");
                Ensure(bufSumAbs > 0f, "LSTM gradient buffer content is all zeros after BPTT.");
                EnsurePositiveFinite(gradNormSq, "LSTM gradient norm squared");
                return $"input_grad_len={inputGrad.Length} grad_norm_sq={gradNormSq:F6} buf_len={bufLen} buf_sum_abs={bufSumAbs.ToString("0.######", CultureInfo.InvariantCulture)} buf_max_abs={bufMaxAbs.ToString("0.######", CultureInfo.InvariantCulture)}";
            });

            RunTest("LSTM apply_gradients changes output", () =>
            {
                var baseline = RequireFloatArray(lstm.Call("forward", x0, h0, c0));
                _ = RequireFloatArray(lstm.Call("forward_sequence", Concat(x0, x1), 2, h0, c0));
                Variant gradBuffer = lstm.Call("create_gradient_buffer");
                var payload = RequireArray(lstm.Call(
                    "accumulate_sequence_gradients",
                    new float[] { 0.25f, -0.1f, -0.05f, 0.2f },
                    gradBuffer,
                    h0,
                    c0));
                gradBuffer = (Variant)RequireFloatArray((Variant)payload[1]);
                lstm.Call("apply_gradients", gradBuffer, 0.001f, 1.0f, 1.0f);
                var after = RequireFloatArray(lstm.Call("forward", x0, h0, c0));
                var delta = MaxAbsDiff(baseline, after);
                Ensure(delta > 1e-8f, "Expected LSTM output to change after apply_gradients.");
                return $"max_abs_delta={delta.ToString("0.######", CultureInfo.InvariantCulture)}";
            });

            RunTest("LSTM serialization preserves output", () =>
            {
                var baseline = RequireFloatArray(lstm.Call("forward", x0, hSeed, cSeed));
                var weights = lstm.Call("get_weights");
                var shapes = lstm.Call("get_shapes");
                lstm.Call("set_weights", weights, shapes);
                var afterSet = RequireFloatArray(lstm.Call("forward", x0, hSeed, cSeed));
                EnsureArraysClose(baseline, afterSet, 1e-6f, "LSTM serialization roundtrip");
                return "roundtrip_output_match=true";
            });

            RunTest("LSTM shape descriptor", () =>
            {
                var shapes = RequireIntArray(lstm.Call("get_shapes"));
                Ensure(shapes.Length == 3, "Expected 3 shape entries for LSTM.");
                Ensure(shapes[0] == 4, "Expected RLLayerKind.Lstm (4).");
                Ensure(shapes[1] == 3 && shapes[2] == 2, "Expected shape [3,2] for LSTM.");
                return $"shapes={FormatIntArray(shapes)}";
            });

            RunTest("LSTM copy_weights_from parity", () =>
            {
                lstmSource = RequireInstance("RlLstmLayer");
                lstmSource.Call("initialize", 3, 2, 0);
                lstm.Call("copy_weights_from", lstmSource);
                var sourceOut = RequireFloatArray(lstmSource.Call("forward", x0, hSeed, cSeed));
                var targetOut = RequireFloatArray(lstm.Call("forward", x0, hSeed, cSeed));
                EnsureArraysClose(sourceOut, targetOut, 1e-6f, "LSTM copy_weights_from parity");
                return "copy_parity=true";
            });

            RunTest("LSTM soft_update tau boundaries", () =>
            {
                lstmSource ??= RequireInstance("RlLstmLayer");
                lstmSource.Call("initialize", 3, 2, 0);

                var before = RequireFloatArray(lstm.Call("forward", x0, hSeed, cSeed));
                lstm.Call("soft_update_from", lstmSource, 0.0f);
                var afterTau0 = RequireFloatArray(lstm.Call("forward", x0, hSeed, cSeed));
                EnsureArraysClose(before, afterTau0, 1e-6f, "LSTM tau=0 should keep target unchanged");

                lstm.Call("soft_update_from", lstmSource, 1.0f);
                var sourceOut = RequireFloatArray(lstmSource.Call("forward", x0, hSeed, cSeed));
                var afterTau1 = RequireFloatArray(lstm.Call("forward", x0, hSeed, cSeed));
                EnsureArraysClose(sourceOut, afterTau1, 1e-6f, "LSTM tau=1 should match source");
                return "tau0_keep=true tau1_match=true";
            });
        }
        finally
        {
            ReleaseGodotObject(lstm);
            ReleaseGodotObject(lstmCopy);
            ReleaseGodotObject(lstmSource);
        }
    }

    private void TestGruLayer()
    {
        GodotObject? gru = null;
        GodotObject? gruSource = null;

        try
        {
            gru = RequireInstance("RlGruLayer");
            gru.Call("initialize", 3, 2, 0);

            var x0 = new float[] { 0.1f, -0.2f, 0.4f };
            var x1 = new float[] { -0.3f, 0.25f, 0.05f };
            var h0 = new float[] { 0f, 0f };
            var hSeed = new float[] { 0.12f, -0.08f };

            RunTest("GRU forward shape", () =>
            {
                var output = RequireFloatArray(gru.Call("forward", x0, h0));
                Ensure(output.Length == 2, "Expected GRU forward length 2.");
                EnsureAllFinite(output, "GRU forward output");
                return $"len={output.Length} sample={FormatArray(output, 2)}";
            });

            RunTest("GRU forward determinism", () =>
            {
                var first = RequireFloatArray(gru.Call("forward", x0, h0));
                var second = RequireFloatArray(gru.Call("forward", x0, h0));
                EnsureArraysClose(first, second, 1e-6f, "GRU deterministic forward");
                return "same_input_same_output=true";
            });

            RunTest("GRU sequence parity", () =>
            {
                var step0 = RequireFloatArray(gru.Call("forward", x0, hSeed));
                var step1 = RequireFloatArray(gru.Call("forward", x1, step0));

                var seqOut = RequireFloatArray(gru.Call("forward_sequence", Concat(x0, x1), 2, hSeed));
                Ensure(seqOut.Length == 4, "Expected forward_sequence to emit T*hidden = 4 values.");
                EnsureSliceClose(seqOut, 0, step0, 1e-5f, "GRU forward_sequence parity step 0");
                EnsureSliceClose(seqOut, 2, step1, 1e-5f, "GRU forward_sequence parity step 1");
                return "forward_sequence_matches_stepwise=true";
            });

            RunTest("GRU sequence gradients mutate buffer", () =>
            {
                _ = RequireFloatArray(gru.Call("forward_sequence", Concat(x0, x1), 2, h0));
                Variant gradBuffer = gru.Call("create_gradient_buffer");
                var payload = RequireArray(gru.Call(
                    "accumulate_sequence_gradients",
                    new float[] { 0.2f, -0.1f, -0.3f, 0.15f },
                    2,
                    gradBuffer,
                    h0));
                Ensure(payload.Count == 2, "GRU accumulate_sequence_gradients returned invalid payload.");
                var inputGrad = RequireFloatArray((Variant)payload[0]);
                gradBuffer = (Variant)RequireFloatArray((Variant)payload[1]);
                var (bufLen, bufSumAbs, bufMaxAbs) = GetBufferStats(gradBuffer);
                var gradNormSq = (float)gru.Call("grad_norm_squared", gradBuffer);
                Ensure(inputGrad.Length == 6, "Expected T*input gradients for GRU sequence.");
                EnsureAllFinite(inputGrad, "GRU sequence input gradients");
                Ensure(bufLen > 0, "GRU gradient buffer is empty after BPTT.");
                Ensure(bufSumAbs > 0f, "GRU gradient buffer content is all zeros after BPTT.");
                EnsurePositiveFinite(gradNormSq, "GRU gradient norm squared");
                return $"input_grad_len={inputGrad.Length} grad_norm_sq={gradNormSq:F6} buf_len={bufLen} buf_sum_abs={bufSumAbs.ToString("0.######", CultureInfo.InvariantCulture)} buf_max_abs={bufMaxAbs.ToString("0.######", CultureInfo.InvariantCulture)}";
            });

            RunTest("GRU apply_gradients changes output", () =>
            {
                var baseline = RequireFloatArray(gru.Call("forward", x0, h0));
                _ = RequireFloatArray(gru.Call("forward_sequence", Concat(x0, x1), 2, h0));
                Variant gradBuffer = gru.Call("create_gradient_buffer");
                var payload = RequireArray(gru.Call(
                    "accumulate_sequence_gradients",
                    new float[] { 0.3f, -0.2f, 0.1f, 0.05f },
                    2,
                    gradBuffer,
                    h0));
                gradBuffer = (Variant)RequireFloatArray((Variant)payload[1]);
                gru.Call("apply_gradients", gradBuffer, 0.001f, 1.0f, 1.0f);
                var after = RequireFloatArray(gru.Call("forward", x0, h0));
                var delta = MaxAbsDiff(baseline, after);
                Ensure(delta > 1e-8f, "Expected GRU output to change after apply_gradients.");
                return $"max_abs_delta={delta.ToString("0.######", CultureInfo.InvariantCulture)}";
            });

            RunTest("GRU serialization preserves output", () =>
            {
                var baseline = RequireFloatArray(gru.Call("forward", x0, hSeed));
                var weights = gru.Call("get_weights");
                var shapes = gru.Call("get_shapes");
                gru.Call("set_weights", weights, shapes);
                var afterSet = RequireFloatArray(gru.Call("forward", x0, hSeed));
                EnsureArraysClose(baseline, afterSet, 1e-6f, "GRU serialization roundtrip");
                return "roundtrip_output_match=true";
            });

            RunTest("GRU shape descriptor", () =>
            {
                var shapes = RequireIntArray(gru.Call("get_shapes"));
                Ensure(shapes.Length == 3, "Expected 3 shape entries for GRU.");
                Ensure(shapes[0] == 5, "Expected RLLayerKind.Gru (5).");
                Ensure(shapes[1] == 3 && shapes[2] == 2, "Expected shape [3,2] for GRU.");
                return $"shapes={FormatIntArray(shapes)}";
            });

            RunTest("GRU copy_weights_from parity", () =>
            {
                gruSource = RequireInstance("RlGruLayer");
                gruSource.Call("initialize", 3, 2, 0);
                gru.Call("copy_weights_from", gruSource);
                var sourceOut = RequireFloatArray(gruSource.Call("forward", x0, hSeed));
                var targetOut = RequireFloatArray(gru.Call("forward", x0, hSeed));
                EnsureArraysClose(sourceOut, targetOut, 1e-6f, "GRU copy_weights_from parity");
                return "copy_parity=true";
            });

            RunTest("GRU soft_update tau boundaries", () =>
            {
                gruSource ??= RequireInstance("RlGruLayer");
                gruSource.Call("initialize", 3, 2, 0);

                var before = RequireFloatArray(gru.Call("forward", x0, hSeed));
                gru.Call("soft_update_from", gruSource, 0.0f);
                var afterTau0 = RequireFloatArray(gru.Call("forward", x0, hSeed));
                EnsureArraysClose(before, afterTau0, 1e-6f, "GRU tau=0 should keep target unchanged");

                gru.Call("soft_update_from", gruSource, 1.0f);
                var sourceOut = RequireFloatArray(gruSource.Call("forward", x0, hSeed));
                var afterTau1 = RequireFloatArray(gru.Call("forward", x0, hSeed));
                EnsureArraysClose(sourceOut, afterTau1, 1e-6f, "GRU tau=1 should match source");
                return "tau0_keep=true tau1_match=true";
            });
        }
        finally
        {
            ReleaseGodotObject(gru);
            ReleaseGodotObject(gruSource);
        }
    }

    private void TestCnnEncoder()
    {
        GodotObject? cnn = null;
        GodotObject? cnnCopy = null;

        try
        {
            cnn = RequireInstance("RlCnnEncoder");
            cnn.Call(
                "initialize",
                4,
                4,
                1,
                new Godot.Collections.Array<int> { 2 },
                new Godot.Collections.Array<int> { 3 },
                new Godot.Collections.Array<int> { 1 },
                5);

            var cnnInputA = BuildRampInput(16);
            var cnnInputB = new float[]
            {
                -0.4f, -0.2f, 0.0f, 0.2f,
                 0.1f,  0.3f, 0.5f, 0.7f,
                -0.7f, -0.5f, 0.4f, 0.6f,
                -0.1f,  0.2f, 0.8f, 1.0f,
            };

            RunTest("CNN forward", () =>
            {
                var output = RequireFloatArray(cnn.Call("forward", cnnInputA));
                Ensure(output.Length == 5, "Expected encoder output length 5.");
                EnsureAllFinite(output, "CNN forward output");
                return $"len={output.Length} sample={FormatArray(output, 5)}";
            });

            RunTest("CNN forward determinism", () =>
            {
                var first = RequireFloatArray(cnn.Call("forward", cnnInputA));
                var second = RequireFloatArray(cnn.Call("forward", cnnInputA));
                EnsureArraysClose(first, second, 1e-6f, "CNN deterministic forward");
                return "same_input_same_output=true";
            });

            RunTest("CNN compute gradient shape", () =>
            {
                _ = RequireFloatArray(cnn.Call("forward", cnnInputA));
                Variant gradBuffer = cnn.Call("create_gradient_buffer");
                var inputGrad = AccumulateWithBuffer(cnn, new float[] { 0.1f, 0.2f, -0.3f, 0.0f, 0.4f }, ref gradBuffer);
                Ensure(inputGrad.Length == 16, "Expected encoder input gradient length 16.");
                EnsureAllFinite(inputGrad, "CNN accumulate_gradients output");
                return $"input_grad_len={inputGrad.Length}";
            });

            RunTest("CNN accumulate buffer nonzero", () =>
            {
                _ = RequireFloatArray(cnn.Call("forward", cnnInputA));
                Variant gradBuffer = cnn.Call("create_gradient_buffer");
                var inputGrad = AccumulateWithBuffer(cnn, new float[] { 0.3f, -0.2f, 0.1f, 0.0f, 0.4f }, ref gradBuffer);
                var (bufLen, bufSumAbs, bufMaxAbs) = GetBufferStats(gradBuffer);
                var gradNormSq = (float)cnn.Call("grad_norm_squared", gradBuffer);

                Ensure(inputGrad.Length == 16, "Expected encoder input gradient length 16.");
                EnsureAllFinite(inputGrad, "CNN input gradient");
                Ensure(bufLen > 0, "CNN gradient buffer is empty after accumulation.");
                Ensure(bufSumAbs > 0f, "CNN gradient buffer content is all zeros after accumulation.");
                EnsurePositiveFinite(gradNormSq, "CNN gradient norm squared");
                return $"input_grad_len={inputGrad.Length} grad_norm_sq={gradNormSq:F6} buf_len={bufLen} buf_sum_abs={bufSumAbs.ToString("0.######", CultureInfo.InvariantCulture)} buf_max_abs={bufMaxAbs.ToString("0.######", CultureInfo.InvariantCulture)}";
            });

            RunTest("CNN apply changes output", () =>
            {
                var before = RequireFloatArray(cnn.Call("forward", cnnInputB));
                _ = RequireFloatArray(cnn.Call("forward", cnnInputA));
                Variant gradBuffer = cnn.Call("create_gradient_buffer");
                _ = AccumulateWithBuffer(cnn, new float[] { -0.2f, 0.1f, 0.05f, -0.1f, 0.3f }, ref gradBuffer);
                cnn.Call("apply_gradients", gradBuffer, 0.0008f, 1.0f);
                var after = RequireFloatArray(cnn.Call("forward", cnnInputB));
                var delta = MaxAbsDiff(before, after);
                Ensure(delta > 1e-8f, "Expected CNN output to change after gradient apply (if accumulate buffer is wired).");
                return $"max_abs_delta={delta.ToString("0.######", CultureInfo.InvariantCulture)}";
            });

            RunTest("CNN serialization roundtrip", () =>
            {
                var weights = cnn.Call("get_weights");
                var shapes = cnn.Call("get_shapes");
                cnn.Call("set_weights", weights, shapes);
                return "weights/get_shapes/set_weights ok";
            });

            RunTest("CNN serialization preserves output", () =>
            {
                var baseline = RequireFloatArray(cnn.Call("forward", cnnInputA));
                var weights = cnn.Call("get_weights");
                var shapes = cnn.Call("get_shapes");
                cnn.Call("set_weights", weights, shapes);
                var afterSet = RequireFloatArray(cnn.Call("forward", cnnInputA));
                EnsureArraysClose(baseline, afterSet, 1e-6f, "CNN serialization roundtrip");
                return "roundtrip_output_match=true";
            });

            RunTest("CNN shape descriptor", () =>
            {
                var shapes = RequireIntArray(cnn.Call("get_shapes"));
                Ensure(shapes.Length == 8, "Expected 8 shape entries for 1-conv CNN descriptor.");
                Ensure(shapes[0] == 1, "Expected one conv layer in descriptor.");
                Ensure(shapes[6] > 0 && shapes[7] == 5, "Expected valid projection sizes ending in output size 5.");
                return $"shapes={FormatIntArray(shapes)}";
            });

            RunTest("CNN set_weights parity across instances", () =>
            {
                cnnCopy = RequireInstance("RlCnnEncoder");
                cnnCopy.Call(
                    "initialize",
                    4,
                    4,
                    1,
                    new Godot.Collections.Array<int> { 2 },
                    new Godot.Collections.Array<int> { 3 },
                    new Godot.Collections.Array<int> { 1 },
                    5);

                var weights = cnn.Call("get_weights");
                var shapes = cnn.Call("get_shapes");
                cnnCopy.Call("set_weights", weights, shapes);

                var originalOut = RequireFloatArray(cnn.Call("forward", cnnInputA));
                var copiedOut = RequireFloatArray(cnnCopy.Call("forward", cnnInputA));
                EnsureArraysClose(originalOut, copiedOut, 1e-6f, "CNN set_weights parity");
                return "copy_parity=true";
            });
        }
        finally
        {
            ReleaseGodotObject(cnn);
            ReleaseGodotObject(cnnCopy);
        }
    }

    private void RunTest(string name, Func<string> test)
    {
        try
        {
            var details = test();
            _passCount++;
            _lines.Add($"[PASS] {name} | {details}");
        }
        catch (Exception ex)
        {
            _failCount++;
            _lines.Add($"[FAIL] {name} | {Sanitize(ex.Message)}");
        }
    }

    private static GodotObject RequireInstance(string className)
    {
        var obj = ClassDB.Instantiate(className).AsGodotObject();
        if (obj is null)
            throw new InvalidOperationException($"Class {className} is unavailable. Build/load native extension first.");
        return obj;
    }

    private static float[] RequireFloatArray(Variant value)
    {
        try
        {
            return (float[])value;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Expected float[] result but got a different Variant type: {ex.Message}");
        }
    }

    private static int[] RequireIntArray(Variant value)
    {
        try
        {
            return (int[])value;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Expected int[] result but got a different Variant type: {ex.Message}");
        }
    }

    private static Godot.Collections.Array RequireArray(Variant value)
    {
        try
        {
            return (Godot.Collections.Array)value;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Expected Array result but got a different Variant type: {ex.Message}");
        }
    }

    private static float[] AccumulateWithBuffer(GodotObject native, float[] outputGrad, ref Variant gradBuffer)
    {
        var payload = (Godot.Collections.Array)native.Call("accumulate_gradients_with_buffer", (Variant)outputGrad, gradBuffer);
        if (payload.Count != 2)
            throw new InvalidOperationException("accumulate_gradients_with_buffer returned invalid payload.");
        var inputGrad = (float[])payload[0];
        var updatedBuffer = (float[])payload[1];
        gradBuffer = (Variant)updatedBuffer;
        return inputGrad;
    }

    private static (int length, float sumAbs, float maxAbs) GetBufferStats(Variant buffer)
    {
        var arr = (float[])buffer;
        var sum = 0f;
        var max = 0f;
        for (var i = 0; i < arr.Length; i++)
        {
            var a = Mathf.Abs(arr[i]);
            sum += a;
            if (a > max) max = a;
        }

        return (arr.Length, sum, max);
    }

    private static float[] FiniteDifferenceInputGradient(
        GodotObject native,
        float[] input,
        float[] outputGrad,
        float epsilon)
    {
        var grad = new float[input.Length];
        var xPlus = (float[])input.Clone();
        var xMinus = (float[])input.Clone();

        for (var i = 0; i < input.Length; i++)
        {
            xPlus[i] = input[i] + epsilon;
            xMinus[i] = input[i] - epsilon;

            var yPlus = RequireFloatArray(native.Call("forward", xPlus));
            var yMinus = RequireFloatArray(native.Call("forward", xMinus));
            var fPlus = Dot(yPlus, outputGrad);
            var fMinus = Dot(yMinus, outputGrad);
            grad[i] = (fPlus - fMinus) / (2f * epsilon);

            xPlus[i] = input[i];
            xMinus[i] = input[i];
        }

        return grad;
    }

    private static float Dot(float[] a, float[] b)
    {
        Ensure(a.Length == b.Length, "Dot requires equal-length vectors.");
        var sum = 0f;
        for (var i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
    }

    private static float[] BuildRampInput(int size)
    {
        var arr = new float[size];
        for (var i = 0; i < size; i++)
            arr[i] = (i % 7 - 3) / 3.0f;
        return arr;
    }

    private static float[] Concat(float[] a, float[] b)
    {
        var result = new float[a.Length + b.Length];
        Array.Copy(a, 0, result, 0, a.Length);
        Array.Copy(b, 0, result, a.Length, b.Length);
        return result;
    }

    private static float[] Slice(float[] input, int offset, int length)
    {
        Ensure(offset >= 0 && length >= 0 && offset + length <= input.Length, "Slice out of range.");
        var result = new float[length];
        Array.Copy(input, offset, result, 0, length);
        return result;
    }

    private static void Ensure(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }

    private static void EnsureAllFinite(float[] values, string label)
    {
        for (var i = 0; i < values.Length; i++)
        {
            if (float.IsNaN(values[i]) || float.IsInfinity(values[i]))
                throw new InvalidOperationException($"{label} contains non-finite value at index {i}.");
        }
    }

    private static void EnsureNonNegativeFinite(float value, string label)
    {
        if (float.IsNaN(value) || float.IsInfinity(value))
            throw new InvalidOperationException($"{label} is non-finite.");
        if (value < 0f)
            throw new InvalidOperationException($"{label} is negative: {value.ToString("0.######", CultureInfo.InvariantCulture)}.");
    }

    private static void EnsurePositiveFinite(float value, string label)
    {
        EnsureNonNegativeFinite(value, label);
        if (value <= 1e-12f)
            throw new InvalidOperationException($"{label} is zero; accumulate_gradients likely did not mutate the gradient buffer.");
    }

    private static void EnsureArraysClose(float[] a, float[] b, float tolerance, string label)
    {
        Ensure(a.Length == b.Length, $"{label} length mismatch: {a.Length} vs {b.Length}.");
        for (var i = 0; i < a.Length; i++)
        {
            var diff = Mathf.Abs(a[i] - b[i]);
            if (diff > tolerance)
                throw new InvalidOperationException(
                    $"{label} mismatch at index {i}: a={a[i].ToString("0.######", CultureInfo.InvariantCulture)} b={b[i].ToString("0.######", CultureInfo.InvariantCulture)} diff={diff.ToString("0.######", CultureInfo.InvariantCulture)} tol={tolerance.ToString("0.######", CultureInfo.InvariantCulture)}.");
        }
    }

    private static void EnsureSliceClose(float[] flat, int offset, float[] expected, float tolerance, string label)
    {
        Ensure(offset + expected.Length <= flat.Length, $"{label} slice out of range.");
        for (var i = 0; i < expected.Length; i++)
        {
            var diff = Mathf.Abs(flat[offset + i] - expected[i]);
            if (diff > tolerance)
                throw new InvalidOperationException(
                    $"{label} mismatch at index {i}: got={flat[offset + i].ToString("0.######", CultureInfo.InvariantCulture)} expected={expected[i].ToString("0.######", CultureInfo.InvariantCulture)} diff={diff.ToString("0.######", CultureInfo.InvariantCulture)} tol={tolerance.ToString("0.######", CultureInfo.InvariantCulture)}.");
        }
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        Ensure(a.Length == b.Length, "Cannot compute diff for arrays with different lengths.");
        var max = 0f;
        for (var i = 0; i < a.Length; i++)
        {
            var d = Mathf.Abs(a[i] - b[i]);
            if (d > max) max = d;
        }
        return max;
    }

    private static float L2Norm(float[] a)
    {
        var sum = 0f;
        for (var i = 0; i < a.Length; i++)
            sum += a[i] * a[i];
        return Mathf.Sqrt(sum);
    }

    private static float MaxAbs(float[] a)
    {
        var max = 0f;
        for (var i = 0; i < a.Length; i++)
        {
            var d = Mathf.Abs(a[i]);
            if (d > max) max = d;
        }
        return max;
    }

    private static string FormatIntArray(int[] values)
    {
        var sb = new StringBuilder();
        sb.Append("[");
        for (var i = 0; i < values.Length; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(values[i]);
        }
        sb.Append("]");
        return sb.ToString();
    }

    private static string FormatArray(float[] values, int max)
    {
        var count = Mathf.Min(values.Length, max);
        var sb = new StringBuilder();
        sb.Append("[");
        for (var i = 0; i < count; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(values[i].ToString("0.0000", CultureInfo.InvariantCulture));
        }

        if (values.Length > count) sb.Append(", ...");
        sb.Append("]");
        return sb.ToString();
    }

    private static string Sanitize(string text)
        => text.Replace("\n", " ").Replace("\r", " ").Trim();

    private static void ReleaseGodotObject(GodotObject? obj)
    {
        if (obj is null) return;
        if (obj is RefCounted) return;
        obj.Free();
    }

    private void LogHeader(DateTime startedAt)
    {
        _lines.Add("================ RL_CPP_VALIDATION_REPORT ================");
        _lines.Add($"utc_start={startedAt:O}");
        _lines.Add($"godot_version={Engine.GetVersionInfo()["string"]}");
        _lines.Add("----------------------------------------------------------");
    }

    private void LogSummary(DateTime startedAt, DateTime endedAt)
    {
        var total = _passCount + _failCount;
        _lines.Add("----------------------------------------------------------");
        _lines.Add($"summary total={total} pass={_passCount} fail={_failCount}");
        _lines.Add($"duration_ms={(endedAt - startedAt).TotalMilliseconds:0}");
        _lines.Add($"result={(_failCount == 0 ? "PASS" : "FAIL")}");
        _lines.Add("==========================================================");
    }

    private void PrintReport()
    {
        GD.Print("\n" + string.Join("\n", _lines) + "\n");

        if (_failCount > 0)
        {
            GD.PushWarning(
                "Native validation reported failures. Copy the full RL_CPP_VALIDATION_REPORT block from the console.");
        }

        CallDeferred(nameof(ExitAfterReport));
    }

    private void ExitAfterReport()
    {
        GetTree().Quit(_failCount > 0 ? 1 : 0);
    }

    // ── Academy extensibility tests ───────────────────────────────────────────

    private void TestAcademyExtensibility()
    {
        var ctx = new StubAcademyContext();

        // ── Token types ───────────────────────────────────────────────────────

        RunTest("Academy: phase tokens are distinct sealed types", () =>
        {
            var a = typeof(PhaseAToken);
            var b = typeof(PhaseBToken);
            var c = typeof(PhaseCToken);
            Ensure(a != b && b != c && a != c, "Phase token types must be distinct.");
            Ensure(a.IsSealed && b.IsSealed && c.IsSealed, "Phase token types must be sealed.");
            return $"{a.Name} / {b.Name} / {c.Name}";
        });

        // ── AcademyEpisodeEndArgs ─────────────────────────────────────────────

        RunTest("Academy: AcademyEpisodeEndArgs default values", () =>
        {
            var args = new AcademyEpisodeEndArgs();
            Ensure(args.Agent is null,                "Default Agent should be null.");
            Ensure(args.GroupId == string.Empty,      "Default GroupId should be empty string.");
            Ensure(args.EpisodeReward == 0f,          "Default EpisodeReward should be 0.");
            Ensure(args.EpisodeSteps == 0,            "Default EpisodeSteps should be 0.");
            Ensure(args.RewardBreakdown is not null,  "Default RewardBreakdown must not be null.");
            Ensure(args.RewardBreakdown!.Count == 0,  "Default RewardBreakdown should be empty.");
            Ensure(args.TotalSteps == 0L,             "Default TotalSteps should be 0.");
            Ensure(args.GroupEpisodeCount == 0L,      "Default GroupEpisodeCount should be 0.");
            Ensure(args.CurriculumProgress == 0f,     "Default CurriculumProgress should be 0.");
            return "all defaults correct";
        });

        RunTest("Academy: AcademyEpisodeEndArgs init values round-trip", () =>
        {
            var breakdown = new Dictionary<string, float> { ["speed"] = 0.5f, ["goal"] = 2.0f };
            var args = new AcademyEpisodeEndArgs
            {
                GroupId            = "player",
                GroupDisplayName   = "Player Group",
                EpisodeReward      = 3.5f,
                EpisodeSteps       = 128,
                RewardBreakdown    = breakdown,
                TotalSteps         = 10_000L,
                GroupEpisodeCount  = 42L,
                CurriculumProgress = 0.75f,
            };
            Ensure(args.GroupId == "player",                         "GroupId mismatch.");
            Ensure(Mathf.Abs(args.EpisodeReward - 3.5f) < 1e-6f,    "EpisodeReward mismatch.");
            Ensure(args.EpisodeSteps == 128,                         "EpisodeSteps mismatch.");
            Ensure(args.RewardBreakdown.Count == 2,                  "RewardBreakdown count mismatch.");
            Ensure(Mathf.Abs(args.RewardBreakdown["goal"] - 2f) < 1e-6f, "RewardBreakdown value mismatch.");
            Ensure(args.TotalSteps == 10_000L,                       "TotalSteps mismatch.");
            Ensure(args.GroupEpisodeCount == 42L,                    "GroupEpisodeCount mismatch.");
            Ensure(Mathf.Abs(args.CurriculumProgress - 0.75f) < 1e-6f, "CurriculumProgress mismatch.");
            return $"reward={args.EpisodeReward} steps={args.EpisodeSteps} progress={args.CurriculumProgress:F2}";
        });

        // ── Base RLAcademy defaults ───────────────────────────────────────────

        RunTest("Academy: base RLAcademy hooks are safe no-ops", () =>
        {
            var acad = new RLAcademy();
            Ensure(!acad.OwnsTrainingStep, "Base OwnsTrainingStep must be false.");
            Ensure(!acad.ShouldStop(ctx),  "Base ShouldStop must return false.");
            // None of these should throw on a detached node with a stub context.
            acad.OnTrainingInitialized(ctx);
            acad.OnBeforeStep(ctx);
            acad.OnAfterStep(ctx);
            acad.OnEpisodeEnd(new AcademyEpisodeEndArgs());
            acad.OnBeforeCheckpoint(ctx);
            acad.TrainingStep(ctx);
            return "all base no-ops safe";
        });

        // ── Tier 1: hook recording subclass ──────────────────────────────────

        RunTest("Academy: Tier 1 subclass hooks fire in order", () =>
        {
            var rec = new HookRecordingAcademy();
            rec.OnTrainingInitialized(ctx);
            rec.OnBeforeStep(ctx);
            rec.OnAfterStep(ctx);
            rec.OnEpisodeEnd(new AcademyEpisodeEndArgs { EpisodeReward = 5f });
            rec.OnBeforeCheckpoint(ctx);

            Ensure(rec.InitCount == 1,           "OnTrainingInitialized not fired.");
            Ensure(rec.BeforeStepCount == 1,     "OnBeforeStep not fired.");
            Ensure(rec.AfterStepCount == 1,      "OnAfterStep not fired.");
            Ensure(rec.EpisodeEndCount == 1,     "OnEpisodeEnd not fired.");
            Ensure(rec.BeforeCheckpointCount == 1, "OnBeforeCheckpoint not fired.");
            Ensure(Mathf.Abs(rec.LastEpisodeReward - 5f) < 1e-6f, "EpisodeReward not propagated.");
            Ensure(!rec.OwnsTrainingStep,        "HookRecordingAcademy must not own training step.");
            Ensure(!rec.ShouldStop(ctx),         "HookRecordingAcademy ShouldStop must return false.");

            var total = rec.InitCount + rec.BeforeStepCount + rec.AfterStepCount
                      + rec.EpisodeEndCount + rec.BeforeCheckpointCount;
            return $"{total} hooks fired correctly";
        });

        // ── Tier 2: custom training loop ─────────────────────────────────────

        RunTest("Academy: Tier 2 OwnsTrainingStep override", () =>
        {
            var t2 = new Tier2TestAcademy();
            Ensure(t2.OwnsTrainingStep,       "Tier2TestAcademy OwnsTrainingStep must be true.");
            Ensure(t2.TrainingStepCount == 0, "TrainingStepCount should start at 0.");
            t2.TrainingStep(ctx);
            t2.TrainingStep(ctx);
            Ensure(t2.TrainingStepCount == 2, "TrainingStep should have been called twice.");
            return $"TrainingStep called {t2.TrainingStepCount} times";
        });

        // ── GameDevAcademy (Demo 11) ──────────────────────────────────────────

        RunTest("Academy: GameDevAcademy inspector property defaults", () =>
        {
            var acad = new GameDevAcademy();
            Ensure(acad.StepBudget == 500_000L,
                $"StepBudget default: expected 500000, got {acad.StepBudget}.");
            Ensure(Mathf.Abs(acad.RewardThreshold - 0.8f) < 1e-6f,
                $"RewardThreshold default: expected 0.8, got {acad.RewardThreshold}.");
            Ensure(Mathf.Abs(acad.CurriculumStep - 0.02f) < 1e-6f,
                $"CurriculumStep default: expected 0.02, got {acad.CurriculumStep}.");
            Ensure(!acad.OwnsTrainingStep, "GameDevAcademy must not own training step.");
            return $"StepBudget={acad.StepBudget} RewardThreshold={acad.RewardThreshold} CurriculumStep={acad.CurriculumStep}";
        });

        RunTest("Academy: GameDevAcademy ShouldStop respects StepBudget", () =>
        {
            var acad = new GameDevAcademy { StepBudget = 1000 };

            ctx.TotalSteps = 999;
            Ensure(!acad.ShouldStop(ctx), "Should not stop before budget.");

            ctx.TotalSteps = 1000;
            Ensure(acad.ShouldStop(ctx),  "Should stop at budget.");

            ctx.TotalSteps = 1001;
            Ensure(acad.ShouldStop(ctx),  "Should stop past budget.");

            acad.StepBudget = 0;
            Ensure(!acad.ShouldStop(ctx), "StepBudget=0 should run forever.");

            ctx.TotalSteps = 0;
            return "ShouldStop boundary correct";
        });

        RunTest("Academy: GameDevAcademy OnEpisodeEnd advances curriculum", () =>
        {
            var acad = new GameDevAcademy { RewardThreshold = 1.0f, CurriculumStep = 0.1f };

            // Below threshold — curriculum must not change.
            var before = acad.CurriculumProgress;
            acad.OnEpisodeEnd(new AcademyEpisodeEndArgs { EpisodeReward = 0.5f, GroupEpisodeCount = 1 });
            Ensure(Mathf.Abs(acad.CurriculumProgress - before) < 1e-6f,
                "Curriculum must not advance below threshold.");

            // At threshold — curriculum must increase.
            acad.OnEpisodeEnd(new AcademyEpisodeEndArgs { EpisodeReward = 1.0f, GroupEpisodeCount = 2 });
            Ensure(acad.CurriculumProgress > before,
                "Curriculum must advance at/above threshold.");
            Ensure(Mathf.Abs(acad.CurriculumProgress - 0.1f) < 1e-6f,
                $"Curriculum should be 0.1 after one win, got {acad.CurriculumProgress}.");

            return $"progress after win={acad.CurriculumProgress:F2}";
        });

        RunTest("Academy: GameDevAcademy OnBeforeCheckpoint resets counters", () =>
        {
            var acad = new GameDevAcademy();
            // Feed two episodes.
            acad.OnEpisodeEnd(new AcademyEpisodeEndArgs { EpisodeReward = 2.0f, GroupEpisodeCount = 1 });
            acad.OnEpisodeEnd(new AcademyEpisodeEndArgs { EpisodeReward = 4.0f, GroupEpisodeCount = 2 });
            // Checkpoint should not throw and should reset internal counters.
            acad.OnBeforeCheckpoint(ctx);
            // Second checkpoint with no new episodes should be a silent no-op.
            acad.OnBeforeCheckpoint(ctx);
            return "checkpoint fired without error, counters reset";
        });

        // ── ResearchAcademy (Demo 12) ─────────────────────────────────────────

        RunTest("Academy: ResearchAcademy inspector property defaults", () =>
        {
            var acad = new ResearchAcademy();
            Ensure(acad.OwnsTrainingStep, "ResearchAcademy must own training step.");
            Ensure(Mathf.Abs(acad.ConvergenceEntropyThreshold - 0.05f) < 1e-6f,
                $"ConvergenceEntropyThreshold default: expected 0.05, got {acad.ConvergenceEntropyThreshold}.");
            Ensure(acad.ConvergenceGracePeriod == 100,
                $"ConvergenceGracePeriod default: expected 100, got {acad.ConvergenceGracePeriod}.");
            return $"threshold={acad.ConvergenceEntropyThreshold} grace={acad.ConvergenceGracePeriod}";
        });

        RunTest("Academy: ResearchAcademy ShouldStop requires grace period", () =>
        {
            var acad = new ResearchAcademy
            {
                ConvergenceEntropyThreshold = 0.5f,
                ConvergenceGracePeriod      = 3,
            };
            // No entropy data yet — must not stop regardless of context.
            Ensure(!acad.ShouldStop(ctx), "Must not stop before any entropy data.");
            return "no early stop without entropy data";
        });
    }

    // ── Stub helpers ──────────────────────────────────────────────────────────

    /// <summary>
    /// Minimal <see cref="IAcademyContext"/> that satisfies the interface contract
    /// without a live <see cref="TrainingBootstrap"/>. Used by the extensibility tests.
    /// </summary>
    private sealed class StubAcademyContext : IAcademyContext
    {
        public long TotalSteps { get; set; }
        public IReadOnlyDictionary<string, long> EpisodeCountByGroup { get; } =
            new Dictionary<string, long>();
        public IReadOnlyList<string> GroupIds { get; } = [];

        public ITrainer? GetTrainer(string groupId) => null;
        public IReadOnlyList<IRLAgent> GetGroupAgents(string groupId) => [];
        public void RunGroupDecisionPipeline(string groupId) { }

        public PhaseAToken EstimateNextValues(string groupId)
            => throw new NotSupportedException("Stub does not support phase execution.");
        public PhaseBToken RecordTransitionsAndReset(string groupId, PhaseAToken phaseA)
            => throw new NotSupportedException("Stub does not support phase execution.");
        public PhaseCToken SampleActions(string groupId, PhaseBToken phaseB)
            => throw new NotSupportedException("Stub does not support phase execution.");
        public void ApplyDecisions(string groupId, PhaseCToken phaseC) { }

        public void TriggerCheckpoint(bool forceWrite = false) { }
        public void LogMetric(string groupId, string metricKey, float value) { }
        public void SetCurriculumProgress(float progress) { }
        public void RequestStop(string reason = "Stopped by custom loop.") { }
    }

    /// <summary>Tier 1 subclass that counts every hook invocation.</summary>
    private sealed partial class HookRecordingAcademy : RLAcademy
    {
        public int   InitCount;
        public int   BeforeStepCount;
        public int   AfterStepCount;
        public int   EpisodeEndCount;
        public int   BeforeCheckpointCount;
        public float LastEpisodeReward;

        public override void OnTrainingInitialized(IAcademyContext ctx) => InitCount++;
        public override void OnBeforeStep(IAcademyContext ctx)           => BeforeStepCount++;
        public override void OnAfterStep(IAcademyContext ctx)            => AfterStepCount++;
        public override void OnBeforeCheckpoint(IAcademyContext ctx)     => BeforeCheckpointCount++;
        public override void OnEpisodeEnd(AcademyEpisodeEndArgs args)
        {
            EpisodeEndCount++;
            LastEpisodeReward = args.EpisodeReward;
        }
    }

    /// <summary>Tier 2 subclass that claims the training loop and counts step calls.</summary>
    private sealed partial class Tier2TestAcademy : RLAcademy
    {
        public int TrainingStepCount;
        public override bool OwnsTrainingStep => true;
        public override void TrainingStep(IAcademyContext ctx) => TrainingStepCount++;
    }
}
