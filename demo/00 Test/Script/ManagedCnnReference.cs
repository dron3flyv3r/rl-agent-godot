using System;
using System.Collections.Generic;

internal sealed class ManagedCnnReference
{
    private const float AdamBeta1 = 0.9f;
    private const float AdamBeta2 = 0.999f;
    private const float AdamEpsilon = 1e-8f;

    private sealed class ConvLayer
    {
        public int InH;
        public int InW;
        public int InC;
        public int OutH;
        public int OutW;
        public int OutC;
        public int Kernel;
        public int Stride;

        public float[] Filters = Array.Empty<float>();
        public float[] Biases = Array.Empty<float>();
        public float[] WeightM = Array.Empty<float>();
        public float[] WeightV = Array.Empty<float>();
        public float[] BiasM = Array.Empty<float>();
        public float[] BiasV = Array.Empty<float>();

        public float[] CachedInput = Array.Empty<float>();
        public float[] CachedPreact = Array.Empty<float>();
        public float[] ColBuffer = Array.Empty<float>();

        public float AdamB1Pow = 1f;
        public float AdamB2Pow = 1f;

        public int FilterCount => OutC * Kernel * Kernel * InC;
        public int OutputCount => OutH * OutW * OutC;
        public int InputCount => InH * InW * InC;
    }

    private sealed class LinearLayer
    {
        public int InSize;
        public int OutSize;

        public float[] Weights = Array.Empty<float>();
        public float[] Biases = Array.Empty<float>();
        public float[] WeightM = Array.Empty<float>();
        public float[] WeightV = Array.Empty<float>();
        public float[] BiasM = Array.Empty<float>();
        public float[] BiasV = Array.Empty<float>();

        public float[] CachedInput = Array.Empty<float>();

        public float AdamB1Pow = 1f;
        public float AdamB2Pow = 1f;
    }

    private readonly ConvLayer[] _convLayers;
    private readonly LinearLayer _projection;

    public int OutputSize => _projection.OutSize;

    public ManagedCnnReference(int width, int height, int channels, RlAgentPlugin.Runtime.RLCnnEncoderDef def)
    {
        _convLayers = new ConvLayer[def.FilterCounts.Length];

        var prevH = width == 0 ? 0 : height;
        var prevW = width;
        var prevC = channels;
        for (var i = 0; i < _convLayers.Length; i++)
        {
            var kernel = def.KernelSizes[i];
            var stride = def.Strides[i];
            var outC = def.FilterCounts[i];
            var outH = (prevH - kernel) / stride + 1;
            var outW = (prevW - kernel) / stride + 1;

            var layer = new ConvLayer
            {
                InH = prevH,
                InW = prevW,
                InC = prevC,
                OutH = outH,
                OutW = outW,
                OutC = outC,
                Kernel = kernel,
                Stride = stride,
                Filters = new float[outC * kernel * kernel * prevC],
                Biases = new float[outC],
                WeightM = new float[outC * kernel * kernel * prevC],
                WeightV = new float[outC * kernel * kernel * prevC],
                BiasM = new float[outC],
                BiasV = new float[outC],
                CachedInput = new float[prevH * prevW * prevC],
                CachedPreact = new float[outH * outW * outC],
                ColBuffer = new float[outH * outW * kernel * kernel * prevC],
            };
            _convLayers[i] = layer;

            prevH = outH;
            prevW = outW;
            prevC = outC;
        }

        var projIn = prevH * prevW * prevC;
        _projection = new LinearLayer
        {
            InSize = projIn,
            OutSize = def.OutputSize,
            Weights = new float[projIn * def.OutputSize],
            Biases = new float[def.OutputSize],
            WeightM = new float[projIn * def.OutputSize],
            WeightV = new float[projIn * def.OutputSize],
            BiasM = new float[def.OutputSize],
            BiasV = new float[def.OutputSize],
            CachedInput = new float[projIn],
        };
    }

    public float[] CreateGradientBuffer()
    {
        var totalSize = 0;
        foreach (var layer in _convLayers)
            totalSize += layer.FilterCount + layer.OutC;
        totalSize += _projection.Weights.Length + _projection.Biases.Length;
        return new float[totalSize];
    }

    public float[] Forward(float[] input)
    {
        var current = input;
        foreach (var layer in _convLayers)
            current = ForwardConv(layer, current);
        return ForwardProjection(current);
    }

    public float[] AccumulateGradients(float[] outputGrad, float[] gradBuffer)
    {
        var projWeightOffset = GetProjectionWeightOffset();
        var projBiasOffset = projWeightOffset + _projection.Weights.Length;

        var currentGrad = new float[_projection.InSize];
        Array.Clear(currentGrad, 0, currentGrad.Length);
        for (var o = 0; o < _projection.OutSize; o++)
        {
            var g = outputGrad[o];
            gradBuffer[projBiasOffset + o] += g;

            var wBase = o * _projection.InSize;
            for (var i = 0; i < _projection.InSize; i++)
            {
                gradBuffer[projWeightOffset + wBase + i] += _projection.CachedInput[i] * g;
                currentGrad[i] += _projection.Weights[wBase + i] * g;
            }
        }

        for (var layerIndex = _convLayers.Length - 1; layerIndex >= 0; layerIndex--)
        {
            currentGrad = BackwardConv(_convLayers[layerIndex], currentGrad, gradBuffer, GetConvFilterOffset(layerIndex));
        }

        return currentGrad;
    }

    public float GradNormSquared(float[] gradBuffer)
    {
        var sum = 0f;
        for (var i = 0; i < gradBuffer.Length; i++)
            sum += gradBuffer[i] * gradBuffer[i];
        return sum;
    }

    public void ApplyGradients(float[] gradBuffer, float learningRate, float gradScale)
    {
        var offset = 0;
        foreach (var layer in _convLayers)
        {
            layer.AdamB1Pow *= AdamBeta1;
            layer.AdamB2Pow *= AdamBeta2;
            var lrCorrected = learningRate * MathF.Sqrt(1f - layer.AdamB2Pow) / (1f - layer.AdamB1Pow);

            ApplyAdam(layer.Filters, layer.WeightM, layer.WeightV, gradBuffer, offset, layer.FilterCount, lrCorrected, gradScale);
            offset += layer.FilterCount;
            ApplyAdam(layer.Biases, layer.BiasM, layer.BiasV, gradBuffer, offset, layer.OutC, lrCorrected, gradScale);
            offset += layer.OutC;
        }

        _projection.AdamB1Pow *= AdamBeta1;
        _projection.AdamB2Pow *= AdamBeta2;
        var projectionLrCorrected = learningRate * MathF.Sqrt(1f - _projection.AdamB2Pow) / (1f - _projection.AdamB1Pow);
        ApplyAdam(_projection.Weights, _projection.WeightM, _projection.WeightV, gradBuffer, offset, _projection.Weights.Length, projectionLrCorrected, gradScale);
        offset += _projection.Weights.Length;
        ApplyAdam(_projection.Biases, _projection.BiasM, _projection.BiasV, gradBuffer, offset, _projection.Biases.Length, projectionLrCorrected, gradScale);
    }

    public void LoadSerialized(IReadOnlyList<float> weights, ref int wi, IReadOnlyList<int> shapes, ref int si)
    {
        var convCount = shapes[si++];
        if (convCount != _convLayers.Length)
            throw new InvalidOperationException("[ManagedCnnReference] Conv layer count mismatch.");

        foreach (var layer in _convLayers)
        {
            var outC = shapes[si++];
            var kH = shapes[si++];
            var kW = shapes[si++];
            var inC = shapes[si++];
            var stride = shapes[si++];
            if (outC != layer.OutC || kH != layer.Kernel || kW != layer.Kernel || inC != layer.InC || stride != layer.Stride)
                throw new InvalidOperationException("[ManagedCnnReference] Conv shape mismatch.");

            for (var i = 0; i < layer.Filters.Length; i++) layer.Filters[i] = weights[wi++];
            for (var i = 0; i < layer.Biases.Length; i++) layer.Biases[i] = weights[wi++];
        }

        var projIn = shapes[si++];
        var projOut = shapes[si++];
        if (projIn != _projection.InSize || projOut != _projection.OutSize)
            throw new InvalidOperationException("[ManagedCnnReference] Projection shape mismatch.");

        for (var i = 0; i < _projection.Weights.Length; i++) _projection.Weights[i] = weights[wi++];
        for (var i = 0; i < _projection.Biases.Length; i++) _projection.Biases[i] = weights[wi++];
        ResetOptimizerState();
    }

    public void AppendSerialized(ICollection<float> weights, ICollection<int> shapes)
    {
        shapes.Add(_convLayers.Length);
        foreach (var layer in _convLayers)
        {
            shapes.Add(layer.OutC);
            shapes.Add(layer.Kernel);
            shapes.Add(layer.Kernel);
            shapes.Add(layer.InC);
            shapes.Add(layer.Stride);
            foreach (var value in layer.Filters) weights.Add(value);
            foreach (var value in layer.Biases) weights.Add(value);
        }

        shapes.Add(_projection.InSize);
        shapes.Add(_projection.OutSize);
        foreach (var value in _projection.Weights) weights.Add(value);
        foreach (var value in _projection.Biases) weights.Add(value);
    }

    public float[] GetFlatGradients(float[] gradBuffer)
    {
        var copy = new float[gradBuffer.Length];
        Array.Copy(gradBuffer, copy, gradBuffer.Length);
        return copy;
    }

    private float[] ForwardConv(ConvLayer layer, float[] input)
    {
        Array.Copy(input, layer.CachedInput, input.Length);
        Im2Col(input, layer);

        var output = new float[layer.OutputCount];
        var colStride = layer.Kernel * layer.Kernel * layer.InC;
        var positions = layer.OutH * layer.OutW;
        for (var m = 0; m < positions; m++)
        {
            var colBase = m * colStride;
            var outBase = m * layer.OutC;
            for (var oc = 0; oc < layer.OutC; oc++)
            {
                var filterBase = oc * colStride;
                var sum = layer.Biases[oc];
                for (var k = 0; k < colStride; k++)
                    sum += layer.ColBuffer[colBase + k] * layer.Filters[filterBase + k];
                layer.CachedPreact[outBase + oc] = sum;
                output[outBase + oc] = sum > 0f ? sum : 0f;
            }
        }
        return output;
    }

    private float[] ForwardProjection(float[] input)
    {
        Array.Copy(input, _projection.CachedInput, input.Length);
        var output = new float[_projection.OutSize];
        for (var o = 0; o < _projection.OutSize; o++)
        {
            var sum = _projection.Biases[o];
            var wBase = o * _projection.InSize;
            for (var i = 0; i < _projection.InSize; i++)
                sum += _projection.CachedInput[i] * _projection.Weights[wBase + i];
            output[o] = _convLayers.Length > 0 ? MathF.Max(sum, 0f) : sum;
        }
        return output;
    }

    private float[] BackwardConv(ConvLayer layer, float[] outputGrad, float[] gradBuffer, int gradOffset)
    {
        var filterGradOffset = gradOffset;
        var biasGradOffset = gradOffset + layer.FilterCount;
        var positions = layer.OutH * layer.OutW;
        var colStride = layer.Kernel * layer.Kernel * layer.InC;
        var colGrad = new float[positions * colStride];

        for (var m = 0; m < positions; m++)
        {
            var colBase = m * colStride;
            var outBase = m * layer.OutC;
            for (var oc = 0; oc < layer.OutC; oc++)
            {
                var outIndex = outBase + oc;
                var reluGrad = layer.CachedPreact[outIndex] > 0f ? outputGrad[outIndex] : 0f;
                gradBuffer[biasGradOffset + oc] += reluGrad;

                var filterBase = oc * colStride;
                for (var k = 0; k < colStride; k++)
                {
                    gradBuffer[filterGradOffset + filterBase + k] += layer.ColBuffer[colBase + k] * reluGrad;
                    colGrad[colBase + k] += layer.Filters[filterBase + k] * reluGrad;
                }
            }
        }

        return Col2Im(colGrad, layer);
    }

    private void Im2Col(float[] input, ConvLayer layer)
    {
        var colStride = layer.Kernel * layer.Kernel * layer.InC;
        for (var oh = 0; oh < layer.OutH; oh++)
        {
            for (var ow = 0; ow < layer.OutW; ow++)
            {
                var row = (oh * layer.OutW + ow) * colStride;
                var cursor = row;
                for (var kh = 0; kh < layer.Kernel; kh++)
                {
                    var ih = oh * layer.Stride + kh;
                    for (var kw = 0; kw < layer.Kernel; kw++)
                    {
                        var iw = ow * layer.Stride + kw;
                        var inputBase = (ih * layer.InW + iw) * layer.InC;
                        for (var ic = 0; ic < layer.InC; ic++)
                            layer.ColBuffer[cursor++] = input[inputBase + ic];
                    }
                }
            }
        }
    }

    private static float[] Col2Im(float[] colGrad, ConvLayer layer)
    {
        var inputGrad = new float[layer.InputCount];
        var colStride = layer.Kernel * layer.Kernel * layer.InC;
        for (var oh = 0; oh < layer.OutH; oh++)
        {
            for (var ow = 0; ow < layer.OutW; ow++)
            {
                var row = (oh * layer.OutW + ow) * colStride;
                var cursor = row;
                for (var kh = 0; kh < layer.Kernel; kh++)
                {
                    var ih = oh * layer.Stride + kh;
                    for (var kw = 0; kw < layer.Kernel; kw++)
                    {
                        var iw = ow * layer.Stride + kw;
                        var inputBase = (ih * layer.InW + iw) * layer.InC;
                        for (var ic = 0; ic < layer.InC; ic++)
                            inputGrad[inputBase + ic] += colGrad[cursor++];
                    }
                }
            }
        }
        return inputGrad;
    }

    private static void ApplyAdam(float[] parameters, float[] m, float[] v, float[] gradBuffer, int gradOffset, int count, float lrCorrected, float gradScale)
    {
        for (var i = 0; i < count; i++)
        {
            var g = gradBuffer[gradOffset + i] * gradScale;
            m[i] = AdamBeta1 * m[i] + (1f - AdamBeta1) * g;
            v[i] = AdamBeta2 * v[i] + (1f - AdamBeta2) * g * g;
            parameters[i] -= lrCorrected * m[i] / (MathF.Sqrt(v[i]) + AdamEpsilon);
        }
    }

    private int GetConvFilterOffset(int layerIndex)
    {
        var offset = 0;
        for (var i = 0; i < layerIndex; i++)
            offset += _convLayers[i].FilterCount + _convLayers[i].OutC;
        return offset;
    }

    private int GetProjectionWeightOffset()
    {
        var offset = 0;
        foreach (var layer in _convLayers)
            offset += layer.FilterCount + layer.OutC;
        return offset;
    }

    private void ResetOptimizerState()
    {
        foreach (var layer in _convLayers)
        {
            Array.Clear(layer.WeightM, 0, layer.WeightM.Length);
            Array.Clear(layer.WeightV, 0, layer.WeightV.Length);
            Array.Clear(layer.BiasM, 0, layer.BiasM.Length);
            Array.Clear(layer.BiasV, 0, layer.BiasV.Length);
            layer.AdamB1Pow = 1f;
            layer.AdamB2Pow = 1f;
        }

        Array.Clear(_projection.WeightM, 0, _projection.WeightM.Length);
        Array.Clear(_projection.WeightV, 0, _projection.WeightV.Length);
        Array.Clear(_projection.BiasM, 0, _projection.BiasM.Length);
        Array.Clear(_projection.BiasV, 0, _projection.BiasV.Length);
        _projection.AdamB1Pow = 1f;
        _projection.AdamB2Pow = 1f;
    }
}
