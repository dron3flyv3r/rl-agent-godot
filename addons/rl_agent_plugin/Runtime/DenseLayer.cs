using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace RlAgentPlugin.Runtime;

internal sealed class LayerCache
{
    public static LayerCache Empty { get; } = new()
    {
        Input = Array.Empty<float>(),
        PreActivation = Array.Empty<float>(),
        Activated = Array.Empty<float>(),
    };

    public float[] Input { get; init; } = Array.Empty<float>();
    public float[] PreActivation { get; init; } = Array.Empty<float>();
    public float[] Activated { get; init; } = Array.Empty<float>();
}

internal sealed class DenseLayer
{
    private const float AdamBeta1 = 0.9f;
    private const float AdamBeta2 = 0.999f;
    private const float AdamEpsilon = 1e-8f;

    private readonly int _inputSize;
    private readonly int _outputSize;
    private readonly RLActivationKind? _activation;
    private readonly float[] _weights;
    private readonly float[] _biases;
    private readonly bool _useAdam;

    // Adam moment vectors (null when using SGD)
    private readonly float[]? _wm;  // weight first moment
    private readonly float[]? _wv;  // weight second moment
    private readonly float[]? _bm;  // bias first moment
    private readonly float[]? _bv;  // bias second moment

    // Iterative bias-correction accumulators: tracks beta^t
    private float _adamB1Pow = 1f;
    private float _adamB2Pow = 1f;

    private readonly RandomNumberGenerator _rng = new();

    public int InputSize => _inputSize;
    public int OutputSize => _outputSize;

    public DenseLayer(int inputSize, int outputSize, RLActivationKind? activation, bool useAdam)
    {
        _inputSize = inputSize;
        _outputSize = outputSize;
        _activation = activation;
        _useAdam = useAdam;
        _weights = new float[inputSize * outputSize];
        _biases = new float[outputSize];
        _rng.Randomize();

        var scale = Mathf.Sqrt(2.0f / Mathf.Max(1, inputSize));
        for (var index = 0; index < _weights.Length; index++)
        {
            _weights[index] = _rng.Randfn(0.0f, scale);
        }

        if (useAdam)
        {
            _wm = new float[_weights.Length];
            _wv = new float[_weights.Length];
            _bm = new float[outputSize];
            _bv = new float[outputSize];
        }
    }

    public LayerCache Forward(float[] input)
    {
        var preActivation = new float[_outputSize];
        for (var outputIndex = 0; outputIndex < _outputSize; outputIndex++)
        {
            var sum = _biases[outputIndex];
            for (var inputIndex = 0; inputIndex < _inputSize; inputIndex++)
            {
                sum += input[inputIndex] * _weights[(outputIndex * _inputSize) + inputIndex];
            }

            preActivation[outputIndex] = sum;
        }

        var activated = preActivation.ToArray();
        if (_activation.HasValue)
        {
            for (var index = 0; index < activated.Length; index++)
            {
                activated[index] = Activate(activated[index], _activation.Value);
            }
        }

        return new LayerCache
        {
            Input = input.ToArray(),
            PreActivation = preActivation,
            Activated = activated,
        };
    }

    public float[] Backward(float[] input, float[] outputGradient, float learningRate, float[]? preActivation = null)
    {
        var localGradient = outputGradient.ToArray();
        if (_activation.HasValue && preActivation is not null)
        {
            for (var index = 0; index < localGradient.Length; index++)
            {
                localGradient[index] *= ActivateDerivative(preActivation[index], _activation.Value);
            }
        }

        // Compute input gradient using original weights (before update)
        var inputGradient = new float[_inputSize];
        for (var outputIndex = 0; outputIndex < _outputSize; outputIndex++)
        {
            for (var inputIndex = 0; inputIndex < _inputSize; inputIndex++)
            {
                inputGradient[inputIndex] += _weights[(outputIndex * _inputSize) + inputIndex] * localGradient[outputIndex];
            }
        }

        if (_useAdam)
        {
            // Advance bias-correction accumulators
            _adamB1Pow *= AdamBeta1;
            _adamB2Pow *= AdamBeta2;
            var b1Corr = 1f - _adamB1Pow;
            var b2Corr = 1f - _adamB2Pow;

            for (var outputIndex = 0; outputIndex < _outputSize; outputIndex++)
            {
                for (var inputIndex = 0; inputIndex < _inputSize; inputIndex++)
                {
                    var wi = (outputIndex * _inputSize) + inputIndex;
                    var g = localGradient[outputIndex] * input[inputIndex];
                    _wm![wi] = AdamBeta1 * _wm[wi] + (1f - AdamBeta1) * g;
                    _wv![wi] = AdamBeta2 * _wv[wi] + (1f - AdamBeta2) * g * g;
                    var mHat = _wm[wi] / b1Corr;
                    var vHat = _wv![wi] / b2Corr;
                    _weights[wi] -= learningRate * mHat / (Mathf.Sqrt(vHat) + AdamEpsilon);
                }

                var bg = localGradient[outputIndex];
                _bm![outputIndex] = AdamBeta1 * _bm[outputIndex] + (1f - AdamBeta1) * bg;
                _bv![outputIndex] = AdamBeta2 * _bv[outputIndex] + (1f - AdamBeta2) * bg * bg;
                var bmHat = _bm[outputIndex] / b1Corr;
                var bvHat = _bv[outputIndex] / b2Corr;
                _biases[outputIndex] -= learningRate * bmHat / (Mathf.Sqrt(bvHat) + AdamEpsilon);
            }
        }
        else
        {
            for (var outputIndex = 0; outputIndex < _outputSize; outputIndex++)
            {
                for (var inputIndex = 0; inputIndex < _inputSize; inputIndex++)
                {
                    var weightIndex = (outputIndex * _inputSize) + inputIndex;
                    _weights[weightIndex] -= learningRate * localGradient[outputIndex] * input[inputIndex];
                }

                _biases[outputIndex] -= learningRate * localGradient[outputIndex];
            }
        }

        return inputGradient;
    }

    /// <summary>
    /// Computes input gradient only — does NOT update weights or biases.
    /// Used when you need dL/dinput through a frozen network (e.g. SAC actor update through Q).
    /// </summary>
    public float[] ComputeInputGrad(float[] input, float[] outputGradient, float[]? preActivation = null)
    {
        var localGradient = outputGradient.ToArray();
        if (_activation.HasValue && preActivation is not null)
        {
            for (var index = 0; index < localGradient.Length; index++)
            {
                localGradient[index] *= ActivateDerivative(preActivation[index], _activation.Value);
            }
        }

        var inputGradient = new float[_inputSize];
        for (var outputIndex = 0; outputIndex < _outputSize; outputIndex++)
        {
            for (var inputIndex = 0; inputIndex < _inputSize; inputIndex++)
            {
                inputGradient[inputIndex] += _weights[(outputIndex * _inputSize) + inputIndex] * localGradient[outputIndex];
            }
        }

        return inputGradient;
    }

    /// <summary>Hard-copies weights and biases from another layer (must have same shape).</summary>
    public void CopyFrom(DenseLayer source)
    {
        Array.Copy(source._weights, _weights, _weights.Length);
        Array.Copy(source._biases, _biases, _biases.Length);
    }

    /// <summary>Polyak-averages weights from source: θ = τ*θ_src + (1-τ)*θ.</summary>
    public void SoftUpdateFrom(DenseLayer source, float tau)
    {
        for (var i = 0; i < _weights.Length; i++)
        {
            _weights[i] = tau * source._weights[i] + (1f - tau) * _weights[i];
        }

        for (var i = 0; i < _biases.Length; i++)
        {
            _biases[i] = tau * source._biases[i] + (1f - tau) * _biases[i];
        }
    }

    public void AppendSerialized(ICollection<float> weights, ICollection<int> shapes)
    {
        shapes.Add(_inputSize);
        shapes.Add(_outputSize);
        shapes.Add(_activation.HasValue ? (int)_activation.Value + 1 : 0);
        foreach (var weight in _weights)
        {
            weights.Add(weight);
        }

        foreach (var bias in _biases)
        {
            weights.Add(bias);
        }
    }

    public void LoadSerialized(IReadOnlyList<float> weights, ref int weightIndex, IReadOnlyList<int> shapes, ref int shapeIndex)
    {
        var serializedInputSize = shapes[shapeIndex++];
        var serializedOutputSize = shapes[shapeIndex++];
        var serializedActivation = shapes[shapeIndex++];
        if (serializedInputSize != _inputSize || serializedOutputSize != _outputSize)
        {
            throw new InvalidOperationException("Checkpoint layer shape does not match the active network.");
        }

        var expectedActivation = _activation.HasValue ? (int)_activation.Value + 1 : 0;
        if (serializedActivation != expectedActivation)
        {
            throw new InvalidOperationException("Checkpoint activation does not match the active network.");
        }

        for (var index = 0; index < _weights.Length; index++)
        {
            _weights[index] = weights[weightIndex++];
        }

        for (var index = 0; index < _biases.Length; index++)
        {
            _biases[index] = weights[weightIndex++];
        }

        // Reset Adam moments on checkpoint load so they warm up from the new weights
        if (_useAdam)
        {
            Array.Clear(_wm!, 0, _wm!.Length);
            Array.Clear(_wv!, 0, _wv!.Length);
            Array.Clear(_bm!, 0, _bm!.Length);
            Array.Clear(_bv!, 0, _bv!.Length);
            _adamB1Pow = 1f;
            _adamB2Pow = 1f;
        }
    }

    private static float Activate(float value, RLActivationKind activation)
    {
        return activation == RLActivationKind.Relu ? Mathf.Max(0.0f, value) : Mathf.Tanh(value);
    }

    private static float ActivateDerivative(float value, RLActivationKind activation)
    {
        if (activation == RLActivationKind.Relu)
        {
            return value > 0.0f ? 1.0f : 0.0f;
        }

        var tanh = Mathf.Tanh(value);
        return 1.0f - (tanh * tanh);
    }
}
