using System;
using Godot;

namespace RlAgentPlugin.Demo.Benchmarks;

public sealed class DiscreteBanditBenchEnvironment : IPlanningBenchEnvironment
{
    public string Name => "DiscreteBandit";
    public int ObservationSize => 1;
    public int DiscreteActionCount => 2;
    public int ContinuousActionDimensions => 0;

    public float[] Reset(int seed)
        => new[] { 1f };

    public AlgorithmBenchStep Step(AlgorithmBenchAction action)
    {
        var reward = action.DiscreteAction == 1 ? 1f : 0f;
        return new AlgorithmBenchStep(new[] { 1f }, reward, true);
    }

    public (float[] nextObservation, float reward, bool done) SimulateStep(float[] observation, int action)
        => (new[] { 1f }, action == 1 ? 1f : 0f, true);
}

public sealed class DiscreteLineWorldBenchEnvironment : IPlanningBenchEnvironment
{
    private const int GoalPosition = 4;
    private const int MaxEpisodeSteps = 8;

    private int _position;
    private int _stepsTaken;

    public string Name => "DiscreteLineWorld";
    public int ObservationSize => 2;
    public int DiscreteActionCount => 2;
    public int ContinuousActionDimensions => 0;

    public float[] Reset(int seed)
    {
        _position = 0;
        _stepsTaken = 0;
        return Encode(_position);
    }

    public AlgorithmBenchStep Step(AlgorithmBenchAction action)
    {
        _stepsTaken++;
        _position = Transition(_position, action.DiscreteAction);
        var reward = _position == GoalPosition ? 1f : -0.05f;
        var done = _position == GoalPosition || _stepsTaken >= MaxEpisodeSteps;
        return new AlgorithmBenchStep(Encode(_position), reward, done);
    }

    public (float[] nextObservation, float reward, bool done) SimulateStep(float[] observation, int action)
    {
        var position = DecodePosition(observation);
        var nextPosition = Transition(position, action);
        var reward = nextPosition == GoalPosition ? 1f : -0.05f;
        var currentSteps = Math.Max(0, Mathf.RoundToInt(observation[1] * MaxEpisodeSteps));
        var nextSteps = Math.Min(MaxEpisodeSteps, currentSteps + 1);
        var done = nextPosition == GoalPosition || nextSteps >= MaxEpisodeSteps;
        return (Encode(nextPosition, nextSteps), reward, done);
    }

    private static int Transition(int position, int action)
    {
        return action switch
        {
            0 => Math.Max(0, position - 1),
            1 => Math.Min(GoalPosition, position + 1),
            _ => position,
        };
    }

    private float[] Encode(int position)
        => Encode(position, _stepsTaken);

    private static float[] Encode(int position, int stepsTaken)
    {
        return new[]
        {
            position / (float)GoalPosition,
            stepsTaken / (float)MaxEpisodeSteps,
        };
    }

    private static int DecodePosition(float[] observation)
        => Mathf.Clamp(Mathf.RoundToInt(observation[0] * GoalPosition), 0, GoalPosition);
}

public sealed class ContinuousTarget1DBenchEnvironment : IAlgorithmBenchEnvironment
{
    private const float StartPosition = -0.75f;
    private const float TargetPosition = 0.75f;
    private const float ActionScale = 0.25f;
    private const int MaxEpisodeSteps = 16;

    private float _position;
    private int _stepsTaken;

    public string Name => "ContinuousTarget1D";
    public int ObservationSize => 3;
    public int DiscreteActionCount => 0;
    public int ContinuousActionDimensions => 1;

    public float[] Reset(int seed)
    {
        _position = StartPosition;
        _stepsTaken = 0;
        return Encode();
    }

    public AlgorithmBenchStep Step(AlgorithmBenchAction action)
    {
        _stepsTaken++;
        var chosen = action.ContinuousActions.Length > 0 ? action.ContinuousActions[0] : 0f;
        _position = Mathf.Clamp(_position + (chosen * ActionScale), -1f, 1f);
        var distance = Mathf.Abs(TargetPosition - _position);
        var reward = 1f - distance;
        var done = distance <= 0.05f || _stepsTaken >= MaxEpisodeSteps;
        return new AlgorithmBenchStep(Encode(), reward, done);
    }

    private float[] Encode()
    {
        return new[]
        {
            _position,
            TargetPosition,
            _stepsTaken / (float)MaxEpisodeSteps,
        };
    }
}

public sealed class SyntheticDiscreteVectorBenchEnvironment : IPlanningBenchEnvironment
{
    private readonly int _observationSize;
    private readonly int _actionCount;
    private readonly int _maxEpisodeSteps;
    private readonly int _computeIterations;

    private float _phase;
    private int _stepsTaken;

    public SyntheticDiscreteVectorBenchEnvironment(
        int observationSize,
        int actionCount,
        int maxEpisodeSteps,
        int computeIterations)
    {
        _observationSize = Math.Max(4, observationSize);
        _actionCount = Math.Max(2, actionCount);
        _maxEpisodeSteps = Math.Max(1, maxEpisodeSteps);
        _computeIterations = Math.Max(0, computeIterations);
    }

    public string Name => "SyntheticDiscreteVector";
    public int ObservationSize => _observationSize;
    public int DiscreteActionCount => _actionCount;
    public int ContinuousActionDimensions => 0;

    public float[] Reset(int seed)
    {
        _phase = SeedToPhase(seed);
        _stepsTaken = 0;
        return EncodeObservation(_phase, _stepsTaken);
    }

    public AlgorithmBenchStep Step(AlgorithmBenchAction action)
    {
        _stepsTaken++;
        var bestAction = ResolveBestAction(_phase);
        var reward = action.DiscreteAction == bestAction ? 1f : -0.25f;
        _phase = AdvancePhase(_phase, action.DiscreteAction);
        var done = _stepsTaken >= _maxEpisodeSteps;
        return new AlgorithmBenchStep(EncodeObservation(_phase, _stepsTaken), reward, done);
    }

    public (float[] nextObservation, float reward, bool done) SimulateStep(float[] observation, int action)
    {
        var phase = observation.Length > 0 ? observation[0] : 0f;
        var steps = observation.Length > 1 ? Math.Max(0, Mathf.RoundToInt(observation[1] * _maxEpisodeSteps)) : 0;
        var bestAction = ResolveBestAction(phase);
        var reward = action == bestAction ? 1f : -0.25f;
        var nextSteps = Math.Min(_maxEpisodeSteps, steps + 1);
        var nextPhase = AdvancePhase(phase, action);
        var done = nextSteps >= _maxEpisodeSteps;
        return (EncodeObservation(nextPhase, nextSteps), reward, done);
    }

    private float[] EncodeObservation(float phase, int stepsTaken)
    {
        var observation = new float[_observationSize];
        observation[0] = phase;
        observation[1] = stepsTaken / (float)_maxEpisodeSteps;

        var accumulator = phase + (stepsTaken * 0.0137f);
        for (var index = 2; index < observation.Length; index++)
        {
            accumulator = FakeCompute(accumulator + (index * 0.001f), _computeIterations);
            observation[index] = accumulator;
        }

        return observation;
    }

    private int ResolveBestAction(float phase)
    {
        var bucket = (int)MathF.Abs(MathF.Floor(phase * 97f));
        return bucket % _actionCount;
    }

    private static float SeedToPhase(int seed)
    {
        var normalized = (seed % 1024) / 1024f;
        return (normalized * 2f) - 1f;
    }

    private static float AdvancePhase(float phase, int action)
    {
        var delta = ((action + 1) * 0.071f) - 0.09f;
        var next = phase + delta;
        if (next > 1f) next -= 2f;
        if (next < -1f) next += 2f;
        return next;
    }

    private static float FakeCompute(float value, int iterations)
    {
        var current = value;
        for (var i = 0; i < iterations; i++)
        {
            current = MathF.Sin(current * 1.173f + 0.31f) + MathF.Cos(current * 0.417f - 0.19f);
            current *= 0.5f;
        }

        return current;
    }
}

public sealed class SyntheticContinuousVectorBenchEnvironment : IAlgorithmBenchEnvironment
{
    private readonly int _observationSize;
    private readonly int _actionDimensions;
    private readonly int _maxEpisodeSteps;
    private readonly int _computeIterations;

    private float _position;
    private float _target;
    private int _stepsTaken;

    public SyntheticContinuousVectorBenchEnvironment(
        int observationSize,
        int actionDimensions,
        int maxEpisodeSteps,
        int computeIterations)
    {
        _observationSize = Math.Max(4, observationSize);
        _actionDimensions = Math.Max(1, actionDimensions);
        _maxEpisodeSteps = Math.Max(1, maxEpisodeSteps);
        _computeIterations = Math.Max(0, computeIterations);
    }

    public string Name => "SyntheticContinuousVector";
    public int ObservationSize => _observationSize;
    public int DiscreteActionCount => 0;
    public int ContinuousActionDimensions => _actionDimensions;

    public float[] Reset(int seed)
    {
        _stepsTaken = 0;
        _position = ((seed % 97) / 97f) * 2f - 1f;
        _target = (((seed + 17) % 89) / 89f) * 2f - 1f;
        return EncodeObservation(_position, _target, _stepsTaken);
    }

    public AlgorithmBenchStep Step(AlgorithmBenchAction action)
    {
        _stepsTaken++;

        var delta = 0f;
        for (var index = 0; index < _actionDimensions; index++)
        {
            var value = index < action.ContinuousActions.Length ? action.ContinuousActions[index] : 0f;
            delta += value;
        }

        delta /= _actionDimensions;
        _position = Mathf.Clamp(_position + (delta * 0.15f), -1f, 1f);

        var distance = Mathf.Abs(_target - _position);
        var reward = 1f - distance;
        var done = distance < 0.05f || _stepsTaken >= _maxEpisodeSteps;
        return new AlgorithmBenchStep(EncodeObservation(_position, _target, _stepsTaken), reward, done);
    }

    private float[] EncodeObservation(float position, float target, int stepsTaken)
    {
        var observation = new float[_observationSize];
        observation[0] = position;
        observation[1] = target;
        observation[2] = stepsTaken / (float)_maxEpisodeSteps;

        var accumulator = position - target;
        for (var index = 3; index < observation.Length; index++)
        {
            accumulator = SyntheticDiscreteVectorBenchEnvironment_FakeCompute(accumulator + (index * 0.002f), _computeIterations);
            observation[index] = accumulator;
        }

        return observation;
    }

    private static float SyntheticDiscreteVectorBenchEnvironment_FakeCompute(float value, int iterations)
    {
        var current = value;
        for (var i = 0; i < iterations; i++)
        {
            current = MathF.Sin(current * 1.071f + 0.27f) - MathF.Cos(current * 0.611f + 0.13f);
            current *= 0.5f;
        }

        return current;
    }
}
