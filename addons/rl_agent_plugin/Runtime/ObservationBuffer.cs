using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

public sealed class ObservationBuffer
{
    private readonly List<float> _values = new();

    public int Count => _values.Count;

    /// <summary>Add a single float value as-is.</summary>
    public void Add(float value) => _values.Add(value);

    /// <summary>Add a boolean as 0 or 1.</summary>
    public void Add(bool value) => _values.Add(value ? 1f : 0f);

    /// <summary>Add a Vector2 as two floats (X, Y).</summary>
    public void Add(Vector2 value)
    {
        _values.Add(value.X);
        _values.Add(value.Y);
    }

    /// <summary>Add a Vector3 as three floats (X, Y, Z).</summary>
    public void Add(Vector3 value)
    {
        _values.Add(value.X);
        _values.Add(value.Y);
        _values.Add(value.Z);
    }

    /// <summary>Add a value linearly mapped from [min, max] to [-1, 1].</summary>
    public void AddNormalized(float value, float min, float max)
    {
        var range = max - min;
        var normalized = range > 0f ? (value - min) / range * 2f - 1f : 0f;
        _values.Add(Mathf.Clamp(normalized, -1f, 1f));
    }

    internal float[] ToArray() => _values.ToArray();
    internal void Clear() => _values.Clear();
}
