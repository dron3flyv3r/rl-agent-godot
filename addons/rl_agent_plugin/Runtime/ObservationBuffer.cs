using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

public sealed class ObservationBuffer
{
    private readonly List<float> _values = new();
    private readonly List<ObservationSegment> _segments = new();

    public int Count => _values.Count;
    public IReadOnlyList<ObservationSegment> Segments => _segments;

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

    public void AddNormalized(Vector2 value, Vector2 min, Vector2 max)
    {
        AddNormalized(value.X, min.X, max.X);
        AddNormalized(value.Y, min.Y, max.Y);
    }

    public void AddNormalized(Vector3 value, Vector3 min, Vector3 max)
    {
        AddNormalized(value.X, min.X, max.X);
        AddNormalized(value.Y, min.Y, max.Y);
        AddNormalized(value.Z, min.Z, max.Z);
    }

    public void AddNormalized(int value, int min, int max) => AddNormalized((float)value, (float)min, (float)max);

    public void AddSensor(IObservationSensor sensor)
    {
        sensor.Write(this);
    }

    public void AddSensor(string name, IObservationSensor sensor)
    {
        var startIndex = Count;
        sensor.Write(this);
        var length = Count - startIndex;
        if (length > 0)
        {
            _segments.Add(new ObservationSegment(name, startIndex, length));
        }
    }

    internal float[] ToArray() => _values.ToArray();

    internal void Clear()
    {
        _values.Clear();
        _segments.Clear();
    }
}
