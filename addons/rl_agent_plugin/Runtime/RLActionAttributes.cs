using System;

namespace RlAgentPlugin.Runtime;

[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
public sealed class DiscreteActionAttribute : Attribute
{
    public DiscreteActionAttribute(int actionCount, params string[] labels)
    {
        ActionCount = Math.Max(0, actionCount);
        Labels = labels ?? Array.Empty<string>();
    }

    public int ActionCount { get; }
    public string Name { get; set; } = string.Empty;
    public string[] Labels { get; }
}

[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
public sealed class ContinuousActionAttribute : Attribute
{
    public ContinuousActionAttribute(int dimensions)
    {
        Dimensions = Math.Max(0, dimensions);
    }

    public int Dimensions { get; }
    public string Name { get; set; } = string.Empty;
}
