using System;

namespace RlAgentPlugin.Runtime;

public enum RLActionVariableType
{
    Discrete = 0,
    Continuous = 1,
}

public readonly struct RLActionDefinition
{
    public RLActionDefinition(string name, RLActionVariableType variableType = RLActionVariableType.Discrete)
    {
        Name = string.IsNullOrWhiteSpace(name) ? "Action" : name;
        VariableType = variableType;
    }

    public string Name { get; }
    public RLActionVariableType VariableType { get; }
}
