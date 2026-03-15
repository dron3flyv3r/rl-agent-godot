using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Godot;

namespace RlAgentPlugin.Runtime;

internal sealed class RLActionBinding
{
    private readonly ActionMemberBinding[] _discreteMembers;
    private readonly ContinuousMemberBinding[] _continuousMembers;

    private RLActionBinding(
        ActionMemberBinding[] discreteMembers,
        ContinuousMemberBinding[] continuousMembers,
        bool supportsOnlyDiscreteActions,
        RLActionDefinition[] actionSpace)
    {
        _discreteMembers = discreteMembers;
        _continuousMembers = continuousMembers;
        SupportsOnlyDiscreteActions = supportsOnlyDiscreteActions;
        ActionSpace = actionSpace;
        ContinuousActionDimensions = continuousMembers.Sum(m => m.Dimensions);
    }

    public bool SupportsOnlyDiscreteActions { get; }
    public RLActionDefinition[] ActionSpace { get; }
    public int ContinuousActionDimensions { get; }

    public static RLActionBinding? Create(Type agentType)
    {
        const BindingFlags bindingFlags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        var members = agentType
            .GetMembers(bindingFlags)
            .Where(member => member.MemberType is MemberTypes.Field or MemberTypes.Property)
            .OrderBy(member => member.MetadataToken)
            .ToArray();

        var discreteMembers = new List<ActionMemberBinding>();
        var continuousMembers = new List<ContinuousMemberBinding>();

        foreach (var member in members)
        {
            var discreteAttribute = member.GetCustomAttribute<DiscreteActionAttribute>();
            if (discreteAttribute is not null)
            {
                var memberType = GetMemberType(member);
                if (memberType is null || !SupportsDiscreteStorage(memberType) || discreteAttribute.ActionCount <= 0)
                {
                    continue;
                }

                discreteMembers.Add(new ActionMemberBinding(member, memberType, discreteAttribute));
            }

            var continuousAttribute = member.GetCustomAttribute<ContinuousActionAttribute>();
            if (continuousAttribute is not null && continuousAttribute.Dimensions > 0)
            {
                var memberType = GetMemberType(member);
                if (memberType is null)
                {
                    continue;
                }

                var dims = ResolveContinuousDimensions(memberType, continuousAttribute.Dimensions);
                if (dims > 0)
                {
                    continuousMembers.Add(new ContinuousMemberBinding(member, memberType, continuousAttribute, dims));
                }
            }
        }

        if (discreteMembers.Count == 0 && continuousMembers.Count == 0)
        {
            return null;
        }

        var supportsOnlyDiscreteActions = continuousMembers.Count == 0;
        var actionSpace = supportsOnlyDiscreteActions
            ? BuildCombinedActionSpace(discreteMembers)
            : Array.Empty<RLActionDefinition>();

        return new RLActionBinding(discreteMembers.ToArray(), continuousMembers.ToArray(), supportsOnlyDiscreteActions, actionSpace);
    }

    public bool TryApply(object agentInstance, int actionIndex)
    {
        if (!SupportsOnlyDiscreteActions || _discreteMembers.Length == 0 || actionIndex < 0 || actionIndex >= ActionSpace.Length)
        {
            return false;
        }

        var remaining = actionIndex;
        for (var index = 0; index < _discreteMembers.Length; index++)
        {
            var member = _discreteMembers[index];
            var memberValue = remaining % member.ActionCount;
            remaining /= member.ActionCount;
            SetDiscreteValue(agentInstance, member, memberValue);
        }

        return true;
    }

    public bool TryApplyContinuous(object agentInstance, float[] actions)
    {
        if (_continuousMembers.Length == 0 || actions.Length < ContinuousActionDimensions)
        {
            return false;
        }

        var offset = 0;
        foreach (var member in _continuousMembers)
        {
            SetContinuousValue(agentInstance, member, actions, offset);
            offset += member.Dimensions;
        }

        return true;
    }

    private static RLActionDefinition[] BuildCombinedActionSpace(IReadOnlyList<ActionMemberBinding> members)
    {
        if (members.Count == 0)
        {
            return Array.Empty<RLActionDefinition>();
        }

        var totalActions = 1;
        foreach (var member in members)
        {
            totalActions *= member.ActionCount;
        }

        var labels = new RLActionDefinition[totalActions];
        for (var flatIndex = 0; flatIndex < totalActions; flatIndex++)
        {
            var remaining = flatIndex;
            var parts = new string[members.Count];
            for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
            {
                var member = members[memberIndex];
                var valueIndex = remaining % member.ActionCount;
                remaining /= member.ActionCount;
                parts[memberIndex] = $"{member.DisplayName}={member.GetValueLabel(valueIndex)}";
            }

            labels[flatIndex] = new RLActionDefinition(string.Join(", ", parts));
        }

        return labels;
    }

    private static void SetDiscreteValue(object agentInstance, ActionMemberBinding member, int actionIndex)
    {
        object boxedValue;
        if (member.MemberType.IsEnum)
        {
            boxedValue = Enum.ToObject(member.MemberType, actionIndex);
        }
        else
        {
            boxedValue = actionIndex;
        }

        switch (member.MemberInfo)
        {
            case PropertyInfo propertyInfo when propertyInfo.SetMethod is not null:
                propertyInfo.SetValue(agentInstance, boxedValue);
                break;
            case FieldInfo fieldInfo:
                fieldInfo.SetValue(agentInstance, boxedValue);
                break;
        }
    }

    private static void SetContinuousValue(object agentInstance, ContinuousMemberBinding member, float[] actions, int offset)
    {
        object? boxedValue = null;

        if (member.MemberType == typeof(float))
        {
            boxedValue = actions[offset];
        }
        else if (member.MemberType == typeof(float[]))
        {
            var arr = new float[member.Dimensions];
            Array.Copy(actions, offset, arr, 0, member.Dimensions);
            boxedValue = arr;
        }
        else if (member.MemberType == typeof(Vector2) && member.Dimensions >= 2)
        {
            boxedValue = new Vector2(actions[offset], actions[offset + 1]);
        }
        else if (member.MemberType == typeof(Vector3) && member.Dimensions >= 3)
        {
            boxedValue = new Vector3(actions[offset], actions[offset + 1], actions[offset + 2]);
        }

        if (boxedValue is null)
        {
            return;
        }

        switch (member.MemberInfo)
        {
            case PropertyInfo propertyInfo when propertyInfo.SetMethod is not null:
                propertyInfo.SetValue(agentInstance, boxedValue);
                break;
            case FieldInfo fieldInfo:
                fieldInfo.SetValue(agentInstance, boxedValue);
                break;
        }
    }

    private static int ResolveContinuousDimensions(Type memberType, int attributeDimensions)
    {
        if (memberType == typeof(float)) return 1;
        if (memberType == typeof(Vector2)) return 2;
        if (memberType == typeof(Vector3)) return 3;
        if (memberType == typeof(float[])) return attributeDimensions;
        return 0;
    }

    private static Type? GetMemberType(MemberInfo memberInfo)
    {
        return memberInfo switch
        {
            PropertyInfo propertyInfo => propertyInfo.PropertyType,
            FieldInfo fieldInfo => fieldInfo.FieldType,
            _ => null,
        };
    }

    private static bool SupportsDiscreteStorage(Type type)
    {
        return type == typeof(int) || type.IsEnum;
    }

    private sealed class ActionMemberBinding
    {
        private readonly string[] _valueLabels;

        public ActionMemberBinding(MemberInfo memberInfo, Type memberType, DiscreteActionAttribute attribute)
        {
            MemberInfo = memberInfo;
            MemberType = memberType;
            ActionCount = attribute.ActionCount;
            DisplayName = string.IsNullOrWhiteSpace(attribute.Name) ? memberInfo.Name : attribute.Name;
            _valueLabels = new string[ActionCount];
            for (var index = 0; index < ActionCount; index++)
            {
                _valueLabels[index] = index < attribute.Labels.Length && !string.IsNullOrWhiteSpace(attribute.Labels[index])
                    ? attribute.Labels[index]
                    : index.ToString();
            }
        }

        public MemberInfo MemberInfo { get; }
        public Type MemberType { get; }
        public int ActionCount { get; }
        public string DisplayName { get; }

        public string GetValueLabel(int index) => _valueLabels[index];
    }

    private sealed class ContinuousMemberBinding
    {
        public ContinuousMemberBinding(MemberInfo memberInfo, Type memberType, ContinuousActionAttribute attribute, int dimensions)
        {
            MemberInfo = memberInfo;
            MemberType = memberType;
            Dimensions = dimensions;
        }

        public MemberInfo MemberInfo { get; }
        public Type MemberType { get; }
        public int Dimensions { get; }
    }
}
