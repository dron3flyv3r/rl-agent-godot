using Godot;

namespace RlAgentPlugin.Runtime;

public static class RLPolicyGroupBindingResolver
{
    public static ResolvedPolicyGroupBinding Resolve(Node sceneRoot, Node agentNode)
    {
        var agentRelativePath = sceneRoot.GetPathTo(agentNode).ToString();
        var groupConfig = ResolvePolicyGroupConfig(agentNode);

        if (groupConfig is not null)
        {
            var key = ResolveExplicitGroupKey(groupConfig, agentRelativePath);
            var displayName = ResolveExplicitDisplayName(groupConfig, key);
            return new ResolvedPolicyGroupBinding
            {
                BindingKey = key,
                DisplayName = displayName,
                SafeGroupId = MakeSafeGroupId(key),
                AgentRelativePath = agentRelativePath,
                Config = groupConfig,
                ConfigPath = groupConfig.ResourcePath,
            };
        }

        var legacyGroup = ResolveLegacyPolicyGroup(agentNode);
        if (!string.IsNullOrWhiteSpace(legacyGroup))
        {
            var trimmedGroup = legacyGroup.Trim();
            return new ResolvedPolicyGroupBinding
            {
                BindingKey = trimmedGroup,
                DisplayName = trimmedGroup,
                SafeGroupId = MakeSafeGroupId(trimmedGroup),
                AgentRelativePath = agentRelativePath,
            };
        }

        var fallbackKey = $"__agent__{agentRelativePath}";
        return new ResolvedPolicyGroupBinding
        {
            BindingKey = fallbackKey,
            DisplayName = agentNode.Name.ToString(),
            SafeGroupId = MakeSafeGroupId(fallbackKey),
            AgentRelativePath = agentRelativePath,
        };
    }

    public static string MakeSafeGroupId(string groupId)
    {
        var safe = new System.Text.StringBuilder(groupId.Length);
        foreach (var c in groupId)
        {
            safe.Append(char.IsLetterOrDigit(c) || c == '-' ? c : '_');
        }

        var result = safe.ToString().Trim('_');
        if (string.IsNullOrEmpty(result))
        {
            result = "default";
        }

        if (result.Length > 64)
        {
            result = result[..64];
        }

        return result;
    }

    private static RLPolicyGroupConfig? ResolvePolicyGroupConfig(Node agentNode)
    {
        var variant = agentNode.Get("AgentConfig");
        if (variant.VariantType != Variant.Type.Object)
        {
            return null;
        }

        return variant.AsGodotObject() is RLAgentConfig agentConfig
            ? agentConfig.PolicyGroupConfig
            : null;
    }

    private static string ResolveLegacyPolicyGroup(Node agentNode)
    {
        var variant = agentNode.Get("PolicyGroup");
        return variant.VariantType == Variant.Type.String ? variant.AsString() : string.Empty;
    }

    private static string ResolveExplicitGroupKey(RLPolicyGroupConfig groupConfig, string agentRelativePath)
    {
        if (!string.IsNullOrWhiteSpace(groupConfig.GroupId))
        {
            return groupConfig.GroupId.Trim();
        }

        if (!string.IsNullOrWhiteSpace(groupConfig.ResourcePath))
        {
            return groupConfig.ResourcePath;
        }

        return $"__policycfg__{agentRelativePath}";
    }

    private static string ResolveExplicitDisplayName(RLPolicyGroupConfig groupConfig, string fallbackKey)
    {
        if (!string.IsNullOrWhiteSpace(groupConfig.GroupId))
        {
            return groupConfig.GroupId.Trim();
        }

        if (!string.IsNullOrWhiteSpace(groupConfig.ResourceName))
        {
            return groupConfig.ResourceName;
        }

        if (!string.IsNullOrWhiteSpace(groupConfig.ResourcePath))
        {
            return groupConfig.ResourcePath;
        }

        return fallbackKey;
    }
}
