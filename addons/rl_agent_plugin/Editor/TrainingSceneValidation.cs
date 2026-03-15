using System.Collections.Generic;
using System.Text;

namespace RlAgentPlugin.Editor;

public sealed class PolicyGroupSummary
{
    public string GroupId { get; set; } = string.Empty;
    public int AgentCount { get; set; }
    public int ActionCount { get; set; }
    public bool IsContinuous { get; set; }
    public int ContinuousActionDimensions { get; set; }
}

public sealed class TrainingSceneValidation
{
    public string ScenePath { get; set; } = string.Empty;
    public string AcademyPath { get; set; } = string.Empty;
    public string TrainerConfigPath { get; set; } = string.Empty;
    public string NetworkConfigPath { get; set; } = string.Empty;
    public string CheckpointPath { get; set; } = string.Empty;
    public string RunPrefix { get; set; } = string.Empty;
    public int CheckpointInterval { get; set; } = 10;
    public float SimulationSpeed { get; set; } = 1.0f;
    public int ActionRepeat { get; set; } = 1;
    public int ExpectedActionCount { get; set; }
    public int TrainAgentCount { get; set; }
    public bool IsValid { get; set; }
    public List<string> AgentNames { get; } = new();
    public List<string> Errors { get; } = new();
    public List<PolicyGroupSummary> PolicyGroups { get; } = new();

    public string BuildSummary()
    {
        var builder = new StringBuilder();
        builder.AppendLine($"Scene: {ScenePath}");
        builder.AppendLine(IsValid ? "Validation: ready to train" : "Validation: blocked");
        if (!string.IsNullOrWhiteSpace(AcademyPath))
        {
            builder.AppendLine($"Academy: {AcademyPath}");
        }

        builder.AppendLine($"Agents: {AgentNames.Count}");
        builder.AppendLine($"Train Agents: {TrainAgentCount}");

        if (PolicyGroups.Count > 1)
        {
            builder.AppendLine($"Policy Groups: {PolicyGroups.Count}");
            foreach (var group in PolicyGroups)
            {
                var actionInfo = group.IsContinuous
                    ? $"{group.ContinuousActionDimensions}D continuous"
                    : $"{group.ActionCount} discrete";
                builder.AppendLine($"  '{group.GroupId}': {group.AgentCount} agent(s), {actionInfo}");
            }
        }

        foreach (var error in Errors)
        {
            builder.AppendLine($"- {error}");
        }

        return builder.ToString().TrimEnd();
    }
}
