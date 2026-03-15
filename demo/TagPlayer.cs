using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

public enum TagAgentRole
{
    Chaser = 0,
    Runner = 1,
}

public partial class TagPlayer : CharacterBody2D
{
    [Export] public TagAgentRole Role { get; set; } = TagAgentRole.Runner;
    [Export] public float MoveSpeed { get; set; } = 210.0f;
    [Export] public bool ManualControlWhenStandalone { get; set; } = true;

    public TagAgent? Agent { get; private set; }
    public string AgentId => Agent?.AgentId ?? Name;

    public override void _Ready()
    {
        Agent = GetNodeOrNull<TagAgent>("Agent");
    }

    public void StepMovement(TagArenaController arena, double delta)
    {
        if (Agent is null)
        {
            return;
        }

        if (!arena.IsTrainingSession
            && Agent.ControlMode == RLAgentControlMode.Human
            && ManualControlWhenStandalone)
        {
            var action = 0;
            if (Input.IsActionPressed("ui_up"))
            {
                action = 1;
            }
            else if (Input.IsActionPressed("ui_down"))
            {
                action = 2;
            }
            else if (Input.IsActionPressed("ui_left"))
            {
                action = 3;
            }
            else if (Input.IsActionPressed("ui_right"))
            {
                action = 4;
            }

            Agent.ApplyAction(action);
        }

        var nextPosition = Position + (Agent.MoveIntent * MoveSpeed * (float)delta);
        Position = arena.ClampToArena(nextPosition);
    }

    public void SetSpawnPosition(Vector2 spawnPosition)
    {
        Position = spawnPosition;
    }
}
