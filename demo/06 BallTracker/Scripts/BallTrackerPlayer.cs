using Godot;

namespace RlAgentPlugin.Demo;

/// <summary>
/// The player entity. Handles movement and exposes state for the agent.
/// The RLAgent2D is a child of this node.
/// </summary>
public partial class BallTrackerPlayer : CharacterBody2D
{
    [Export] public float ArenaMinX { get; set; } = 40f;
    [Export] public float ArenaMaxX { get; set; } = 760f;
    [Export] public float ArenaMinY { get; set; } = 40f;
    [Export] public float ArenaMaxY { get; set; } = 560f;
    [Export] public float MoveSpeed { get; set; } = 180f;
    [Export] public NodePath BallPath { get; set; } = "";

    public BallTrackerBall? Ball { get; private set; }
    public Vector2 MoveInput { get; set; }

    public override void _Ready()
    {
        if (!string.IsNullOrEmpty(BallPath))
            Ball = GetNodeOrNull<BallTrackerBall>(BallPath);

        // Fallback: find the first BallTrackerBall sibling in the parent node.
        if (Ball is null)
            Ball = FindSiblingBall();

        if (Ball is null)
            GD.PushWarning("[BallTrackerPlayer] Ball node not found — assign BallPath in the inspector.");
    }

    private BallTrackerBall? FindSiblingBall()
    {
        var parent = GetParent();
        if (parent is null) return null;
        foreach (var child in parent.GetChildren())
            if (child is BallTrackerBall ball) return ball;
        return null;
    }

    public override void _PhysicsProcess(double delta)
    {
        var vel    = MoveInput * MoveSpeed * (float)delta;
        var newPos = Position + vel;
        newPos.X   = Mathf.Clamp(newPos.X, ArenaMinX, ArenaMaxX);
        newPos.Y   = Mathf.Clamp(newPos.Y, ArenaMinY, ArenaMaxY);
        Position   = newPos;
    }

    public void ResetPosition()
    {
        Position = new Vector2(
            (float)GD.RandRange(ArenaMinX + 40f, ArenaMaxX - 40f),
            (float)GD.RandRange(ArenaMinY + 40f, ArenaMaxY - 40f));
        MoveInput = Vector2.Zero;
    }
}
