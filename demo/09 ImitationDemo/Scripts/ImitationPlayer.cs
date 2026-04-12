using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

/// <summary>
/// 2D top-down player controller for the Imitation Learning demo.
/// Receives a move direction from <see cref="ImitationAgent"/> and moves within a bounded arena.
/// Handles keyboard input when the agent is in Human mode, because the RLAcademy's Human mode
/// is not activated during the RecordingBootstrap session (agents default to Auto at scene load
/// and the academy caches this at _Ready time). Input action indices match ImitationAgent's
/// MoveAction enum: Stay=0, Up=1, Down=2, Left=3, Right=4.
/// </summary>
public partial class ImitationPlayer : CharacterBody2D
{
    [Export] public float ArenaLeft   = 100f;
    [Export] public float ArenaRight  = 700f;
    [Export] public float ArenaTop    = 60f;
    [Export] public float ArenaBottom = 540f;
    [Export] public float Speed       = 200f;
    [Export] public float GoalRadius  = 24f;

    private Vector2 _moveDir = Vector2.Zero;
    private Vector2 _goalPosition;
    private Node2D? _goalMarker;
    private IRLAgent? _agent;

    // Action indices — must match ImitationAgent.MoveAction enum order.
    private const int Stay  = 0;
    private const int Up    = 1;
    private const int Down  = 2;
    private const int Left  = 3;
    private const int Right = 4;

    public Vector2 GoalPosition => _goalPosition;
    public bool IsAtGoal => Position.DistanceTo(_goalPosition) < GoalRadius;

    public float ArenaWidth  => ArenaRight  - ArenaLeft;
    public float ArenaHeight => ArenaBottom - ArenaTop;
    public Vector2 ArenaCenter => new((ArenaLeft + ArenaRight) * 0.5f, (ArenaTop + ArenaBottom) * 0.5f);

    public override void _Ready()
    {
        foreach (var child in GetChildren())
        {
            if (child is IRLAgent a) { _agent = a; break; }
        }

        _goalMarker = GetParent()?.FindChild("GoalMarker", true, false) as Node2D;
        ResetEpisodeState();
    }

    /// <summary>Called by ImitationAgent each step to set the movement direction.</summary>
    public void SetMoveDirection(Vector2 dir)
        => _moveDir = dir;

    public override void _PhysicsProcess(double delta)
    {
        // Handle human input for recording mode.
        // RecordingBootstrap forces ControlMode = Human but the RLAcademy's Human mode pipeline
        // is not activated at that point, so this player script reads keyboard directly.
        if (_agent is not null && _agent.ControlMode == RLAgentControlMode.Human)
        {
            int action;

            // Step mode: RecordingBootstrap injects a specific action via PendingStepAction.
            if (_agent.PendingStepAction >= 0)
            {
                action = _agent.PendingStepAction;
                _agent.PendingStepAction = -1; // consume
            }
            else
            {
                // Normal Human mode: read keyboard.
                if      (Input.IsActionPressed("ui_up"))    action = Up;
                else if (Input.IsActionPressed("ui_down"))  action = Down;
                else if (Input.IsActionPressed("ui_left"))  action = Left;
                else if (Input.IsActionPressed("ui_right")) action = Right;
                else                                        action = Stay;
            }

            _agent.ApplyAction(action);
        }

        Velocity = _moveDir * Speed;
        MoveAndSlide();

        // Clamp position to arena bounds.
        var pos = Position;
        pos.X = Mathf.Clamp(pos.X, ArenaLeft, ArenaRight);
        pos.Y = Mathf.Clamp(pos.Y, ArenaTop, ArenaBottom);
        Position = pos;
    }

    public void ResetEpisodeState()
    {
        Position = new Vector2(
            ArenaCenter.X + (GD.Randf() - 0.5f) * ArenaWidth  * 0.6f,
            ArenaCenter.Y + (GD.Randf() - 0.5f) * ArenaHeight * 0.6f);

        _moveDir = Vector2.Zero;
        RandomizeGoal();
    }

    private void RandomizeGoal()
    {
        do
        {
            _goalPosition = new Vector2(
                ArenaLeft + GD.Randf() * ArenaWidth,
                ArenaTop  + GD.Randf() * ArenaHeight);
        }
        while (Position.DistanceTo(_goalPosition) < 80f);

        if (_goalMarker is not null)
            _goalMarker.GlobalPosition = _goalPosition;
    }
}
