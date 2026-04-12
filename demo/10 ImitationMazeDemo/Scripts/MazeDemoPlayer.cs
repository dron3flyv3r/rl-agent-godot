using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

/// <summary>
/// 2D top-down player for the Imitation Maze demo.
///
/// The arena is divided by a horizontal wall with a gap on the right side.
/// Each episode the agent spawns on one side and the goal on the other,
/// forcing the agent to navigate through the gap.
///
/// Wall:    x=100–540, y=300  (gap from x=540–700)
/// </summary>
public partial class MazeDemoPlayer : CharacterBody2D
{
    [Export] public float ArenaLeft   = 100f;
    [Export] public float ArenaRight  = 700f;
    [Export] public float ArenaTop    =  60f;
    [Export] public float ArenaBottom = 540f;
    [Export] public float Speed       = 200f;
    [Export] public float GoalRadius  =  24f;

    /// <summary>Y coordinate of the dividing wall.</summary>
    [Export] public float WallY     = 300f;
    /// <summary>X where the wall ends and the gap begins (gap runs to ArenaRight).</summary>
    [Export] public float GapStartX = 540f;

    private Vector2 _moveDir;
    private Vector2 _goalPosition;
    private Node2D? _goalMarker;
    private IRLAgent? _agent;

    private const int Stay  = 0;
    private const int Up    = 1;
    private const int Down  = 2;
    private const int Left  = 3;
    private const int Right = 4;

    public Vector2 GoalPosition => _goalPosition;
    public bool    IsAtGoal     => Position.DistanceTo(_goalPosition) < GoalRadius;
    public float   ArenaWidth   => ArenaRight  - ArenaLeft;
    public float   ArenaHeight  => ArenaBottom - ArenaTop;
    public Vector2 ArenaCenter  => new((ArenaLeft + ArenaRight) * 0.5f, (ArenaTop + ArenaBottom) * 0.5f);

    public override void _Ready()
    {
        foreach (var child in GetChildren())
            if (child is IRLAgent a) { _agent = a; break; }

        _goalMarker = GetParent()?.FindChild("GoalMarker", true, false) as Node2D;
        ResetEpisodeState();
    }

    public void SetMoveDirection(Vector2 dir) => _moveDir = dir;

    public override void _PhysicsProcess(double delta)
    {
        if (_agent is not null && _agent.ControlMode == RLAgentControlMode.Human)
        {
            int action;
            if (_agent.PendingStepAction >= 0)
            {
                action = _agent.PendingStepAction;
                _agent.PendingStepAction = -1;
            }
            else
            {
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

        // Clamp to arena bounds (wall collision is handled by physics).
        var pos = Position;
        pos.X = Mathf.Clamp(pos.X, ArenaLeft, ArenaRight);
        pos.Y = Mathf.Clamp(pos.Y, ArenaTop, ArenaBottom);
        Position = pos;
    }

    public void ResetEpisodeState()
    {
        const float margin = 30f;

        var topMinY = ArenaTop  + margin;
        var topMaxY = WallY     - margin;
        var botMinY = WallY     + margin;
        var botMaxY = ArenaBottom - margin;

        // Agent and goal always spawn on opposite sides of the wall.
        var agentInTop = GD.Randf() < 0.5f;
        float agentY, goalY;
        if (agentInTop)
        {
            agentY = topMinY + GD.Randf() * (topMaxY - topMinY);
            goalY  = botMinY + GD.Randf() * (botMaxY - botMinY);
        }
        else
        {
            agentY = botMinY + GD.Randf() * (botMaxY - botMinY);
            goalY  = topMinY + GD.Randf() * (topMaxY - topMinY);
        }

        var minX = ArenaLeft  + margin;
        var maxX = ArenaRight - margin;
        Position      = new Vector2(minX + GD.Randf() * (maxX - minX), agentY);
        _goalPosition = new Vector2(minX + GD.Randf() * (maxX - minX), goalY);
        _moveDir = Vector2.Zero;

        if (_goalMarker is not null)
            _goalMarker.GlobalPosition = _goalPosition;
    }
}
