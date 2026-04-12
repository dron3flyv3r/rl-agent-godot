using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

/// <summary>
/// Agent for the Imitation Maze demo. Demonstrates that BC can learn a
/// non-trivial policy that goes AROUND a wall rather than straight to the goal.
///
/// The arena is split by a horizontal wall with a gap on the right side.
/// Agent and goal always spawn on opposite sides, so the agent must find the gap.
///
/// Observations (6):
///   [0] agent_x             normalized to [-1, 1] over arena width
///   [1] agent_y             normalized to [-1, 1] over arena height
///   [2] goal_x              normalized to [-1, 1] over arena width
///   [3] goal_y              normalized to [-1, 1] over arena height
///   [4] gap_start_x - agent_x  normalized: positive = agent is left of gap (must go right)
///   [5] agent_y - wall_y    normalized: negative = above wall, positive = below
///
/// Actions (5 discrete): Stay / Up / Down / Left / Right.
///
/// Scripted policy: right until past the gap, then cross, then toward goal.
/// </summary>
public partial class MazeDemoAgent : RLAgent2D
{
    private enum MoveAction { Stay = 0, Up = 1, Down = 2, Left = 3, Right = 4 }

    private MazeDemoPlayer? _player;
    private Vector2 _prevPos;

    public override void _Ready()
    {
        base._Ready();
        _player = GetParent() as MazeDemoPlayer;
    }

    public override void DefineActions(ActionSpaceBuilder builder)
        => builder.AddDiscrete<MoveAction>();

    // ── RL interface ──────────────────────────────────────────────────────────

    protected override void OnActionsReceived(ActionBuffer actions)
    {
        if (_player is null) return;
        var dir = actions.GetDiscreteAsEnum<MoveAction>() switch
        {
            MoveAction.Up => Vector2.Up,
            MoveAction.Down => Vector2.Down,
            MoveAction.Left => Vector2.Left,
            MoveAction.Right => Vector2.Right,
            _ => Vector2.Zero,
        };
        _player.SetMoveDirection(dir);
    }

    public override void CollectObservations(ObservationBuffer obs)
    {
        if (_player is null) return;

        var halfW = _player.ArenaWidth * 0.5f;
        var halfH = _player.ArenaHeight * 0.5f;
        var center = _player.ArenaCenter;
        var pos = _player.Position;
        var goal = _player.GoalPosition;

        obs.AddNormalized(pos.X - center.X, -halfW, halfW);
        obs.AddNormalized(pos.Y - center.Y, -halfH, halfH);
        obs.AddNormalized(goal.X - center.X, -halfW, halfW);
        obs.AddNormalized(goal.Y - center.Y, -halfH, halfH);

        // How far left of the gap: positive means must move right to find it.
        obs.AddNormalized(_player.GapStartX - pos.X, -_player.ArenaWidth, _player.ArenaWidth);
        // Signed distance to wall: negative = above wall, positive = below.
        obs.AddNormalized(pos.Y - _player.WallY, -halfH, halfH);
    }

    public override void OnStep()
    {
        if (_player is null) return;

        var distNow = _player.Position.DistanceTo(_player.GoalPosition);
        var distPrev = _prevPos.DistanceTo(_player.GoalPosition);
        var arenaDiag = Mathf.Sqrt(
            _player.ArenaWidth * _player.ArenaWidth +
            _player.ArenaHeight * _player.ArenaHeight);

        AddReward((distPrev - distNow) / arenaDiag, "progress");
        AddReward(-0.001f, "step_penalty");

        if (_player.IsAtGoal)
        {
            AddReward(1.0f, "goal_reached");
            EndEpisode();
        }

        _prevPos = _player.Position;
    }

    public override void OnEpisodeBegin()
    {
        _player ??= GetParent() as MazeDemoPlayer;
        _player?.ResetEpisodeState();
        _prevPos = _player?.Position ?? Vector2.Zero;
    }

    // ── Scripted heuristic ────────────────────────────────────────────────────

    /// <summary>
    /// Phase 1: move right until past the gap.
    /// Phase 2: cross the wall vertically.
    /// Phase 3 (same side): go straight to goal.
    /// </summary>
    protected override void OnScriptedInput()
    {
        if (_player is null) return;

        var pos = _player.Position;
        var goal = _player.GoalPosition;
        var wallY = _player.WallY;
        var gapStartX = _player.GapStartX;
        var gapCenterX = gapStartX + (_player.ArenaRight - gapStartX) * 0.5f;
        const float gapAlignTolerance = 12f;

        var goalAboveWall = goal.Y < wallY;
        const float wallClearance = 20f;
        var crossTargetY = goalAboveWall ? wallY - wallClearance : wallY + wallClearance;
        var clearedWall = goalAboveWall ? pos.Y <= crossTargetY : pos.Y >= crossTargetY;

        MoveAction action;

        if (clearedWall)
        {
            // Same side — move directly toward the goal.
            var diff = goal - pos;
            action = Mathf.Abs(diff.X) >= Mathf.Abs(diff.Y)
                ? (diff.X > 0 ? MoveAction.Right : MoveAction.Left)
                : (diff.Y > 0 ? MoveAction.Down : MoveAction.Up);
        }
        else
        {
            // Opposite sides — first align with the center of the opening, then cross.
            var gapOffsetX = gapCenterX - pos.X;
            if (Mathf.Abs(gapOffsetX) > gapAlignTolerance)
            {
                action = gapOffsetX > 0f ? MoveAction.Right : MoveAction.Left;
            }
            else
            {
                action = goalAboveWall ? MoveAction.Up : MoveAction.Down;
            }
        }

        ApplyAction((int)action);
    }

    protected override void OnHumanInput()
    {
        base.OnHumanInput();
    }
}
