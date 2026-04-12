using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

/// <summary>
/// Simple 2D top-down navigation agent for the Imitation Learning demo.
///
/// Demonstrates the full Record → BC Train → Infer workflow:
///   1. Open ImitationDemo scene → RL Imitation dock → Record tab → Start Recording.
///   2. Drive the agent with arrow keys to demonstrate reaching the goal.
///   3. Stop recording, switch to Train tab, pick the .rldem file and hit Train.
///   4. Point InferenceModelPath at the resulting .rlcheckpoint → Run Inference.
///
/// Observations (4): agent_x, agent_y, goal_x, goal_y — all normalized to [-1, 1].
/// Actions (5 discrete): Stay / Up / Down / Left / Right.
/// Human controls: Arrow keys (ui_up / ui_down / ui_left / ui_right).
/// </summary>
public partial class ImitationAgent : RLAgent2D
{
    private enum MoveAction
    {
        Stay  = 0,
        Up    = 1,
        Down  = 2,
        Left  = 3,
        Right = 4,
    }

    private ImitationPlayer? _player;
    private Vector2 _prevPos;

    public override void _Ready()
    {
        base._Ready();
        _player = GetParent() as ImitationPlayer;
    }

    public override void DefineActions(ActionSpaceBuilder builder)
        => builder.AddDiscrete<MoveAction>();

    // ── RL interface ──────────────────────────────────────────────────────────

    protected override void OnActionsReceived(ActionBuffer actions)
    {
        if (_player is null) return;
        var dir = actions.GetDiscreteAsEnum<MoveAction>() switch
        {
            MoveAction.Up    => Vector2.Up,
            MoveAction.Down  => Vector2.Down,
            MoveAction.Left  => Vector2.Left,
            MoveAction.Right => Vector2.Right,
            _                => Vector2.Zero,
        };
        _player.SetMoveDirection(dir);
    }

    public override void CollectObservations(ObservationBuffer obs)
    {
        if (_player is null) return;

        var halfW  = _player.ArenaWidth  * 0.5f;
        var halfH  = _player.ArenaHeight * 0.5f;
        var center = _player.ArenaCenter;

        obs.AddNormalized(_player.Position.X    - center.X, -halfW, halfW);
        obs.AddNormalized(_player.Position.Y    - center.Y, -halfH, halfH);
        obs.AddNormalized(_player.GoalPosition.X - center.X, -halfW, halfW);
        obs.AddNormalized(_player.GoalPosition.Y - center.Y, -halfH, halfH);
    }

    public override void OnStep()
    {
        if (_player is null) return;

        var distNow  = _player.Position.DistanceTo(_player.GoalPosition);
        var distPrev = _prevPos.DistanceTo(_player.GoalPosition);
        var arenaDiag = Mathf.Sqrt(
            _player.ArenaWidth  * _player.ArenaWidth +
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
        _player ??= GetParent() as ImitationPlayer;
        _player?.ResetEpisodeState();
        _prevPos = _player?.Position ?? Vector2.Zero;
    }

    /// <summary>
    /// Scripted heuristic policy: move in the axis with the greatest distance to the goal.
    /// Called by RecordingBootstrap in Script mode to auto-generate demonstrations.
    /// </summary>
    protected override void OnScriptedInput()
    {
        if (_player is null) return;

        var diff = _player.GoalPosition - _player.Position;
        MoveAction action;

        if (Mathf.Abs(diff.X) >= Mathf.Abs(diff.Y))
            action = diff.X > 0 ? MoveAction.Right : MoveAction.Left;
        else
            action = diff.Y > 0 ? MoveAction.Down : MoveAction.Up;

        ApplyAction((int)action);
    }
}
