using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

public partial class ReachTargetAgent : RLAgent2D
{

    private int _moveActionIndex;
    private ReachTargetPlayer? _player;

    public float MoveDirection => _moveActionIndex == 0 ? -1.0f : 1.0f;

    public override void _Ready()
    {
        base._Ready();
        _player = GetParent() as ReachTargetPlayer;
    }

    public override void DefineActions(ActionSpaceBuilder builder)
    {
        builder.AddDiscrete("Move", "Move Left", "Move Right");
    }

    protected override void OnActionsReceived(ActionBuffer actions)
    {
        _moveActionIndex = actions.GetDiscrete("Move");
    }

    public override void CollectObservations(ObservationBuffer obs)
    {
        if (_player is null) return;
        var laneMin = Mathf.Min(_player.LaneMinX, _player.LaneMaxX);
        var laneMax = Mathf.Max(_player.LaneMinX, _player.LaneMaxX);
        var laneWidth = Mathf.Max(1.0f, laneMax - laneMin);

        obs.AddNormalized(_player.Position.X, laneMin, laneMax);
        obs.AddNormalized(_player.GoalX, laneMin, laneMax);
        obs.AddNormalized(_player.GoalX - _player.Position.X, -laneWidth, laneWidth);
    }

    public override void OnStep()
    {
        if (_player is null) return;
        AddReward(_player.ConsumeStepReward());
        if (_player.IsAtGoal) EndEpisode();
    }

    public override void OnEpisodeBegin()
    {
        _player ??= GetParent() as ReachTargetPlayer;
        _player?.ResetEpisodeState();
    }
}
