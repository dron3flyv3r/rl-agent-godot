using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

public partial class ReachTargetAgent : RLAgent2D
{

    [DiscreteAction(2, "Move Left", "Move Right", Name = "Move")]
    private int _movementAction;

    private ReachTargetPlayer? _player;

    public float MoveDirection => _movementAction == 0 ? -1.0f : 1.0f;

    public override void _Ready()
    {
        base._Ready();
        _player = GetParent() as ReachTargetPlayer;
    }

    public override void CollectObservations(ObservationBuffer obs)
    {
        if (_player is null) return;
        obs.AddNormalized(_player.Position.X, 60f, 580f);
        obs.AddNormalized(_player.GoalX, 60f, 580f);
        obs.AddNormalized(_player.GoalX - _player.Position.X, -520f, 520f);
    }

    public override void OnStep()
    {
        if (_player is null) return;
        AddReward(_player.ConsumeStepReward());
        if (_player.IsAtGoal) EndEpisode();
    }

    public override void OnEpisodeBegin()
    {
        _player?.ResetEpisodeState();
    }
}
