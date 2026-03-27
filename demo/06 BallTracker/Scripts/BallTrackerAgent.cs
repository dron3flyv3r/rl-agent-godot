using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

/// <summary>
/// RL agent attached as a child of <see cref="BallTrackerPlayer"/>.
/// Observes the arena through a <see cref="RLCameraSensor2D"/> child and
/// outputs a continuous 2-D movement vector.
/// </summary>
public partial class BallTrackerAgent : RLAgent2D
{
    [Export] private RLCameraSensor2D? _camera;

    private BallTrackerPlayer? _player;
    private float              _arenaDiag;

    public override void _Ready()
    {
        base._Ready();
        _camera  ??= GetNodeOrNull<RLCameraSensor2D>("Camera");
        _player    = GetParent<BallTrackerPlayer>();
        var arenaW = _player.ArenaMaxX - _player.ArenaMinX;
        var arenaH = _player.ArenaMaxY - _player.ArenaMinY;
        _arenaDiag = Mathf.Sqrt(arenaW * arenaW + arenaH * arenaH);
    }

    public override void DefineActions(ActionSpaceBuilder builder)
    {
        builder.AddContinuous("move", 2);
    }

    protected override void OnActionsReceived(ActionBuffer actions)
    {
        if (_player is null) return;
        var h = actions.GetContinuous("move", 0);
        var v = actions.GetContinuous("move", 1);
        _player.MoveInput = new Vector2(h, v).LimitLength(1f);
    }

    public override void CollectObservations(ObservationBuffer obs)
    {
        if (_camera is null) return;
        obs.AddImage("camera", _camera);
    }

    public override void OnStep()
    {
        if (_player?.Ball is null) return;
        var dist     = _player.Position.DistanceTo(_player.Ball.Position);
        var normDist = dist / _arenaDiag;
        AddReward(-normDist, "distance");
    }

    public override void OnEpisodeBegin()
    {
        _player ??= GetParent<BallTrackerPlayer>();
        GD.Randomize();
        _player?.ResetPosition();
        _player?.Ball?.Randomize();
    }
}
