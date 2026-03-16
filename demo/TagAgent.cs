using System;
using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

public partial class TagAgent : RLAgent2D
{
    private int _moveActionIndex;

    enum MoveAction
    {
        Idle,
        Up,
        Down,
        Left,
        Right
    }

    public Vector2 MoveIntent => _moveActionIndex switch
    {
        1 => Vector2.Up,
        2 => Vector2.Down,
        3 => Vector2.Left,
        4 => Vector2.Right,
        _ => Vector2.Zero,
    };

    private TagPlayer? _player;
    private TagArenaController? _arena;

    public override void _Ready()
    {
        base._Ready();
        _player = GetParent() as TagPlayer;
        _arena = _player?.GetParent() as TagArenaController;
    }

    public override void DefineActions(ActionSpaceBuilder builder)
    {
        builder.AddDiscrete<MoveAction>();
    }

    protected override void OnActionsReceived(ActionBuffer actions)
    {
        _moveActionIndex = (int)actions.GetDiscreteAsEnum<MoveAction>();;
    }

    public override void CollectObservations(ObservationBuffer obs)
    {
        if (_player is not null)
        {
            _arena?.CollectObservations(_player, obs);
        }
    }

    public override void OnStep()
    {
        if (_arena is null || _player is null)
        {
            return;
        }

        AddReward(_arena.ConsumeStepReward(_player));
        if (_arena.IsEpisodeResolvedFor(_player))
        {
            EndEpisode();
        }
    }

    public override void OnEpisodeBegin()
    {
        _player ??= GetParent() as TagPlayer;
        _arena ??= _player?.GetParent() as TagArenaController;
        if (_player is not null)
        {
            _arena?.HandleAgentEpisodeBegin(_player);
        }
    }
}
