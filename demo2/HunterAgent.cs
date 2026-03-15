using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo2;

/// <summary>
/// One of two cooperative hunters in the Cooperative Tag demo.
/// Both hunters share PolicyGroup = "Hunters", so they share a single policy brain.
///
/// Observation (6 values):
///   [own_x, own_y, other_hunter_x, other_hunter_y, prey_x, prey_y]
///   All normalized to [-1, 1] within the arena bounds.
///
/// Actions (5):
///   0 = Stay, 1 = Up, 2 = Down, 3 = Left, 4 = Right
///
/// Episode ends when BOTH hunters are within CatchRadius of the prey simultaneously.
/// </summary>
public partial class HunterAgent : RLAgent2D
{
    [DiscreteAction(5, "Stay", "Up", "Down", "Left", "Right", Name = "Move")]
    private int _moveAction;

    internal const float ArenaMinX = 64f;
    internal const float ArenaMaxX = 576f;
    internal const float ArenaMinY = 40f;
    internal const float ArenaMaxY = 320f;

    private CoopTagGame? _game;

    /// <summary>Velocity direction vector based on current action (not yet scaled by speed).</summary>
    public Vector2 MoveVelocity => _moveAction switch
    {
        1 => Vector2.Up,
        2 => Vector2.Down,
        3 => Vector2.Left,
        4 => Vector2.Right,
        _ => Vector2.Zero,
    };

    public override void _Ready()
    {
        base._Ready();
        _game = GetParent() as CoopTagGame;
    }

    public override void CollectObservations(ObservationBuffer obs)
    {
        if (_game is null) return;

        var other = _game.GetOtherHunterPosition(this);
        var prey = _game.PreyPosition;

        obs.AddNormalized(Position.X, ArenaMinX, ArenaMaxX);
        obs.AddNormalized(Position.Y, ArenaMinY, ArenaMaxY);
        obs.AddNormalized(other.X, ArenaMinX, ArenaMaxX);
        obs.AddNormalized(other.Y, ArenaMinY, ArenaMaxY);
        obs.AddNormalized(prey.X, ArenaMinX, ArenaMaxX);
        obs.AddNormalized(prey.Y, ArenaMinY, ArenaMaxY);
    }

    public override void OnStep()
    {
        if (_game is null) return;
        AddReward(_game.ConsumeReward(this));
        if (_game.IsEpisodeDone) EndEpisode();
    }

    public override void OnEpisodeBegin()
    {
        _game?.ResetEpisode();
    }
}
