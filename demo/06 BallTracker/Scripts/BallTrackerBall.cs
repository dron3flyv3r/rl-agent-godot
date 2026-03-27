using Godot;

namespace RlAgentPlugin.Demo;

/// <summary>
/// A ball that bounces around the arena with a random walk velocity.
/// The tracker agent must follow it using only camera observations.
/// </summary>
public partial class BallTrackerBall : CharacterBody2D
{
    [Export] public float ArenaMinX { get; set; } = 40f;
    [Export] public float ArenaMaxX { get; set; } = 760f;
    [Export] public float ArenaMinY { get; set; } = 40f;
    [Export] public float ArenaMaxY { get; set; } = 560f;
    [Export] public float Speed     { get; set; } = 200f;

    private readonly RandomNumberGenerator _rng = new();

    public override void _Ready() => Randomize();

    public override void _PhysicsProcess(double delta)
    {
        var motion = Velocity * (float)delta;
        var col    = MoveAndCollide(motion);
        if (col is not null)
        {
            Velocity = Velocity.Bounce(col.GetNormal());
        }

        // Clamp inside arena in case of tunnelling.
        var pos = Position;
        if (pos.X < ArenaMinX) { pos.X = ArenaMinX; Velocity = new Vector2(Mathf.Abs(Velocity.X),  Velocity.Y); }
        if (pos.X > ArenaMaxX) { pos.X = ArenaMaxX; Velocity = new Vector2(-Mathf.Abs(Velocity.X), Velocity.Y); }
        if (pos.Y < ArenaMinY) { pos.Y = ArenaMinY; Velocity = new Vector2(Velocity.X,  Mathf.Abs(Velocity.Y)); }
        if (pos.Y > ArenaMaxY) { pos.Y = ArenaMaxY; Velocity = new Vector2(Velocity.X, -Mathf.Abs(Velocity.Y)); }
        Position = pos;
    }

    /// <summary>Teleport to a random position and give the ball a new random velocity.</summary>
    public void Randomize()
    {
        _rng.Randomize();
        Position = new Vector2(
            _rng.RandfRange(ArenaMinX + 60f, ArenaMaxX - 60f),
            _rng.RandfRange(ArenaMinY + 60f, ArenaMaxY - 60f));
        var angle = _rng.RandfRange(0f, Mathf.Tau);
        Velocity  = new Vector2(Mathf.Cos(angle), Mathf.Sin(angle)) * Speed;
    }
}
