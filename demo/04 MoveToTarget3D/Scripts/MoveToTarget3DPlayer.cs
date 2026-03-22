using Godot;

namespace RlAgentPlugin.Demo;

/// <summary>
/// Minimal CharacterBody3D controller for the MoveToTarget3D debug environment.
/// Receives a move intent vector each physics frame and applies movement + gravity.
/// </summary>
public partial class MoveToTarget3DPlayer : CharacterBody3D
{
    [Export] public float MoveSpeed { get; set; } = 4.0f;
    [Export] public float Gravity   { get; set; } = 9.8f;

    private Vector3 _moveIntent = Vector3.Zero;

    /// <summary>Horizontal velocity (XZ plane, no Y component).</summary>
    public Vector3 HorizontalVelocity => new(Velocity.X, 0f, Velocity.Z);

    /// <summary>Called by the agent each physics frame before MoveAndSlide.</summary>
    public void SetMoveIntent(Vector3 direction)
    {
        // Only use XZ; Y is controlled by gravity
        _moveIntent = new Vector3(direction.X, 0f, direction.Z);
    }

    public override void _PhysicsProcess(double delta)
    {
        var vel = Velocity;

        // Gravity
        if (!IsOnFloor())
            vel.Y -= Gravity * (float)delta;
        else
            vel.Y = 0f;

        // Horizontal movement driven by agent intent
        vel.X = _moveIntent.X * MoveSpeed;
        vel.Z = _moveIntent.Z * MoveSpeed;

        Velocity = vel;
        MoveAndSlide();
    }
}
