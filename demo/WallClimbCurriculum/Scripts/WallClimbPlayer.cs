using System;
using Godot;

namespace RlAgentPlugin.Demo;

public partial class WallClimbPlayer : CharacterBody3D
{
    [Export] public float MoveSpeed { get; set; } = 5f;
    [Export] public float JumpVelocity { get; set; } = 7f;

    private const float Gravity = -20f;
    private Vector3 _moveIntent;
    private bool _jumpIntent;

    public Vector3 PlayerVelocity => Velocity;
    public bool IsGrounded => IsOnFloor();

    public override void _PhysicsProcess(double delta)
    {
        var vel = Velocity;

        // Gravity
        if (!IsOnFloor())
            vel.Y += Gravity * (float)delta;

        // Jump
        if (_jumpIntent && IsOnFloor())
            vel.Y = JumpVelocity;

        // Horizontal movement
        vel.X = _moveIntent.X * MoveSpeed;
        vel.Z = _moveIntent.Z * MoveSpeed;

        Velocity = vel;
        MoveAndSlide();

        // Push any RigidBody3D we're sliding against.
        for (var i = 0; i < GetSlideCollisionCount(); i++)
        {
            var collision = GetSlideCollision(i);
            if (collision.GetCollider() is RigidBody3D rigidBody)
            {
                var pushDir = -collision.GetNormal();
                pushDir.Y = 0f; // horizontal push only
                rigidBody.ApplyImpulse(pushDir * MoveSpeed * (float)delta,
                    collision.GetPosition() - rigidBody.GlobalPosition);
            }
        }

        _jumpIntent = false;
    }

    public void SetMoveIntent(Vector3 direction, bool jump)
    {
        _moveIntent = direction;
        if (jump) _jumpIntent = true;
    }
}
