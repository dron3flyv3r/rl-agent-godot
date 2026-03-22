using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

public partial class WallClimbAgent : RLAgent3D
{
    private enum MoveAction { Idle = 0, Forward = 1, Back = 2, Left = 3, Right = 4, ForwardLeft = 5, ForwardRight = 6, BackLeft = 7, BackRight = 8 }
    private enum JumpAction { NoJump = 0, Jump = 1 }

    private WallClimbPlayer? _player;
    private WallClimbArenaController? _arena;
    private RLRaycastSensor3D? _sensor;
    private RigidBody3D? _pushBox;

    // Arena is roughly ±5 in X/Z, 0–4 in Y.
    // Positions use asymmetric Y bounds (never below 0) but symmetric X/Z.
    private const float PosXZ  = 5f;   // half-extent of arena in X and Z
    private const float PosYMax = 5f;   // max expected height
    private const float VelNorm = 6f;   // max expected speed in any axis
    private const float WallHeightNorm = 3.5f;

    public override void _Ready()
    {
        base._Ready();
        _player   = GetParent() as WallClimbPlayer;
        _arena    = _player?.GetParent() as WallClimbArenaController;
        _sensor   = GetNodeOrNull<RLRaycastSensor3D>("RLRaycastSensor3D");
        _pushBox  = _arena?.GetNodeOrNull<RigidBody3D>("PushBox");
    }

    public override void DefineActions(ActionSpaceBuilder builder)
    {
        builder.AddDiscrete<MoveAction>();
        builder.AddDiscrete<JumpAction>();
    }

    protected override void OnActionsReceived(ActionBuffer actions)
    {
        if (_player is null) return;

        var move = actions.GetDiscreteAsEnum<MoveAction>();
        var jump = actions.GetDiscreteAsEnum<JumpAction>();

        var direction = move switch
        {
            MoveAction.Forward => new Vector3(1f, 0f, 0f),
            MoveAction.Back    => new Vector3(-1f, 0f, 0f),
            MoveAction.Left    => new Vector3(0f, 0f, -1f),
            MoveAction.Right   => new Vector3(0f, 0f, 1f),
            MoveAction.ForwardLeft  => new Vector3(1f, 0f, -1f).Normalized(),
            MoveAction.ForwardRight => new Vector3(1f, 0f, 1f).Normalized(),
            MoveAction.BackLeft     => new Vector3(-1f, 0f, -1f).Normalized(),
            MoveAction.BackRight    => new Vector3(-1f, 0f, 1f).Normalized(),
            _                  => Vector3.Zero,
        };

        _player.SetMoveIntent(direction, jump == JumpAction.Jump);
    }

    public override void CollectObservations(ObservationBuffer obs)
    {
        _sensor  ??= GetNodeOrNull<RLRaycastSensor3D>("RLRaycastSensor3D");
        _pushBox ??= _arena?.GetNodeOrNull<RigidBody3D>("PushBox");
        if (_player is null || _arena is null) return;

        var playerPos = _player.GlobalPosition;
        var playerVel = _player.PlayerVelocity;
        var boxPos    = _pushBox?.GlobalPosition ?? Vector3.Zero;
        var boxVel    = _pushBox?.LinearVelocity ?? Vector3.Zero;
        var goalPos   = _arena.GoalWorldPosition;

        // Symmetric bounds: arena is ±PosXZ in X/Z, 0–PosYMax in Y.
        var posMin = new Vector3(-PosXZ, 0f,    -PosXZ);
        var posMax = new Vector3( PosXZ, PosYMax, PosXZ);
        // Relative vectors can span twice the arena in any axis.
        var relMin = new Vector3(-PosXZ * 2f, -PosYMax, -PosXZ * 2f);
        var relMax = new Vector3( PosXZ * 2f,  PosYMax,  PosXZ * 2f);
        // Velocity is signed.
        var velMin = new Vector3(-VelNorm, -VelNorm, -VelNorm);
        var velMax = new Vector3( VelNorm,  VelNorm,  VelNorm);

        // [0-2] Player position
        obs.AddNormalized(playerPos, posMin, posMax);

        // [3] Is grounded
        obs.Add(_player.IsGrounded ? 1f : 0f);

        // [4-6] Box position
        obs.AddNormalized(boxPos, posMin, posMax);

        // [7-9] Box velocity (signed)
        obs.AddNormalized(boxVel, velMin, velMax);

        // [10-12] Player velocity (signed)
        obs.AddNormalized(playerVel, velMin, velMax);

        // [13-15] Vector player → box (signed relative)
        obs.AddNormalized(boxPos - playerPos, relMin, relMax);

        // [16-18] Vector player → goal (signed relative)
        obs.AddNormalized(goalPos - playerPos, relMin, relMax);

        // [19] Normalised wall height — tells the agent how tall the obstacle is
        obs.AddNormalized(_arena.CurrentWallHeight, 0f, WallHeightNorm);

        // raycast sensor — per ray: [dist, hit_flag] when IncludeHitClass, else [dist]
        // if (_sensor is not null)
        //     obs.AddSensor("rays", _sensor);
    }

    public override void OnStep()
    {
        if (_arena is null) return;

        // Step penalty
        AddReward(-0.001f, "step_penalty");

        // Consume shaping rewards from arena
        var (_, breakdown) = _arena.ConsumeStepRewards();
        foreach (var (tag, amount) in breakdown)
            AddReward(amount, tag);

        if (_arena.IsGoalReached || _arena.IsOutOfBounds)
            EndEpisode();
    }

    protected override void OnHumanInput()
    {
        if (_player is null) return;

        var input = Vector3.Zero;
        if (Input.IsKeyPressed(Key.W) || Input.IsKeyPressed(Key.Up))    input.X += 1f;
        if (Input.IsKeyPressed(Key.S) || Input.IsKeyPressed(Key.Down))  input.X -= 1f;
        if (Input.IsKeyPressed(Key.A) || Input.IsKeyPressed(Key.Left))  input.Z -= 1f;
        if (Input.IsKeyPressed(Key.D) || Input.IsKeyPressed(Key.Right)) input.Z += 1f;

        var jump = Input.IsKeyPressed(Key.Space);
        _player.SetMoveIntent(input.Normalized(), jump);
    }

    public override void OnEpisodeBegin()
    {
        _player  ??= GetParent() as WallClimbPlayer;
        _arena   ??= _player?.GetParent() as WallClimbArenaController;
        _sensor  ??= GetNodeOrNull<RLRaycastSensor3D>("RLRaycastSensor3D");
        _pushBox ??= _arena?.GetNodeOrNull<RigidBody3D>("PushBox");
        _arena?.HandleAgentEpisodeBegin();
    }

    public override void OnTrainingProgress(float progress)
    {
        _arena?.ApplyCurriculumProgress(progress);
    }
}
