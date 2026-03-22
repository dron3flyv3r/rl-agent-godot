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

    private const float PosNorm = 10f;
    private const float VelNorm = 10f;
    private const float WallHeightNorm = 3.5f;

    public override void _Ready()
    {
        base._Ready();
        _player = GetParent() as WallClimbPlayer;
        _arena = _player?.GetParent() as WallClimbArenaController;
        _sensor = GetNodeOrNull<RLRaycastSensor3D>("RLRaycastSensor3D");
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
        _sensor ??= GetNodeOrNull<RLRaycastSensor3D>("RLRaycastSensor3D");
        if (_player is null || _arena is null) return;

        var playerPos = _player.GlobalPosition;
        var playerVel = _player.PlayerVelocity;

        var pushBox = _arena.GetNodeOrNull<RigidBody3D>("PushBox");
        var boxPos = pushBox?.GlobalPosition ?? Vector3.Zero;
        var boxVel = pushBox?.LinearVelocity ?? Vector3.Zero;
        var goalPos = _arena.GoalWorldPosition;
        var posNorm = new Vector3(PosNorm, PosNorm, PosNorm);

        // player position (normalized)
        obs.AddNormalized(playerPos, Vector3.Zero, posNorm);


        // is_on_floor
        obs.Add(_player.IsGrounded ? 1f : 0f);

        // box position (normalized)
        obs.AddNormalized(boxPos, Vector3.Zero, posNorm);

        // box linear velocity (normalized)
        obs.AddNormalized(boxVel, Vector3.Zero, new Vector3(VelNorm, VelNorm, VelNorm));

        // player velocity (normalized)
        obs.AddNormalized(playerVel, Vector3.Zero, new Vector3(VelNorm, VelNorm, VelNorm));

        // vector player → box
        var playerToBox = boxPos - playerPos;
        obs.AddNormalized(playerToBox, Vector3.Zero, posNorm);

        // vector player → goal
        var playerToGoal = goalPos - playerPos;
        obs.AddNormalized(playerToGoal, Vector3.Zero, posNorm);

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
        _player ??= GetParent() as WallClimbPlayer;
        _arena ??= _player?.GetParent() as WallClimbArenaController;
        _sensor ??= GetNodeOrNull<RLRaycastSensor3D>("RLRaycastSensor3D");
        _arena?.HandleAgentEpisodeBegin();
    }

    public override void OnTrainingProgress(float progress)
    {
        _arena?.ApplyCurriculumProgress(progress);
    }
}
