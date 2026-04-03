using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

public partial class WallClimbArenaController : Node3D, IRLCurriculumConsumer
{
    [Export] public float WallHeightMin { get; set; } = 0f;
    [Export] public float WallHeightMax { get; set; } = 3.0f;

    // Spawn positions
    private static readonly Vector3 PlayerSpawn = new(-3f, 0.5f, 0f);
    private static readonly Vector3 BoxSpawn = new(-1.5f, 0.5f, 0f);
    private static readonly Vector3 GoalPosition = new(3f, 0.1f, 3f);

    private StaticBody3D? _wall;
    private CollisionShape3D? _wallCollision;
    private MeshInstance3D? _wallMesh;
    private StaticBody3D? _platform;
    private CollisionShape3D? _platformCollision;
    private MeshInstance3D? _platformMesh;
    private WallClimbPlayer? _player;
    private RigidBody3D? _pushBox;
    private Area3D? _goal;

    private float _accumulatedReward;
    private readonly System.Collections.Generic.Dictionary<string, float> _rewardBreakdown = new();
    private Vector3 _wallBasePosition;
    private float _wallBaseBottomY;

    // Shaping state
    private float _prevPlayerToBoxDist = float.MaxValue;
    private float _prevBoxToWallDist = float.MaxValue;
    private float _prevPlayerToGoalDist = float.MaxValue;
    private bool _onBoxBonusGiven;
    private bool _clearedWallBonusGiven;

    public float CurriculumProgress { get; private set; }
    public float CurrentWallHeight { get; private set; }
    public Vector3 GoalWorldPosition => _goal?.GlobalPosition ?? ToGlobal(GoalPosition);
    public bool IsGoalReached { get; private set; }
    public bool IsOutOfBounds { get; private set; }

    private const float KillPlaneY = -1.0f;
    private const float GoalReachRadius = 0.9f;
    private const float ProgressRewardScale = 0.02f;
    private const float GoalProgressRewardScale = 0.05f;

    public override void _Ready()
    {
        _wall = GetNodeOrNull<StaticBody3D>("Wall");
        _wallCollision = _wall?.GetNodeOrNull<CollisionShape3D>("Collision");
        _wallMesh = _wall?.GetNodeOrNull<MeshInstance3D>("Mesh");
        _platform = GetNodeOrNull<StaticBody3D>("Platform");
        _platformCollision = _platform?.GetNodeOrNull<CollisionShape3D>("Collision");
        _platformMesh = _platform?.GetNodeOrNull<MeshInstance3D>("Mesh");
        _player = GetNodeOrNull<WallClimbPlayer>("WallClimbPlayer");
        _pushBox = GetNodeOrNull<RigidBody3D>("PushBox");
        _goal = GetNodeOrNull<Area3D>("Goal");
        CaptureWallAnchor();
    }

    public void NotifyCurriculumProgress(float p)
    {
        CurriculumProgress = Mathf.Clamp(p, 0f, 1f);
        SetWallHeight(Mathf.Lerp(WallHeightMin, WallHeightMax, CurriculumProgress));
    }

    public void HandleAgentEpisodeBegin()
    {
        // Reset player
        if (_player is not null)
        {
            _player.Position = PlayerSpawn;
            _player.Velocity = Vector3.Zero;
        }

        // Reset box
        if (_pushBox is not null)
        {
            _pushBox.Position = BoxSpawn;
            _pushBox.LinearVelocity = Vector3.Zero;
            _pushBox.AngularVelocity = Vector3.Zero;
        }

        IsGoalReached = false;
        IsOutOfBounds = false;
        _accumulatedReward = 0f;
        _rewardBreakdown.Clear();
        _prevPlayerToBoxDist = float.MaxValue;
        _prevBoxToWallDist = float.MaxValue;
        _prevPlayerToGoalDist = float.MaxValue;
        _onBoxBonusGiven = false;
        _clearedWallBonusGiven = false;
    }

    public override void _PhysicsProcess(double delta)
    {
        if (_player is null || _pushBox is null || _wall is null) return;

        if (!IsOutOfBounds && _player.GlobalPosition.Y < KillPlaneY)
        {
            IsOutOfBounds = true;
            AccumulateReward(-2.0f, "out_of_bounds");
        }

        var playerPos = _player.GlobalPosition;
        var boxPos = _pushBox.GlobalPosition;
        var wallX = _wall.GlobalPosition.X;
        var goalPos = GoalWorldPosition;

        // wallFactor scales sub-task rewards (box/player shaping) so they are silent at
        // curriculum=0 (no wall) and grow proportionally as the wall rises.
        // At curriculum=0 the only relevant gradient is goal_progress; at full height all
        // sub-tasks are fully active.
        var wallFactor = WallHeightMax > 0f ? CurrentWallHeight / WallHeightMax : 0f;

        // Player-to-box proximity shaping (only useful when there is a wall to climb)
        var playerToBoxDist = playerPos.DistanceTo(boxPos);
        if (wallFactor > 0f && _prevPlayerToBoxDist < float.MaxValue)
        {
            var delta2 = _prevPlayerToBoxDist - playerToBoxDist;
            if (delta2 != 0f)
                AccumulateReward(delta2 * ProgressRewardScale * wallFactor, "player_to_box");
        }
        _prevPlayerToBoxDist = playerToBoxDist;

        // Box-toward-wall shaping (only useful when there is a wall to climb)
        var boxToWallDist = Mathf.Abs(boxPos.X - wallX);
        if (wallFactor > 0f && _prevBoxToWallDist < float.MaxValue)
        {
            var delta3 = _prevBoxToWallDist - boxToWallDist;
            if (delta3 != 0f)
                AccumulateReward(delta3 * ProgressRewardScale * wallFactor, "box_to_wall");
        }
        _prevBoxToWallDist = boxToWallDist;

        // One-shot: player on top of box
        if (!_onBoxBonusGiven && playerPos.Y > boxPos.Y + 0.3f && playerToBoxDist < 1.2f)
        {
            _onBoxBonusGiven = true;
            AccumulateReward(0.1f * wallFactor, "on_box");
        }

        // One-shot: player cleared the wall (X > wallX + 0.5)
        if (!_clearedWallBonusGiven && playerPos.X > wallX + 0.5f)
        {
            _clearedWallBonusGiven = true;
            AccumulateReward(0.5f, "cleared_wall");
        }

        // Shape toward goal using XZ-only distance so that falling (Y decrease) does not
        // generate large negative goal_progress and inadvertently reinforce dying.
        // GoalProgressRewardScale is larger than the sub-task scale so the goal is always dominant.
        {
            var playerToGoalDist = new Vector2(playerPos.X - goalPos.X, playerPos.Z - goalPos.Z).Length();
            if (_prevPlayerToGoalDist < float.MaxValue)
            {
                var goalDelta = _prevPlayerToGoalDist - playerToGoalDist;
                if (goalDelta != 0f)
                    AccumulateReward(goalDelta * GoalProgressRewardScale, "goal_progress");
            }
            _prevPlayerToGoalDist = playerToGoalDist;
        }

        // Goal detection via distance check avoids C# signal marshalling edge cases.
        if (!IsGoalReached)
        {
            if (playerPos.DistanceTo(goalPos) <= GoalReachRadius)
            {
                IsGoalReached = true;
                AccumulateReward(2.0f, "goal_reached");
            }
        }
    }

    public (float reward, System.Collections.Generic.Dictionary<string, float> breakdown) ConsumeStepRewards()
    {
        var reward = _accumulatedReward;
        var breakdown = new System.Collections.Generic.Dictionary<string, float>(_rewardBreakdown);
        _accumulatedReward = 0f;
        _rewardBreakdown.Clear();
        return (reward, breakdown);
    }

    private void SetWallHeight(float h)
    {
        if (_wall is null) return;

        CurrentWallHeight = Mathf.Max(0.01f, h);
        _wall.Position = new Vector3(
            _wallBasePosition.X,
            _wallBaseBottomY + (CurrentWallHeight * 0.5f),
            _wallBasePosition.Z);

        if (_wallCollision?.Shape is BoxShape3D boxShape)
        {
            var size = boxShape.Size;
            boxShape.Size = new Vector3(size.X, CurrentWallHeight, size.Z);
            _wallCollision.Position = Vector3.Zero;
        }

        if (_wallMesh?.Mesh is BoxMesh boxMesh)
        {
            boxMesh.Size = new Vector3(boxMesh.Size.X, CurrentWallHeight, boxMesh.Size.Z);
            _wallMesh.Position = Vector3.Zero;
        }
    }

    private void CaptureWallAnchor()
    {
        if (_wall is null) return;

        _wallBasePosition = _wall.Position;
        _wallBaseBottomY = TryGetPlatformTopY(out var platformTopY)
            ? platformTopY
            : _wall.Position.Y - (GetWallReferenceHeight() * 0.5f);
    }

    private bool TryGetPlatformTopY(out float platformTopY)
    {
        platformTopY = 0f;
        if (_platform is null)
            return false;

        var platformHeight = 0f;
        if (_platformCollision?.Shape is BoxShape3D collisionShape)
            platformHeight = collisionShape.Size.Y;
        else if (_platformMesh?.Mesh is BoxMesh mesh)
            platformHeight = mesh.Size.Y;

        if (platformHeight <= 0.01f)
            return false;

        platformTopY = _platform.Position.Y + (platformHeight * 0.5f);
        return true;
    }

    private float GetWallReferenceHeight()
    {
        if (_wallCollision?.Shape is BoxShape3D collisionShape)
            return Mathf.Max(0.01f, collisionShape.Size.Y);

        if (_wallMesh?.Mesh is BoxMesh mesh)
            return Mathf.Max(0.01f, mesh.Size.Y);

        return 0.01f;
    }

    private void AccumulateReward(float amount, string tag)
    {
        _accumulatedReward += amount;
        _rewardBreakdown.TryGetValue(tag, out var current);
        _rewardBreakdown[tag] = current + amount;
    }
}
