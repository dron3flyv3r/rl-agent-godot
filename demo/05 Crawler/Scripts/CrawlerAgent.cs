using System;
using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

/// <summary>
/// SAC agent for the four-legged crawler locomotion demo.
///
/// OBSERVATION LAYOUT (26 floats):
///   [0-2]   Torso linear velocity         — normalized to [-1, 1] over ±5 m/s
///   [3-5]   Torso angular velocity        — normalized to [-1, 1] over ±5 rad/s
///   [6-8]   Torso up-vector (Basis.Y)     — raw unit vector, already in [-1, 1]
///   [9]     Torso height                  — normalized to [-1, 1] over 0–1.5 m
///   [10-17] Joint angles (8)              — normalized to [-1, 1] over ±π/2 rad
///   [18-25] Joint angular velocities (8)  — normalized to [-1, 1] over ±5 rad/s
///
/// ACTION LAYOUT (8 continuous, each in [-1, 1]):
///   [0] FL hip · [1] FL knee · [2] FR hip · [3] FR knee
///   [4] BL hip · [5] BL knee · [6] BR hip · [7] BR knee
///   Values are scaled to ±CrawlerBody.MaxJointVelocity rad/s by CrawlerBody.
///
/// REWARD BREAKDOWN:
///   forward_vel — torso velocity in +X (walk direction); main training signal
///   upright     — bonus when Basis.Y·Up > 0.5; discourages tipping
///   energy      — penalty proportional to mean squared joint command
///   alive       — small constant per step to discourage instant collapse
///
/// TERMINATION:
///   Torso Y drops below CollapseHeight (creature fell or tipped irrecoverably).
/// </summary>
public partial class CrawlerAgent : RLAgent3D
{
    [Export] public float CollapseHeight { get; set; } = 0.15f;

    private CrawlerBody? _body;
    private float[] _lastActions = new float[8];

    private static readonly Vector3 VelBound = Vector3.One * 5f;

    // ── RLAgent3D overrides ───────────────────────────────────────────────

    public override void _Ready()
    {
        base._Ready();
        _body = GetParent<CrawlerBody>();
        if (_body is null)
            GD.PushError("[CrawlerAgent] Parent must be a CrawlerBody node.");
    }

    public override void DefineActions(ActionSpaceBuilder builder)
    {
        // 8 joint motors: FL hip, FL knee, FR hip, FR knee, BL hip, BL knee, BR hip, BR knee
        builder.AddContinuous("joints", 8);
    }

    protected override void OnActionsReceived(ActionBuffer actions)
    {
        _body ??= GetParent<CrawlerBody>();
        if (_body is null) return;

        var joints = actions.GetContinuous("joints");
        _body.SetJointTargetVelocities(joints);
        joints.CopyTo(_lastActions, 0);
    }

    public override void CollectObservations(ObservationBuffer obs)
    {
        _body ??= GetParent<CrawlerBody>();
        if (_body is null) return;

        var torso = _body.Torso;

        // [0-2] Torso linear velocity
        obs.AddNormalized(torso.LinearVelocity, -VelBound, VelBound);

        // [3-5] Torso angular velocity
        obs.AddNormalized(torso.AngularVelocity, -VelBound, VelBound);

        // [6-8] Torso up-vector — tells the agent how much the creature is tipping
        obs.Add(torso.GlobalBasis.Y);

        // [9] Torso height
        obs.AddNormalized(torso.GlobalPosition.Y, 0f, 1.5f);

        // [10-17] Joint angles
        foreach (var a in _body.GetJointAngles())
            obs.AddNormalized(a, -Mathf.Pi / 2f, Mathf.Pi / 2f);

        // [18-25] Joint angular velocities
        foreach (var v in _body.GetJointAngularVelocities())
            obs.AddNormalized(v, -5f, 5f);
    }

    public override void OnStep()
    {
        _body ??= GetParent<CrawlerBody>();
        if (_body is null) return;

        var torso = _body.Torso;

        // Forward velocity in +X is the primary training signal.
        // Doubled scale so locomotion dominates over upright-only strategies.
        AddReward(torso.LinearVelocity.X * 0.1f, "forward_vel");

        // Upright bonus: smooth gradient above 0.5 alignment to discourage tipping.
        var uprightDot = torso.GlobalBasis.Y.Dot(Vector3.Up);
        if (uprightDot > 0.5f)
            AddReward((uprightDot - 0.5f) * 0.05f, "upright");

        // Energy penalty: discourages unnecessary motor activity and aids smooth gaits.
        var meanSqAction = 0f;
        foreach (var a in _lastActions) meanSqAction += a * a;
        AddReward(-(meanSqAction / 8f) * 0.005f, "energy");

        // NOTE: alive bonus removed — it rewarded doing nothing (standing still upright
        // was near-optimal), trapping the policy in a local minimum with no locomotion.

        // Terminate when the torso is too close to the ground (tipped/collapsed).
        if (torso.GlobalPosition.Y < CollapseHeight)
            EndEpisode();
    }

    public override void OnEpisodeBegin()
    {
        _body ??= GetParent<CrawlerBody>();
        _body?.Reset();
        Array.Clear(_lastActions, 0, _lastActions.Length);
    }

    protected override void OnHumanInput()
    {
        var input = Vector2.Zero;
        if (Input.IsActionPressed("ui_right")) input.X += 1f;
        if (Input.IsActionPressed("ui_left"))  input.X -= 1f;
        if (Input.IsActionPressed("ui_down"))  input.Y += 1f;
        if (Input.IsActionPressed("ui_up"))    input.Y -= 1f;

        // Simple heuristic: set hip motors based on input, knees based on hip angles.
        var joints = new float[8];
        joints[0] = input.X; // FL hip
        joints[2] = input.X; // FR hip
        joints[4] = input.X; // BL hip
        joints[6] = input.X; // BR hip

        // Knee targets are negative when hips are near zero, positive when hips are flexed.
        for (int i = 0; i < 4; i++)
            joints[i * 2 + 1] = -Mathf.Sign(joints[i * 2]) * Mathf.Abs(joints[i * 2]);

        _body?.SetJointTargetVelocities(joints);
        joints.CopyTo(_lastActions, 0);
    }
    
}
