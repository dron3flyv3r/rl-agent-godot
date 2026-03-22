using System;
using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

/// <summary>
/// Minimal 3D RL environment: agent must walk to a randomly-placed target on a flat platform.
///
/// DESIGN GOALS (keep it simple to expose system bugs):
///   - No physics complexity (no rigidbodies, no jumping, no walls)
///   - Continuous 2D actions: move_x, move_z in [-1, 1]
///   - 8 observations — all clearly labeled and normalized
///   - Dense distance-shaping reward + goal bonus + OOB penalty
///   - Heavy debug logging to isolate training instability
///
/// OBSERVATION LAYOUT (8 floats):
///   [0] agent_x       — normalized to [-1, 1] over arena bounds
///   [1] agent_z       — normalized to [-1, 1] over arena bounds
///   [2] target_x      — normalized to [-1, 1] over arena bounds
///   [3] target_z      — normalized to [-1, 1] over arena bounds
///   [4] delta_x       — (target - agent).X, normalized to [-1, 1] over 2× arena
///   [5] delta_z       — (target - agent).Z, normalized to [-1, 1] over 2× arena
///   [6] vel_x         — agent horizontal velocity X, normalized
///   [7] vel_z         — agent horizontal velocity Z, normalized
///
/// REWARD BREAKDOWN:
///   step_penalty    — small constant per step to encourage speed
///   dist_progress   — (prev_dist - cur_dist) shaped reward (dense, signed)
///   goal_reached    — +1.0 one-shot bonus on arrival
///   out_of_bounds   — -1.0 penalty when agent leaves arena
/// </summary>
public partial class MoveToTarget3DAgent : RLAgent3D
{
    // ── Inspector exports ──────────────────────────────────────────────────────

    /// <summary>Half-width/depth of the playable arena (platform is 2× this in each axis).</summary>
    [Export] public float ArenaHalfSize { get; set; } = 4.5f;

    /// <summary>Distance threshold to consider the target reached.</summary>
    [Export] public float GoalRadius { get; set; } = 0.6f;

    /// <summary>Print episode summary (steps, total reward, outcome) at episode end.</summary>
    [Export] public bool DebugPrintEpisode { get; set; } = true;

    /// <summary>Print raw observation values every N steps. 0 = disabled.</summary>
    [Export] public int DebugObsPrintInterval { get; set; } = 100;

    /// <summary>Print action values every N steps. 0 = disabled.</summary>
    [Export] public int DebugActionPrintInterval { get; set; } = 100;

    // ── Private state ─────────────────────────────────────────────────────────

    private MoveToTarget3DPlayer? _player;
    private MeshInstance3D? _targetMarker;

    private Vector3 _targetPos;
    private float _prevDist = float.MaxValue;
    private int _episodeCount;
    private string _lastOutcome = "none";

    // Self-tracked debug stats — the framework clears EpisodeReward/EpisodeSteps
    // before OnEpisodeBegin(), so we maintain our own counters to log episode summaries.
    private int _debugSteps;
    private float _debugReward;
    private readonly System.Collections.Generic.Dictionary<string, float> _debugBreakdown = new();

    // Normalization constants (must match scene arena size)
    private float NormBound => ArenaHalfSize;          // for position obs
    private float DeltaNorm => ArenaHalfSize * 2f;     // for relative vector obs
    private const float VelNorm = 6.0f;                // max expected speed (MoveSpeed + margin)

    // Agent spawn (center of platform, just above surface)
    private readonly Vector3 AgentSpawn = new(0f, 0.5f, 0f);
    // Kill plane below which we consider the agent OOB vertically
    private const float KillPlaneY = -1.5f;

    // ── RLAgent3D overrides ───────────────────────────────────────────────────

    public override void _Ready()
    {
        base._Ready();
        _player = GetParent() as MoveToTarget3DPlayer;

        // Target marker is a sibling of the Player, one level up in the arena
        _targetMarker = GetParent()?.GetParent()?.GetNodeOrNull<MeshInstance3D>("TargetMarker");

        if (_player is null)
            GD.PushError("[MoveToTarget3D] Agent could not find MoveToTarget3DPlayer parent.");
        if (_targetMarker is null)
            GD.PrintRich("[color=yellow][MoveToTarget3D] TargetMarker not found — target position won't be visualized.[/color]");
    }

    public override void DefineActions(ActionSpaceBuilder builder)
    {
        // Continuous 2D movement in [-1, 1] on the XZ plane
        builder.AddContinuous("move", dimensions: 2);
    }

    protected override void OnActionsReceived(ActionBuffer actions)
    {
        if (_player is null) return;
        var move = actions.GetContinuous("move");  // float[2]: [move_x, move_z]
        _player.SetMoveIntent(new Vector3(move[0], 0f, move[1]));
    }

    public override void CollectObservations(ObservationBuffer obs)
    {
        if (_player is null) return;

        var agentPos = _player.GlobalPosition;
        var vel      = _player.HorizontalVelocity;
        var delta    = _targetPos - agentPos;

        // [0-1] Agent XZ position, normalized to [-1, 1] over arena bounds
        obs.AddNormalized(agentPos.X, -NormBound, NormBound);
        obs.AddNormalized(agentPos.Z, -NormBound, NormBound);

        // [2-3] Target XZ position, normalized
        obs.AddNormalized(_targetPos.X, -NormBound, NormBound);
        obs.AddNormalized(_targetPos.Z, -NormBound, NormBound);

        // [4-5] Relative vector agent → target, normalized
        obs.AddNormalized(delta.X, -DeltaNorm, DeltaNorm);
        obs.AddNormalized(delta.Z, -DeltaNorm, DeltaNorm);

        // [6-7] Agent horizontal velocity, normalized
        obs.AddNormalized(vel.X, -VelNorm, VelNorm);
        obs.AddNormalized(vel.Z, -VelNorm, VelNorm);
    }

    public override void OnStep()
    {
        if (_player is null) return;

        _debugSteps++;
        var agentPos = _player.GlobalPosition;

        // ── Step penalty (encourages reaching target quickly) ──────────────
        Track(-0.001f, "step_penalty");

        // ── Dense distance-shaping ─────────────────────────────────────────
        var curDist = agentPos.DistanceTo(_targetPos);
        if (_prevDist < float.MaxValue)
        {
            var progress = _prevDist - curDist;   // positive = moved closer
            Track(progress, "dist_progress"); // ~0.003-0.01 per step when moving well
        }
        _prevDist = curDist;

        // ── Out-of-bounds detection ────────────────────────────────────────
        var oobX = Mathf.Abs(agentPos.X) > ArenaHalfSize + 1f;
        var oobZ = Mathf.Abs(agentPos.Z) > ArenaHalfSize + 1f;
        var oobY = agentPos.Y < KillPlaneY;

        if (oobX || oobZ || oobY)
        {
            _lastOutcome = oobY ? "fell_off" : "walked_off";
            Track(-1.0f, "out_of_bounds");
            EndEpisode();
            return;
        }

        // ── Goal reached ───────────────────────────────────────────────────
        if (curDist < GoalRadius)
        {
            float bonus = Math.Max(0f, 1.0f * (1f - EpisodeSteps / 500f)); // up to 100% bonus for fast solutions
            _lastOutcome = "goal";
            Track(1.0f + bonus, "goal_reached");
            EndEpisode();
        }
    }

    /// <summary>AddReward + mirror into our own debug counters.</summary>
    private void Track(float amount, string tag)
    {
        AddReward(amount, tag);
        _debugReward += amount;
        _debugBreakdown.TryGetValue(tag, out var cur);
        _debugBreakdown[tag] = cur + amount;
    }

    public override void OnEpisodeBegin()
    {
        // Re-cache references in case of re-use across batch instances
        _player       ??= GetParent() as MoveToTarget3DPlayer;
        _targetMarker ??= GetParent()?.GetParent()?.GetNodeOrNull<MeshInstance3D>("TargetMarker");

        // Print summary for the episode that just ended.
        // NOTE: The framework clears EpisodeReward/EpisodeSteps before OnEpisodeBegin(),
        // so we use our own _debug* counters which we reset manually below.
        if (DebugPrintEpisode && _episodeCount > 0)
        {
            var parts = new System.Text.StringBuilder();
            foreach (var (tag, amount) in _debugBreakdown)
                parts.Append($"  {tag}={amount:+0.000;-0.000}");
        }

        _episodeCount++;
        _lastOutcome = "timeout";

        // Reset our debug counters for the new episode
        _debugSteps = 0;
        _debugReward = 0f;
        _debugBreakdown.Clear();

        // Reset player
        if (_player is not null)
        {
            _player.GlobalPosition = AgentSpawn;
            _player.Velocity       = Vector3.Zero;
        }

        // Randomize target (ensure minimum distance from spawn)
        var rng = new RandomNumberGenerator();
        rng.Randomize();
        float tx, tz;
        do
        {
            tx = rng.RandfRange(-NormBound + 0.5f, NormBound - 0.5f);
            tz = rng.RandfRange(-NormBound + 0.5f, NormBound - 0.5f);
        }
        while (new Vector2(tx - AgentSpawn.X, tz - AgentSpawn.Z).Length() < 2.0f);

        _targetPos = new Vector3(tx, 0.1f, tz);
        if (_targetMarker is not null)
            _targetMarker.GlobalPosition = _targetPos;

        _prevDist = float.MaxValue;
    }

    protected override void OnHumanInput()
    {
        if (_player is null) return;

        var input = Vector3.Zero;
        if (Input.IsKeyPressed(Key.W) || Input.IsKeyPressed(Key.Up))    input.X += 1f;
        if (Input.IsKeyPressed(Key.S) || Input.IsKeyPressed(Key.Down))  input.X -= 1f;
        if (Input.IsKeyPressed(Key.A) || Input.IsKeyPressed(Key.Left))  input.Z -= 1f;
        if (Input.IsKeyPressed(Key.D) || Input.IsKeyPressed(Key.Right)) input.Z += 1f;

        _player.SetMoveIntent(input.Normalized());
    }
}
