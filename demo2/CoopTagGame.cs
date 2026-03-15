using Godot;

namespace RlAgentPlugin.Demo2;

/// <summary>
/// Scene root for the Cooperative Tag demo.
///
/// Two hunters (HunterAgent) must coordinate to simultaneously get within
/// CatchRadius of a randomly-walking prey. The prey bounces off arena walls
/// and occasionally changes direction.
///
/// Reward design:
///   +2.0  to each hunter when both are within CatchRadius (episode ends)
///   -0.005 per step time penalty (encourages speed)
///   +cooperative shaping: closeness1 * closeness2 * 0.03 each step
///         (rewards both agents being near the prey at the same time)
/// </summary>
public partial class CoopTagGame : Node2D
{
    [Export] public float HunterSpeed { get; set; } = 110f;
    [Export] public float PreySpeed { get; set; } = 75f;
    [Export] public float CatchRadius { get; set; } = 36f;

    // Arena bounds — kept in sync with HunterAgent constants
    private const float MinX = HunterAgent.ArenaMinX;
    private const float MaxX = HunterAgent.ArenaMaxX;
    private const float MinY = HunterAgent.ArenaMinY;
    private const float MaxY = HunterAgent.ArenaMaxY;
    private static readonly float MaxDist =
        new Vector2(MaxX - MinX, MaxY - MinY).Length();

    public Vector2 PreyPosition { get; private set; }
    public bool IsEpisodeDone { get; private set; }

    private HunterAgent? _hunter1;
    private HunterAgent? _hunter2;
    private Node2D? _preyNode;
    private Label? _statusLabel;

    private float _hunter1Reward;
    private float _hunter2Reward;
    private Vector2 _preyVelocity;
    private int _dirChangeCooldown;

    private readonly RandomNumberGenerator _rng = new();

    // Frame guard: prevents double-reset when both agents call OnEpisodeBegin in the same frame.
    private ulong _lastResetFrame = ulong.MaxValue;

    public override void _Ready()
    {
        _rng.Randomize();
        _hunter1 = GetNode<HunterAgent>("Hunter1");
        _hunter2 = GetNode<HunterAgent>("Hunter2");
        _preyNode = GetNodeOrNull<Node2D>("Prey");
        _statusLabel = GetNodeOrNull<Label>("UI/Panel/Margin/VBox/StatusLabel");
        ResetEpisodeInternal();
    }

    public override void _PhysicsProcess(double delta)
    {
        if (_hunter1 is null || _hunter2 is null) return;

        var dt = (float)delta;

        MoveHunter(_hunter1, dt);
        MoveHunter(_hunter2, dt);
        MovePrey(dt);

        var d1 = _hunter1.Position.DistanceTo(PreyPosition);
        var d2 = _hunter2.Position.DistanceTo(PreyPosition);
        var caught = d1 <= CatchRadius && d2 <= CatchRadius;

        if (caught)
        {
            _hunter1Reward += 2f;
            _hunter2Reward += 2f;
            IsEpisodeDone = true;
            UpdateStatus("CAUGHT!");
        }
        else
        {
            // Time penalty
            _hunter1Reward -= 0.005f;
            _hunter2Reward -= 0.005f;

            // Cooperative shaping: reward grows when BOTH hunters close in simultaneously.
            // closeness = 1 when on top of prey, 0 when at max distance.
            var c1 = 1f - d1 / MaxDist;
            var c2 = 1f - d2 / MaxDist;
            var bonus = c1 * c2 * 0.03f;
            _hunter1Reward += bonus;
            _hunter2Reward += bonus;

            UpdateStatus($"Hunting… d1={d1:F0}  d2={d2:F0}");
        }
    }

    /// <summary>Returns the position of the other hunter (from the caller's perspective).</summary>
    public Vector2 GetOtherHunterPosition(HunterAgent caller) =>
        caller == _hunter1 ? (_hunter2?.Position ?? Vector2.Zero)
                           : (_hunter1?.Position ?? Vector2.Zero);

    /// <summary>Drains and returns the accumulated reward for this agent.</summary>
    public float ConsumeReward(HunterAgent caller)
    {
        if (caller == _hunter1)
        {
            var r = _hunter1Reward;
            _hunter1Reward = 0f;
            return r;
        }
        else
        {
            var r = _hunter2Reward;
            _hunter2Reward = 0f;
            return r;
        }
    }

    /// <summary>
    /// Called from each hunter's OnEpisodeBegin().
    /// Frame-guard ensures the env is only reset once per physics frame
    /// even when both agents trigger the reset simultaneously.
    /// </summary>
    public void ResetEpisode()
    {
        var frame = Engine.GetPhysicsFrames();
        if (frame == _lastResetFrame) return;
        _lastResetFrame = frame;
        ResetEpisodeInternal();
    }

    // ── Private ───────────────────────────────────────────────────────────────

    private void ResetEpisodeInternal()
    {
        IsEpisodeDone = false;
        _hunter1Reward = 0f;
        _hunter2Reward = 0f;

        if (_hunter1 is not null) _hunter1.Position = RandomPos();
        if (_hunter2 is not null) _hunter2.Position = RandomPos();
        PreyPosition = RandomPos();
        if (_preyNode is not null) _preyNode.Position = PreyPosition;

        var angle = _rng.RandfRange(0f, Mathf.Tau);
        _preyVelocity = new Vector2(Mathf.Cos(angle), Mathf.Sin(angle));
        _dirChangeCooldown = (int)_rng.RandfRange(40f, 100f);

        UpdateStatus("Episode started");
    }

    private void MovePrey(float dt)
    {
        _dirChangeCooldown--;
        if (_dirChangeCooldown <= 0)
        {
            var angle = _rng.RandfRange(0f, Mathf.Tau);
            _preyVelocity = new Vector2(Mathf.Cos(angle), Mathf.Sin(angle));
            _dirChangeCooldown = (int)_rng.RandfRange(40f, 100f);
        }

        var newPos = PreyPosition + _preyVelocity * PreySpeed * dt;

        if (newPos.X < MinX || newPos.X > MaxX) _preyVelocity.X *= -1f;
        if (newPos.Y < MinY || newPos.Y > MaxY) _preyVelocity.Y *= -1f;

        PreyPosition = new Vector2(
            Mathf.Clamp(newPos.X, MinX, MaxX),
            Mathf.Clamp(newPos.Y, MinY, MaxY));

        if (_preyNode is not null) _preyNode.Position = PreyPosition;
    }

    private void MoveHunter(HunterAgent agent, float dt)
    {
        var vel = agent.MoveVelocity * HunterSpeed * dt;
        agent.Position = new Vector2(
            Mathf.Clamp(agent.Position.X + vel.X, MinX, MaxX),
            Mathf.Clamp(agent.Position.Y + vel.Y, MinY, MaxY));
    }

    private Vector2 RandomPos() =>
        new(_rng.RandfRange(MinX + 40f, MaxX - 40f),
            _rng.RandfRange(MinY + 40f, MaxY - 40f));

    private void UpdateStatus(string msg)
    {
        if (_statusLabel is not null) _statusLabel.Text = msg;
    }
}
