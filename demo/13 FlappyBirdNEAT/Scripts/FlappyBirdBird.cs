using Godot;

namespace RlAgentPlugin.Demo;

/// <summary>
/// A single Flappy Bird agent body. Node2D with manual physics — no physics engine collision.
/// The controller queries bounding-box overlaps to detect pipe and boundary hits.
/// </summary>
public partial class FlappyBirdBird : Node2D
{
    // ── Tuning ────────────────────────────────────────────────────────────────

    public const float Gravity       = 1200f;  // px/s²
    public const float FlapVelocity  = -520f;  // px/s  (upward)
    public const float MaxFallSpeed  = 700f;   // px/s  (downward cap)

    public const float BirdRadius    = 14f;    // collision half-extent (square check)

    // ── State ────────────────────────────────────────────────────────────────

    public float VelocityY  { get; private set; }
    public bool  IsDead     { get; private set; }

    /// <summary>Set by FlappyBirdAgent each physics frame before Update() is called.</summary>
    public bool WantFlap { get; set; }

    // visual references, assigned in _Ready
    private Node2D? _visual;

    // ── Godot lifecycle ──────────────────────────────────────────────────────

    public override void _Ready()
    {
        _visual = GetNodeOrNull<Node2D>("Visual");
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// <summary>Advance physics by one frame. Called by the controller.</summary>
    public void Update(double delta)
    {
        if (IsDead) return;

        if (WantFlap)
            VelocityY = FlapVelocity;

        VelocityY += Gravity * (float)delta;
        if (VelocityY > MaxFallSpeed) VelocityY = MaxFallSpeed;

        Position += new Vector2(0f, VelocityY * (float)delta);

        // Tilt visual to match velocity
        if (_visual != null)
        {
            float tilt = Mathf.Clamp(VelocityY / MaxFallSpeed, -1f, 1f) * 40f;
            _visual.Rotation = Mathf.DegToRad(tilt);
        }

        WantFlap = false;
    }

    /// <summary>Kill this bird (called by the controller on collision).</summary>
    public void Die()
    {
        if (IsDead) return;
        IsDead = true;
        // Fade to ghost so the player can still see dead birds
        Modulate = new Color(1f, 1f, 1f, 0.35f);
    }

    /// <summary>Reset position, velocity, and alive state for a new episode.</summary>
    public void ResetBird(float startX, float startY)
    {
        GlobalPosition = new Vector2(startX, startY);
        VelocityY  = 0f;
        IsDead     = false;
        WantFlap   = false;
        Modulate   = Colors.White;
        if (_visual != null) _visual.Rotation = 0f;
    }
}
