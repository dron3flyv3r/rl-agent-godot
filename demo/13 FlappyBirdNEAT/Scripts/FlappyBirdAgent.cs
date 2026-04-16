using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

/// <summary>
/// One Flappy Bird agent. Parented inside a BirdGroup alongside a <see cref="FlappyBirdBird"/>.
///
/// Observations (6 floats, all normalised to [-1, 1]):
///   0. Bird Y position
///   1. Bird Y velocity
///   2. Next pipe horizontal distance
///   3. Next pipe gap centre Y
///   4. Second pipe horizontal distance
///   5. Second pipe gap centre Y
///
/// Discrete actions:
///   0 = do nothing
///   1 = flap
///
/// Rewards:
///   +0.01 per step survived
///   +2.0  on passing a pipe
///   -1.0  on death (applied once when the episode ends, not every frame)
///
/// Episode ends via generation barrier — all birds must die before EndEpisode is called.
/// </summary>
public partial class FlappyBirdAgent : RLAgent2D
{
    // ── State ────────────────────────────────────────────────────────────────

    private FlappyBirdBird?       _bird;
    private FlappyBirdController? _controller;

    // Screen bounds for normalisation (must match controller)
    private const float TopY    = 35f;
    private const float BotY    = 555f;
    private const float MaxDx   = 860f;
    private const float MaxVel  = FlappyBirdBird.MaxFallSpeed;
    private const float MinVel  = FlappyBirdBird.FlapVelocity;

    // ── Debug ────────────────────────────────────────────────────────────────

    // Set to true at runtime to print per-bird action logs for the next episode.
    public static bool DebugActions = true;

    // Slot assigned by the training bootstrap (logged for verification).
    private int _debugSlot = -1;
    private int _debugActionFrames;

    // ── RLAgent2D overrides ───────────────────────────────────────────────────

    public override void DefineActions(ActionSpaceBuilder builder)
    {
        builder.AddDiscrete("flap", 2);  // 0 = nothing, 1 = flap
    }

    public override void _Ready()
    {
        base._Ready();

        // Bird is a sibling named "Bird" inside the same BirdGroup node
        _bird = GetParent().GetNodeOrNull<FlappyBirdBird>("Bird");

        // Controller is somewhere above — walk up until found
        _controller = FindAncestor<FlappyBirdController>();

        if (_bird == null)
            GD.PushError("[FlappyBirdAgent] Could not find sibling 'Bird' node. Check scene structure.");
        if (_controller == null)
            GD.PushError("[FlappyBirdAgent] Could not find FlappyBirdController ancestor.");
    }

    private static T? FindAncestor<T>(Node start) where T : Node
    {
        var current = start.GetParent();
        while (current != null)
        {
            if (current is T found) return found;
            current = current.GetParent();
        }
        return null;
    }

    private T? FindAncestor<T>() where T : Node => FindAncestor<T>(this);

    protected override void OnActionsReceived(ActionBuffer actions)
    {
        if (_bird == null || _bird.IsDead) return;

        int action = actions.GetDiscrete("flap");

        // Debug: print the first 5 actions this bird receives each episode.
        if (DebugActions && _debugActionFrames < 5)
        {
            _debugActionFrames++;
            // Read the training slot from PolicyGroupConfig.AgentId if available,
            // or fall back to the node path as a slot proxy.
            GD.Print($"[NEAT-DBG] {GetParent().Name}/{Name}  action={action}  birdPos={_bird.GlobalPosition}  slot={_debugSlot}");
            if (_debugActionFrames == 5)
                GD.Print($"[NEAT-DBG] {GetParent().Name}/{Name}  (suppressing further per-step logs)");
        }

        _bird.WantFlap = action == 1;
    }

    public override void CollectObservations(ObservationBuffer obs)
    {
        if (_bird == null || _controller == null)
        {
            // Zeros as safe fallback
            for (int i = 0; i < 6; i++) obs.Add(0f);
            return;
        }

        float bx = _bird.GlobalPosition.X;
        float by = _bird.GlobalPosition.Y;

        var (p1dx, p1gy) = _controller.GetNextPipe(bx);
        var (p2dx, p2gy) = _controller.GetSecondPipe(bx);

        obs.AddNormalized(by,              TopY,  BotY);
        obs.AddNormalized(_bird.VelocityY, MinVel, MaxVel);
        obs.AddNormalized(p1dx,            0f,    MaxDx);
        obs.AddNormalized(p1gy,            TopY,  BotY);
        obs.AddNormalized(p2dx,            0f,    MaxDx * 2f);
        obs.AddNormalized(p2gy,            TopY,  BotY);
    }

    public override void OnStep()
    {
        if (_bird == null || _controller == null) return;

        if (_bird.IsDead)
        {
            if (_controller.IsGenerationOver)
            {
                AddReward(-1.0f, "death");
                EndEpisode();
            }
            return;
        }

        AddReward(0.01f, "survival");

        // Check if we just passed any pipe
        if (_controller.CheckAndMarkPassed(_bird.GlobalPosition.X, _bird))
            AddReward(2.0f, "pipe_passed");

        if (_controller.IsGenerationOver)
            EndEpisode();
    }

    public override void OnEpisodeBegin()
    {
        _debugActionFrames = 0;
        _controller?.OnBirdReset();
    }

    /// <summary>Called by FlappyBirdController to inform this agent of its slot index.</summary>
    public void SetDebugSlot(int slot) => _debugSlot = slot;
}
