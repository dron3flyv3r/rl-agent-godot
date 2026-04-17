using System.Collections.Generic;
using System.Globalization;
using System.Linq;
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
///   +0.0010 per step survived
///   +0.0030 scaled centering bonus when approaching the next pipe
///   +5.0    on passing a pipe
///   -2.0    on death (applied once when the episode ends, not every frame)
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
    public static bool DebugActions = false;
    private const int MaxDecisionTraceEntries = 12;

    // Slot assigned by the training bootstrap (logged for verification).
    private int _debugSlot = -1;
    private int _debugActionFrames;
    private int _debugFlapCount;
    private int _debugIdleCount;
    private readonly Queue<string> _decisionTrace = new();

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
        if (action == 1) _debugFlapCount++;
        else _debugIdleCount++;
        AppendDecisionTrace(action);

        // Debug: print the first 5 actions this bird receives each episode.
        if (DebugActions && _debugActionFrames < 5)
        {
            _debugActionFrames++;
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

        float nextGapOffset   = p1gy - by;
        float secondGapOffset = p2gy - by;
        float maxGapOffset    = BotY - TopY;

        obs.AddNormalized(by,               TopY,         BotY);
        obs.AddNormalized(_bird.VelocityY,  MinVel,       MaxVel);
        obs.AddNormalized(p1dx,             0f,           MaxDx);
        obs.AddNormalized(nextGapOffset,   -maxGapOffset, maxGapOffset);
        obs.AddNormalized(p2dx,             0f,           MaxDx * 2f);
        obs.AddNormalized(secondGapOffset, -maxGapOffset, maxGapOffset);
    }

    public override void OnStep()
    {
        if (_bird == null || _controller == null) return;

        if (_bird.IsDead)
        {
            if (_controller.IsGenerationOver)
            {
                AddReward(-2.0f, "death");
                EndEpisode();
            }
            return;
        }

        AddReward(0.0010f, "survival");

        var (nextPipeDx, nextGapY) = _controller.GetNextPipe(_bird.GlobalPosition.X);
        float nextGapOffset = Mathf.Abs(nextGapY - _bird.GlobalPosition.Y);
        float approachWeight = 1f - Mathf.Clamp(nextPipeDx / (_controller.PipeSpacing * 1.25f), 0f, 1f);
        float centering = 1f - Mathf.Clamp(nextGapOffset / Mathf.Max(1f, _controller.GetCurrentGapHeight() * 0.75f), 0f, 1f);
        AddReward(0.0030f * approachWeight * centering, "approach_centering");

        // Check if we just passed any pipe
        if (_controller.CheckAndMarkPassed(_bird.GlobalPosition.X, _bird))
            AddReward(5.0f, "pipe_passed");

        if (_controller.IsGenerationOver)
            EndEpisode();
    }

    public override void OnEpisodeBegin()
    {
        _debugActionFrames = 0;
        _debugFlapCount = 0;
        _debugIdleCount = 0;
        _decisionTrace.Clear();
        _controller?.OnBirdReset();
    }

    /// <summary>Called by FlappyBirdController to inform this agent of its slot index.</summary>
    public void SetDebugSlot(int slot) => _debugSlot = slot;

    public int GetDebugSlot() => _debugSlot;

    public int GetDebugFlapCount() => _debugFlapCount;

    public int GetDebugIdleCount() => _debugIdleCount;

    public string GetDecisionTrace() => string.Join(";", _decisionTrace);

    public string GetLastObservationSummary()
    {
        var observation = GetLastObservation();
        if (observation.Length == 0) return "none";
        return string.Join(",", observation.Select(v => v.ToString("F3", CultureInfo.InvariantCulture)));
    }

    private void AppendDecisionTrace(int action)
    {
        var observation = GetLastObservation();
        string obsSummary = observation.Length == 0
            ? "none"
            : string.Join(",", observation.Select(v => v.ToString("F2", CultureInfo.InvariantCulture)));
        string trace =
            $"step={EpisodeSteps}" +
            $"|a={action}" +
            $"|y={_bird?.GlobalPosition.Y.ToString("F1", CultureInfo.InvariantCulture) ?? "n/a"}" +
            $"|vy={_bird?.VelocityY.ToString("F1", CultureInfo.InvariantCulture) ?? "n/a"}" +
            $"|obs={obsSummary}";

        if (_decisionTrace.Count >= MaxDecisionTraceEntries)
            _decisionTrace.Dequeue();
        _decisionTrace.Enqueue(trace);
    }

    protected override void OnHumanInput()
    {
        if (Input.IsActionJustPressed("ui_accept") || Input.IsActionJustPressed("ui_select") || Input.IsActionJustPressed("ui_up"))
        {
            if (_bird != null)
                _bird.WantFlap = true;
        }
    }
}
