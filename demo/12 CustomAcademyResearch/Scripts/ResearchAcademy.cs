using System.Collections.Generic;
using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

/// <summary>
/// Demo 12 — Research Academy  (Tier 2 + 3: custom training loop)
///
/// Shows how a researcher can take full control of the training loop:
///
///   Tier 2 — <see cref="OwnsTrainingStep"/> = true
///     TrainingBootstrap calls <see cref="TrainingStep"/> instead of its own
///     four-phase dispatch.  Groups are processed in reverse-registration order
///     to demonstrate custom scheduling.
///
///   Tier 3 — phase-split API + direct <see cref="ITrainer"/> access
///     Each group is run through the individual phase methods
///     (EstimateNextValues → RecordTransitionsAndReset → SampleActions → ApplyDecisions)
///     rather than the convenience wrapper.  This mirrors the structure needed for
///     future multi-threaded rollouts: phases A and C are pure math and can be
///     parallelised across groups on worker threads without Godot API access.
///
///     After the pipeline, <see cref="ITrainer.TryUpdate"/> is called directly and
///     the resulting entropy / loss values are written as custom metrics via
///     <see cref="IAcademyContext.LogMetric"/> — giving the researcher full control
///     over what lands in the JSONL log.
///
///     NOTE: because TryUpdate is drained here, the bootstrap's own TryUpdate pass
///     this frame will find an empty buffer and return null.  Standard metric
///     columns (episode_reward, etc.) are still written by OnEpisodeEnd handling
///     inside Phase B — only the per-update loss columns move to our custom log.
///
///   Stopping condition
///     Stops when entropy has remained below <see cref="ConvergenceEntropyThreshold"/>
///     for every group over a <see cref="ConvergenceGracePeriod"/> window.
///
/// How to use:
///   1. Attach this script to the Academy node in place of RLAcademy.cs.
///   2. Tune the convergence knobs in the Inspector.
///   3. Run training — groups will process in the reverse of the order they were
///      registered, and entropy will appear as a custom metric in RLDash.
/// </summary>
[GlobalClass]
public partial class ResearchAcademy : RLAcademy
{
    [ExportGroup("Convergence")]
    /// <summary>
    /// Training stops when every group's last-known entropy is at or below this value
    /// for <see cref="ConvergenceGracePeriod"/> consecutive frames.
    /// </summary>
    [Export(PropertyHint.Range, "0.001,2.0,0.001")]
    public float ConvergenceEntropyThreshold { get; set; } = 0.05f;

    /// <summary>
    /// Number of consecutive frames all groups must stay below
    /// <see cref="ConvergenceEntropyThreshold"/> before training stops.
    /// Prevents spurious stops during early stochastic dips.
    /// </summary>
    [Export(PropertyHint.Range, "1,500,1")]
    public int ConvergenceGracePeriod { get; set; } = 100;

    // ── Internal state ────────────────────────────────────────────────────────
    private readonly Dictionary<string, float> _lastEntropyByGroup = new();
    private int _convergenceFrames;

    // ─────────────────────────────────────────────────────────────────────────
    // Tier 2: claim the training loop
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Returning true tells TrainingBootstrap to call <see cref="TrainingStep"/>
    /// each physics frame instead of its own four-phase group loop.
    /// Frozen / self-play opponents are still handled by the bootstrap.
    /// </summary>
    public override bool OwnsTrainingStep => true;

    // ─────────────────────────────────────────────────────────────────────────
    // Tier 1: lifecycle hooks
    // ─────────────────────────────────────────────────────────────────────────

    public override void OnTrainingInitialized(IAcademyContext ctx)
    {
        GD.Print("[ResearchAcademy] Custom training loop active (Tier 2 + 3).");
        GD.Print($"  Groups (will run in reverse order): " +
                 $"{string.Join(", ", ctx.GroupIds)}");
        GD.Print($"  Convergence : H ≤ {ConvergenceEntropyThreshold:F3} " +
                 $"for {ConvergenceGracePeriod} consecutive frames");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tier 2 + 3: custom training step
    // ─────────────────────────────────────────────────────────────────────────

    public override void TrainingStep(IAcademyContext ctx)
    {
        var groups = ctx.GroupIds;
        if (groups.Count == 0) return;
        var n = groups.Count;

        // ── Phase A: estimate bootstrap values ───────────────────────────────
        // Pure math — no Godot API. In a real parallel pipeline each call could
        // be dispatched to a Task and awaited before Phase B.
        var tokensA = new PhaseAToken[n];
        for (var i = n - 1; i >= 0; i--)           // ← reversed group priority
            tokensA[i] = ctx.EstimateNextValues(groups[i]);

        // ── Phase B: record transitions + reset done agents ──────────────────
        // Touches the Godot scene tree — must stay on the main thread.
        // OnEpisodeEnd is fired from inside this phase.
        var tokensB = new PhaseBToken[n];
        for (var i = n - 1; i >= 0; i--)
            tokensB[i] = ctx.RecordTransitionsAndReset(groups[i], tokensA[i]);

        // ── Phase C: sample new actions ──────────────────────────────────────
        // Pure math — no Godot API, parallelisable like Phase A.
        var tokensC = new PhaseCToken[n];
        for (var i = n - 1; i >= 0; i--)
            tokensC[i] = ctx.SampleActions(groups[i], tokensB[i]);

        // ── Phase D: apply decisions to agents ───────────────────────────────
        // Writes back to the scene tree — main thread only.
        for (var i = n - 1; i >= 0; i--)
            ctx.ApplyDecisions(groups[i], tokensC[i]);

        // ── Tier 3: drive updates and log custom metrics ─────────────────────
        // Calling TryUpdate here drains the rollout buffer.  The bootstrap's own
        // TryUpdate pass (which still runs after TrainingStep returns) will find
        // the buffer empty and return null — so we own both the update and the
        // per-update metric entries written to the JSONL log.
        foreach (var gid in groups)
        {
            var trainer = ctx.GetTrainer(gid);
            if (trainer is null) continue;

            var episodeCount = ctx.EpisodeCountByGroup.GetValueOrDefault(gid, 0L);
            var stats = trainer.TryUpdate(gid, ctx.TotalSteps, episodeCount);
            if (stats is null) continue;   // buffer not full yet this frame

            // Write exactly the metrics we care about; ignore the rest.
            ctx.LogMetric(gid, "entropy",      stats.Entropy);
            ctx.LogMetric(gid, "policy_loss",  stats.PolicyLoss);
            ctx.LogMetric(gid, "value_loss",   stats.ValueLoss);

            _lastEntropyByGroup[gid] = stats.Entropy;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tier 1: convergence stop
    // ─────────────────────────────────────────────────────────────────────────

    public override bool ShouldStop(IAcademyContext ctx)
    {
        // No entropy data yet — can't judge convergence.
        if (_lastEntropyByGroup.Count < ctx.GroupIds.Count) return false;

        var allConverged = true;
        foreach (var gid in ctx.GroupIds)
        {
            if (!_lastEntropyByGroup.TryGetValue(gid, out var h) ||
                h > ConvergenceEntropyThreshold)
            {
                allConverged = false;
                break;
            }
        }

        if (allConverged)
        {
            _convergenceFrames++;
            if (_convergenceFrames < ConvergenceGracePeriod) return false;

            GD.Print(
                $"[ResearchAcademy] Convergence detected at step {ctx.TotalSteps:N0}. " +
                $"All groups held H ≤ {ConvergenceEntropyThreshold:F3} " +
                $"for {_convergenceFrames} consecutive frames.");
            return true;
        }

        _convergenceFrames = 0;
        return false;
    }
}
