using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

/// <summary>
/// Demo 11 — Game-Dev Academy  (Tier 1: lifecycle hooks only)
///
/// Shows how a game developer can add adaptive curriculum and a step-budget
/// stop condition without touching any training-loop code.
///
/// How to use:
///   1. In your training scene, select the Academy node.
///   2. Replace the script from RLAcademy.cs to this file.
///   3. Assign an RLCurriculumConfig to the academy and wire the curriculum
///      consumers (e.g. arena controllers) to read CurriculumProgress.
///   4. Tune StepBudget, CurriculumStep, and RewardThreshold in the Inspector.
///   5. Hit "Start Training" — everything else is handled automatically.
///
/// Hooks used:
///   OnTrainingInitialized  — prints a one-time summary to the Godot output.
///   OnEpisodeEnd           — bumps curriculum when the reward goal is beaten.
///   OnBeforeCheckpoint     — logs an inter-checkpoint reward average.
///   ShouldStop             — stops the run when the step budget is exhausted.
/// </summary>
[GlobalClass]
public partial class GameDevAcademy : RLAcademy
{
    [ExportGroup("Step Budget")]
    /// <summary>
    /// Training stops automatically after this many total environment steps.
    /// Set to 0 to run indefinitely (until a manual stop or other condition).
    /// </summary>
    [Export(PropertyHint.Range, "0,10000000,1,or_greater")]
    public long StepBudget { get; set; } = 500_000;

    [ExportGroup("Adaptive Curriculum")]
    /// <summary>
    /// Reward the agent must reach in an episode for the curriculum to advance.
    /// </summary>
    [Export(PropertyHint.Range, "0,100,0.1,or_greater")]
    public float RewardThreshold { get; set; } = 0.8f;

    /// <summary>
    /// How much to increase CurriculumProgress when an episode beats the threshold.
    /// Curriculum is clamped to [0, 1] automatically.
    /// </summary>
    [Export(PropertyHint.Range, "0.001,0.1,0.001")]
    public float CurriculumStep { get; set; } = 0.02f;

    // ── Internal state ────────────────────────────────────────────────────────
    private int   _episodesSinceCheckpoint;
    private float _rewardSumSinceCheckpoint;

    // ─────────────────────────────────────────────────────────────────────────
    // Tier 1 hooks
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Runs once after all trainers and agents are ready.
    /// A good place to print a run summary or initialise any external loggers.
    /// </summary>
    public override void OnTrainingInitialized(IAcademyContext ctx)
    {
        GD.Print("[GameDevAcademy] Training started.");
        GD.Print($"  Policy groups : {string.Join(", ", ctx.GroupIds)}");
        GD.Print($"  Step budget   : {(StepBudget > 0 ? StepBudget.ToString("N0") : "unlimited")}");
        GD.Print($"  Reward goal   : >= {RewardThreshold:F2}  " +
                 $"(curriculum +{CurriculumStep:P0} per win)");
    }

    /// <summary>
    /// Called once per episode completion, possibly multiple times per physics frame
    /// when several agents finish simultaneously.
    ///
    /// Here we drive curriculum purely from episode reward: every time an agent
    /// beats <see cref="RewardThreshold"/> the global difficulty nudges up.
    /// </summary>
    public override void OnEpisodeEnd(AcademyEpisodeEndArgs args)
    {
        _episodesSinceCheckpoint++;
        _rewardSumSinceCheckpoint += args.EpisodeReward;

        if (args.EpisodeReward < RewardThreshold) return;

        var next = Mathf.Clamp(args.CurriculumProgress + CurriculumStep, 0f, 1f);
        SetCurriculumProgress(next);

        if (args.GroupEpisodeCount % 50 == 0)   // throttle console noise
        {
            GD.Print($"[GameDevAcademy] Curriculum → {next:P0}  " +
                     $"(ep reward {args.EpisodeReward:F2}, step {args.TotalSteps:N0})");
        }
    }

    /// <summary>
    /// Runs immediately before a checkpoint is written to disk.
    /// Prints the average reward over all episodes since the last checkpoint —
    /// a quick sanity check that learning is progressing.
    /// </summary>
    public override void OnBeforeCheckpoint(IAcademyContext ctx)
    {
        if (_episodesSinceCheckpoint == 0) return;

        var avg = _rewardSumSinceCheckpoint / _episodesSinceCheckpoint;
        GD.Print($"[GameDevAcademy] Checkpoint @ step {ctx.TotalSteps:N0}  " +
                 $"— avg reward: {avg:F3}  ({_episodesSinceCheckpoint} episodes)  " +
                 $"curriculum: {CurriculumProgress:P0}");

        _episodesSinceCheckpoint   = 0;
        _rewardSumSinceCheckpoint  = 0f;
    }

    /// <summary>
    /// Evaluated every frame after the training step.
    /// Returning true here is equivalent to clicking "Stop Training" — the
    /// bootstrap will write a final checkpoint and cleanly exit.
    /// </summary>
    public override bool ShouldStop(IAcademyContext ctx)
        => StepBudget > 0 && ctx.TotalSteps >= StepBudget;
}
