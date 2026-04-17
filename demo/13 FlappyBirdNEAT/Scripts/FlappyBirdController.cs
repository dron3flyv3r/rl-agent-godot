using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

/// <summary>
/// Root controller for the Flappy Bird NEAT demo.
///
/// Scene structure expected (controller is the root node):
///   FlappyBirdNEATDemo  (this script)
///     Academy           (RLAcademy)
///     Background        (Polygon2D)
///     Ground            (Polygon2D)
///     Ceiling           (Polygon2D)
///     Pipes             (Node2D — pipe ColorRects are added/removed here at runtime)
///     BirdGroup0        (Node2D)
///       Bird            (FlappyBirdBird)
///       Agent           (FlappyBirdAgent)
///     BirdGroup1 ... BirdGroup19
///     CanvasLayer
///       Panel / labels
///
/// The controller:
///   - Scrolls pipes and detects collisions (no physics engine used)
///   - Implements a generation barrier: waits until every bird has died,
///     then allows all agents to call EndEpisode simultaneously
///   - Provides pipe observation helpers used by FlappyBirdAgent
/// </summary>
public partial class FlappyBirdController : Node2D, INeatLiveStatusProvider
{
    // ── Tuning exports ────────────────────────────────────────────────────────

    [Export] public float PipeScrollSpeed { get; set; } = 200f;
    [Export] public float PipeSpacing     { get; set; } = 260f;   // horizontal gap between pipes
    [Export] public float GapHeight       { get; set; } = 130f;   // vertical opening in each pipe pair
    [Export] public float PipeWidth       { get; set; } = 60f;
    [Export] public float GroundY         { get; set; } = 555f;
    [Export] public float CeilingY        { get; set; } = 35f;
    [Export] public float BirdStartX      { get; set; } = 180f;
    [Export] public float BirdStartY      { get; set; } = 300f;
    [Export] public float PipeSpawnX      { get; set; } = 880f;
    [Export] public int   InitialPipeCount { get; set; } = 3;
    [Export] public int   EasyStartPipeCount { get; set; } = 2;
    [Export] public float PipeGapStepLimit   { get; set; } = 70f;

    [ExportGroup("Debug")]
    [Export] public bool LogGenerationSummaries { get; set; } = true;
    [Export] public bool LogTrainerGenerations  { get; set; } = true;
    /// <summary>
    /// Prints an extended per-slot fitness table, reproduction breakdown (elite/crossover/mutate counts),
    /// species composition, and champion network sample outputs every generation.
    /// Useful for diagnosing whether the NEAT algorithm and slot mapping are working correctly.
    /// </summary>
    [Export] public bool LogDeepStats           { get; set; } = true;
    [Export] public bool LogGenerationStarts    { get; set; } = true;
    [Export] public bool LogDeathEvents         { get; set; } = true;
    [Export] public bool LogEpisodeDetails      { get; set; } = true;
    [Export] public bool LogPipeLayouts         { get; set; } = true;

    // ── Public state ──────────────────────────────────────────────────────────

    /// <summary>True once every bird has died. Agents end their episodes when this is set.</summary>
    public bool IsGenerationOver { get; private set; }

    public int Generation { get; private set; }

    // ── Private state ─────────────────────────────────────────────────────────

    private readonly List<FlappyBirdBird>  _birds   = new();
    private readonly Dictionary<FlappyBirdBird, FlappyBirdAgent> _birdAgents = new();
    private readonly Dictionary<FlappyBirdAgent, FlappyBirdBird> _agentBirds = new();
    private readonly List<PipePair>        _pipes   = new();
    private Node2D? _pipeContainer;
    private Label?  _genLabel;
    private Label?  _aliveLabel;
    // Pooled visual rects — reused every frame instead of destroyed/recreated.
    private readonly List<(ColorRect Top, ColorRect Bot)> _pipeRects = [];

    private int _totalBirds;
    private int _deadCount;
    private int _groundDeaths;
    private int _ceilingDeaths;
    private int _pipeDeaths;
    private bool _generationPrepared;
    private readonly Dictionary<FlappyBirdBird, int> _birdLifetimeSteps = new();
    private readonly Dictionary<FlappyBirdBird, int> _birdPipePasses = new();
    private readonly Dictionary<FlappyBirdBird, DeathSnapshot> _deathSnapshots = new();

    // Per-bird "has this bird passed this pipe" — key: (birdId, pipeId)
    private readonly HashSet<(int, int)> _passedSet = new();
    private int _nextPipeId;
    private static readonly Random _rng = new();

    // ── Pipe data ─────────────────────────────────────────────────────────────

    private sealed class PipePair
    {
        public int   Id;
        public float X;
        public float GapCenterY;
    }

    private enum DeathCause
    {
        Pipe,
        Ground,
        Ceiling,
    }

    private sealed class DeathSnapshot
    {
        public DeathCause Cause;
        public int Steps;
        public int Passes;
        public float Y;
        public float VelocityY;
        public float NextPipeDx;
        public float NextGapOffset;
        public int LastAction;
    }

    // ── Godot lifecycle ───────────────────────────────────────────────────────

    public override void _Ready()
    {
        _pipeContainer = GetNodeOrNull<Node2D>("Pipes");
        _genLabel      = GetNodeOrNull<Label>("CanvasLayer/Panel/Margin/VBox/GenLabel");
        _aliveLabel    = GetNodeOrNull<Label>("CanvasLayer/Panel/Margin/VBox/AliveLabel");

        // Collect birds from all sources:
        //  a) children of RLAgentSpawner siblings (spawner pattern)
        //  b) direct BirdGroup children (manual scene placement)
        CollectBirds();

        _totalBirds = _birds.Count;
        _deadCount  = 0;
        NeatTrainer.DebugGenerationStats = LogTrainerGenerations;
        NeatTrainer.DebugDeepStats       = LogDeepStats;
        ResetGenerationDebugState();

        if (_totalBirds == 0)
            GD.PushWarning("[FlappyBirdController] No FlappyBirdBird nodes found. " +
                           "Add an RLAgentSpawner with BirdGroup.tscn, or place BirdGroup nodes manually.");

        SpawnInitialPipes();
    }

    private void CollectBirds()
    {
        foreach (Node child in GetChildren())
        {
            // Spawner pattern: RLAgentSpawner holds BirdGroup instances
            if (child is RlAgentPlugin.Runtime.RLAgentSpawner spawner)
            {
                foreach (Node spawned in spawner.GetChildren())
                {
                    var bird = spawned.GetNodeOrNull<FlappyBirdBird>("Bird");
                    if (bird != null)
                    {
                        _birds.Add(bird);
                        var agent = spawned.GetNodeOrNull<FlappyBirdAgent>("Agent");
                        if (agent != null)
                        {
                            _birdAgents[bird] = agent;
                            _agentBirds[agent] = bird;
                        }
                    }
                }
                continue;
            }

            // Manual placement: BirdGroup nodes directly under the controller
            if (child is Node2D group)
            {
                var bird = group.GetNodeOrNull<FlappyBirdBird>("Bird");
                if (bird != null)
                {
                    _birds.Add(bird);
                    var agent = group.GetNodeOrNull<FlappyBirdAgent>("Agent");
                    if (agent != null)
                    {
                        _birdAgents[bird] = agent;
                        _agentBirds[agent] = bird;
                    }
                }
            }
        }
    }

    public override void _PhysicsProcess(double delta)
    {
        if (IsGenerationOver) return;

        ScrollPipes(delta);
        UpdateBirds(delta);
        CheckCollisions();
        RecyclePipes();
        UpdateLabels();
    }

    // ── Bird update ───────────────────────────────────────────────────────────

    private void UpdateBirds(double delta)
    {
        foreach (var bird in _birds)
        {
            if (!bird.IsDead && _birdLifetimeSteps.TryGetValue(bird, out var steps))
                _birdLifetimeSteps[bird] = steps + 1;
            bird.Update(delta);
        }
    }

    private void CheckCollisions()
    {
        float r = FlappyBirdBird.BirdRadius;

        foreach (var bird in _birds)
        {
            if (bird.IsDead) continue;

            float bx = bird.GlobalPosition.X;
            float by = bird.GlobalPosition.Y;

            // Ground / ceiling
            if (by + r >= GroundY || by - r <= CeilingY)
            {
                KillBird(bird, by + r >= GroundY ? DeathCause.Ground : DeathCause.Ceiling);
                continue;
            }

            // Pipes
            foreach (var pipe in _pipes)
            {
                float left  = pipe.X - PipeWidth / 2f;
                float right = pipe.X + PipeWidth / 2f;
                if (bx + r < left || bx - r > right) continue;

                float topGap = pipe.GapCenterY - GapHeight / 2f;
                float botGap = pipe.GapCenterY + GapHeight / 2f;

                if (by - r < topGap || by + r > botGap)
                {
                    KillBird(bird, DeathCause.Pipe);
                    break;
                }
            }
        }
    }

    private void KillBird(FlappyBirdBird bird, DeathCause cause)
    {
        var nextPipe = GetNextPipe(bird.GlobalPosition.X);
        _deathSnapshots[bird] = new DeathSnapshot
        {
            Cause = cause,
            Steps = _birdLifetimeSteps.GetValueOrDefault(bird),
            Passes = _birdPipePasses.GetValueOrDefault(bird),
            Y = bird.GlobalPosition.Y,
            VelocityY = bird.VelocityY,
            NextPipeDx = nextPipe.dx,
            NextGapOffset = nextPipe.gapY - bird.GlobalPosition.Y,
            LastAction = _birdAgents.TryGetValue(bird, out var agent) ? agent.CurrentActionIndex : -1,
        };

        if (LogDeathEvents)
            EmitDeathLog(bird, _deathSnapshots[bird]);

        bird.Die();
        _deadCount++;
        switch (cause)
        {
            case DeathCause.Ground:  _groundDeaths++; break;
            case DeathCause.Ceiling: _ceilingDeaths++; break;
            default:                 _pipeDeaths++; break;
        }
        if (_deadCount >= _totalBirds)
        {
            IsGenerationOver = true;
            EmitGenerationSummary();
        }
    }

    // ── Called by FlappyBirdAgent.OnEpisodeBegin ──────────────────────────────

    public void OnBirdReset()
    {
        if (_generationPrepared && !IsGenerationOver) return;

        _deadCount       = 0;
        IsGenerationOver = false;
        Generation++;
        _generationPrepared = true;
        _passedSet.Clear();
        ResetGenerationDebugState();

        foreach (var bird in _birds)
            bird.ResetBird(BirdStartX, BirdStartY);

        ResetPipes();
        EmitGenerationStartLog();
    }

    // ── Pipe-passing reward helper ─────────────────────────────────────────────

    /// <summary>
    /// Returns true (once per bird per pipe) when <paramref name="bird"/> has crossed
    /// the right edge of any pipe it hasn't been rewarded for yet.
    /// </summary>
    public bool CheckAndMarkPassed(float birdX, FlappyBirdBird bird)
    {
        int birdId = bird.GetInstanceId().GetHashCode();
        foreach (var pipe in _pipes)
        {
            float right = pipe.X + PipeWidth / 2f;
            if (birdX > right && _passedSet.Add((birdId, pipe.Id)))
            {
                if (_birdPipePasses.TryGetValue(bird, out var count))
                    _birdPipePasses[bird] = count + 1;
                return true;
            }
        }
        return false;
    }

    // ── Observation helpers ───────────────────────────────────────────────────

    public (float dx, float gapY) GetNextPipe(float birdX)
    {
        foreach (var p in _pipes)
            if (p.X + PipeWidth / 2f > birdX)
                return (p.X - birdX, p.GapCenterY);
        return (PipeSpawnX, (GroundY + CeilingY) / 2f);
    }

    public (float dx, float gapY) GetSecondPipe(float birdX)
    {
        int skipped = 0;
        foreach (var p in _pipes)
        {
            if (p.X + PipeWidth / 2f > birdX)
            {
                if (skipped == 1) return (p.X - birdX, p.GapCenterY);
                skipped++;
            }
        }
        return (PipeSpawnX + PipeSpacing, (GroundY + CeilingY) / 2f);
    }

    // ── Pipe management ───────────────────────────────────────────────────────

    private float RandomGapY() =>
        ClampGapY((float)(_rng.NextDouble() * (GroundY - CeilingY - GapHeight - 80f)
                + CeilingY + 40f + GapHeight / 2f));

    private float ClampGapY(float gapY)
    {
        float minGapY = CeilingY + 40f + GapHeight / 2f;
        float maxGapY = GroundY - 40f - GapHeight / 2f;
        return Mathf.Clamp(gapY, minGapY, maxGapY);
    }

    private float InitialGapYForIndex(int pipeIndex, float previousGapY)
    {
        if (pipeIndex < EasyStartPipeCount)
            return BirdStartY;
        return NextGapY(previousGapY);
    }

    private float NextGapY(float previousGapY)
    {
        float delta = ((float)_rng.NextDouble() * 2f - 1f) * PipeGapStepLimit;
        return ClampGapY(previousGapY + delta);
    }

    private void SpawnInitialPipes()
    {
        _pipes.Clear();
        int pipeCount = Mathf.Max(2, InitialPipeCount);
        float previousGapY = BirdStartY;
        for (int i = 0; i < pipeCount; i++)
        {
            float gapY = InitialGapYForIndex(i, previousGapY);
            _pipes.Add(new PipePair
            {
                Id = _nextPipeId++,
                X = PipeSpawnX + i * PipeSpacing,
                GapCenterY = gapY
            });
            previousGapY = gapY;
        }
        RebuildPipeVisuals();
    }

    private void ResetPipes()
    {
        _pipes.Clear();
        int pipeCount = Mathf.Max(2, InitialPipeCount);
        float previousGapY = BirdStartY;
        for (int i = 0; i < pipeCount; i++)
        {
            float gapY = InitialGapYForIndex(i, previousGapY);
            _pipes.Add(new PipePair
            {
                Id = _nextPipeId++,
                X = PipeSpawnX + i * PipeSpacing,
                GapCenterY = gapY
            });
            previousGapY = gapY;
        }
        RebuildPipeVisuals();
    }

    private void ScrollPipes(double delta)
    {
        float dx = PipeScrollSpeed * (float)delta;
        foreach (var p in _pipes) p.X -= dx;
    }

    private void RecyclePipes()
    {
        bool changed = false;

        // Remove off-screen pipes and clear their pass markers to avoid unbounded growth.
        for (int i = _pipes.Count - 1; i >= 0; i--)
        {
            if (_pipes[i].X + PipeWidth / 2f < 0f)
            {
                int removedId = _pipes[i].Id;
                _pipes.RemoveAt(i);
                _passedSet.RemoveWhere(entry => entry.Item2 == removedId);
                changed = true;
            }
        }

        float rightmostX = PipeSpawnX - PipeSpacing;
        for (int i = 0; i < _pipes.Count; i++)
        {
            if (_pipes[i].X > rightmostX)
                rightmostX = _pipes[i].X;
        }

        // Keep at least one pipe queued off-screen to the right at all times.
        while (rightmostX < PipeSpawnX)
        {
            rightmostX += PipeSpacing;
            float previousGapY = _pipes.Count > 0 ? _pipes[^1].GapCenterY : BirdStartY;
            _pipes.Add(new PipePair { Id = _nextPipeId++, X = rightmostX, GapCenterY = NextGapY(previousGapY) });
            changed = true;
        }

        // Keep a minimum count for stability if a large delta removes multiple pipes.
        while (_pipes.Count < 2)
        {
            rightmostX += PipeSpacing;
            float previousGapY = _pipes.Count > 0 ? _pipes[^1].GapCenterY : BirdStartY;
            _pipes.Add(new PipePair { Id = _nextPipeId++, X = rightmostX, GapCenterY = NextGapY(previousGapY) });
            changed = true;
        }

        _pipes.Sort((a, b) => a.X.CompareTo(b.X));
        if (changed) RebuildPipeVisuals();
    }

    // ── Pipe visuals ─────────────────────────────────────────────────────────

    private void RebuildPipeVisuals()
    {
        // Grow the pool if needed — never shrink/free nodes during gameplay.
        if (_pipeContainer == null) return;
        var green = new Color(0.22f, 0.70f, 0.29f);
        while (_pipeRects.Count < _pipes.Count)
        {
            var top = new ColorRect { Color = green, Size = new Vector2(PipeWidth, 0f) };
            var bot = new ColorRect { Color = green, Size = new Vector2(PipeWidth, 0f) };
            _pipeContainer.AddChild(top);
            _pipeContainer.AddChild(bot);
            _pipeRects.Add((top, bot));
        }

        // Hide excess pool slots, position active ones.
        for (int i = 0; i < _pipeRects.Count; i++)
        {
            var (top, bot) = _pipeRects[i];
            if (i < _pipes.Count)
            {
                SyncPipeRect(_pipes[i], top, bot);
                top.Visible = true;
                bot.Visible = true;
            }
            else
            {
                top.Visible = false;
                bot.Visible = false;
            }
        }
    }

    // Each visual frame we reposition existing ColorRects to match scrolled pipe X values.
    public override void _Process(double delta)
    {
        for (int i = 0; i < _pipes.Count && i < _pipeRects.Count; i++)
            SyncPipeRect(_pipes[i], _pipeRects[i].Top, _pipeRects[i].Bot);
    }

    private void SyncPipeRect(PipePair p, ColorRect top, ColorRect bot)
    {
        float pipeLeft = p.X - PipeWidth / 2f;
        float topH     = Mathf.Max(0f, p.GapCenterY - GapHeight / 2f - CeilingY);
        float botH     = Mathf.Max(0f, GroundY - (p.GapCenterY + GapHeight / 2f));
        top.Position = new Vector2(pipeLeft, CeilingY);
        top.Size     = new Vector2(PipeWidth, topH);
        bot.Position = new Vector2(pipeLeft, p.GapCenterY + GapHeight / 2f);
        bot.Size     = new Vector2(PipeWidth, botH);
    }

    // ── Labels ────────────────────────────────────────────────────────────────

    private int _lastLabelGen  = -1;
    private int _lastLabelAlive = -1;

    private void UpdateLabels()
    {
        int alive = _totalBirds - _deadCount;
        if (_genLabel   != null && Generation != _lastLabelGen)
        {
            _genLabel.Text  = $"Generation: {Generation}";
            _lastLabelGen   = Generation;
        }
        if (_aliveLabel != null && alive != _lastLabelAlive)
        {
            _aliveLabel.Text = $"Alive: {alive} / {_totalBirds}";
            _lastLabelAlive  = alive;
        }
    }

    public bool TryGetLiveAliveCount(out int aliveCount)
    {
        aliveCount = Math.Max(0, _totalBirds - _deadCount);
        return _totalBirds > 0;
    }

    public float GetCurrentGapHeight() => GapHeight;

    private void ResetGenerationDebugState()
    {
        _groundDeaths = 0;
        _ceilingDeaths = 0;
        _pipeDeaths = 0;
        _birdLifetimeSteps.Clear();
        _birdPipePasses.Clear();
        _deathSnapshots.Clear();

        foreach (var bird in _birds)
        {
            _birdLifetimeSteps[bird] = 0;
            _birdPipePasses[bird] = 0;
        }
    }

    private void EmitGenerationStartLog()
    {
        if (!LogGenerationStarts || _totalBirds == 0) return;

        GD.Print(
            $"[FlappyBird/start] gen={Generation}" +
            $" birds={_totalBirds}" +
            $" gap={Fmt(GapHeight)}" +
            $" gap_step={Fmt(PipeGapStepLimit)}" +
            $" easy_pipes={EasyStartPipeCount}" +
            $" pipe_spacing={Fmt(PipeSpacing)}" +
            $" scroll={Fmt(PipeScrollSpeed)}" +
            $" pipe_layout={FormatPipeLayout()}");
    }

    private void EmitDeathLog(FlappyBirdBird bird, DeathSnapshot snapshot)
    {
        string birdName = DescribeBird(bird);
        int slot = GetBirdSlot(bird);

        GD.Print(
            $"[FlappyBird/death] gen={Generation}" +
            $" slot={slot}" +
            $" bird={birdName}" +
            $" cause={snapshot.Cause.ToString().ToLowerInvariant()}" +
            $" steps={snapshot.Steps}" +
            $" passes={snapshot.Passes}" +
            $" y={Fmt(snapshot.Y)}" +
            $" vy={Fmt(snapshot.VelocityY)}" +
            $" next_dx={Fmt(snapshot.NextPipeDx)}" +
            $" next_gap_offset={Fmt(snapshot.NextGapOffset)}" +
            $" last_action={snapshot.LastAction}");
    }

    private void EmitGenerationSummary()
    {
        if (!LogGenerationSummaries || _totalBirds == 0) return;

        int totalLifetimeSteps = 0;
        int bestLifetimeSteps = 0;
        int totalPipePasses = 0;
        int bestPipePasses = 0;

        foreach (var bird in _birds)
        {
            int lifetime = _birdLifetimeSteps.GetValueOrDefault(bird);
            int passes = _birdPipePasses.GetValueOrDefault(bird);
            totalLifetimeSteps += lifetime;
            totalPipePasses += passes;
            if (lifetime > bestLifetimeSteps) bestLifetimeSteps = lifetime;
            if (passes > bestPipePasses) bestPipePasses = passes;
        }

        float avgLifetimeSteps = _totalBirds > 0 ? totalLifetimeSteps / (float)_totalBirds : 0f;
        float avgPipePasses = _totalBirds > 0 ? totalPipePasses / (float)_totalBirds : 0f;
        GD.Print(
            $"[FlappyBird] gen={Generation} avg_steps={avgLifetimeSteps:F1} best_steps={bestLifetimeSteps} " +
            $"avg_passes={avgPipePasses:F2} best_passes={bestPipePasses} " +
            $"gap={GapHeight:F0} step={PipeGapStepLimit:F0} " +
            $"deaths(pipe={_pipeDeaths}, ground={_groundDeaths}, ceiling={_ceilingDeaths})");

        if (LogPipeLayouts)
            GD.Print($"[FlappyBird/pipes] gen={Generation} layout={FormatPipeLayout()}");
    }

    public void OnAgentEpisodeEnd(AcademyEpisodeEndArgs args)
    {
        if (!LogEpisodeDetails) return;
        if (args.Agent is not FlappyBirdAgent agent) return;
        if (!_agentBirds.TryGetValue(agent, out var bird)) return;

        _deathSnapshots.TryGetValue(bird, out var death);
        string rewardBreakdown = FormatRewardBreakdown(args.RewardBreakdown);

        GD.Print(
            $"[FlappyBird/episode] gen={Generation}" +
            $" slot={GetBirdSlot(bird)}" +
            $" bird={DescribeBird(bird)}" +
            $" group={args.GroupId}" +
            $" reward={Fmt(args.EpisodeReward)}" +
            $" steps={args.EpisodeSteps}" +
            $" passes={_birdPipePasses.GetValueOrDefault(bird)}" +
            $" death={death?.Cause.ToString().ToLowerInvariant() ?? "unknown"}" +
            $" death_y={Fmt(death?.Y ?? float.NaN)}" +
            $" death_vy={Fmt(death?.VelocityY ?? float.NaN)}" +
            $" death_dx={Fmt(death?.NextPipeDx ?? float.NaN)}" +
            $" death_gap_offset={Fmt(death?.NextGapOffset ?? float.NaN)}" +
            $" action_idle={agent.GetDebugIdleCount()}" +
            $" action_flap={agent.GetDebugFlapCount()}" +
            $" last_action={agent.CurrentActionIndex}" +
            $" last_obs={agent.GetLastObservationSummary()}" +
            $" reward_breakdown={rewardBreakdown}" +
            $" trace={agent.GetDecisionTrace()}");
    }

    private int GetBirdSlot(FlappyBirdBird bird)
    {
        if (_birdAgents.TryGetValue(bird, out var agent))
            return agent.GetDebugSlot();
        return -1;
    }

    private string DescribeBird(FlappyBirdBird bird) =>
        bird.GetParent()?.Name.ToString() ?? bird.Name.ToString();

    private string FormatPipeLayout()
    {
        var sb = new StringBuilder();
        for (int i = 0; i < _pipes.Count; i++)
        {
            if (i > 0) sb.Append(";");
            sb.Append($"id={_pipes[i].Id},x={Fmt(_pipes[i].X)},gap_y={Fmt(_pipes[i].GapCenterY)}");
        }
        return sb.ToString();
    }

    private static string FormatRewardBreakdown(IReadOnlyDictionary<string, float> breakdown)
    {
        if (breakdown.Count == 0) return "none";
        return string.Join("|",
            breakdown.OrderBy(kv => kv.Key)
                     .Select(kv => $"{kv.Key}={Fmt(kv.Value)}"));
    }

    private static string Fmt(float value) =>
        float.IsFinite(value)
            ? value.ToString("F3", CultureInfo.InvariantCulture)
            : "nan";
}
