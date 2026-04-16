using System;
using System.Collections.Generic;
using Godot;

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
public partial class FlappyBirdController : Node2D
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

    // ── Public state ──────────────────────────────────────────────────────────

    /// <summary>True once every bird has died. Agents end their episodes when this is set.</summary>
    public bool IsGenerationOver { get; private set; }

    public int Generation { get; private set; }

    // ── Private state ─────────────────────────────────────────────────────────

    private readonly List<FlappyBirdBird>  _birds   = new();
    private readonly List<PipePair>        _pipes   = new();
    private Node2D? _pipeContainer;
    private Label?  _genLabel;
    private Label?  _aliveLabel;

    private int _totalBirds;
    private int _deadCount;
    private int _resetCount;

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
                    if (bird != null) _birds.Add(bird);
                }
                continue;
            }

            // Manual placement: BirdGroup nodes directly under the controller
            if (child is Node2D group)
            {
                var bird = group.GetNodeOrNull<FlappyBirdBird>("Bird");
                if (bird != null) _birds.Add(bird);
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
            bird.Update(delta);
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
                KillBird(bird);
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
                    KillBird(bird);
                    break;
                }
            }
        }
    }

    private void KillBird(FlappyBirdBird bird)
    {
        bird.Die();
        _deadCount++;
        if (_deadCount >= _totalBirds)
            IsGenerationOver = true;
    }

    // ── Called by FlappyBirdAgent.OnEpisodeBegin ──────────────────────────────

    public void OnBirdReset()
    {
        _resetCount++;
        if (_resetCount < _totalBirds) return;

        _resetCount      = 0;
        _deadCount       = 0;
        IsGenerationOver = false;
        Generation++;
        _passedSet.Clear();

        foreach (var bird in _birds)
            bird.ResetBird(BirdStartX, BirdStartY);

        ResetPipes();
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
                return true;
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
        (float)(_rng.NextDouble() * (GroundY - CeilingY - GapHeight - 80f)
                + CeilingY + 40f + GapHeight / 2f);

    private void SpawnInitialPipes()
    {
        _pipes.Clear();
        _pipes.Add(new PipePair { Id = _nextPipeId++, X = PipeSpawnX,              GapCenterY = RandomGapY() });
        _pipes.Add(new PipePair { Id = _nextPipeId++, X = PipeSpawnX + PipeSpacing, GapCenterY = RandomGapY() });
        RebuildPipeVisuals();
    }

    private void ResetPipes()
    {
        _pipes.Clear();
        _pipes.Add(new PipePair { Id = _nextPipeId++, X = PipeSpawnX,              GapCenterY = RandomGapY() });
        _pipes.Add(new PipePair { Id = _nextPipeId++, X = PipeSpawnX + PipeSpacing, GapCenterY = RandomGapY() });
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
        for (int i = _pipes.Count - 1; i >= 0; i--)
        {
            if (_pipes[i].X + PipeWidth / 2f < 0f)
            {
                _pipes.RemoveAt(i);
                _pipes.Add(new PipePair { Id = _nextPipeId++, X = PipeSpawnX, GapCenterY = RandomGapY() });
                changed = true;
            }
        }
        _pipes.Sort((a, b) => a.X.CompareTo(b.X));
        if (changed) RebuildPipeVisuals();
    }

    // ── Pipe visuals ─────────────────────────────────────────────────────────

    private void RebuildPipeVisuals()
    {
        // Called on topology changes (reset, recycle). No-op if container missing.
        if (_pipeContainer == null) return;
        foreach (Node child in _pipeContainer.GetChildren())
            child.QueueFree();

        var green = new Color(0.22f, 0.70f, 0.29f);
        foreach (var p in _pipes)
        {
            float topH     = Mathf.Max(0f, p.GapCenterY - GapHeight / 2f - CeilingY);
            float botH     = Mathf.Max(0f, GroundY - (p.GapCenterY + GapHeight / 2f));
            float pipeLeft = p.X - PipeWidth / 2f;

            _pipeContainer.AddChild(new ColorRect
            {
                Color = green, Size = new Vector2(PipeWidth, topH),
                Position = new Vector2(pipeLeft, CeilingY),
                Name = $"PipeTop_{p.Id}"
            });
            _pipeContainer.AddChild(new ColorRect
            {
                Color = green, Size = new Vector2(PipeWidth, botH),
                Position = new Vector2(pipeLeft, p.GapCenterY + GapHeight / 2f),
                Name = $"PipeBot_{p.Id}"
            });
        }
    }

    // Each visual frame we reposition existing ColorRects to match scrolled pipe X values.
    public override void _Process(double delta)
    {
        if (_pipeContainer == null) return;
        foreach (var p in _pipes)
        {
            float pipeLeft = p.X - PipeWidth / 2f;
            var top = _pipeContainer.GetNodeOrNull<ColorRect>($"PipeTop_{p.Id}");
            var bot = _pipeContainer.GetNodeOrNull<ColorRect>($"PipeBot_{p.Id}");
            if (top != null) top.Position = new Vector2(pipeLeft, CeilingY);
            if (bot != null) bot.Position = new Vector2(pipeLeft, p.GapCenterY + GapHeight / 2f);
        }
    }

    // ── Labels ────────────────────────────────────────────────────────────────

    private void UpdateLabels()
    {
        int alive = _totalBirds - _deadCount;
        if (_genLabel   != null) _genLabel.Text   = $"Generation: {Generation}";
        if (_aliveLabel != null) _aliveLabel.Text = $"Alive: {alive} / {_totalBirds}";
    }
}
