using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

public enum StandaloneControlledAgent
{
    None = 0,
    ChaserA = 1,
    ChaserB = 2,
    RunnerA = 3,
    RunnerB = 4,
}

public partial class TagArenaController : Node2D
{
    [ExportGroup("Arena")]
    [Export] public Vector2 ArenaMin { get; set; } = new(96.0f, 96.0f);
    [Export] public Vector2 ArenaMax { get; set; } = new(704.0f, 416.0f);
    [Export] public float TagRadius { get; set; } = 30.0f;
    [Export] public float SpawnMargin { get; set; } = 48.0f;
    [Export] public float MinimumSpawnSeparation { get; set; } = 72.0f;

    [ExportGroup("Episode")]
    [Export] public int EpisodeStepLimit { get; set; } = 600;
    [Export] public float ChaserDistanceRewardScale { get; set; } = 0.02f;
    [Export] public float RunnerDistanceRewardScale { get; set; } = 0.02f;
    [Export] public float ChaserStepPenalty { get; set; } = 0.005f;
    [Export] public float RunnerSurvivalReward { get; set; } = 0.003f;
    [Export] public float ChaserTagReward { get; set; } = 1.25f;
    [Export] public float RunnerTaggedPenalty { get; set; } = -1.25f;
    [Export] public float RunnerTimeoutReward { get; set; } = 0.6f;
    [Export] public float ChaserTimeoutPenalty { get; set; } = -0.6f;

    [ExportGroup("Standalone")]
    [Export] public StandaloneControlledAgent ControlledAgent { get; set; } = StandaloneControlledAgent.RunnerA;
    [Export] public bool AllowStandaloneReset { get; set; } = true;

    private readonly RandomNumberGenerator _rng = new();
    private readonly List<TagPlayer> _players = new();
    private readonly Dictionary<TagPlayer, Dictionary<string, float>> _stepRewardComponents = new();
    private readonly Dictionary<TagPlayer, bool> _resetRequested = new();
    private readonly Dictionary<TagPlayer, Vector2> _terminalPositions = new();

    private int _episodeStep;
    private bool _episodeResolved;
    private bool _nextEpisodePrepared;
    private string _resolutionMessage = "Episode ready.";

    private Label? _modeLabel;
    private Label? _statusLabel;
    private Label? _footerLabel;

    public override void _EnterTree()
    {
        _players.Clear();
        DiscoverPlayers(this);
        ConfigureStandaloneControl();
    }

    public override void _Ready()
    {
        _rng.Randomize();

        foreach (var player in _players)
        {
            _stepRewardComponents[player] = new Dictionary<string, float>(StringComparer.Ordinal);
            _resetRequested[player] = false;
        }

        _modeLabel = GetNodeOrNull<Label>("CanvasLayer/Panel/Margin/VBox/ModeLabel");
        _statusLabel = GetNodeOrNull<Label>("CanvasLayer/Panel/Margin/VBox/StatusLabel");
        _footerLabel = GetNodeOrNull<Label>("CanvasLayer/Panel/Margin/VBox/FooterLabel");
        BeginEpisode("Episode ready.");
    }

    public override void _PhysicsProcess(double delta)
    {
        if (_players.Count == 0)
        {
            return;
        }

        if (!IsTrainingRun() && AllowStandaloneReset && Input.IsActionJustPressed("ui_accept"))
        {
            ResetStandaloneEpisode();
        }

        if (_episodeResolved)
        {
            UpdateHud();
            return;
        }

        var beforePositions = CaptureCurrentPositions();
        var beforeNearestOpponentDistance = ComputeNearestOpponentDistances(beforePositions);

        foreach (var player in _players)
        {
            player.StepMovement(this, delta);
            _stepRewardComponents[player].Clear();
        }

        _episodeStep += 1;

        var afterPositions = CaptureCurrentPositions();
        var afterNearestOpponentDistance = ComputeNearestOpponentDistances(afterPositions);

        foreach (var player in _players)
        {
            var beforeDistance = beforeNearestOpponentDistance[player];
            var afterDistance = afterNearestOpponentDistance[player];

            if (player.Role == TagAgentRole.Chaser)
            {
                var progress = beforeDistance - afterDistance;
                AddStepReward(player, progress * ChaserDistanceRewardScale, "distance_progress");
                AddStepReward(player, -ChaserStepPenalty, "step_penalty");
            }
            else
            {
                var escape = afterDistance - beforeDistance;
                AddStepReward(player, escape * RunnerDistanceRewardScale, "distance_escape");
                AddStepReward(player, RunnerSurvivalReward, "survival_reward");
            }
        }

        var tagPair = FindFirstTag(afterPositions);
        if (tagPair is not null)
        {
            ResolveTag(tagPair.Value.chaser, tagPair.Value.runner, afterPositions);
        }
        else if (_episodeStep >= Math.Max(1, EpisodeStepLimit))
        {
            ResolveTimeout(afterPositions);
        }

        UpdateHud();
    }

    public bool IsTrainingSession => IsTrainingRun();

    public void CollectObservations(TagPlayer self, ObservationBuffer obs)
    {
        if (!_players.Contains(self))
        {
            return;
        }

        var positions = ShouldUseTerminalSnapshot(self) ? _terminalPositions : CaptureCurrentPositions();
        if (!positions.TryGetValue(self, out var selfPosition))
        {
            selfPosition = self.Position;
        }

        var arenaSize = new Vector2(
            Math.Max(1.0f, ArenaMax.X - ArenaMin.X),
            Math.Max(1.0f, ArenaMax.Y - ArenaMin.Y));

        obs.AddNormalized(selfPosition, ArenaMin, ArenaMax);

        foreach (var delta in GetRelativePositions(self, positions, self.Role == TagAgentRole.Chaser
                     ? TagAgentRole.Runner
                     : TagAgentRole.Chaser, 2))
        {
            obs.AddNormalized(delta.X, -arenaSize.X, arenaSize.X);
            obs.AddNormalized(delta.Y, -arenaSize.Y, arenaSize.Y);
        }

        foreach (var delta in GetRelativePositions(self, positions, self.Role, 1))
        {
            obs.AddNormalized(delta.X, -arenaSize.X, arenaSize.X);
            obs.AddNormalized(delta.Y, -arenaSize.Y, arenaSize.Y);
        }

        var remainingSteps = _episodeResolved && ShouldUseTerminalSnapshot(self)
            ? 0
            : Math.Max(0, EpisodeStepLimit - _episodeStep);
        obs.AddNormalized(remainingSteps, 0, Math.Max(1, EpisodeStepLimit));
        obs.Add(self.Role == TagAgentRole.Chaser);
    }

    public IReadOnlyDictionary<string, float> ConsumeStepRewardBreakdown(TagPlayer player)
    {
        if (_resetRequested.GetValueOrDefault(player))
        {
            return new Dictionary<string, float>(StringComparer.Ordinal);
        }

        if (!_stepRewardComponents.TryGetValue(player, out var breakdown))
        {
            return new Dictionary<string, float>(StringComparer.Ordinal);
        }

        return new Dictionary<string, float>(breakdown, StringComparer.Ordinal);
    }

    public bool IsEpisodeResolvedFor(TagPlayer player)
    {
        return _episodeResolved && !_resetRequested.GetValueOrDefault(player);
    }

    public void HandleAgentEpisodeBegin(TagPlayer player)
    {
        if (!_players.Contains(player) || !_episodeResolved || _resetRequested.GetValueOrDefault(player))
        {
            return;
        }

        _resetRequested[player] = true;
        PrepareNextEpisodeIfNeeded();

        if (_resetRequested.Values.All(value => value))
        {
            FinalizePreparedEpisode();
        }
    }

    private void ResetStandaloneEpisode()
    {
        BeginEpisode("Standalone reset.");
        foreach (var player in _players)
        {
            player.Agent?.ResetEpisode();
        }
    }

    private void BeginEpisode(string message)
    {
        _episodeStep = 0;
        _episodeResolved = false;
        _nextEpisodePrepared = false;
        _resolutionMessage = message;
        _terminalPositions.Clear();

        foreach (var player in _players)
        {
            _stepRewardComponents[player].Clear();
            _resetRequested[player] = false;
        }

        foreach (var (player, spawnPosition) in CreateSpawnPositions())
        {
            player.SetSpawnPosition(spawnPosition);
        }

        UpdateHud();
    }

    private void PrepareNextEpisodeIfNeeded()
    {
        if (_nextEpisodePrepared)
        {
            return;
        }

        foreach (var (player, spawnPosition) in CreateSpawnPositions())
        {
            player.SetSpawnPosition(spawnPosition);
        }

        _episodeStep = 0;
        _nextEpisodePrepared = true;
    }

    private void FinalizePreparedEpisode()
    {
        _episodeResolved = false;
        _nextEpisodePrepared = false;
        _resolutionMessage = "New episode.";
        _terminalPositions.Clear();

        foreach (var player in _players)
        {
            _stepRewardComponents[player].Clear();
            _resetRequested[player] = false;
        }

        UpdateHud();
    }

    private void ResolveTag(TagPlayer chaser, TagPlayer runner, IReadOnlyDictionary<TagPlayer, Vector2> positions)
    {
        _episodeResolved = true;
        CaptureTerminalPositions(positions);

        foreach (var player in _players)
        {
            if (player.Role == TagAgentRole.Chaser)
            {
                AddStepReward(player, ChaserTagReward, "tag_reward");
            }
            else
            {
                AddStepReward(player, RunnerTaggedPenalty, "tagged_penalty");
            }
        }

        _resolutionMessage = $"{chaser.PlayerId} tagged {runner.PlayerId} on step {_episodeStep}.";
    }

    private void ResolveTimeout(IReadOnlyDictionary<TagPlayer, Vector2> positions)
    {
        _episodeResolved = true;
        CaptureTerminalPositions(positions);

        foreach (var player in _players)
        {
            if (player.Role == TagAgentRole.Chaser)
            {
                AddStepReward(player, ChaserTimeoutPenalty, "timeout_penalty");
            }
            else
            {
                AddStepReward(player, RunnerTimeoutReward, "timeout_reward");
            }
        }

        _resolutionMessage = $"Runners survived {_episodeStep} steps.";
    }

    private void CaptureTerminalPositions(IReadOnlyDictionary<TagPlayer, Vector2> positions)
    {
        _terminalPositions.Clear();
        foreach (var (player, position) in positions)
        {
            _terminalPositions[player] = position;
            _resetRequested[player] = false;
        }

        _nextEpisodePrepared = false;
    }

    private void DiscoverPlayers(Node node)
    {
        if (node is TagPlayer player)
        {
            _players.Add(player);
        }

        foreach (var child in node.GetChildren())
        {
            if (child is Node childNode)
            {
                DiscoverPlayers(childNode);
            }
        }
    }

    private void AddStepReward(TagPlayer player, float amount, string tag)
    {
        if (!_stepRewardComponents.TryGetValue(player, out var breakdown))
        {
            breakdown = new Dictionary<string, float>(StringComparer.Ordinal);
            _stepRewardComponents[player] = breakdown;
        }

        breakdown.TryGetValue(tag, out var currentAmount);
        breakdown[tag] = currentAmount + amount;
    }

    private void ConfigureStandaloneControl()
    {
        if (IsTrainingRun())
        {
            return;
        }

        var controlledAgentId = ControlledAgent switch
        {
            StandaloneControlledAgent.ChaserA => "ChaserA",
            StandaloneControlledAgent.ChaserB => "ChaserB",
            StandaloneControlledAgent.RunnerA => "RunnerA",
            StandaloneControlledAgent.RunnerB => "RunnerB",
            _ => string.Empty,
        };

        foreach (var player in _players)
        {
            var agent = player.Agent ?? player.GetNodeOrNull<TagAgent>("Agent");
            if (agent is null)
            {
                continue;
            }

            if (!string.IsNullOrEmpty(controlledAgentId) && player.PlayerId == controlledAgentId)
            {
                agent.ControlMode = RLAgentControlMode.Human;
            }
            else if (agent.ControlMode == RLAgentControlMode.Train)
            {
                agent.ControlMode = RLAgentControlMode.Inference;
            }
        }
    }

    private Dictionary<TagPlayer, Vector2> CreateSpawnPositions()
    {
        var spawnPositions = new Dictionary<TagPlayer, Vector2>();
        foreach (var player in _players.OrderBy(player => player.Role).ThenBy(player => player.PlayerId))
        {
            var attempts = 0;
            var spawn = Vector2.Zero;

            do
            {
                attempts += 1;
                var horizontalPadding = Math.Max(0.0f, SpawnMargin);
                var verticalPadding = Math.Max(0.0f, SpawnMargin);
                var midX = (ArenaMin.X + ArenaMax.X) * 0.5f;

                var minX = player.Role == TagAgentRole.Chaser ? ArenaMin.X + horizontalPadding : midX + horizontalPadding * 0.25f;
                var maxX = player.Role == TagAgentRole.Chaser ? midX - horizontalPadding * 0.25f : ArenaMax.X - horizontalPadding;

                spawn = new Vector2(
                    _rng.RandfRange(Math.Min(minX, maxX), Math.Max(minX, maxX)),
                    _rng.RandfRange(ArenaMin.Y + verticalPadding, ArenaMax.Y - verticalPadding));
            }
            while (attempts < 32 && spawnPositions.Values.Any(existing => existing.DistanceTo(spawn) < MinimumSpawnSeparation));

            spawnPositions[player] = spawn;
        }

        return spawnPositions;
    }

    private Dictionary<TagPlayer, Vector2> CaptureCurrentPositions()
    {
        var positions = new Dictionary<TagPlayer, Vector2>(_players.Count);
        foreach (var player in _players)
        {
            positions[player] = player.Position;
        }

        return positions;
    }

    private Dictionary<TagPlayer, float> ComputeNearestOpponentDistances(IReadOnlyDictionary<TagPlayer, Vector2> positions)
    {
        var result = new Dictionary<TagPlayer, float>(_players.Count);
        foreach (var player in _players)
        {
            var nearestDistance = float.MaxValue;
            foreach (var other in _players)
            {
                if (other == player || other.Role == player.Role)
                {
                    continue;
                }

                nearestDistance = Math.Min(nearestDistance, positions[player].DistanceTo(positions[other]));
            }

            result[player] = nearestDistance == float.MaxValue ? 0f : nearestDistance;
        }

        return result;
    }

    private (TagPlayer chaser, TagPlayer runner)? FindFirstTag(IReadOnlyDictionary<TagPlayer, Vector2> positions)
    {
        foreach (var chaser in _players.Where(player => player.Role == TagAgentRole.Chaser))
        {
            foreach (var runner in _players.Where(player => player.Role == TagAgentRole.Runner))
            {
                if (positions[chaser].DistanceTo(positions[runner]) <= TagRadius)
                {
                    return (chaser, runner);
                }
            }
        }

        return null;
    }

    private IEnumerable<Vector2> GetRelativePositions(
        TagPlayer self,
        IReadOnlyDictionary<TagPlayer, Vector2> positions,
        TagAgentRole role,
        int count)
    {
        var selfPosition = positions[self];
        var ordered = _players
            .Where(player => player != self && player.Role == role)
            .OrderBy(player => selfPosition.DistanceTo(positions[player]))
            .Take(count)
            .Select(player => positions[player] - selfPosition)
            .ToList();

        while (ordered.Count < count)
        {
            ordered.Add(Vector2.Zero);
        }

        return ordered;
    }

    public Vector2 ClampToArena(Vector2 position)
    {
        return new Vector2(
            Mathf.Clamp(position.X, ArenaMin.X, ArenaMax.X),
            Mathf.Clamp(position.Y, ArenaMin.Y, ArenaMax.Y));
    }

    private bool ShouldUseTerminalSnapshot(TagPlayer player)
    {
        return _episodeResolved
            && !_resetRequested.GetValueOrDefault(player)
            && _terminalPositions.ContainsKey(player);
    }

    private bool IsTrainingRun()
    {
        var current = GetParent();
        while (current is not null)
        {
            if (current.GetType().Name == "TrainingBootstrap")
            {
                return true;
            }

            current = current.GetParent();
        }

        return false;
    }

    private void UpdateHud()
    {
        if (_modeLabel is not null)
        {
            var controlledAgentName = ControlledAgent == StandaloneControlledAgent.None
                ? "none"
                : ControlledAgent.ToString();
            _modeLabel.Text = IsTrainingRun()
                ? "Mode: training self-play"
                : $"Mode: standalone, player = {controlledAgentName}";
        }

        if (_statusLabel is not null)
        {
            _statusLabel.Text = $"Step {_episodeStep}/{EpisodeStepLimit} | {_resolutionMessage}";
        }

        if (_footerLabel is not null)
        {
            _footerLabel.Text = IsTrainingRun()
                ? "Training uses four train-mode agents with separate policies and shared arena resets."
                : ControlledAgent == StandaloneControlledAgent.None
                    ? "Controls: Enter resets the arena. Spy overlay stays visible; set ControlledAgent to a player to enable manual control."
                    : "Controls: arrows move the selected human agent, Enter resets the arena, Tab cycles the spy overlay.";
        }
    }
}
