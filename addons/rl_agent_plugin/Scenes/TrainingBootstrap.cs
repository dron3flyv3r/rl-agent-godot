using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace RlAgentPlugin.Runtime;

public partial class TrainingBootstrap : Node
{
    private TrainingLaunchManifest? _manifest;
    private readonly List<RLAcademy> _academies = new();
    private readonly List<EnvironmentRuntime> _environments = new();
    private List<RLAgent2D> _allTrainAgents = new();
    private RunMetricsWriter? _statusWriter;
    private readonly RandomNumberGenerator _selfPlayRng = new();

    // Per-group state
    private readonly Dictionary<string, PolicyGroupConfig> _groupConfigsByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, ResolvedPolicyGroupBinding> _groupBindingsByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, ITrainer> _trainersByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, RunMetricsWriter> _metricsWritersByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long> _episodeCountByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long> _updateCountByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float> _lastPolicyLossByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float> _lastValueLossByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float> _lastEntropyByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float?> _lastClipFractionByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _selfPlayOpponentByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float> _historicalOpponentRateByLearnerGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _frozenCheckpointIntervalByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, SelfPlayPairRuntime> _selfPlayPairsByKey = new(StringComparer.Ordinal);
    private readonly Dictionary<string, OpponentBankRuntime> _opponentBanksByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, IInferencePolicy> _frozenPoliciesBySnapshotKey = new(StringComparer.Ordinal);
    private readonly HashSet<string> _selfPlayParticipantGroups = new(StringComparer.Ordinal);
    private readonly Dictionary<string, EloTracker> _eloTrackersByGroup  = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float>      _winThresholdByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, bool>       _pfspEnabledByGroup  = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float>      _pfspAlphaByGroup    = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int>        _maxPoolSizeByGroup  = new(StringComparer.Ordinal);

    // Per-agent state
    private readonly Dictionary<RLAgent2D, AgentRuntimeState> _agentStates = new();

    private long _totalSteps;
    private double _previousTimeScale = 1.0;
    private int _previousMaxPhysicsStepsPerFrame = 8;
    private int _checkpointInterval = 10;
    private int _actionRepeat = 1;
    private int _batchSize = 1;
    private bool _showBatchGrid;
    private readonly List<SubViewport> _viewports = new();

    public override void _Ready()
    {
        _selfPlayRng.Randomize();

        _manifest = TrainingLaunchManifest.LoadFromUserStorage();
        if (_manifest is null)
        {
            GD.PushError("RL Agent Plugin could not load the training manifest.");
            GetTree().Quit(1);
            return;
        }

        var packedScene = GD.Load<PackedScene>(_manifest.ScenePath);
        if (packedScene is null)
        {
            GD.PushError($"Could not load training scene {_manifest.ScenePath}.");
            GetTree().Quit(1);
            return;
        }

        var firstSceneInstance = packedScene.Instantiate();
        var firstAcademy = FindNodeByPath(firstSceneInstance, _manifest.AcademyNodePath) as RLAcademy;
        if (firstAcademy is null)
        {
            GD.PushError("[RL] Could not resolve RLAcademy in the training scene.");
            firstSceneInstance.QueueFree();
            GetTree().Quit(1);
            return;
        }

        _batchSize = Math.Max(1, firstAcademy.BatchSize);
        _checkpointInterval = Math.Max(1, firstAcademy.CheckpointInterval);
        _actionRepeat = Math.Max(1, firstAcademy.ActionRepeat);
        _showBatchGrid = firstAcademy.ShowBatchGrid;

        _previousTimeScale = Engine.TimeScale;
        _previousMaxPhysicsStepsPerFrame = Engine.MaxPhysicsStepsPerFrame;
        var simulationSpeed = Math.Max(0.1f, firstAcademy.SimulationSpeed);
        Engine.TimeScale = simulationSpeed;
        Engine.MaxPhysicsStepsPerFrame = Math.Max(8, (int)Math.Ceiling(simulationSpeed) + 1);

        if (_manifest.BatchSize != _batchSize
            || _manifest.CheckpointInterval != _checkpointInterval
            || _manifest.ActionRepeat != _actionRepeat
            || Math.Abs(_manifest.SimulationSpeed - simulationSpeed) > 0.001f)
        {
            GD.PushWarning(
                $"[RL] Training manifest settings differed from scene academy settings. " +
                $"Using scene values: batch={_batchSize}, checkpoint_interval={_checkpointInterval}, action_repeat={_actionRepeat}, simulation_speed={simulationSpeed:0.###}.");
        }

        RLTrainingConfig? trainingConfig = null;
        RLTrainerConfig? trainerConfig = null;
        var groupedAgents = new Dictionary<string, List<RLAgent2D>>(StringComparer.Ordinal);

        for (var batchIdx = 0; batchIdx < _batchSize; batchIdx++)
        {
            var sceneInstance = batchIdx == 0 ? firstSceneInstance : packedScene.Instantiate();

            var viewport = new SubViewport();
            viewport.RenderTargetUpdateMode = SubViewport.UpdateMode.Disabled;
            viewport.HandleInputLocally = false;
            viewport.AddChild(sceneInstance);
            _viewports.Add(viewport);

            var academy = batchIdx == 0
                ? firstAcademy
                : FindNodeByPath(sceneInstance, _manifest.AcademyNodePath) as RLAcademy;
            if (academy is null)
            {
                GD.PushError($"Could not resolve RLAcademy in batch copy {batchIdx}.");
                GetTree().Quit(1);
                return;
            }

            _academies.Add(academy);

            if (batchIdx == 0)
            {
                trainingConfig = academy.TrainingConfig;
                if (trainingConfig is null && !string.IsNullOrWhiteSpace(_manifest.TrainingConfigPath))
                {
                    trainingConfig = GD.Load<RLTrainingConfig>(_manifest.TrainingConfigPath);
                }

                trainerConfig = academy.ResolveTrainerConfig()
                    ?? trainingConfig?.ToTrainerConfig()
                    ?? GD.Load<RLTrainerConfig>(_manifest.TrainerConfigPath);
            }

            var environment = new EnvironmentRuntime
            {
                Index = batchIdx,
                SceneRoot = sceneInstance,
                Academy = academy,
            };

            var batchAgents = academy.GetAgents(RLAgentControlMode.Train);
            foreach (var agent in batchAgents)
            {
                var binding = RLPolicyGroupBindingResolver.Resolve(sceneInstance, agent);
                if (binding is null)
                {
                    GD.PushError($"[RL] Agent '{agent.Name}' has no PolicyGroupConfig assigned and will not be trained.");
                    continue;
                }

                if (!environment.AgentsByGroup.TryGetValue(binding.BindingKey, out var environmentGroup))
                {
                    environmentGroup = new List<RLAgent2D>();
                    environment.AgentsByGroup[binding.BindingKey] = environmentGroup;
                }

                environmentGroup.Add(agent);

                if (!groupedAgents.TryGetValue(binding.BindingKey, out var grouped))
                {
                    grouped = new List<RLAgent2D>();
                    groupedAgents[binding.BindingKey] = grouped;
                    _groupBindingsByGroup[binding.BindingKey] = binding;
                }

                grouped.Add(agent);
            }

            _environments.Add(environment);
        }

        SetupBatchDisplay();

        if (_academies.Count == 0)
        {
            GD.PushError("No academy instances could be created.");
            GetTree().Quit(1);
            return;
        }

        var agents = _academies.SelectMany(a => a.GetAgents(RLAgentControlMode.Train)).ToList();
        if (trainerConfig is null || agents.Count == 0)
        {
            GD.PushError("Training scene is missing RL configuration or agents.");
            GetTree().Quit(1);
            return;
        }

        var algorithm = trainerConfig.Algorithm;
        foreach (var (groupId, groupAgents) in groupedAgents)
        {
            var binding = _groupBindingsByGroup[groupId];
            var firstAgent = groupAgents[0];
            if (!ObservationSizeInference.TryInferAgentObservationSize(firstAgent, out var obsSize, out var observationError))
            {
                GD.PushError($"[RL] Group '{groupId}': observation inference failed: {observationError}");
                GetTree().Quit(1);
                return;
            }

            var discreteCount = firstAgent.GetDiscreteActionCount();
            var continuousDims = firstAgent.GetContinuousActionDimensions();
            var actionDefinitions = firstAgent.GetActionSpace();

            if (algorithm == RLAlgorithmKind.PPO && continuousDims > 0)
            {
                GD.PushError($"[RL] Group '{groupId}': PPO does not support continuous actions.");
                GetTree().Quit(1);
                return;
            }

            if (algorithm == RLAlgorithmKind.PPO && discreteCount <= 0)
            {
                GD.PushError($"[RL] Group '{groupId}': PPO requires at least one discrete action.");
                GetTree().Quit(1);
                return;
            }

            if (algorithm == RLAlgorithmKind.SAC && discreteCount > 0 && continuousDims > 0)
            {
                GD.PushError($"[RL] Group '{groupId}': SAC does not support mixing discrete and continuous actions.");
                GetTree().Quit(1);
                return;
            }

            foreach (var agent in groupAgents)
            {
                if (!ObservationSizeInference.TryInferAgentObservationSize(agent, out var agentObservationSize, out observationError))
                {
                    GD.PushError($"[RL] Group '{groupId}': observation inference failed for '{agent.Name}': {observationError}");
                    GetTree().Quit(1);
                    return;
                }

                if (agentObservationSize != obsSize)
                {
                    GD.PushError($"[RL] Group '{groupId}': all agents must emit the same observation vector length.");
                    GetTree().Quit(1);
                    return;
                }

                if (algorithm == RLAlgorithmKind.PPO && agent.GetDiscreteActionCount() != discreteCount)
                {
                    GD.PushError($"[RL] Group '{groupId}': all agents must have the same discrete action count.");
                    GetTree().Quit(1);
                    return;
                }
            }

            var safeGroupId = binding.SafeGroupId;
            var checkpointPath = $"{_manifest.RunDirectory}/checkpoint__{safeGroupId}.json";
            var metricsPath = $"{_manifest.RunDirectory}/metrics__{safeGroupId}.jsonl";

            var groupConfig = new PolicyGroupConfig
            {
                GroupId = groupId,
                RunId = _manifest.RunId,
                Algorithm = algorithm,
                SharedPolicy = binding.Config,
                TrainerConfig = trainerConfig,
                NetworkGraph = binding.Config?.ResolvedNetworkGraph ?? RLNetworkGraph.CreateDefault(),
                ActionDefinitions = actionDefinitions,
                ObservationSize = obsSize,
                DiscreteActionCount = discreteCount,
                ContinuousActionDimensions = continuousDims,
                CheckpointPath = checkpointPath,
                MetricsPath = metricsPath,
            };

            _groupConfigsByGroup[groupId] = groupConfig;
            _trainersByGroup[groupId] = TrainerFactory.Create(groupConfig);
            _metricsWritersByGroup[groupId] = new RunMetricsWriter(metricsPath, _manifest.StatusPath);
            _episodeCountByGroup[groupId] = 0;
            _updateCountByGroup[groupId] = 0;
            _lastPolicyLossByGroup[groupId] = 0f;
            _lastValueLossByGroup[groupId] = 0f;
            _lastEntropyByGroup[groupId] = 0f;
            _lastClipFractionByGroup[groupId] = algorithm == RLAlgorithmKind.PPO ? 0f : null;

            GD.Print($"[RL] Group '{binding.DisplayName}': {groupAgents.Count} agent(s), {algorithm}, obs={obsSize}, discrete={discreteCount}, continuous={continuousDims}");
            GD.Print($"[RL]   Checkpoint: {checkpointPath}");
            GD.Print($"[RL]   Metrics:    {metricsPath}");
        }

        if (!TryConfigureSelfPlay(out var selfPlayError))
        {
            GD.PushError($"[RL] {selfPlayError}");
            GetTree().Quit(1);
            return;
        }

        ConfigureEnvironmentRoles();
        InitializeOpponentBanks();
        SaveInitialCheckpoints();

        foreach (var environment in _environments)
        {
            if (!TryInitializeEnvironment(environment, out var initializationError))
            {
                GD.PushError($"[RL] {initializationError}");
                GetTree().Quit(1);
                return;
            }
        }

        _allTrainAgents = _academies.SelectMany(a => a.GetAgents(RLAgentControlMode.Train)).ToList();

        _statusWriter = new RunMetricsWriter(string.Empty, _manifest.StatusPath);
        _statusWriter.WriteStatus("running", _manifest.ScenePath, _totalSteps, 0, "Training started.");
        GD.Print($"[RL] Run: {_manifest.RunId}");
        if (_batchSize > 1)
        {
            GD.Print($"[RL] Batch size: {_batchSize} ({_allTrainAgents.Count} total agents)");
        }
    }

    public override void _PhysicsProcess(double delta)
    {
        if (_academies.Count == 0 || _manifest is null || _statusWriter is null)
        {
            return;
        }

        var pendingLearningDecisionsByGroup = new Dictionary<string, List<PendingDecisionContext>>(StringComparer.Ordinal);
        var pendingFrozenDecisions = new List<PendingDecisionContext>();

        foreach (var agent in _allTrainAgents)
        {
            if (!_agentStates.TryGetValue(agent, out var state))
            {
                continue;
            }

            agent.TickStep();
            var reward = agent.ConsumePendingReward();
            var rewardBreakdown = agent.ConsumePendingRewardBreakdown();
            agent.AccumulateReward(reward, rewardBreakdown);
            state.WindowReward += reward;
            state.StepsSinceDecision++;

            var done = agent.ConsumeDonePending() || agent.HasReachedEpisodeLimit();
            if (done || state.StepsSinceDecision >= _actionRepeat)
            {
                var nextObservation = agent.CollectObservationArray();
                var pending = new PendingDecisionContext
                {
                    Agent = agent,
                    State = state,
                    TransitionObservation = nextObservation,
                    Done = done,
                };

                var role = GetEnvironmentRole(state.EnvironmentIndex, state.GroupId);
                if (role.Control == EnvironmentGroupControl.FrozenOpponent)
                {
                    pendingFrozenDecisions.Add(pending);
                }
                else
                {
                    if (!pendingLearningDecisionsByGroup.TryGetValue(state.GroupId, out var pendingGroup))
                    {
                        pendingGroup = new List<PendingDecisionContext>();
                        pendingLearningDecisionsByGroup[state.GroupId] = pendingGroup;
                    }

                    pendingGroup.Add(pending);
                }
            }
            else
            {
                ReapplyAction(agent, state);
            }
        }

        foreach (var (groupId, pendingDecisions) in pendingLearningDecisionsByGroup)
        {
            if (_trainersByGroup.TryGetValue(groupId, out var trainer))
            {
                ProcessLearningDecisions(groupId, trainer, pendingDecisions);
            }
        }

        if (pendingFrozenDecisions.Count > 0)
        {
            ProcessFrozenDecisions(pendingFrozenDecisions);
        }

        RLCheckpoint? lastCheckpoint = null;
        foreach (var (groupId, trainer) in _trainersByGroup)
        {
            var episodeCount = _episodeCountByGroup[groupId];
            var updateStats = trainer.TryUpdate(groupId, _totalSteps, episodeCount);
            if (updateStats is null)
            {
                continue;
            }

            _updateCountByGroup[groupId] += 1;
            _lastPolicyLossByGroup[groupId] = updateStats.PolicyLoss;
            _lastValueLossByGroup[groupId] = updateStats.ValueLoss;
            _lastEntropyByGroup[groupId] = updateStats.Entropy;
            _lastClipFractionByGroup[groupId] = updateStats.ClipFraction;

            var currentCheckpoint = trainer.CreateCheckpoint(groupId, _totalSteps, episodeCount, _updateCountByGroup[groupId]);
            lastCheckpoint = currentCheckpoint;

            PersistCheckpoint(groupId, currentCheckpoint, _updateCountByGroup[groupId]);
        }

        if (lastCheckpoint is not null)
        {
            foreach (var academy in _academies)
            {
                academy.Checkpoint = lastCheckpoint;
            }
        }

        var trainerConfig = _academies.Count > 0 ? _academies[0].ResolveTrainerConfig() : null;
        if (trainerConfig is not null && _totalSteps % Math.Max(1, trainerConfig.StatusWriteIntervalSteps) == 0)
        {
            var totalEpisodes = _episodeCountByGroup.Values.Sum();
            _statusWriter.WriteStatus("running", _manifest.ScenePath, _totalSteps, totalEpisodes,
                $"Training update {_updateCountByGroup.Values.Sum()}");
        }
    }

    public override void _ExitTree()
    {
        Engine.TimeScale = _previousTimeScale;
        Engine.MaxPhysicsStepsPerFrame = _previousMaxPhysicsStepsPerFrame;

        if (_manifest is null)
        {
            return;
        }

        foreach (var (groupId, trainer) in _trainersByGroup)
        {
            var episodeCount = _episodeCountByGroup.GetValueOrDefault(groupId);
            var updateCount = _updateCountByGroup.GetValueOrDefault(groupId);
            var finalCheckpoint = trainer.CreateCheckpoint(groupId, _totalSteps, episodeCount, updateCount);
            PersistCheckpoint(groupId, finalCheckpoint, updateCount, forceLatestWrite: true, allowFrozenSnapshot: false);

            foreach (var academy in _academies)
            {
                academy.Checkpoint = finalCheckpoint;
            }
        }

        var totalEpisodes = _episodeCountByGroup.Values.Sum();
        _statusWriter?.WriteStatus("stopped", _manifest.ScenePath, _totalSteps, totalEpisodes, "Training ended.");
    }

    private bool TryConfigureSelfPlay(out string error)
    {
        error = string.Empty;
        _selfPlayOpponentByGroup.Clear();
        _historicalOpponentRateByLearnerGroup.Clear();
        _frozenCheckpointIntervalByGroup.Clear();
        _selfPlayPairsByKey.Clear();
        _selfPlayParticipantGroups.Clear();
        _winThresholdByGroup.Clear();
        _pfspEnabledByGroup.Clear();
        _pfspAlphaByGroup.Clear();
        _maxPoolSizeByGroup.Clear();
        _eloTrackersByGroup.Clear();

        var configuredPairings = GetConfiguredSelfPlayPairings();
        return configuredPairings.Count == 0
            ? true
            : TryConfigureSelfPlayFromPairings(configuredPairings, out error);
    }

    private bool TryConfigureSelfPlayFromPairings(IReadOnlyList<RLPolicyPairingConfig> configuredPairings, out string error)
    {
        error = string.Empty;
        foreach (var pairing in configuredPairings)
        {
            var groupAId = pairing.ResolvedGroupA is null ? string.Empty : ResolveGroupIdForConfig(pairing.ResolvedGroupA);
            var groupBId = pairing.ResolvedGroupB is null ? string.Empty : ResolveGroupIdForConfig(pairing.ResolvedGroupB);

            if (string.IsNullOrWhiteSpace(groupAId) || string.IsNullOrWhiteSpace(groupBId))
            {
                error = $"Pairing '{ResolvePairingDisplayName(pairing)}' must reference two policy groups used by train-mode agents in the scene.";
                return false;
            }

            if (string.Equals(groupAId, groupBId, StringComparison.Ordinal))
            {
                error = $"Pairing '{ResolvePairingDisplayName(pairing)}' cannot use the same group for both sides.";
                return false;
            }

            if (!pairing.TrainGroupA && !pairing.TrainGroupB)
            {
                error = $"Pairing '{ResolvePairingDisplayName(pairing)}' must train at least one side.";
                return false;
            }

            if (_selfPlayParticipantGroups.Contains(groupAId) || _selfPlayParticipantGroups.Contains(groupBId))
            {
                error =
                    $"Groups '{GetGroupDisplayName(groupAId)}' and '{GetGroupDisplayName(groupBId)}' already belong to another self-play pairing. " +
                    "v1 supports disjoint 2-group pairings only.";
                return false;
            }

            var pairKey = MakePairKey(groupAId, groupBId);
            _selfPlayPairsByKey[pairKey] = new SelfPlayPairRuntime
            {
                PairKey = pairKey,
                GroupA = groupAId,
                GroupB = groupBId,
                TrainGroupA = pairing.TrainGroupA,
                TrainGroupB = pairing.TrainGroupB,
            };

            _selfPlayParticipantGroups.Add(groupAId);
            _selfPlayParticipantGroups.Add(groupBId);
            _frozenCheckpointIntervalByGroup[groupAId] = Math.Max(1, pairing.FrozenCheckpointInterval);
            _frozenCheckpointIntervalByGroup[groupBId] = Math.Max(1, pairing.FrozenCheckpointInterval);

            _maxPoolSizeByGroup[groupAId]  = pairing.MaxPoolSize;
            _maxPoolSizeByGroup[groupBId]  = pairing.MaxPoolSize;
            _pfspEnabledByGroup[groupAId]  = pairing.PfspEnabled;
            _pfspEnabledByGroup[groupBId]  = pairing.PfspEnabled;
            _pfspAlphaByGroup[groupAId]    = pairing.PfspAlpha;
            _pfspAlphaByGroup[groupBId]    = pairing.PfspAlpha;

            if (pairing.TrainGroupA)
            {
                _selfPlayOpponentByGroup[groupAId] = groupBId;
                _historicalOpponentRateByLearnerGroup[groupAId] = Mathf.Clamp(pairing.HistoricalOpponentRate, 0f, 1f);
                _winThresholdByGroup[groupAId] = pairing.WinThreshold;
            }

            if (pairing.TrainGroupB)
            {
                _selfPlayOpponentByGroup[groupBId] = groupAId;
                _historicalOpponentRateByLearnerGroup[groupBId] = Mathf.Clamp(pairing.HistoricalOpponentRate, 0f, 1f);
                _winThresholdByGroup[groupBId] = pairing.WinThreshold;
            }
        }

        return ValidateSelfPlayBatchSize(out error);
    }

    private bool ValidateSelfPlayBatchSize(out string error)
    {
        error = string.Empty;
        var requiredBatchCopies = _selfPlayPairsByKey.Values.Count == 0
            ? 1
            : _selfPlayPairsByKey.Values.Max(pair => pair.LearnerCount);
        if (_batchSize < requiredBatchCopies)
        {
            error =
                $"Self-play requires at least {requiredBatchCopies} batch copies for the configured rival groups, " +
                $"but BatchSize is {_batchSize}.";
            return false;
        }

        return true;
    }

    private void ConfigureEnvironmentRoles()
    {
        foreach (var environment in _environments)
        {
            foreach (var groupId in environment.AgentsByGroup.Keys)
            {
                environment.GroupRoles[groupId] = new EnvironmentGroupRole
                {
                    GroupId = groupId,
                    Control = EnvironmentGroupControl.LiveTrainer,
                };
            }

            foreach (var pair in _selfPlayPairsByKey.Values)
            {
                var learnerGroupId = pair.GetLearnerForEnvironment(environment.Index);
                var frozenGroupId = pair.GetOpponentForLearner(learnerGroupId);

                if (!environment.GroupRoles.TryGetValue(learnerGroupId, out var learnerRole)
                    || !environment.GroupRoles.TryGetValue(frozenGroupId, out var frozenRole))
                {
                    continue;
                }

                learnerRole.IsSelfPlayLearner = true;
                learnerRole.OpponentGroupId = frozenGroupId;
                learnerRole.HistoricalOpponentRate = _historicalOpponentRateByLearnerGroup.GetValueOrDefault(learnerGroupId);

                frozenRole.Control = EnvironmentGroupControl.FrozenOpponent;
                frozenRole.LearnerGroupId = learnerGroupId;
            }

            environment.NeedsMatchupRefresh = environment.GroupRoles.Values.Any(role => role.IsSelfPlayLearner);
        }
    }

    private void InitializeOpponentBanks()
    {
        foreach (var groupId in _selfPlayParticipantGroups)
        {
            var maxPool     = _maxPoolSizeByGroup.GetValueOrDefault(groupId, 20);
            var pfspEnabled = _pfspEnabledByGroup.GetValueOrDefault(groupId, true);
            var pfspAlpha   = _pfspAlphaByGroup.GetValueOrDefault(groupId, 4.0f);
            var pool        = new PolicyPool(maxPool, pfspEnabled, pfspAlpha, _selfPlayRng)
            {
                LatestCheckpointPath = GetGroupCheckpointPath(groupId) ?? string.Empty,
            };

            // Re-hydrate historical snapshots from disk (win/loss state starts fresh at Laplace prior).
            var directory = GetSelfPlayBankDirectory(groupId);
            var dir       = DirAccess.Open(directory);
            if (dir is not null)
            {
                var entries = new List<string>();
                dir.ListDirBegin();
                while (true)
                {
                    var entry = dir.GetNext();
                    if (string.IsNullOrEmpty(entry)) break;
                    if (dir.CurrentIsDir() || entry.StartsWith(".") || !entry.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
                        continue;
                    entries.Add(entry);
                }
                dir.ListDirEnd();
                entries.Sort(StringComparer.Ordinal);

                foreach (var entry in entries)
                {
                    var filePath    = $"{directory}/{entry}";
                    var snapshotKey = ExtractSnapshotKeyFromFileName(entry, filePath);
                    pool.AddSnapshot(filePath, snapshotKey, EloTracker.InitialRating);
                }
            }

            _opponentBanksByGroup[groupId] = new OpponentBankRuntime
            {
                GroupId = groupId,
                Pool    = pool,
            };
            _eloTrackersByGroup[groupId] = new EloTracker();
        }
    }

    /// <summary>
    /// Derives a snapshot key from a historical checkpoint filename.
    /// Expected format: <c>opponent__u{updateCount:D6}.json</c>
    /// </summary>
    private static string ExtractSnapshotKeyFromFileName(string fileName, string filePath)
    {
        const string prefix = "opponent__u";
        const string suffix = ".json";
        if (fileName.StartsWith(prefix, StringComparison.Ordinal)
            && fileName.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
        {
            var countStr = fileName[prefix.Length..^suffix.Length];
            if (long.TryParse(countStr, out var updateCount))
                return $"{filePath}::{updateCount}";
        }

        return $"{filePath}::0";
    }

    private void SaveInitialCheckpoints()
    {
        foreach (var (groupId, trainer) in _trainersByGroup)
        {
            var checkpoint = trainer.CreateCheckpoint(groupId, _totalSteps, _episodeCountByGroup[groupId], _updateCountByGroup[groupId]);
            PersistCheckpoint(groupId, checkpoint, _updateCountByGroup[groupId], forceLatestWrite: true, allowFrozenSnapshot: false);
        }
    }

    private bool TryInitializeEnvironment(EnvironmentRuntime environment, out string error)
    {
        error = string.Empty;

        foreach (var groupAgents in environment.AgentsByGroup.Values)
        {
            foreach (var agent in groupAgents)
            {
                agent.ResetEpisode();
            }
        }

        if (!TryRefreshEnvironmentMatchups(environment, out error))
        {
            return false;
        }

        var entropyByGroup = new Dictionary<string, float>(StringComparer.Ordinal);
        var decisionCountByGroup = new Dictionary<string, int>(StringComparer.Ordinal);

        foreach (var (groupId, groupAgents) in environment.AgentsByGroup)
        {
            var role = GetEnvironmentRole(environment.Index, groupId);
            var observations = new List<float[]>(groupAgents.Count);
            foreach (var agent in groupAgents)
            {
                observations.Add(agent.CollectObservationArray());
            }

            if (role.Control == EnvironmentGroupControl.FrozenOpponent)
            {
                if (role.FrozenPolicy is null)
                {
                    error = $"Environment {environment.Index}: frozen opponent policy for group '{GetGroupDisplayName(groupId)}' was not prepared.";
                    return false;
                }

                for (var index = 0; index < groupAgents.Count; index++)
                {
                    var agent = groupAgents[index];
                    var observation = observations[index];
                    var decision = role.FrozenPolicy.Predict(observation);
                    _agentStates[agent] = CreateAgentState(groupId, environment.Index, observation, decision, isLearningEnabled: false);
                    ApplyDecision(agent, decision);
                }

                continue;
            }

            if (!_trainersByGroup.TryGetValue(groupId, out var trainer))
            {
                error = $"Environment {environment.Index}: no trainer exists for group '{GetGroupDisplayName(groupId)}'.";
                return false;
            }

            var decisions = trainer.SampleActions(VectorBatch.FromRows(observations));
            var entropySum = 0f;
            for (var index = 0; index < groupAgents.Count; index++)
            {
                var agent = groupAgents[index];
                var observation = observations[index];
                var decision = decisions[index];
                entropySum += decision.Entropy;
                _agentStates[agent] = CreateAgentState(groupId, environment.Index, observation, decision, isLearningEnabled: true);
                ApplyDecision(agent, decision);
            }

            entropyByGroup.TryGetValue(groupId, out var currentEntropy);
            entropyByGroup[groupId] = currentEntropy + entropySum;
            decisionCountByGroup.TryGetValue(groupId, out var currentCount);
            decisionCountByGroup[groupId] = currentCount + groupAgents.Count;
        }

        foreach (var (groupId, entropySum) in entropyByGroup)
        {
            var count = Math.Max(1, decisionCountByGroup.GetValueOrDefault(groupId));
            _lastEntropyByGroup[groupId] = entropySum / count;
        }

        return true;
    }

    private void ProcessLearningDecisions(string groupId, ITrainer trainer, List<PendingDecisionContext> pendingDecisions)
    {
        var nextValues = new float[pendingDecisions.Count];
        var nonTerminalObservations = new List<float[]>();
        var nonTerminalIndices = new List<int>();

        for (var index = 0; index < pendingDecisions.Count; index++)
        {
            if (!pendingDecisions[index].Done)
            {
                nonTerminalIndices.Add(index);
                nonTerminalObservations.Add(pendingDecisions[index].TransitionObservation);
            }
        }

        if (nonTerminalObservations.Count > 0)
        {
            var estimatedValues = trainer.EstimateValues(VectorBatch.FromRows(nonTerminalObservations));
            for (var index = 0; index < nonTerminalIndices.Count; index++)
            {
                nextValues[nonTerminalIndices[index]] = estimatedValues[index];
            }
        }

        var decisionObservations = new List<float[]>(pendingDecisions.Count);
        for (var index = 0; index < pendingDecisions.Count; index++)
        {
            var pending = pendingDecisions[index];
            var state = pending.State;
            var role = GetEnvironmentRole(state.EnvironmentIndex, state.GroupId);

            var transition = new Transition
            {
                Observation = state.LastObservation,
                DiscreteAction = state.PendingAction,
                ContinuousActions = state.PendingContinuousActions,
                Reward = state.WindowReward,
                Done = pending.Done,
                NextObservation = pending.TransitionObservation,
                OldLogProbability = state.LastLogProbability,
                Value = state.LastValue,
                NextValue = pending.Done ? 0f : nextValues[index],
            };
            trainer.RecordTransition(transition);
            _totalSteps += 1;

            state.WindowReward = 0f;
            state.StepsSinceDecision = 0;

            if (pending.Done)
            {
                _episodeCountByGroup[groupId] += 1;
                var episodeCount = _episodeCountByGroup[groupId];

                if (role.IsSelfPlayLearner)
                {
                    var won   = pending.Agent.EpisodeReward > _winThresholdByGroup.GetValueOrDefault(groupId, 0f);
                    var score = won ? 1.0f : 0.0f;
                    if (_opponentBanksByGroup.TryGetValue(role.OpponentGroupId, out var ob))
                        ob.Pool.RecordOutcome(role.ActiveSnapshotKey, won);
                    if (_eloTrackersByGroup.TryGetValue(groupId, out var elo))
                    {
                        var rec = ob?.Pool.Records.FirstOrDefault(r => r.SnapshotKey == role.ActiveSnapshotKey);
                        elo.Update(rec?.SnapshotElo ?? EloTracker.InitialRating, score);
                    }
                }

                _metricsWritersByGroup[groupId].AppendMetric(
                    pending.Agent.EpisodeReward,
                    pending.Agent.EpisodeSteps,
                    _lastPolicyLossByGroup[groupId],
                    _lastValueLossByGroup[groupId],
                    _lastEntropyByGroup[groupId],
                    _lastClipFractionByGroup[groupId],
                    _totalSteps,
                    episodeCount,
                    pending.Agent.GetEpisodeRewardBreakdown(),
                    policyGroup: GetGroupDisplayName(groupId),
                    opponentGroup: role.IsSelfPlayLearner ? GetGroupDisplayName(role.OpponentGroupId) : string.Empty,
                    opponentSource: role.IsSelfPlayLearner ? role.ActiveOpponentSource : string.Empty,
                    opponentCheckpointPath: role.IsSelfPlayLearner ? role.ActiveOpponentCheckpointPath : string.Empty,
                    opponentUpdateCount: role.IsSelfPlayLearner ? role.ActiveOpponentUpdateCount : null,
                    learnerElo:  role.IsSelfPlayLearner && _eloTrackersByGroup.TryGetValue(groupId, out var ep)
                        ? ep.Rating : null,
                    poolWinRate: role.IsSelfPlayLearner && _opponentBanksByGroup.TryGetValue(role.OpponentGroupId, out var opb)
                        ? opb.Pool.AverageWinRate : null);

                PrepareEnvironmentForNextEpisode(state.EnvironmentIndex);
                if (!EnsureEnvironmentMatchupsReady(state.EnvironmentIndex))
                {
                    GD.PushWarning($"[RL] Could not refresh self-play matchup for environment {state.EnvironmentIndex}; reusing the previous opponent policy.");
                }

                pending.Agent.ResetEpisode();
                decisionObservations.Add(pending.Agent.CollectObservationArray());
            }
            else
            {
                decisionObservations.Add(pending.TransitionObservation);
            }
        }

        var decisions = trainer.SampleActions(VectorBatch.FromRows(decisionObservations));
        var entropySum = 0f;
        for (var index = 0; index < pendingDecisions.Count; index++)
        {
            var pending = pendingDecisions[index];
            var state = pending.State;
            var decision = decisions[index];
            state.LastObservation = decisionObservations[index];
            state.PendingAction = decision.DiscreteAction;
            state.PendingContinuousActions = decision.ContinuousActions;
            state.LastLogProbability = decision.LogProbability;
            state.LastValue = decision.Value;
            entropySum += decision.Entropy;
            ApplyDecision(pending.Agent, decision);
        }

        _lastEntropyByGroup[groupId] = pendingDecisions.Count > 0 ? entropySum / pendingDecisions.Count : 0f;
    }

    private void ProcessFrozenDecisions(List<PendingDecisionContext> pendingDecisions)
    {
        foreach (var pending in pendingDecisions)
        {
            var state = pending.State;
            state.WindowReward = 0f;
            state.StepsSinceDecision = 0;

            if (pending.Done)
            {
                PrepareEnvironmentForNextEpisode(state.EnvironmentIndex);
                if (!EnsureEnvironmentMatchupsReady(state.EnvironmentIndex))
                {
                    GD.PushWarning($"[RL] Could not refresh self-play matchup for environment {state.EnvironmentIndex}; reusing the previous opponent policy.");
                }

                pending.Agent.ResetEpisode();
            }

            var role = GetEnvironmentRole(state.EnvironmentIndex, state.GroupId);
            if (role.FrozenPolicy is null)
            {
                GD.PushWarning($"[RL] Frozen opponent policy missing for group '{GetGroupDisplayName(state.GroupId)}'.");
                continue;
            }

            var decisionObservation = pending.Done
                ? pending.Agent.CollectObservationArray()
                : pending.TransitionObservation;
            var decision = role.FrozenPolicy.Predict(decisionObservation);
            state.LastObservation = decisionObservation;
            state.PendingAction = decision.DiscreteAction;
            state.PendingContinuousActions = decision.ContinuousActions;
            state.LastLogProbability = 0f;
            state.LastValue = 0f;
            ApplyDecision(pending.Agent, decision);
        }
    }

    private void PersistCheckpoint(
        string groupId,
        RLCheckpoint checkpoint,
        long updateCount,
        bool forceLatestWrite = false,
        bool allowFrozenSnapshot = true)
    {
        var checkpointPath = GetGroupCheckpointPath(groupId);
        var participatesInSelfPlay = _selfPlayParticipantGroups.Contains(groupId);
        var shouldWriteLatest = forceLatestWrite
            || participatesInSelfPlay
            || updateCount == 0
            || updateCount % _checkpointInterval == 0;

        if (!string.IsNullOrEmpty(checkpointPath) && shouldWriteLatest)
        {
            RLCheckpoint.SaveToFile(checkpoint, checkpointPath);
        }

        if (!_opponentBanksByGroup.TryGetValue(groupId, out var bank) || string.IsNullOrEmpty(checkpointPath))
        {
            return;
        }

        bank.Pool.LatestCheckpointPath = checkpointPath;

        if (!allowFrozenSnapshot)
        {
            return;
        }

        var frozenInterval = Math.Max(1, _frozenCheckpointIntervalByGroup.GetValueOrDefault(groupId, 10));
        if (updateCount <= 0 || updateCount % frozenInterval != 0)
        {
            return;
        }

        var frozenPath   = $"{GetSelfPlayBankDirectory(groupId)}/opponent__u{updateCount:D6}.json";
        var snapshotKey  = $"{frozenPath}::{updateCount}";
        var currentElo   = _eloTrackersByGroup.TryGetValue(groupId, out var eloTracker)
            ? eloTracker.Rating
            : EloTracker.InitialRating;
        RLCheckpoint.SaveToFile(checkpoint, frozenPath);
        bank.Pool.AddSnapshot(frozenPath, snapshotKey, currentElo);
    }

    private void PrepareEnvironmentForNextEpisode(int environmentIndex)
    {
        if (environmentIndex < 0 || environmentIndex >= _environments.Count)
        {
            return;
        }

        _environments[environmentIndex].NeedsMatchupRefresh = _environments[environmentIndex].GroupRoles.Values.Any(role => role.IsSelfPlayLearner);
    }

    private bool EnsureEnvironmentMatchupsReady(int environmentIndex)
    {
        if (environmentIndex < 0 || environmentIndex >= _environments.Count)
        {
            return true;
        }

        var environment = _environments[environmentIndex];
        if (!environment.NeedsMatchupRefresh)
        {
            return true;
        }

        return TryRefreshEnvironmentMatchups(environment, out _);
    }

    private bool TryRefreshEnvironmentMatchups(EnvironmentRuntime environment, out string error)
    {
        error = string.Empty;
        if (!environment.NeedsMatchupRefresh)
        {
            return true;
        }

        foreach (var learnerRole in environment.GroupRoles.Values.Where(role => role.IsSelfPlayLearner))
        {
            if (!TrySelectOpponentSnapshot(learnerRole.GroupId, learnerRole.OpponentGroupId, learnerRole.HistoricalOpponentRate, out var snapshot, out error))
            {
                return false;
            }

            if (!environment.GroupRoles.TryGetValue(learnerRole.OpponentGroupId, out var frozenRole))
            {
                error = $"Environment {environment.Index}: opponent group '{GetGroupDisplayName(learnerRole.OpponentGroupId)}' is missing.";
                return false;
            }

            frozenRole.FrozenPolicy = snapshot.Policy;
            frozenRole.ActiveOpponentCheckpointPath = snapshot.CheckpointPath;
            frozenRole.ActiveOpponentSource = snapshot.Source;
            frozenRole.ActiveOpponentUpdateCount = snapshot.UpdateCount;
            frozenRole.ActiveSnapshotKey = snapshot.SnapshotKey;

            learnerRole.ActiveOpponentCheckpointPath = snapshot.CheckpointPath;
            learnerRole.ActiveOpponentSource = snapshot.Source;
            learnerRole.ActiveOpponentUpdateCount = snapshot.UpdateCount;
            learnerRole.ActiveSnapshotKey = snapshot.SnapshotKey;
        }

        environment.NeedsMatchupRefresh = false;
        return true;
    }

    private bool TrySelectOpponentSnapshot(
        string learnerGroupId,
        string opponentGroupId,
        float historicalOpponentRate,
        out LoadedInferenceSnapshot snapshot,
        out string error)
    {
        snapshot = default;
        error = string.Empty;

        if (!_opponentBanksByGroup.TryGetValue(opponentGroupId, out var bank))
        {
            error = $"Group '{GetGroupDisplayName(opponentGroupId)}' has no opponent bank.";
            return false;
        }

        var useHistorical = bank.Pool.Records.Count > 0
            && _selfPlayRng.Randf() < Mathf.Clamp(historicalOpponentRate, 0f, 1f);

        OpponentRecord? historicalRecord = null;
        string selectedPath;
        if (useHistorical)
        {
            historicalRecord = bank.Pool.SampleHistorical();
            selectedPath     = historicalRecord?.CheckpointPath ?? bank.Pool.LatestCheckpointPath;
        }
        else
        {
            selectedPath = bank.Pool.LatestCheckpointPath;
        }

        if (string.IsNullOrWhiteSpace(selectedPath))
        {
            error = $"Group '{GetGroupDisplayName(opponentGroupId)}' has no checkpoint available for self-play.";
            return false;
        }

        if (TryLoadInferenceSnapshot(selectedPath, opponentGroupId, useHistorical ? "historical" : "latest", out snapshot, out error))
        {
            return true;
        }

        if (!useHistorical || string.Equals(selectedPath, bank.Pool.LatestCheckpointPath, StringComparison.Ordinal))
        {
            error = $"Could not load opponent snapshot for learner '{GetGroupDisplayName(learnerGroupId)}': {error}";
            return false;
        }

        return TryLoadInferenceSnapshot(bank.Pool.LatestCheckpointPath, opponentGroupId, "latest", out snapshot, out error);
    }

    private bool TryLoadInferenceSnapshot(
        string checkpointPath,
        string groupId,
        string source,
        out LoadedInferenceSnapshot snapshot,
        out string error)
    {
        snapshot = default;
        error = string.Empty;

        RLCheckpoint? checkpoint = checkpointPath.EndsWith(".rlmodel", StringComparison.OrdinalIgnoreCase)
            ? RLModelLoader.LoadFromFile(checkpointPath)
            : RLCheckpoint.LoadFromFile(checkpointPath);

        if (checkpoint is null)
        {
            error = $"checkpoint '{checkpointPath}' could not be loaded.";
            return false;
        }

        var cacheKey = $"{checkpointPath}::{checkpoint.UpdateCount}";
        if (!_frozenPoliciesBySnapshotKey.TryGetValue(cacheKey, out var policy))
        {
            var fallbackGraph = _groupBindingsByGroup.TryGetValue(groupId, out var binding)
                ? binding.Config?.ResolvedNetworkGraph
                : null;

            policy = InferencePolicyFactory.Create(checkpoint, fallbackGraph);
            policy.LoadCheckpoint(checkpoint);
            _frozenPoliciesBySnapshotKey[cacheKey] = policy;
        }

        snapshot = new LoadedInferenceSnapshot
        {
            CheckpointPath = checkpointPath,
            SnapshotKey    = cacheKey,
            Source         = source,
            UpdateCount    = checkpoint.UpdateCount,
            Policy         = policy,
        };
        return true;
    }

    private List<RLPolicyPairingConfig> GetConfiguredSelfPlayPairings()
    {
        return _academies.Count > 0
            ? _academies[0].GetResolvedSelfPlayPairings()
            : new List<RLPolicyPairingConfig>();
    }

    private static string ResolvePairingDisplayName(RLPolicyPairingConfig pairing)
    {
        if (!string.IsNullOrWhiteSpace(pairing.PairingId))
        {
            return pairing.PairingId.Trim();
        }

        var groupA = pairing.ResolvedGroupA?.ResourceName;
        var groupB = pairing.ResolvedGroupB?.ResourceName;
        if (!string.IsNullOrWhiteSpace(groupA) && !string.IsNullOrWhiteSpace(groupB))
        {
            return $"{groupA.Trim()} vs {groupB.Trim()}";
        }

        return "<unnamed pairing>";
    }

    private string ResolveGroupIdForConfig(RLPolicyGroupConfig config)
    {
        foreach (var (groupId, binding) in _groupBindingsByGroup)
        {
            if (ReferenceEquals(binding.Config, config))
            {
                return groupId;
            }
        }

        if (!string.IsNullOrWhiteSpace(config.ResourcePath))
        {
            foreach (var (groupId, binding) in _groupBindingsByGroup)
            {
                if (string.Equals(binding.ConfigPath, config.ResourcePath, StringComparison.Ordinal))
                {
                    return groupId;
                }
            }
        }

        return string.Empty;
    }

    private string GetGroupDisplayName(string groupId)
    {
        return _groupBindingsByGroup.TryGetValue(groupId, out var binding)
            ? binding.DisplayName
            : groupId;
    }

    private EnvironmentGroupRole GetEnvironmentRole(int environmentIndex, string groupId)
    {
        var environment = _environments[environmentIndex];
        return environment.GroupRoles[groupId];
    }

    private string GetSelfPlayBankDirectory(string groupId)
    {
        var safeId = _groupBindingsByGroup.TryGetValue(groupId, out var binding)
            ? binding.SafeGroupId
            : RLPolicyGroupBindingResolver.MakeSafeGroupId(groupId);
        return $"{_manifest?.RunDirectory}/selfplay/{safeId}";
    }

    private string? GetGroupCheckpointPath(string groupId)
    {
        var safeId = _groupBindingsByGroup.TryGetValue(groupId, out var binding)
            ? binding.SafeGroupId
            : RLPolicyGroupBindingResolver.MakeSafeGroupId(groupId);
        return $"{_manifest?.RunDirectory}/checkpoint__{safeId}.json";
    }

    private static void ApplyDecision(RLAgent2D agent, PolicyDecision decision)
    {
        if (decision.DiscreteAction >= 0)
        {
            agent.ApplyAction(decision.DiscreteAction);
        }
        else if (decision.ContinuousActions.Length > 0)
        {
            agent.ApplyAction(decision.ContinuousActions);
        }
    }

    private static void ReapplyAction(RLAgent2D agent, AgentRuntimeState state)
    {
        if (state.PendingAction >= 0)
        {
            agent.ApplyAction(state.PendingAction);
        }
        else if (state.PendingContinuousActions.Length > 0)
        {
            agent.ApplyAction(state.PendingContinuousActions);
        }
    }

    private static AgentRuntimeState CreateAgentState(string groupId, int environmentIndex, float[] observation, PolicyDecision decision, bool isLearningEnabled)
    {
        return new AgentRuntimeState
        {
            GroupId = groupId,
            EnvironmentIndex = environmentIndex,
            IsLearningEnabled = isLearningEnabled,
            LastObservation = observation,
            PendingAction = decision.DiscreteAction,
            PendingContinuousActions = decision.ContinuousActions,
            LastLogProbability = decision.LogProbability,
            LastValue = decision.Value,
        };
    }

    private static string MakePairKey(string left, string right)
    {
        return string.CompareOrdinal(left, right) <= 0
            ? $"{left}|{right}"
            : $"{right}|{left}";
    }

    private void SetupBatchDisplay()
    {
        var canvasLayer = new CanvasLayer();
        AddChild(canvasLayer);

        if (!_showBatchGrid || _viewports.Count == 1)
        {
            // Single view: only env 0 renders, fullscreen. All others are update-disabled orphans.
            var container = new SubViewportContainer();
            container.Stretch = true;
            container.SetAnchorsAndOffsetsPreset(Control.LayoutPreset.FullRect);
            _viewports[0].RenderTargetUpdateMode = SubViewport.UpdateMode.Always;
            container.AddChild(_viewports[0]);
            canvasLayer.AddChild(container);

            for (var i = 1; i < _viewports.Count; i++)
            {
                AddChild(_viewports[i]);
            }
        }
        else
        {
            // Grid view: all envs rendered at full resolution, displayed scaled-to-fit with padding.
            var cols = (int)Math.Ceiling(Math.Sqrt(_viewports.Count));
            var rows = (int)Math.Ceiling((float)_viewports.Count / cols);
            var windowSize = DisplayServer.WindowGetSize();
            const int padding = 8;
            var cellW = (windowSize.X - padding * (cols + 1)) / cols;
            var cellH = (windowSize.Y - padding * (rows + 1)) / rows;

            var root = new Control();
            root.SetAnchorsAndOffsetsPreset(Control.LayoutPreset.FullRect);
            canvasLayer.AddChild(root);

            for (var i = 0; i < _viewports.Count; i++)
            {
                var col = i % cols;
                var row = i / cols;

                // Render at full window resolution so the camera sees the whole scene.
                _viewports[i].Size = windowSize;
                _viewports[i].RenderTargetUpdateMode = SubViewport.UpdateMode.Always;
                AddChild(_viewports[i]);

                // Display the viewport texture scaled to fit the cell, keeping aspect ratio.
                var rect = new TextureRect();
                rect.Texture = _viewports[i].GetTexture();
                rect.ExpandMode = TextureRect.ExpandModeEnum.IgnoreSize;
                rect.StretchMode = TextureRect.StretchModeEnum.KeepAspectCentered;
                rect.Position = new Vector2(padding + col * (cellW + padding), padding + row * (cellH + padding));
                rect.Size = new Vector2(cellW, cellH);
                root.AddChild(rect);
            }
        }
    }

    private static Node? FindNodeByPath(Node root, string nodePath)
    {
        if (string.IsNullOrWhiteSpace(nodePath))
        {
            return null;
        }

        return root.GetNodeOrNull(new NodePath(nodePath));
    }

    private sealed class AgentRuntimeState
    {
        public string GroupId { get; init; } = string.Empty;
        public int EnvironmentIndex { get; init; }
        public bool IsLearningEnabled { get; init; }
        public float[] LastObservation { get; set; } = Array.Empty<float>();
        public int PendingAction { get; set; } = -1;
        public float[] PendingContinuousActions { get; set; } = Array.Empty<float>();
        public float LastLogProbability { get; set; }
        public float LastValue { get; set; }
        public float WindowReward { get; set; }
        public int StepsSinceDecision { get; set; }
    }

    private sealed class PendingDecisionContext
    {
        public RLAgent2D Agent { get; init; } = null!;
        public AgentRuntimeState State { get; init; } = null!;
        public float[] TransitionObservation { get; init; } = Array.Empty<float>();
        public bool Done { get; init; }
    }

    private sealed class EnvironmentRuntime
    {
        public int Index { get; init; }
        public Node SceneRoot { get; init; } = null!;
        public RLAcademy Academy { get; init; } = null!;
        public Dictionary<string, List<RLAgent2D>> AgentsByGroup { get; } = new(StringComparer.Ordinal);
        public Dictionary<string, EnvironmentGroupRole> GroupRoles { get; } = new(StringComparer.Ordinal);
        public bool NeedsMatchupRefresh { get; set; }
    }

    private sealed class EnvironmentGroupRole
    {
        public string GroupId { get; init; } = string.Empty;
        public EnvironmentGroupControl Control { get; set; } = EnvironmentGroupControl.LiveTrainer;
        public bool IsSelfPlayLearner { get; set; }
        public string OpponentGroupId { get; set; } = string.Empty;
        public string LearnerGroupId { get; set; } = string.Empty;
        public float HistoricalOpponentRate { get; set; }
        public IInferencePolicy? FrozenPolicy { get; set; }
        public string ActiveOpponentCheckpointPath { get; set; } = string.Empty;
        public string ActiveOpponentSource { get; set; } = string.Empty;
        public long? ActiveOpponentUpdateCount { get; set; }
        public string ActiveSnapshotKey { get; set; } = string.Empty;
    }

    private enum EnvironmentGroupControl
    {
        LiveTrainer = 0,
        FrozenOpponent = 1,
    }

    private sealed class OpponentBankRuntime
    {
        public string GroupId { get; init; } = string.Empty;
        public PolicyPool Pool { get; init; } = null!;
    }

    private sealed class SelfPlayPairRuntime
    {
        public string PairKey { get; init; } = string.Empty;
        public string GroupA { get; init; } = string.Empty;
        public string GroupB { get; init; } = string.Empty;
        public bool TrainGroupA { get; set; }
        public bool TrainGroupB { get; set; }

        public int LearnerCount => (TrainGroupA ? 1 : 0) + (TrainGroupB ? 1 : 0);

        public string GetLearnerForEnvironment(int environmentIndex)
        {
            if (TrainGroupA && TrainGroupB)
            {
                return environmentIndex % 2 == 0 ? GroupA : GroupB;
            }

            return TrainGroupA ? GroupA : GroupB;
        }

        public string GetOpponentForLearner(string learnerGroupId)
        {
            return string.Equals(learnerGroupId, GroupA, StringComparison.Ordinal)
                ? GroupB
                : GroupA;
        }
    }

    private readonly struct LoadedInferenceSnapshot
    {
        public string CheckpointPath { get; init; }
        public string SnapshotKey { get; init; }
        public string Source { get; init; }
        public long UpdateCount { get; init; }
        public IInferencePolicy Policy { get; init; }
    }
}
