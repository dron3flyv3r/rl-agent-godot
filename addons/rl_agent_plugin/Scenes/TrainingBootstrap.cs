using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace RlAgentPlugin.Runtime;

public partial class TrainingBootstrap : Node
{
    private TrainingLaunchManifest? _manifest;
    private readonly List<RLAcademy> _academies = new();
    private List<RLAgent2D> _allTrainAgents = new();
    private RunMetricsWriter? _statusWriter;

    // Per-group state
    private readonly Dictionary<string, ITrainer> _trainersByGroup = new();
    private readonly Dictionary<string, RunMetricsWriter> _metricsWritersByGroup = new();
    private readonly Dictionary<string, long> _episodeCountByGroup = new();
    private readonly Dictionary<string, long> _updateCountByGroup = new();
    private readonly Dictionary<string, float> _lastPolicyLossByGroup = new();
    private readonly Dictionary<string, float> _lastValueLossByGroup = new();
    private readonly Dictionary<string, float> _lastEntropyByGroup = new();

    // Per-agent state
    private readonly Dictionary<RLAgent2D, AgentRuntimeState> _agentStates = new();

    private long _totalSteps;
    private double _previousTimeScale = 1.0;
    private int _previousMaxPhysicsStepsPerFrame = 8;
    private int _checkpointInterval = 10;
    private int _actionRepeat = 1;
    private int _batchSize = 1;

    public override void _Ready()
    {
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

        _batchSize = Math.Max(1, _manifest.BatchSize);
        _checkpointInterval = Math.Max(1, _manifest.CheckpointInterval);
        _actionRepeat = Math.Max(1, _manifest.ActionRepeat);

        // Apply simulation speed. MaxPhysicsStepsPerFrame must be >= SimulationSpeed so the
        // engine can actually run that many physics ticks per rendered frame.
        _previousTimeScale = Engine.TimeScale;
        _previousMaxPhysicsStepsPerFrame = Engine.MaxPhysicsStepsPerFrame;
        var simulationSpeed = Math.Max(0.1f, _manifest.SimulationSpeed);
        Engine.TimeScale = simulationSpeed;
        Engine.MaxPhysicsStepsPerFrame = Math.Max(8, (int)Math.Ceiling(simulationSpeed) + 1);

        RLTrainingConfig? trainingConfig = null;
        RLTrainerConfig? trainerConfig = null;
        RLNetworkConfig? networkConfig = null;

        // Group agents by explicit shared-policy config when present, then legacy PolicyGroup,
        // and finally by scene-relative path as a deterministic fallback.
        var groupedAgents = new Dictionary<string, List<RLAgent2D>>();
        var groupBindings = new Dictionary<string, ResolvedPolicyGroupBinding>();

        for (var batchIdx = 0; batchIdx < _batchSize; batchIdx++)
        {
            var sceneInstance = packedScene.Instantiate();

            // Offset each batch copy so their physics bodies don't visually overlap.
            if (sceneInstance is Node2D node2D && _batchSize > 1)
            {
                node2D.Position = new Vector2(batchIdx * 1200, 0);
            }

            AddChild(sceneInstance);

            var academy = FindNodeByPath(sceneInstance, _manifest.AcademyNodePath) as RLAcademy;
            if (academy is null)
            {
                GD.PushError($"Could not resolve RLAcademy in batch copy {batchIdx}.");
                GetTree().Quit(1);
                return;
            }

            _academies.Add(academy);

            // Use config from the first batch copy.
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
                networkConfig = academy.ResolveNetworkConfig()
                    ?? trainingConfig?.ToNetworkConfig()
                    ?? GD.Load<RLNetworkConfig>(_manifest.NetworkConfigPath);
            }

            var batchAgents = academy.GetAgents(RLAgentControlMode.Train);
            foreach (var agent in batchAgents)
            {
                var binding = RLPolicyGroupBindingResolver.Resolve(sceneInstance, agent);
                var group = binding.BindingKey;

                if (!groupedAgents.ContainsKey(group))
                {
                    groupedAgents[group] = new List<RLAgent2D>();
                    groupBindings[group] = binding;
                }

                groupedAgents[group].Add(agent);
            }
        }

        if (_academies.Count == 0)
        {
            GD.PushError("No academy instances could be created.");
            GetTree().Quit(1);
            return;
        }

        var agents = _academies.SelectMany(a => a.GetAgents(RLAgentControlMode.Train)).ToList();

        if (trainerConfig is null || networkConfig is null || agents.Count == 0)
        {
            GD.PushError("Training scene is missing RL configuration or agents.");
            GetTree().Quit(1);
            return;
        }

        var algorithm = trainerConfig.Algorithm;

        // Validate and create trainers per group
        foreach (var (groupId, groupAgents) in groupedAgents)
        {
            var binding = groupBindings[groupId];
            var firstAgent = groupAgents[0];
            var obsSize = firstAgent.CollectObservationArray().Length;
            var discreteCount = firstAgent.GetDiscreteActionCount();
            var continuousDims = firstAgent.GetContinuousActionDimensions();
            var actionDefinitions = firstAgent.GetActionSpace();

            // PPO: discrete only
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

            // SAC: no mixing discrete + continuous within the same group
            if (algorithm == RLAlgorithmKind.SAC && discreteCount > 0 && continuousDims > 0)
            {
                GD.PushError($"[RL] Group '{groupId}': SAC does not support mixing discrete and continuous actions.");
                GetTree().Quit(1);
                return;
            }

            // Validate all agents in group are consistent
            foreach (var agent in groupAgents)
            {
                if (agent.CollectObservationArray().Length != obsSize)
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
                NetworkConfig = networkConfig,
                ActionDefinitions = actionDefinitions,
                ObservationSize = obsSize,
                DiscreteActionCount = discreteCount,
                ContinuousActionDimensions = continuousDims,
                CheckpointPath = checkpointPath,
                MetricsPath = metricsPath,
                SelfPlay = binding.Config?.SelfPlay ?? false,
                HistoricalOpponentRate = binding.Config?.HistoricalOpponentRate ?? 0f,
                FrozenCheckpointInterval = binding.Config?.FrozenCheckpointInterval ?? 10,
            };

            var trainer = TrainerFactory.Create(groupConfig);
            _trainersByGroup[groupId] = trainer;
            _metricsWritersByGroup[groupId] = new RunMetricsWriter(metricsPath, _manifest.StatusPath);
            _episodeCountByGroup[groupId] = 0;
            _updateCountByGroup[groupId] = 0;
            _lastPolicyLossByGroup[groupId] = 0f;
            _lastValueLossByGroup[groupId] = 0f;
            _lastEntropyByGroup[groupId] = 0f;

            // Initialize agent states for this group
            foreach (var agent in groupAgents)
            {
                agent.ResetEpisode();
                var obs = agent.CollectObservationArray();
                var decision = trainer.SampleAction(obs);
                _agentStates[agent] = new AgentRuntimeState
                {
                    GroupId = groupId,
                    LastObservation = obs,
                    PendingAction = decision.DiscreteAction,
                    PendingContinuousActions = decision.ContinuousActions,
                    LastLogProbability = decision.LogProbability,
                    LastValue = decision.Value,
                };
                ApplyDecision(agent, decision);
            }

            GD.Print($"[RL] Group '{binding.DisplayName}': {groupAgents.Count} agent(s), {algorithm}, obs={obsSize}, discrete={discreteCount}, continuous={continuousDims}");
            GD.Print($"[RL]   Checkpoint: {checkpointPath}");
            GD.Print($"[RL]   Metrics:    {metricsPath}");
            if (groupConfig.SelfPlay)
            {
                GD.Print($"[RL]   Self-play configured (historical rate={groupConfig.HistoricalOpponentRate:0.00}, freeze interval={groupConfig.FrozenCheckpointInterval})");
            }
        }

        // Cache the full list of train agents across all batch copies.
        _allTrainAgents = _academies.SelectMany(a => a.GetAgents(RLAgentControlMode.Train)).ToList();

        _statusWriter = new RunMetricsWriter(string.Empty, _manifest.StatusPath);
        _statusWriter.WriteStatus("running", _manifest.ScenePath, _totalSteps, 0, "Training started.");
        GD.Print($"[RL] Run: {_manifest.RunId}");
        if (_batchSize > 1)
            GD.Print($"[RL] Batch size: {_batchSize} ({_allTrainAgents.Count} total agents)");
    }

    public override void _PhysicsProcess(double delta)
    {
        if (_academies.Count == 0 || _manifest is null || _statusWriter is null)
        {
            return;
        }

        foreach (var agent in _allTrainAgents)
        {
            if (!_agentStates.TryGetValue(agent, out var state))
            {
                continue;
            }

            if (!_trainersByGroup.TryGetValue(state.GroupId, out var trainer))
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
                var nextValue = done ? 0.0f : trainer.EstimateValue(nextObservation);

                var transition = new Transition
                {
                    Observation = state.LastObservation,
                    DiscreteAction = state.PendingAction,
                    ContinuousActions = state.PendingContinuousActions,
                    Reward = state.WindowReward,
                    Done = done,
                    NextObservation = nextObservation,
                    OldLogProbability = state.LastLogProbability,
                    Value = state.LastValue,
                    NextValue = nextValue,
                };
                trainer.RecordTransition(transition);
                _totalSteps += 1;

                state.WindowReward = 0f;
                state.StepsSinceDecision = 0;

                if (done)
                {
                    _episodeCountByGroup[state.GroupId] += 1;
                    var episodeCount = _episodeCountByGroup[state.GroupId];
                    _metricsWritersByGroup[state.GroupId].AppendMetric(
                        agent.EpisodeReward, agent.EpisodeSteps,
                        _lastPolicyLossByGroup[state.GroupId],
                        _lastValueLossByGroup[state.GroupId],
                        _lastEntropyByGroup[state.GroupId],
                        _totalSteps, episodeCount,
                        agent.GetEpisodeRewardBreakdown());

                    agent.ResetEpisode();
                    nextObservation = agent.CollectObservationArray();
                }

                var nextDecision = trainer.SampleAction(nextObservation);
                state.LastObservation = nextObservation;
                state.PendingAction = nextDecision.DiscreteAction;
                state.PendingContinuousActions = nextDecision.ContinuousActions;
                state.LastLogProbability = nextDecision.LogProbability;
                state.LastValue = nextDecision.Value;
                _lastEntropyByGroup[state.GroupId] = nextDecision.Entropy;
                ApplyDecision(agent, nextDecision);
            }
            else
            {
                // Re-apply the same action so physics-driven movement continues each tick.
                if (state.PendingAction >= 0)
                    agent.ApplyAction(state.PendingAction);
                else if (state.PendingContinuousActions.Length > 0)
                    agent.ApplyAction(state.PendingContinuousActions);
            }
        }

        // Per-group updates
        RLCheckpoint? lastCheckpoint = null;
        foreach (var (groupId, trainer) in _trainersByGroup)
        {
            var episodeCount = _episodeCountByGroup[groupId];
            var updateStats = trainer.TryUpdate(groupId, _totalSteps, episodeCount);
            if (updateStats is not null)
            {
                _updateCountByGroup[groupId] += 1;
                _lastPolicyLossByGroup[groupId] = updateStats.PolicyLoss;
                _lastValueLossByGroup[groupId] = updateStats.ValueLoss;
                _lastEntropyByGroup[groupId] = updateStats.Entropy;
                var currentCheckpoint = trainer.CreateCheckpoint(groupId, _totalSteps, episodeCount, _updateCountByGroup[groupId]);
                lastCheckpoint = currentCheckpoint;

                var checkpointPath = GetGroupCheckpointPath(groupId);
                if (checkpointPath is not null && _updateCountByGroup[groupId] % _checkpointInterval == 0)
                {
                    RLCheckpoint.SaveToFile(currentCheckpoint, checkpointPath);
                }
            }
        }

        if (lastCheckpoint is not null)
        {
            foreach (var academy in _academies)
                academy.Checkpoint = lastCheckpoint;
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

        if (_manifest is null) return;

        foreach (var (groupId, trainer) in _trainersByGroup)
        {
            var episodeCount = _episodeCountByGroup.GetValueOrDefault(groupId);
            var updateCount = _updateCountByGroup.GetValueOrDefault(groupId);
            var finalCheckpoint = trainer.CreateCheckpoint(groupId, _totalSteps, episodeCount, updateCount);

            var checkpointPath = GetGroupCheckpointPath(groupId);
            if (!string.IsNullOrEmpty(checkpointPath))
            {
                RLCheckpoint.SaveToFile(finalCheckpoint, checkpointPath);
            }

            foreach (var academy in _academies)
                academy.Checkpoint = finalCheckpoint;
        }

        var totalEpisodes = _episodeCountByGroup.Values.Sum();
        _statusWriter?.WriteStatus("stopped", _manifest.ScenePath, _totalSteps, totalEpisodes, "Training ended.");
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

    private string? GetGroupCheckpointPath(string groupId)
    {
        var safeId = RLPolicyGroupBindingResolver.MakeSafeGroupId(groupId);
        return $"{_manifest?.RunDirectory}/checkpoint__{safeId}.json";
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
        public float[] LastObservation { get; set; } = Array.Empty<float>();
        public int PendingAction { get; set; } = -1;
        public float[] PendingContinuousActions { get; set; } = Array.Empty<float>();
        public float LastLogProbability { get; set; }
        public float LastValue { get; set; }
        public float WindowReward { get; set; }
        public int StepsSinceDecision { get; set; }
    }
}
