using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
public partial class RLAcademy : Node
{
    private RLTrainingConfig? _trainingConfig;

    [ExportGroup("Configuration")]
    [Export]
    public RLTrainingConfig? TrainingConfig
    {
        get => _trainingConfig;
        set { _trainingConfig = value; UpdateConfigurationWarnings(); }
    }

    [ExportGroup("Training Run")]
    [Export] public string RunPrefix { get; set; } = string.Empty;
    [Export] public float SimulationSpeed { get; set; } = 1.0f;
    // Number of gradient update steps between checkpoint saves
    [Export] public int CheckpointInterval { get; set; } = 10;
    // How many physics ticks to repeat each action before sampling the next one (1 = every tick)
    [Export] public int ActionRepeat { get; set; } = 1;
    // Global episode length cap used when agents and policy groups do not override it (0 = no cap).
    [Export(PropertyHint.Range, "0,100000,1,or_greater")] public int MaxEpisodeSteps { get; set; } = 0;
    // Number of parallel scene instances to run during training (vectorised environments)
    [Export(PropertyHint.Range, "1, 256, 8, or_greater")] public int BatchSize { get; set; } = 1;

    [ExportGroup("Self-Play")]
    [Export] public Godot.Collections.Array<RLPolicyPairingConfig> SelfPlayPairings { get; set; } = new();

    [ExportGroup("Inference")]
    [Export] public RLCheckpoint? Checkpoint { get; set; }

    [ExportGroup("Debug")]
    /// <summary>Show the observation/reward/action spy overlay when running outside of training. Press Tab to cycle agents.</summary>
    [Export] public bool EnableSpyOverlay { get; set; } = false;
    /// <summary>Render all batch environments in a grid during training. Off by default (only env 0 is rendered).</summary>
    [Export] public bool ShowBatchGrid { get; set; } = false;

    public bool InferenceActive { get; private set; }

    public override string[] _GetConfigurationWarnings()
    {
        var warnings = new List<string>();
        if (TrainingConfig is null && ResolveTrainerConfig() is null)
            warnings.Add("TrainingConfig is not assigned. Assign an RLTrainingConfig resource.");
        return warnings.ToArray();
    }

    private RLAgentSpyOverlay? _spyOverlay;
    private readonly Dictionary<IRLAgent, IInferencePolicy> _agentInferencePolicies = new();
    private readonly Dictionary<IRLAgent, string> _agentObservationGroups = new();
    private readonly Dictionary<IRLAgent, int> _inferenceStepCounters = new();
    private readonly Dictionary<string, string> _observationGroupDisplayNames = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _validatedObservationSizesByGroup = new(StringComparer.Ordinal);
    private double _previousTimeScale = 1.0;
    private int _previousMaxPhysicsStepsPerFrame = 8;

    public override void _Ready()
    {
        TryInitializeInference();

        // Apply simulation speed when running outside of TrainingBootstrap (which manages it on its own).
        if (!IsInsideTrainingBootstrap())
        {
            _previousTimeScale = Engine.TimeScale;
            _previousMaxPhysicsStepsPerFrame = Engine.MaxPhysicsStepsPerFrame;
            Engine.TimeScale = Math.Max(0.1f, SimulationSpeed);
            Engine.MaxPhysicsStepsPerFrame = Math.Max(8, (int)Math.Ceiling(SimulationSpeed) + 1);

            if (EnableSpyOverlay)
            {
                _spyOverlay = new RLAgentSpyOverlay();
                _spyOverlay.Initialize(this);
                AddChild(_spyOverlay);
            }
        }
    }

    public override void _ExitTree()
    {
        if (!IsInsideTrainingBootstrap())
        {
            Engine.TimeScale = _previousTimeScale;
            Engine.MaxPhysicsStepsPerFrame = _previousMaxPhysicsStepsPerFrame;
        }
    }

    public override void _PhysicsProcess(double delta)
    {
        if (!InferenceActive)
        {
            return;
        }

        var repeat = Math.Max(1, ActionRepeat);

        foreach (var pair in _agentInferencePolicies)
        {
            var agent = pair.Key;
            var policy = pair.Value;

            agent.TickStep();
            var stepReward = agent.ConsumePendingReward();
            var stepBreakdown = agent.ConsumePendingRewardBreakdown();
            agent.AccumulateReward(stepReward, stepBreakdown.Count > 0 ? stepBreakdown : null);
            if (agent.ConsumeDonePending() || agent.HasReachedEpisodeLimit())
            {
                agent.ResetEpisode();
                _inferenceStepCounters[agent] = 0;
            }

            _inferenceStepCounters.TryGetValue(agent, out var stepCount);
            stepCount++;
            _inferenceStepCounters[agent] = stepCount;

            if (stepCount >= repeat)
            {
                _inferenceStepCounters[agent] = 0;
                var observation = CollectValidatedObservation(agent);
                if (observation.Length == 0)
                {
                    continue;
                }

                var decision = policy.Predict(observation);
                if (decision.DiscreteAction >= 0)
                {
                    agent.ApplyAction(decision.DiscreteAction);
                }
                else if (decision.ContinuousActions.Length > 0)
                {
                    agent.ApplyAction(decision.ContinuousActions);
                }
            }
            else
            {
                // Re-apply current action so physics-driven movement continues each tick.
                if (agent.CurrentActionIndex >= 0)
                {
                    agent.ApplyAction(agent.CurrentActionIndex);
                }
                else if (agent.CurrentContinuousActions.Length > 0)
                {
                    agent.ApplyAction(agent.CurrentContinuousActions);
                }
            }
        }
    }

    public IReadOnlyList<IRLAgent> GetAgents()
    {
        var agents = new List<IRLAgent>();
        var sceneRoot = ResolveSceneRoot();
        CollectAgents(sceneRoot, agents);
        return agents;
    }

    public IReadOnlyList<IRLAgent> GetAgents(RLAgentControlMode controlMode)
    {
        var agents = new List<IRLAgent>();
        foreach (var agent in GetAgents())
        {
            if (agent.ControlMode == controlMode)
            {
                agents.Add(agent);
            }
            // Auto agents join Train groups during training and Inference groups during normal play.
            else if (agent.ControlMode == RLAgentControlMode.Auto
                     && (controlMode == RLAgentControlMode.Train || controlMode == RLAgentControlMode.Inference))
            {
                agents.Add(agent);
            }
        }

        return agents;
    }

    public void ResetAllAgents()
    {
        foreach (var agent in GetAgents())
        {
            agent.ResetEpisode();
        }
    }

    public List<RLPolicyPairingConfig> GetResolvedSelfPlayPairings()
    {
        return new List<RLPolicyPairingConfig>(SelfPlayPairings);
    }

    public ObservationSizeInferenceResult InferObservationSizes(RLAgentControlMode? controlMode = null, bool resetEpisodes = true)
    {
        var agents = controlMode.HasValue ? GetAgents(controlMode.Value) : GetAgents();
        return ObservationSizeInference.Infer(ResolveSceneRoot(), agents, resetEpisodes);
    }

    private static void CollectAgents(Node node, ICollection<IRLAgent> agents)
    {
        if (node is IRLAgent agent)
        {
            agents.Add(agent);
        }

        foreach (var child in node.GetChildren())
        {
            if (child is Node childNode)
            {
                CollectAgents(childNode, agents);
            }
        }
    }

    private bool IsInsideTrainingBootstrap()
    {
        var current = GetParent();
        while (current is not null)
        {
            if (current is TrainingBootstrap) return true;
            current = current.GetParent();
        }
        return false;
    }

    private Node ResolveSceneRoot()
    {
        var current = this as Node;
        while (current.GetParent() is Node parent
               && parent is not TrainingBootstrap
               && parent is not InferenceBootstrap
               && parent is not SubViewport)
        {
            current = parent;
        }

        return current;
    }

    private void TryInitializeInference()
    {
        if (IsInsideTrainingBootstrap())
        {
            return;
        }

        var fallbackNetworkGraph = RLNetworkGraph.CreateDefault();
        var agents = GetAgents();
        if (agents.Count == 0)
        {
            return;
        }

        // Warn about any agents left in Train mode when running outside of the training bootstrap.
        // Auto mode is intentionally excluded — it's designed to work in both contexts.
        foreach (var agent in agents)
        {
            if (agent.ControlMode == RLAgentControlMode.Train)
            {
                GD.PushWarning(
                    $"[RLAcademy] Agent '{agent.AsNode().Name}' is set to Train mode, but the game was " +
                    "started normally. Training will not run. Use the 'Start Training' button in " +
                    "the RL Agent Plugin dock, or switch the agent to Inference or Auto mode.");
            }
        }

        RLCheckpoint? firstLoadedCheckpoint = null;
        // GetAgents(Inference) includes Auto-mode agents so their observation sizes are inferred too.
        var observationInference = InferObservationSizes(RLAgentControlMode.Inference);
        foreach (var error in observationInference.Errors)
        {
            GD.PushError($"[RLAcademy] {error}");
        }

        foreach (var agent in agents)
        {
            if (agent.ControlMode != RLAgentControlMode.Inference
                && agent.ControlMode != RLAgentControlMode.Auto)
            {
                continue;
            }

            var checkpointPath = agent.GetInferenceModelPath();

            // For .rlmodel files resolve path directly; for .json fall back to registry.
            RLCheckpoint? checkpoint;
            if (checkpointPath.EndsWith(".rlmodel", StringComparison.OrdinalIgnoreCase))
            {
                if (!FileAccess.FileExists(checkpointPath))
                {
                    GD.PushWarning($"[RLAcademy] .rlmodel not found for agent '{agent.AsNode().Name}': {checkpointPath}");
                    continue;
                }

                checkpoint = RLModelLoader.LoadFromFile(checkpointPath);
            }
            else
            {
                var resolvedPath = CheckpointRegistry.ResolveCheckpointPath(checkpointPath);
                if (string.IsNullOrWhiteSpace(resolvedPath))
                {
                    continue;
                }

                checkpoint = RLCheckpoint.LoadFromFile(resolvedPath);
            }

            if (checkpoint is null)
            {
                GD.PushWarning($"[RLAcademy] Could not load checkpoint for agent '{agent.AsNode().Name}'.");
                continue;
            }

            var obsSize = checkpoint.ObservationSize;
            var actionCount = checkpoint.DiscreteActionCount > 0
                ? checkpoint.DiscreteActionCount
                : checkpoint.ContinuousActionDimensions;
            observationInference.AgentBindings.TryGetValue(agent, out var binding);
            if (binding is not null)
            {
                _agentObservationGroups[agent] = binding.BindingKey;
                _observationGroupDisplayNames[binding.BindingKey] = binding.DisplayName;
                if (observationInference.GroupSizes.TryGetValue(binding.BindingKey, out var groupObservationSize))
                {
                    _validatedObservationSizesByGroup[binding.BindingKey] = groupObservationSize;
                }
            }

            var agentObsSize = observationInference.AgentSizes.TryGetValue(agent, out var inferredObservationSize)
                ? inferredObservationSize
                : 0;
            var agentDiscreteCount = agent.GetDiscreteActionCount();
            var agentContinuousDims = agent.GetContinuousActionDimensions();

            if (agentObsSize <= 0)
            {
                GD.PushError($"[RLAcademy] Could not infer observations for agent '{agent.AsNode().Name}'.");
                continue;
            }

            if (obsSize != agentObsSize)
            {
                GD.PushError(
                    $"[RLAcademy] Checkpoint observation size {obsSize} does not match agent '{agent.AsNode().Name}' " +
                    $"observation size {agentObsSize}.");
                continue;
            }

            if (string.Equals(checkpoint.Algorithm, RLCheckpoint.PpoAlgorithm, StringComparison.OrdinalIgnoreCase)
                && agentContinuousDims > 0)
            {
                GD.PushError($"[RLAcademy] Checkpoint for '{agent.AsNode().Name}' is PPO, but the agent expects continuous actions.");
                continue;
            }

            if (checkpoint.ContinuousActionDimensions > 0 && checkpoint.ContinuousActionDimensions != agentContinuousDims)
            {
                GD.PushError(
                    $"[RLAcademy] Continuous action mismatch for '{agent.AsNode().Name}': " +
                    $"checkpoint={checkpoint.ContinuousActionDimensions}, agent={agentContinuousDims}.");
                continue;
            }

            if (checkpoint.DiscreteActionCount > 0 && checkpoint.DiscreteActionCount != agentDiscreteCount)
            {
                GD.PushError(
                    $"[RLAcademy] Discrete action mismatch for '{agent.AsNode().Name}': " +
                    $"checkpoint={checkpoint.DiscreteActionCount}, agent={agentDiscreteCount}.");
                continue;
            }

            if (obsSize <= 0 || actionCount <= 0)
            {
                GD.PushWarning($"[RLAcademy] Checkpoint for agent '{agent.AsNode().Name}' has invalid dimensions (obs={obsSize}, actions={actionCount}).");
                continue;
            }

            try
            {
                var policy = InferencePolicyFactory.Create(checkpoint, fallbackNetworkGraph);
                policy.LoadCheckpoint(checkpoint);
                _agentInferencePolicies[agent] = policy;
                _inferenceStepCounters[agent] = 0;
                firstLoadedCheckpoint ??= checkpoint;
                GD.Print(
                    $"[RLAcademy] Loaded {checkpoint.Algorithm} inference model for '{agent.AsNode().Name}' " +
                    $"(obs={obsSize}, actions={actionCount}).");
            }
            catch (Exception ex)
            {
                GD.PushError($"[RLAcademy] Failed to load checkpoint for agent '{agent.AsNode().Name}': {ex.Message} — " +
                             "Verify that the checkpoint metadata matches the active agent.");
            }
        }

        InferenceActive = _agentInferencePolicies.Count > 0;
        if (InferenceActive && firstLoadedCheckpoint is not null)
        {
            Checkpoint = firstLoadedCheckpoint;
        }
    }

    public RLTrainerConfig? ResolveTrainerConfig()
    {
        return TrainingConfig?.ToTrainerConfig();
    }

    private float[] CollectValidatedObservation(IRLAgent agent)
    {
        var observation = agent.CollectObservationArray();
        ValidateGroupObservationSize(agent, observation.Length);
        return observation;
    }

    private void ValidateGroupObservationSize(IRLAgent agent, int observationSize)
    {
        if (!_agentObservationGroups.TryGetValue(agent, out var groupKey))
        {
            return;
        }

        if (!_validatedObservationSizesByGroup.TryGetValue(groupKey, out var expectedSize))
        {
            _validatedObservationSizesByGroup[groupKey] = observationSize;
            return;
        }

        if (expectedSize == observationSize)
        {
            return;
        }

        var displayName = _observationGroupDisplayNames.TryGetValue(groupKey, out var name)
            ? name
            : groupKey;
        GD.PushError(
            $"[RLAcademy] Agent '{agent.AsNode().Name}' in policy group '{displayName}' emitted {observationSize} observations, " +
            $"expected {expectedSize}. Observation size must remain stable for the whole group.");
    }
}
