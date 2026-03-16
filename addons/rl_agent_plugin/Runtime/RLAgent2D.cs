using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
public partial class RLAgent2D : Node2D
{
    private ActionSpaceBuilder? _explicitActionSpace;
    private bool _explicitActionSpaceResolved;

    private readonly ObservationBuffer _observationBuffer = new();
    private int? _validatedObservationSize;
    private float _pendingReward;
    private readonly Dictionary<string, float> _pendingRewardComponents = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float> _episodeRewardComponents = new(StringComparer.Ordinal);
    private bool _donePending;
    private string _agentId = "Agent";
    private RLAgentControlMode _inlineControlMode = RLAgentControlMode.Train;
    private string _inlinePolicyGroup = string.Empty;
    private string _inlineInferenceCheckpointPath = string.Empty;
    private RLAgentConfig? _agentConfig;

    [Export] public string AgentId { get => string.IsNullOrEmpty(_agentId) ? "Agent" : _agentId; set => _agentId = value; }
    /// <summary>Maximum steps per episode. Set to 0 to disable the built-in limit (e.g. when a game controller manages episode length).</summary>
    [Export] public int MaxEpisodeSteps { get; set; } = 0;

    [ExportGroup("Control")]
    [Export]
    public RLAgentControlMode ControlMode
    {
        get => _agentConfig?.ControlMode ?? _inlineControlMode;
        set { _inlineControlMode = value; UpdateConfigurationWarnings(); }
    }

    [Export]
    public string PolicyGroup
    {
        get => _agentConfig?.PolicyGroup ?? _inlinePolicyGroup;
        set => _inlinePolicyGroup = value;
    }

    [Export(PropertyHint.File, "*.json,*.rlmodel")]
    public string InferenceCheckpointPath
    {
        get => _agentConfig?.InferenceCheckpointPath ?? _inlineInferenceCheckpointPath;
        set { _inlineInferenceCheckpointPath = value; UpdateConfigurationWarnings(); }
    }

    [ExportGroup("Advanced")]
    [Export]
    public RLAgentConfig? AgentConfig
    {
        get => _agentConfig;
        set { _agentConfig = value; UpdateConfigurationWarnings(); }
    }

    public int EpisodeSteps { get; private set; }
    public float EpisodeReward { get; private set; }
    public int CurrentActionIndex { get; private set; }

    public override void _Ready()
    {
        AddToGroup("rl_agent_plugin_agent");
    }

    // ── New API ───────────────────────────────────────────────────────────────

    /// <summary>Override to fill the observation buffer each step.</summary>
    public virtual void CollectObservations(ObservationBuffer buffer) { }

    /// <summary>
    /// Called each physics step. Override to compute rewards and call
    /// EndEpisode() when the episode is complete.
    /// </summary>
    public virtual void OnStep() { }

    /// <summary>Called at the start of every episode. Override to reset scene state.</summary>
    public virtual void OnEpisodeBegin() { }

    /// <summary>Accumulate reward for the current step.</summary>
    protected void AddReward(float reward) => _pendingReward += reward;

    /// <summary>Accumulate a named reward component for the current step.</summary>
    protected void AddReward(float reward, string tag)
    {
        _pendingReward += reward;
        AddRewardComponent(_pendingRewardComponents, tag, reward);
    }

    /// <summary>Replace the current pending reward with a specific value.</summary>
    protected void SetReward(float reward)
    {
        _pendingReward = reward;
        _pendingRewardComponents.Clear();
    }

    /// <summary>Replace the current pending reward with a single named reward component.</summary>
    protected void SetReward(float reward, string tag)
    {
        _pendingReward = reward;
        _pendingRewardComponents.Clear();
        AddRewardComponent(_pendingRewardComponents, tag, reward);
    }

    /// <summary>Signal that the episode has ended. Can be called from external scripts (e.g. a game controller).</summary>
    public void EndEpisode() => _donePending = true;

    /// <summary>Returns true if EndEpisode() has been called and the done signal is pending. Does not consume it.</summary>
    public bool IsDone => _donePending;

    // ── Action API ────────────────────────────────────────────────────────────

    public virtual void DefineActions(ActionSpaceBuilder builder) { }

    protected virtual void OnActionsReceived(ActionBuffer actions) { }

    public virtual RLActionDefinition[] GetActionSpace()
    {
        return ResolveExplicitActionSpace()?.Build() ?? Array.Empty<RLActionDefinition>();
    }

    public virtual string[] GetDiscreteActionLabels()
    {
        return ResolveExplicitActionSpace()?.BuildDiscreteActionLabels() ?? Array.Empty<string>();
    }

    public int GetDiscreteActionCount()
    {
        return ResolveExplicitActionSpace()?.GetDiscreteActionCount() ?? 0;
    }

    public bool SupportsOnlyDiscreteActions()
    {
        return ResolveExplicitActionSpace()?.SupportsOnlyDiscreteActions() ?? false;
    }

    public virtual void ApplyAction(int action)
    {
        var explicitActionSpace = ResolveExplicitActionSpace();
        var discreteActionCount = GetDiscreteActionCount();
        if (discreteActionCount > 0 && (action < 0 || action >= discreteActionCount))
        {
            return;
        }

        CurrentActionIndex = action;
        if (explicitActionSpace is not null)
        {
            OnActionsReceived(explicitActionSpace.CreateDiscreteActionBuffer(action));
        }
    }

    public virtual void ApplyAction(float[] continuousActions)
    {
        var explicitActionSpace = ResolveExplicitActionSpace();
        if (explicitActionSpace is not null)
        {
            OnActionsReceived(explicitActionSpace.CreateContinuousActionBuffer(continuousActions));
        }
    }

    public int GetContinuousActionDimensions()
    {
        return ResolveExplicitActionSpace()?.GetContinuousActionDimensions() ?? 0;
    }

    public virtual void ResetEpisode()
    {
        EpisodeSteps = 0;
        EpisodeReward = 0.0f;
        _episodeRewardComponents.Clear();
        CurrentActionIndex = 0;
        _pendingReward = 0f;
        _pendingRewardComponents.Clear();
        _donePending = false;
        OnEpisodeBegin();
        if (GetDiscreteActionCount() > 0)
        {
            ApplyAction(CurrentActionIndex);
        }
    }

    public void AccumulateReward(float reward, IReadOnlyDictionary<string, float>? rewardBreakdown = null)
    {
        EpisodeReward += reward;
        EpisodeSteps += 1;

        if (rewardBreakdown is null)
        {
            return;
        }

        foreach (var (tag, amount) in rewardBreakdown)
        {
            AddRewardComponent(_episodeRewardComponents, tag, amount);
        }
    }

    public bool HasReachedEpisodeLimit()
    {
        return MaxEpisodeSteps > 0 && EpisodeSteps >= MaxEpisodeSteps;
    }

    public string GetInferenceCheckpointPath() => InferenceCheckpointPath;

    public override string[] _GetConfigurationWarnings()
    {
        var warnings = new List<string>();
        if (ControlMode == RLAgentControlMode.Inference && string.IsNullOrEmpty(GetInferenceCheckpointPath()))
            warnings.Add("ControlMode is Inference but no checkpoint path is set.");
        return warnings.ToArray();
    }

    // ── Framework-internal ────────────────────────────────────────────────────

    /// <summary>Called by the training/inference loop each physics step.</summary>
    internal void TickStep() => OnStep();

    /// <summary>Returns accumulated reward and clears the pending buffer.</summary>
    internal float ConsumePendingReward()
    {
        var reward = _pendingReward;
        _pendingReward = 0f;
        return reward;
    }

    internal Dictionary<string, float> ConsumePendingRewardBreakdown()
    {
        var breakdown = new Dictionary<string, float>(_pendingRewardComponents, StringComparer.Ordinal);
        _pendingRewardComponents.Clear();
        return breakdown;
    }

    /// <summary>Returns the done flag and clears it.</summary>
    internal bool ConsumeDonePending()
    {
        var done = _donePending;
        _donePending = false;
        return done;
    }

    internal float[] CollectObservationArray()
    {
        _observationBuffer.Clear();
        CollectObservations(_observationBuffer);

        var observation = _observationBuffer.ToArray();
        if (_validatedObservationSize is null)
        {
            _validatedObservationSize = observation.Length;
        }
        else if (_validatedObservationSize.Value != observation.Length)
        {
            GD.PushError(
                $"[RLAgent2D] Agent '{Name}' changed observation size from {_validatedObservationSize.Value} to {observation.Length}. " +
                "Observation size must remain stable across steps and episodes.");
        }

        return observation;
    }

    public IReadOnlyDictionary<string, float> GetEpisodeRewardBreakdown()
    {
        return new Dictionary<string, float>(_episodeRewardComponents, StringComparer.Ordinal);
    }

    // ── Private ───────────────────────────────────────────────────────────────

    private ActionSpaceBuilder? ResolveExplicitActionSpace()
    {
        if (_explicitActionSpaceResolved)
        {
            return _explicitActionSpace;
        }

        var builder = new ActionSpaceBuilder();
        DefineActions(builder);
        _explicitActionSpace = builder.HasActions ? builder : null;
        _explicitActionSpaceResolved = true;
        return _explicitActionSpace;
    }

    private static void AddRewardComponent(IDictionary<string, float> target, string tag, float amount)
    {
        if (string.IsNullOrWhiteSpace(tag))
        {
            return;
        }

        target.TryGetValue(tag, out var current);
        target[tag] = current + amount;
    }
}
