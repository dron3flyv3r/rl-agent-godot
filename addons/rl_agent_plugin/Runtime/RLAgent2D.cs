using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
public partial class RLAgent2D : Node2D
{
    private RLActionBinding? _actionBinding;
    private bool _actionBindingResolved;

    private readonly ObservationBuffer _observationBuffer = new();
    private float _pendingReward;
    private bool _donePending;
    private string _agentId = "Agent";
    private RLAgentControlMode _inlineControlMode = RLAgentControlMode.Train;
    private string _inlinePolicyGroup = string.Empty;
    private string _inlineInferenceCheckpointPath = string.Empty;
    private RLAgentConfig? _agentConfig;

    [Export] public string AgentId { get => string.IsNullOrEmpty(_agentId) ? "Agent" : _agentId; set => _agentId = value; }
    [Export] public int MaxEpisodeSteps { get; set; } = 1024;

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

    /// <summary>
    /// Override to fill the observation buffer each step.
    /// Preferred over GetObservation().
    /// </summary>
    public virtual void CollectObservations(ObservationBuffer buffer) { }

    /// <summary>
    /// Called each physics step. Override to compute rewards and call
    /// EndEpisode() when the episode is complete.
    /// Default falls back to ComputeReward() / IsEpisodeDone() for compatibility.
    /// </summary>
    public virtual void OnStep()
    {
        var reward = ComputeReward();
        if (reward != 0f) AddReward(reward);
        if (IsEpisodeDone()) EndEpisode();
    }

    /// <summary>Called at the start of every episode. Override to reset scene state.</summary>
    public virtual void OnEpisodeBegin() { }

    /// <summary>Accumulate reward for the current step.</summary>
    protected void AddReward(float reward) => _pendingReward += reward;

    /// <summary>Replace the current pending reward with a specific value.</summary>
    protected void SetReward(float reward) => _pendingReward = reward;

    /// <summary>Signal that the episode has ended.</summary>
    protected void EndEpisode() => _donePending = true;

    // ── Legacy / Fallback API ─────────────────────────────────────────────────

    /// <summary>
    /// Override to return an observation array directly.
    /// Default calls CollectObservations() internally.
    /// Prefer CollectObservations() for new agents.
    /// </summary>
    public virtual float[] GetObservation()
    {
        _observationBuffer.Clear();
        CollectObservations(_observationBuffer);
        return _observationBuffer.ToArray();
    }

    /// <summary>Legacy: return a reward scalar. Prefer OnStep() + AddReward().</summary>
    public virtual float ComputeReward() => 0.0f;

    /// <summary>Legacy: return true when episode ends. Prefer OnStep() + EndEpisode().</summary>
    public virtual bool IsEpisodeDone() => false;

    // ── Action API ────────────────────────────────────────────────────────────

    public virtual RLActionDefinition[] GetActionSpace()
    {
        return ResolveActionBinding()?.ActionSpace ?? Array.Empty<RLActionDefinition>();
    }

    public virtual string[] GetDiscreteActionLabels()
    {
        var actionSpace = GetActionSpace();
        if (actionSpace.Length == 0)
        {
            return Array.Empty<string>();
        }

        var labels = new List<string>(actionSpace.Length);
        foreach (var action in actionSpace)
        {
            if (action.VariableType == RLActionVariableType.Discrete)
            {
                labels.Add(action.Name);
            }
        }

        return labels.ToArray();
    }

    public int GetDiscreteActionCount()
    {
        return GetDiscreteActionLabels().Length;
    }

    public bool SupportsOnlyDiscreteActions()
    {
        var actionBinding = ResolveActionBinding();
        if (actionBinding is not null)
        {
            return actionBinding.SupportsOnlyDiscreteActions;
        }

        foreach (var action in GetActionSpace())
        {
            if (action.VariableType != RLActionVariableType.Discrete)
            {
                return false;
            }
        }

        return true;
    }

    public virtual void ApplyAction(int action)
    {
        if (GetDiscreteActionCount() > 0 && (action < 0 || action >= GetDiscreteActionCount()))
        {
            return;
        }

        CurrentActionIndex = action;
        if (!TryApplyActionSettings(action))
        {
            OnActionApplied(action);
        }
    }

    public virtual void ApplyAction(float[] continuousActions)
    {
        if (!TryApplyContinuousActionSettings(continuousActions))
        {
            OnContinuousActionApplied(continuousActions);
        }
    }

    public int GetContinuousActionDimensions()
    {
        return ResolveActionBinding()?.ContinuousActionDimensions ?? 0;
    }

    public virtual void ResetEpisode()
    {
        EpisodeSteps = 0;
        EpisodeReward = 0.0f;
        CurrentActionIndex = 0;
        _pendingReward = 0f;
        _donePending = false;
        OnEpisodeBegin();
        if (GetDiscreteActionCount() > 0)
        {
            ApplyAction(CurrentActionIndex);
        }
    }

    public void AccumulateReward(float reward)
    {
        EpisodeReward += reward;
        EpisodeSteps += 1;
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

    /// <summary>Returns the done flag and clears it.</summary>
    internal bool ConsumeDonePending()
    {
        var done = _donePending;
        _donePending = false;
        return done;
    }

    // ── Private ───────────────────────────────────────────────────────────────

    private RLActionBinding? ResolveActionBinding()
    {
        if (_actionBindingResolved)
        {
            return _actionBinding;
        }

        _actionBinding = RLActionBinding.Create(GetType());
        _actionBindingResolved = true;
        return _actionBinding;
    }

    private bool TryApplyActionSettings(int action)
    {
        var actionBinding = ResolveActionBinding();
        return actionBinding is not null && actionBinding.TryApply(this, action);
    }

    private bool TryApplyContinuousActionSettings(float[] actions)
    {
        var actionBinding = ResolveActionBinding();
        return actionBinding is not null && actionBinding.TryApplyContinuous(this, actions);
    }

    protected virtual void OnActionApplied(int action) { }
    protected virtual void OnContinuousActionApplied(float[] continuousActions) { }
}
