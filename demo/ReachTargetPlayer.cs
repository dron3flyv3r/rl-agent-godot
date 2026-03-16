using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

public partial class ReachTargetPlayer : CharacterBody2D
{
    [ExportGroup("Lane")]
    [Export] public float LaneMinX { get; set; } = 126.0f;
    [Export] public float LaneMaxX { get; set; } = 674.0f;
    [Export] public float LaneY { get; set; } = 338.0f;
    [Export] public float SpawnPadding { get; set; } = 54.0f;
    [Export] public float MinGoalSeparation { get; set; } = 120.0f;

    [Export] public float MoveStep { get; set; } = 18.0f;
    [Export] public float SuccessThreshold { get; set; } = 18.0f;
    [Export] public bool ManualControlWhenStandalone { get; set; } = true;

    public float GoalX { get; private set; }
    public bool IsAtGoal { get; private set; }

    private readonly RandomNumberGenerator _rng = new();
    private float _stepReward;
    private float _lastDistance;
    private ReachTargetAgent? _agent;
    private RLAcademy? _academy;
    private Label? _distanceLabel;
    private Label? _footerLabel;
    private Node2D? _goalMarker;

    public override void _Ready()
    {
        _rng.Randomize();
        _agent = GetNode<ReachTargetAgent>("Agent");
        _academy = GetNode<RLAcademy>("../Academy");
        _distanceLabel = GetNode<Label>("../CanvasLayer/Panel/Margin/VBox/DistanceLabel");
        _footerLabel = GetNode<Label>("../CanvasLayer/Panel/Margin/VBox/FooterLabel");
        _goalMarker = GetNode<Node2D>("../GoalMarker");
        ResetEpisodeState();
    }

    public override void _PhysicsProcess(double delta)
    {
        if (_agent is null)
        {
            return;
        }

        if (_agent.ControlMode == RLAgentControlMode.Human
            && !IsTrainingRun()
            && !(_academy?.InferenceActive ?? false)
            && ManualControlWhenStandalone)
        {
            if (Input.IsActionPressed("ui_left"))
            {
                _agent.ApplyAction(0);
            }
            else if (Input.IsActionPressed("ui_right"))
            {
                _agent.ApplyAction(1);
            }
        }

        var previousDistance = _lastDistance;
        Position = new Vector2(
            Mathf.Clamp(Position.X + (_agent.MoveDirection * MoveStep), GetLaneMinX(), GetLaneMaxX()),
            LaneY);

        _lastDistance = Mathf.Abs(GoalX - Position.X);
        _stepReward = Mathf.Clamp((previousDistance - _lastDistance) / MoveStep, -1.0f, 1.0f) * 0.25f;
        _stepReward -= 0.01f;

        if (_lastDistance <= SuccessThreshold)
        {
            IsAtGoal = true;
            _stepReward += 1.0f;
        }

        if (!IsTrainingRun() && Input.IsActionJustPressed("ui_accept"))
        {
            _agent.ResetEpisode();
        }

        UpdateUi();
    }

    /// <summary>Returns the reward accumulated this physics step and resets it.</summary>
    public float ConsumeStepReward()
    {
        var reward = _stepReward;
        _stepReward = 0.0f;
        return reward;
    }

    public void ResetEpisodeState()
    {
        IsAtGoal = false;
        _stepReward = 0.0f;
        var laneMin = GetLaneMinX();
        var laneMax = GetLaneMaxX();
        var spawnMin = Mathf.Min(laneMin + SpawnPadding, laneMax);
        var spawnMax = Mathf.Max(laneMax - SpawnPadding, laneMin);
        if (spawnMin > spawnMax)
        {
            spawnMin = laneMin;
            spawnMax = laneMax;
        }

        Position = new Vector2(_rng.RandfRange(spawnMin, spawnMax), LaneY);
        GoalX = _rng.RandfRange(spawnMin, spawnMax);

        while (Mathf.Abs(GoalX - Position.X) < MinGoalSeparation)
        {
            GoalX = _rng.RandfRange(spawnMin, spawnMax);
        }

        _lastDistance = Mathf.Abs(GoalX - Position.X);
        if (_goalMarker is not null)
        {
            _goalMarker.Position = new Vector2(GoalX, LaneY);
        }

        UpdateUi();
    }

    private void UpdateUi()
    {
        if (_distanceLabel is not null)
        {
            _distanceLabel.Text = $"Distance: {_lastDistance:F1}";
        }

        if (_footerLabel is not null)
        {
            _footerLabel.Text = IsTrainingRun()
                ? "Training run active. Metrics are still being written to the run folder."
                : "Controls: Left/Right move, Enter resets. Use Start Training in the plugin to switch to learning.";
        }
    }

    private float GetLaneMinX()
    {
        return Mathf.Min(LaneMinX, LaneMaxX);
    }

    private float GetLaneMaxX()
    {
        return Mathf.Max(LaneMinX, LaneMaxX);
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
}
