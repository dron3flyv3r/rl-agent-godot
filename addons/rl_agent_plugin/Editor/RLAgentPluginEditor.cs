using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Godot;
using RlAgentPlugin.Editor;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin;

[Tool]
public partial class RLAgentPluginEditor : EditorPlugin
{
    private const string AgentScriptPath = "res://addons/rl_agent_plugin/Runtime/RLAgent2D.cs";
    private const string AcademyScriptPath = "res://addons/rl_agent_plugin/Runtime/RLAcademy.cs";
    private const string PluginIconPath = "res://icon.svg";
    private static readonly Lazy<Dictionary<string, Type>> ScriptTypeIndex = new(BuildScriptTypeIndex);

    private Texture2D? _pluginIcon;
    private RLModelFormatLoader? _rlModelFormatLoader;
    private EditorDock? _setupEditorDock;
    private RLSetupDock? _setupDock;
    private RLDashboard? _dashboard;
    private Button? _startTrainingButton;
    private Button? _stopTrainingButton;
    private TrainingSceneValidation? _lastValidation;
    private bool _launchedTrainingRun;
    private string _lastAutoScenePath = string.Empty;
    private string _lastValidationSignature = string.Empty;

    public override void _EnterTree()
    {
        _pluginIcon = GD.Load<Texture2D>(PluginIconPath);
        _rlModelFormatLoader = new RLModelFormatLoader();
        ResourceLoader.AddResourceFormatLoader(_rlModelFormatLoader, true);

        _setupDock = new RLSetupDock();
        _setupDock.StartTrainingRequested += OnStartTrainingRequested;
        _setupDock.StopTrainingRequested += OnStopTrainingRequested;

        _setupEditorDock = new EditorDock
        {
            Title = "RL Setup",
            DefaultSlot = EditorDock.DockSlot.RightUl,
        };
        _setupEditorDock.AddChild(_setupDock);

        AddDock(_setupEditorDock);

        _startTrainingButton = new Button { Text = "Start Training", TooltipText = "Launch the configured training run." };
        _startTrainingButton.Pressed += OnStartTrainingRequested;
        AddControlToContainer(CustomControlContainer.Toolbar, _startTrainingButton);

        _stopTrainingButton = new Button { Text = "Stop Training", TooltipText = "Stop the active training run." };
        _stopTrainingButton.Pressed += OnStopTrainingRequested;
        AddControlToContainer(CustomControlContainer.Toolbar, _stopTrainingButton);

        _dashboard = new RLDashboard { Name = "RLDash" };
        EditorInterface.Singleton.GetEditorMainScreen().AddChild(_dashboard);
        _MakeVisible(false);

        RegisterCustomTypes();
        SetProcess(true);
        RefreshValidationFromActiveScene();
    }

    public override void _ExitTree()
    {
        SetProcess(false);
        UnregisterCustomTypes();

        if (_rlModelFormatLoader is not null)
        {
            ResourceLoader.RemoveResourceFormatLoader(_rlModelFormatLoader);
            _rlModelFormatLoader = null;
        }

        if (_startTrainingButton is not null)
        {
            RemoveControlFromContainer(CustomControlContainer.Toolbar, _startTrainingButton);
            _startTrainingButton.QueueFree();
        }

        if (_stopTrainingButton is not null)
        {
            RemoveControlFromContainer(CustomControlContainer.Toolbar, _stopTrainingButton);
            _stopTrainingButton.QueueFree();
        }

        if (_setupDock is not null)
        {
            _setupDock = null;
        }

        if (_setupEditorDock is not null)
        {
            RemoveDock(_setupEditorDock);
            _setupEditorDock.QueueFree();
        }

        if (_dashboard is not null)
        {
            _dashboard.QueueFree();
            _dashboard = null;
        }
    }

    public override bool _HasMainScreen() => true;

    public override string _GetPluginName() => "RLDash";

    public override Texture2D? _GetPluginIcon() => _pluginIcon;

    public override void _MakeVisible(bool visible)
    {
        if (_dashboard is not null)
            _dashboard.Visible = visible;
    }

    public override bool _Build()
    {
        // Training validation runs inside OnStartTrainingRequested(), not here.
        // Blocking the build would prevent normal "Run Project" from working.
        return true;
    }

    public override void _Process(double delta)
    {
        if (_setupDock is null)
        {
            return;
        }

        var isPlaying = EditorInterface.Singleton.IsPlayingScene();
        if (!isPlaying)
        {
            _launchedTrainingRun = false;
        }

        if (_startTrainingButton is not null)
        {
            var validationBlocked = _lastValidation is { IsValid: false };
            _startTrainingButton.Disabled = _launchedTrainingRun || validationBlocked;
            _startTrainingButton.TooltipText = validationBlocked && _lastValidation!.Errors.Count > 0
                ? _lastValidation.Errors[0]
                : "Launch the configured training run.";
        }

        if (_stopTrainingButton is not null)
        {
            _stopTrainingButton.Disabled = !_launchedTrainingRun;
        }

        // Auto-refresh validation when the edited scene changes.
        var currentScenePath = ResolveTrainingScenePath();
        if (currentScenePath != _lastAutoScenePath)
        {
            _lastAutoScenePath = currentScenePath;
            RefreshValidationFromActiveScene();
            return;
        }

        var currentSignature = BuildValidationSignature(EditorInterface.Singleton.GetEditedSceneRoot());
        if (currentSignature != _lastValidationSignature)
        {
            RefreshValidationFromActiveScene();
        }
    }

    private void RegisterCustomTypes()
    {
        var agentScript = GD.Load<Script>(AgentScriptPath);
        var academyScript = GD.Load<Script>(AcademyScriptPath);

        if (agentScript is not null)
        {
            AddCustomType(nameof(RLAgent2D), nameof(Node2D), agentScript, _pluginIcon);
        }

        if (academyScript is not null)
        {
            AddCustomType(nameof(RLAcademy), nameof(Node), academyScript, _pluginIcon);
        }
    }

    private void UnregisterCustomTypes()
    {
        RemoveCustomType(nameof(RLAgent2D));
        RemoveCustomType(nameof(RLAcademy));
    }

    private void RefreshValidationFromActiveScene()
    {
        if (_setupDock is null)
        {
            return;
        }

        var scenePath = ResolveTrainingScenePath();
        if (string.IsNullOrWhiteSpace(scenePath))
        {
            _setupDock.SetScenePath(string.Empty);
            _setupDock.SetValidationSummary("No active scene. Open a scene or set a main scene in Project Settings.");
            return;
        }

        _setupDock.SetScenePath(scenePath);
        var validation = ValidateSceneSafely(scenePath, "auto validation");
        _lastValidationSignature = BuildValidationSignature(EditorInterface.Singleton.GetEditedSceneRoot());
        UpdateValidationUi(validation);
    }

    private void OnStartTrainingRequested()
    {
        if (_setupDock is null)
        {
            return;
        }

        var scenePath = ResolveTrainingScenePath();
        if (string.IsNullOrWhiteSpace(scenePath))
        {
            _setupDock.SetLaunchStatus("No active scene or main scene configured.");
            return;
        }

        var validation = ValidateSceneSafely(scenePath, "training launch");
        UpdateValidationUi(validation);
        if (!validation.IsValid)
        {
            _setupDock.SetLaunchStatus("Training launch blocked by scene validation errors.");
            return;
        }

        var manifest = TrainingLaunchManifest.CreateDefault();
        var runPrefix = SanitizeRunPrefix(validation.RunPrefix);
        var runIdPrefix = string.IsNullOrWhiteSpace(runPrefix) ? "run" : runPrefix;
        var checkpointFileName = string.IsNullOrWhiteSpace(runPrefix) ? "checkpoint.json" : $"{runPrefix}_checkpoint.json";
        manifest.ScenePath = scenePath;
        manifest.AcademyNodePath = validation.AcademyPath;
        manifest.RunId = $"{runIdPrefix}_{Time.GetUnixTimeFromSystem()}";
        manifest.RunDirectory = $"res://RL-Agent-Training/runs/{manifest.RunId}";
        manifest.CheckpointPath = $"{manifest.RunDirectory}/{checkpointFileName}";
        manifest.TrainingConfigPath = validation.TrainingConfigPath;
        manifest.TrainerConfigPath = validation.TrainerConfigPath;
        manifest.NetworkConfigPath = validation.NetworkConfigPath;
        manifest.MetricsPath = $"{manifest.RunDirectory}/metrics.jsonl";
        manifest.StatusPath = $"{manifest.RunDirectory}/status.json";
        manifest.CheckpointInterval = validation.CheckpointInterval;
        manifest.SimulationSpeed = validation.SimulationSpeed;
        manifest.ActionRepeat = validation.ActionRepeat;
        manifest.BatchSize = Math.Max(1, validation.BatchSize);

        var writeError = manifest.SaveToUserStorage();
        if (writeError != Error.Ok)
        {
            _setupDock.SetLaunchStatus($"Failed to write training manifest: {writeError}");
            return;
        }

        _setupDock.SetLaunchStatus($"Launching {manifest.RunId}\n{ProjectSettings.GlobalizePath(manifest.RunDirectory)}");

        // Write initial meta.json so the dashboard can show agent names on export.
        WriteRunMeta(manifest.RunId, validation.AgentNames, validation.AgentGroups);

        var bootstrapScene = "res://addons/rl_agent_plugin/Scenes/TrainingBootstrap.tscn";
        EditorInterface.Singleton.PlayCustomScene(bootstrapScene);
        _launchedTrainingRun = true;

        // Notify dashboard immediately so it can auto-select the run and show LIVE badge.
        _dashboard?.OnTrainingStarted(manifest.RunId);
    }

    private static void WriteRunMeta(string runId, IReadOnlyList<string> agentNames, IReadOnlyList<string> agentGroups)
    {
        try
        {
            var runDirAbs = ProjectSettings.GlobalizePath($"res://RL-Agent-Training/runs/{runId}");
            System.IO.Directory.CreateDirectory(runDirAbs);

            var agentArr = new Godot.Collections.Array();
            foreach (var name in agentNames) agentArr.Add(Variant.From(name));

            var groupArr = new Godot.Collections.Array();
            foreach (var g in agentGroups) groupArr.Add(Variant.From(g));

            var d = new Godot.Collections.Dictionary
            {
                { "display_name",  "" },
                { "agent_names",   agentArr },
                { "agent_groups",  groupArr },
            };

            System.IO.File.WriteAllText(
                System.IO.Path.Combine(runDirAbs, "meta.json"),
                Json.Stringify(d));
        }
        catch (Exception ex)
        {
            GD.PushWarning($"[RLAgentPluginEditor] Failed to write meta.json: {ex.Message}");
        }
    }

    private void OnStopTrainingRequested()
    {
        if (!_launchedTrainingRun)
        {
            return;
        }

        EditorInterface.Singleton.StopPlayingScene();
        _launchedTrainingRun = false;
        _setupDock?.SetLaunchStatus("Training run stopped.");
    }

    private void UpdateValidationUi(TrainingSceneValidation validation)
    {
        _lastValidation = validation;
        _setupDock?.SetValidationSummary(validation.BuildSummary(), validation.IsValid);
        _setupDock?.SetConfigSummary(validation.TrainerConfigPath, validation.NetworkConfigPath, validation.CheckpointPath);
    }

    private static string ResolveTrainingScenePath()
    {
        // Prefer the currently open/edited scene.
        var editedRoot = EditorInterface.Singleton.GetEditedSceneRoot();
        if (editedRoot is not null && !string.IsNullOrWhiteSpace(editedRoot.SceneFilePath))
        {
            return editedRoot.SceneFilePath;
        }

        // Fall back to the project's configured main scene.
        var mainScene = ProjectSettings.GetSetting("application/run/main_scene").ToString();
        return mainScene ?? string.Empty;
    }

    private static TrainingSceneValidation ValidateSceneSafely(string scenePath, string operation)
    {
        try
        {
            var validation = ValidateScene(scenePath);
            LogValidationMessages(operation, validation);
            return validation;
        }
        catch (Exception exception)
        {
            return BuildValidationCrashResult(scenePath, operation, exception);
        }
    }

    private static TrainingSceneValidation ValidateScene(string scenePath)
    {
        var validation = new TrainingSceneValidation
        {
            ScenePath = scenePath,
        };

        var packedScene = GD.Load<PackedScene>(scenePath);
        if (packedScene is null)
        {
            validation.Errors.Add($"Could not load scene: {scenePath}");
            return validation;
        }

        var root = packedScene.Instantiate();
        try
        {
            Node? academy = null;
            // groupId → list of agent nodes
            var agentsByGroup = new System.Collections.Generic.Dictionary<string, System.Collections.Generic.List<Node>>();
            var groupBindings = new System.Collections.Generic.Dictionary<string, ResolvedPolicyGroupBinding>();

            Traverse(root, node =>
            {
                if (IsAcademyNode(node))
                {
                    if (academy is null)
                    {
                        academy = node;
                    }
                    else
                    {
                        validation.Errors.Add("More than one RLAcademy was found. Only one academy is supported.");
                    }
                }

                if (IsAgentNode(node))
                {
                    var agentId = ReadStringProperty(node, "AgentId");
                    var agentName = string.IsNullOrWhiteSpace(agentId) ? node.Name.ToString() : agentId;
                    validation.AgentNames.Add(agentName);

                    var controlMode = ReadAgentControlMode(node);
                    var binding = RLPolicyGroupBindingResolver.Resolve(root, node);
                    validation.AgentGroups.Add(binding.SafeGroupId);

                    if (controlMode == RLAgentControlMode.Train)
                    {
                        validation.TrainAgentCount += 1;

                        if (!agentsByGroup.ContainsKey(binding.BindingKey))
                        {
                            agentsByGroup[binding.BindingKey] = new System.Collections.Generic.List<Node>();
                            groupBindings[binding.BindingKey] = binding;
                        }

                        agentsByGroup[binding.BindingKey].Add(node);
                    }
                    else if (controlMode == RLAgentControlMode.Inference)
                    {
                        ValidateInferenceCheckpoint(node, root, validation);
                    }
                }
            });

            // Check for duplicate AgentIds across different policy groups (same group = shared brain = OK).
            var agentIdToGroup = new System.Collections.Generic.Dictionary<string, string>(StringComparer.Ordinal);
            for (var i = 0; i < validation.AgentNames.Count; i++)
            {
                var id    = validation.AgentNames[i];
                var group = i < validation.AgentGroups.Count ? validation.AgentGroups[i] : string.Empty;
                if (agentIdToGroup.TryGetValue(id, out var existingGroup))
                {
                    if (existingGroup != group)
                        validation.Errors.Add($"Duplicate AgentId \"{id}\" used by agents in different policy groups. Each brain must have a unique AgentId.");
                }
                else
                {
                    agentIdToGroup[id] = group;
                }
            }

            if (academy is null)
            {
                validation.Errors.Add("No RLAcademy node was found in the selected scene.");
            }
            else
            {
                validation.AcademyPath = root.GetPathTo(academy).ToString();
                var trainingConfigRes = ReadResourceProperty(academy, "TrainingConfig");
                var trainerConfigRes = ReadResourceProperty(academy, "TrainerConfig");
                var networkConfig = ReadResourceProperty(academy, "NetworkConfig");
                var checkpoint = ReadResourceProperty(academy, "Checkpoint");

                validation.TrainingConfigPath = trainingConfigRes?.ResourcePath ?? string.Empty;
                validation.TrainerConfigPath = trainerConfigRes?.ResourcePath ?? validation.TrainingConfigPath;
                validation.NetworkConfigPath = networkConfig?.ResourcePath ?? validation.TrainingConfigPath;
                validation.CheckpointPath = checkpoint?.ResourcePath ?? string.Empty;
                validation.RunPrefix = ReadStringProperty(academy, "RunPrefix");
                validation.CheckpointInterval = ReadIntProperty(academy, "CheckpointInterval", 10);
                validation.SimulationSpeed = ReadFloatProperty(academy, "SimulationSpeed", 1.0f);
                validation.ActionRepeat = ReadIntProperty(academy, "ActionRepeat", 1);
                validation.BatchSize = ReadIntProperty(academy, "BatchSize", 1);

                if (trainingConfigRes is null && trainerConfigRes is null)
                {
                    validation.Errors.Add("RLAcademy is missing an RLTrainingConfig resource or a legacy RLTrainerConfig resource.");
                }

                if (trainingConfigRes is null && networkConfig is null)
                {
                    validation.Errors.Add("RLAcademy is missing an RLTrainingConfig resource or a legacy RLNetworkConfig resource.");
                }

                // Determine algorithm from trainer config
                var algorithm = RLAlgorithmKind.PPO;
                if (trainingConfigRes is RLTrainingConfig trainingConfig)
                {
                    algorithm = trainingConfig.Algorithm;
                }
                else if (trainerConfigRes is RLTrainerConfig trainerConfig)
                {
                    algorithm = trainerConfig.Algorithm;
                }

                // Per-group validation
                foreach (var (groupId, groupNodes) in agentsByGroup)
                {
                    var binding = groupBindings[groupId];
                    var firstNode = groupNodes[0];
                    var firstObservationSize = ReadAgentObservationSize(firstNode, root, validation);
                    var firstActionCount = ReadAgentActionCount(firstNode);
                    var firstIsDiscrete = SupportsOnlyDiscreteActions(firstNode);
                    var firstContinuousDims = ReadAgentContinuousDims(firstNode);

                    var groupSummary = new PolicyGroupSummary
                    {
                        GroupId = binding.DisplayName,
                        AgentCount = groupNodes.Count,
                        ObservationSize = firstObservationSize,
                        ActionCount = firstActionCount,
                        IsContinuous = !firstIsDiscrete,
                        ContinuousActionDimensions = firstContinuousDims,
                        UsesExplicitConfig = binding.UsesExplicitConfig,
                        PolicyConfigPath = binding.ConfigPath,
                        SelfPlay = binding.Config?.SelfPlay ?? false,
                    };
                    foreach (var node in groupNodes)
                    {
                        groupSummary.AgentPaths.Add(root.GetPathTo(node).ToString());
                    }

                    validation.PolicyGroups.Add(groupSummary);

                    if (binding.UsesExplicitConfig
                        && string.IsNullOrWhiteSpace(binding.Config?.GroupId)
                        && string.IsNullOrWhiteSpace(binding.ConfigPath))
                    {
                        validation.Errors.Add($"Group '{groupSummary.GroupId}': PolicyGroupConfig should set GroupId or be saved as a standalone resource so grouping remains stable across scene copies.");
                    }

                    // PPO: discrete only
                    if (algorithm == RLAlgorithmKind.PPO && !firstIsDiscrete)
                    {
                        validation.Errors.Add($"Group '{groupSummary.GroupId}': PPO requires discrete-only actions.");
                    }

                    if (algorithm == RLAlgorithmKind.PPO && firstActionCount == 0 && firstIsDiscrete)
                    {
                        validation.Errors.Add($"Group '{groupSummary.GroupId}': define at least one discrete action.");
                    }

                    // Skip when firstObservationSize == -1 (cast unavailable; not a real empty obs vector).
                    if (firstObservationSize == 0)
                    {
                        validation.Errors.Add($"Group '{groupSummary.GroupId}': could not infer a non-zero observation size.");
                    }

                    // SAC: no mixing discrete + continuous
                    if (algorithm == RLAlgorithmKind.SAC && firstActionCount > 0 && firstContinuousDims > 0)
                    {
                        validation.Errors.Add($"Group '{groupSummary.GroupId}': SAC does not support mixing discrete and continuous actions.");
                    }

                    // All agents in group must be consistent
                    foreach (var node in groupNodes)
                    {
                        var nodePath = root.GetPathTo(node).ToString();
                        var observationSize = ReadAgentObservationSize(node, root, validation);
                        var actionCount = ReadAgentActionCount(node);
                        var isDiscrete = SupportsOnlyDiscreteActions(node);
                        if (isDiscrete != firstIsDiscrete)
                        {
                            validation.Errors.Add($"Group '{groupSummary.GroupId}': {nodePath}: all agents in a group must use the same action type (discrete vs continuous).");
                        }

                        if (firstObservationSize >= 0 && observationSize >= 0 && observationSize != firstObservationSize)
                        {
                            validation.Errors.Add($"Group '{groupSummary.GroupId}': {nodePath}: all agents must emit the same observation vector length.");
                        }

                        if (algorithm == RLAlgorithmKind.PPO && isDiscrete && firstActionCount >= 0 && actionCount >= 0 && actionCount != firstActionCount)
                        {
                            validation.Errors.Add($"Group '{groupSummary.GroupId}': {nodePath}: all agents must share the same discrete action count.");
                        }
                    }
                }

                // For backward compat, set ExpectedActionCount from first group
                if (agentsByGroup.Count > 0)
                {
                    validation.ExpectedActionCount = validation.PolicyGroups.Count > 0
                        ? validation.PolicyGroups[0].ActionCount
                        : 0;
                }
            }

            if (validation.AgentNames.Count == 0)
            {
                validation.Errors.Add("No RLAgent2D nodes were found in the selected scene.");
            }
            else if (validation.TrainAgentCount == 0)
            {
                validation.Errors.Add("No agents are set to Train mode.");
            }

            validation.IsValid = validation.Errors.Count == 0;
            return validation;
        }
        finally
        {
            root.QueueFree();
        }
    }

    private static void Traverse(Node node, System.Action<Node> visitor)
    {
        visitor(node);
        foreach (var child in node.GetChildren())
        {
            if (child is Node childNode)
            {
                Traverse(childNode, visitor);
            }
        }
    }

    private static bool IsAcademyNode(Node node)
    {
        if (node is RLAcademy)
        {
            return true;
        }

        var managedType = ResolveManagedScriptType(node);
        if (managedType is not null && typeof(RLAcademy).IsAssignableFrom(managedType))
        {
            return true;
        }

        return ScriptInheritsPath(GetNodeScript(node), AcademyScriptPath);
    }

    private static bool IsAgentNode(Node node)
    {
        if (node is RLAgent2D)
        {
            return true;
        }

        var managedType = ResolveManagedScriptType(node);
        return managedType is not null && typeof(RLAgent2D).IsAssignableFrom(managedType);
    }

    private static int ReadAgentActionCount(Node node)
    {
        if (node is RLAgent2D agent)
            return agent.GetDiscreteActionCount();
        // C# cast unavailable in editor context (e.g. assembly not yet fully loaded).
        // Return -1 so callers can skip checks rather than produce false-positive errors.
        return -1;
    }

    private static int ReadAgentContinuousDims(Node node)
    {
        return node is RLAgent2D agent ? agent.GetContinuousActionDimensions() : 0;
    }

    private static int ReadAgentObservationSize(Node node, Node root, TrainingSceneValidation validation)
    {
        if (node is not RLAgent2D agent)
        {
            // -1 = unknown (cast unavailable). 0 = cast succeeded but returned empty obs.
            return -1;
        }

        try
        {
            agent.ResetEpisode();
            return agent.CollectObservationArray().Length;
        }
        catch (Exception exception)
        {
            validation.Errors.Add($"Agent '{root.GetPathTo(node)}': observation inference failed: {exception.Message}");
            return 0;
        }
    }

    private static bool SupportsOnlyDiscreteActions(Node node)
    {
        if (node is RLAgent2D agent)
            return agent.SupportsOnlyDiscreteActions();
        // Cast unavailable: assume discrete to avoid false-positive "PPO requires discrete-only" errors.
        return IsAgentNode(node);
    }

    private static void ValidateInferenceCheckpoint(Node node, Node root, TrainingSceneValidation validation)
    {
        if (node is not RLAgent2D agent)
        {
            return;
        }

        var checkpointPath = agent.GetInferenceCheckpointPath();
        var nodePath = root.GetPathTo(node).ToString();
        if (string.IsNullOrWhiteSpace(checkpointPath))
        {
            validation.Errors.Add($"Agent '{nodePath}' is in Inference mode but has no checkpoint path.");
            return;
        }

        RLCheckpoint? checkpoint;
        if (checkpointPath.EndsWith(".rlmodel", StringComparison.OrdinalIgnoreCase))
        {
            checkpoint = RLModelLoader.LoadFromFile(checkpointPath);
        }
        else
        {
            var resolvedPath = CheckpointRegistry.ResolveCheckpointPath(checkpointPath);
            checkpoint = string.IsNullOrWhiteSpace(resolvedPath)
                ? null
                : RLCheckpoint.LoadFromFile(resolvedPath);
        }

        if (checkpoint is null)
        {
            validation.Errors.Add($"Agent '{nodePath}': failed to load inference checkpoint '{checkpointPath}'.");
            return;
        }

        var observationSize = ReadAgentObservationSize(node, root, validation);
        var discreteCount = agent.GetDiscreteActionCount();
        var continuousDims = agent.GetContinuousActionDimensions();

        if (checkpoint.ObservationSize != observationSize)
        {
            validation.Errors.Add(
                $"Agent '{nodePath}': checkpoint observation size {checkpoint.ObservationSize} " +
                $"does not match agent observation size {observationSize}.");
        }

        if (string.Equals(checkpoint.Algorithm, RLCheckpoint.PpoAlgorithm, StringComparison.OrdinalIgnoreCase)
            && continuousDims > 0)
        {
            validation.Errors.Add($"Agent '{nodePath}': PPO checkpoints cannot drive continuous actions.");
        }

        if (checkpoint.DiscreteActionCount > 0 && checkpoint.DiscreteActionCount != discreteCount)
        {
            validation.Errors.Add(
                $"Agent '{nodePath}': checkpoint discrete action count {checkpoint.DiscreteActionCount} " +
                $"does not match agent count {discreteCount}.");
        }

        if (checkpoint.ContinuousActionDimensions > 0 && checkpoint.ContinuousActionDimensions != continuousDims)
        {
            validation.Errors.Add(
                $"Agent '{nodePath}': checkpoint continuous action dims {checkpoint.ContinuousActionDimensions} " +
                $"does not match agent dims {continuousDims}.");
        }
    }

    private static RLAgentControlMode ReadAgentControlMode(Node node)
    {
        if (node is RLAgent2D agent)
        {
            return agent.ControlMode;
        }

        var variant = node.Get("ControlMode");
        return variant.VariantType == Variant.Type.Int
            ? (RLAgentControlMode)(int)variant
            : RLAgentControlMode.Train;
    }

    private static Resource? ReadResourceProperty(Node node, string propertyName)
    {
        var variant = node.Get(propertyName);
        return variant.VariantType == Variant.Type.Object ? variant.AsGodotObject() as Resource : null;
    }

    private static string ReadStringProperty(Node node, string propertyName)
    {
        var variant = node.Get(propertyName);
        return variant.VariantType == Variant.Type.String ? variant.AsString() : string.Empty;
    }

    private static int ReadIntProperty(Node node, string propertyName, int defaultValue)
    {
        var variant = node.Get(propertyName);
        return variant.VariantType == Variant.Type.Int ? (int)variant : defaultValue;
    }

    private static float ReadFloatProperty(Node node, string propertyName, float defaultValue)
    {
        var variant = node.Get(propertyName);
        return variant.VariantType switch
        {
            Variant.Type.Float => (float)(double)variant,
            Variant.Type.Int => (int)variant,
            _ => defaultValue,
        };
    }

    private static Type? ResolveManagedScriptType(Node node)
    {
        if (node.GetType() != typeof(Node) && node.GetType() != typeof(Node2D))
        {
            return node.GetType();
        }

        var script = GetNodeScript(node);
        if (script is null)
        {
            return null;
        }

        return ResolveManagedScriptType(script);
    }

    private static Type? ResolveManagedScriptType(Script script)
    {
        if (!string.IsNullOrWhiteSpace(script.ResourcePath)
            && ScriptTypeIndex.Value.TryGetValue(script.ResourcePath, out var managedType))
        {
            return managedType;
        }

        var baseScript = script.GetBaseScript() as Script;
        return baseScript is null ? null : ResolveManagedScriptType(baseScript);
    }

    private static Script? GetNodeScript(Node node)
    {
        var scriptVariant = node.GetScript();
        return scriptVariant.VariantType == Variant.Type.Object ? scriptVariant.AsGodotObject() as Script : null;
    }

    private static bool ScriptInheritsPath(Script? script, string targetPath)
    {
        var current = script;
        while (current is not null)        {
            if (current.ResourcePath == targetPath)
            {
                return true;
            }

            current = current.GetBaseScript() as Script;
        }

        return false;
    }

    private static string BuildValidationSignature(Node? editedRoot)
    {
        if (editedRoot is null)
        {
            return string.Empty;
        }

        var builder = new System.Text.StringBuilder();
        Traverse(editedRoot, node =>
        {
            if (IsAcademyNode(node))
            {
                builder.Append("academy|");
                builder.Append(node.GetPath());
                builder.Append('|');
                builder.Append(ReadStringProperty(node, "RunPrefix"));
                builder.Append('|');
                builder.Append(ReadIntProperty(node, "CheckpointInterval", 10));
                builder.Append('|');
                builder.Append(ReadIntProperty(node, "ActionRepeat", 1));
                builder.Append('|');
                builder.Append(ReadIntProperty(node, "BatchSize", 1));
                builder.Append('|');
                builder.Append(ReadFloatProperty(node, "SimulationSpeed", 1.0f));
                builder.Append('|');
                builder.Append(ReadResourceProperty(node, "TrainingConfig")?.ResourcePath ?? string.Empty);
                builder.Append('|');
                builder.Append(ReadResourceProperty(node, "TrainerConfig")?.ResourcePath ?? string.Empty);
                builder.Append('|');
                builder.Append(ReadResourceProperty(node, "NetworkConfig")?.ResourcePath ?? string.Empty);
                builder.Append('|');
                builder.Append(ReadResourceProperty(node, "Checkpoint")?.ResourcePath ?? string.Empty);
                builder.AppendLine();
            }

            if (IsAgentNode(node))
            {
                builder.Append("agent|");
                builder.Append(node.GetPath());
                builder.Append('|');
                builder.Append(ReadStringProperty(node, "AgentId"));
                builder.Append('|');
                builder.Append((int)ReadAgentControlMode(node));
                builder.Append('|');
                builder.Append(ReadStringProperty(node, "PolicyGroup"));
                builder.Append('|');
                builder.Append(ReadAgentActionCount(node));
                builder.Append('|');
                builder.Append(ReadAgentContinuousDims(node));
                builder.AppendLine();
            }
        });

        return builder.ToString();
    }

    private static Dictionary<string, Type> BuildScriptTypeIndex()
    {
        var index = new Dictionary<string, Type>(StringComparer.Ordinal);
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            foreach (var type in SafeGetTypes(assembly))
            {
                foreach (var scriptPath in GetScriptPaths(type))
                {
                    if (!string.IsNullOrWhiteSpace(scriptPath))
                    {
                        index[scriptPath] = type;
                    }
                }
            }
        }

        return index;
    }

    private static IEnumerable<string> GetScriptPaths(MemberInfo member)
    {
        return member
            .GetCustomAttributes<ScriptPathAttribute>(false)
            .Select(attribute => attribute.Path)
            .Where(path => !string.IsNullOrWhiteSpace(path))
            .Distinct(StringComparer.Ordinal);
    }

    private static IEnumerable<Type> SafeGetTypes(Assembly assembly)
    {
        try
        {
            return assembly.GetTypes();
        }
        catch (ReflectionTypeLoadException exception)
        {
            return exception.Types.Where(type => type is not null)!;
        }
    }

    private static TrainingSceneValidation BuildValidationCrashResult(string scenePath, string operation, Exception exception)
    {
        var message = $"{operation} failed while validating {scenePath}: {exception.Message}";
        GD.PushError(message);
        GD.PrintErr(message);
        GD.PrintErr(exception.ToString());

        var validation = new TrainingSceneValidation
        {
            ScenePath = scenePath,
            IsValid = false,
        };
        validation.Errors.Add(message);
        return validation;
    }

    private static void LogValidationMessages(string operation, TrainingSceneValidation validation)
    {
        if (validation.Errors.Count == 0)
        {
            return;
        }

        foreach (var error in validation.Errors)
        {
            var message = $"{operation}: {error}";
            GD.PushError(message);
            GD.PrintErr(message);
        }
    }

    private static string SanitizeRunPrefix(string prefix)
    {
        if (string.IsNullOrWhiteSpace(prefix))
        {
            return string.Empty;
        }

        var builder = new System.Text.StringBuilder(prefix.Length);
        foreach (var character in prefix)
        {
            if (char.IsLetterOrDigit(character))
            {
                builder.Append(char.ToLowerInvariant(character));
            }
            else if (character is '-' or '_')
            {
                builder.Append(character);
            }
        }

        return builder.ToString().Trim('_', '-');
    }
}
