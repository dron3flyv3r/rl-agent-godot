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
            _startTrainingButton.Disabled = _launchedTrainingRun;
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
        manifest.TrainerConfigPath = validation.TrainerConfigPath;
        manifest.NetworkConfigPath = validation.NetworkConfigPath;
        manifest.MetricsPath = $"{manifest.RunDirectory}/metrics.jsonl";
        manifest.StatusPath = $"{manifest.RunDirectory}/status.json";
        manifest.CheckpointSaveIntervalUpdates = validation.CheckpointSaveIntervalUpdates;
        manifest.SimulationSpeed = validation.SimulationSpeed;

        var writeError = manifest.SaveToUserStorage();
        if (writeError != Error.Ok)
        {
            _setupDock.SetLaunchStatus($"Failed to write training manifest: {writeError}");
            return;
        }

        _setupDock.SetLaunchStatus($"Launching {manifest.RunId}\n{ProjectSettings.GlobalizePath(manifest.RunDirectory)}");

        // Write initial meta.json so the dashboard can show agent names on export.
        WriteRunMeta(manifest.RunId, validation.AgentNames);

        var bootstrapScene = "res://addons/rl_agent_plugin/Scenes/TrainingBootstrap.tscn";
        EditorInterface.Singleton.PlayCustomScene(bootstrapScene);
        _launchedTrainingRun = true;

        // Notify dashboard immediately so it can auto-select the run and show LIVE badge.
        _dashboard?.OnTrainingStarted(manifest.RunId);
    }

    private static void WriteRunMeta(string runId, IReadOnlyList<string> agentNames)
    {
        try
        {
            var runDirAbs = ProjectSettings.GlobalizePath($"res://RL-Agent-Training/runs/{runId}");
            System.IO.Directory.CreateDirectory(runDirAbs);

            var agentArr = new Godot.Collections.Array();
            foreach (var name in agentNames) agentArr.Add(Variant.From(name));

            var d = new Godot.Collections.Dictionary
            {
                { "display_name", "" },
                { "agent_names",  agentArr },
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
                    var agentName = node.Name.ToString();
                    var agentConfig = ReadResourceProperty(node, "AgentConfig") as RLAgentConfig;
                    validation.AgentNames.Add(string.IsNullOrWhiteSpace(agentName) ? "RLAgent2D" : agentName);

                    var controlMode = agentConfig?.ControlMode ?? RLAgentControlMode.Train;
                    if (controlMode == RLAgentControlMode.Train)
                    {
                        validation.TrainAgentCount += 1;
                        var group = agentConfig?.PolicyGroup ?? string.Empty;
                        if (string.IsNullOrEmpty(group))
                        {
                            group = $"__agent__{node.Name}";
                        }

                        if (!agentsByGroup.ContainsKey(group))
                        {
                            agentsByGroup[group] = new System.Collections.Generic.List<Node>();
                        }

                        agentsByGroup[group].Add(node);
                    }
                }
            });

            if (academy is null)
            {
                validation.Errors.Add("No RLAcademy node was found in the selected scene.");
            }
            else
            {
                validation.AcademyPath = root.GetPathTo(academy).ToString();
                var trainerConfigRes = ReadResourceProperty(academy, "TrainerConfig");
                var networkConfig = ReadResourceProperty(academy, "NetworkConfig");
                var checkpoint = ReadResourceProperty(academy, "Checkpoint");

                validation.TrainerConfigPath = trainerConfigRes?.ResourcePath ?? string.Empty;
                validation.NetworkConfigPath = networkConfig?.ResourcePath ?? string.Empty;
                validation.CheckpointPath = checkpoint?.ResourcePath ?? string.Empty;
                validation.RunPrefix = ReadStringProperty(academy, "RunPrefix");
                validation.CheckpointSaveIntervalUpdates = ReadIntProperty(academy, "CheckpointSaveIntervalUpdates", 10);
                validation.SimulationSpeed = ReadFloatProperty(academy, "SimulationSpeed", 1.0f);

                if (trainerConfigRes is null)
                {
                    validation.Errors.Add("RLAcademy is missing an RLTrainerConfig resource.");
                }

                if (networkConfig is null)
                {
                    validation.Errors.Add("RLAcademy is missing an RLNetworkConfig resource.");
                }

                // Determine algorithm from trainer config
                var algorithm = RLAlgorithmKind.PPO;
                if (trainerConfigRes is RLTrainerConfig trainerConfig)
                {
                    algorithm = trainerConfig.Algorithm;
                }

                // Per-group validation
                foreach (var (groupId, groupNodes) in agentsByGroup)
                {
                    var firstNode = groupNodes[0];
                    var firstActionCount = ReadAgentActionCount(firstNode);
                    var firstIsDiscrete = SupportsOnlyDiscreteActions(firstNode);
                    var firstContinuousDims = ReadAgentContinuousDims(firstNode);

                    var groupSummary = new PolicyGroupSummary
                    {
                        GroupId = groupId.StartsWith("__agent__") ? groupNodes[0].Name.ToString() : groupId,
                        AgentCount = groupNodes.Count,
                        ActionCount = firstActionCount,
                        IsContinuous = !firstIsDiscrete,
                        ContinuousActionDimensions = firstContinuousDims,
                    };
                    validation.PolicyGroups.Add(groupSummary);

                    // PPO: discrete only
                    if (algorithm == RLAlgorithmKind.PPO && !firstIsDiscrete)
                    {
                        validation.Errors.Add($"Group '{groupSummary.GroupId}': PPO requires discrete-only actions.");
                    }

                    if (algorithm == RLAlgorithmKind.PPO && firstActionCount <= 0 && firstIsDiscrete)
                    {
                        validation.Errors.Add($"Group '{groupSummary.GroupId}': define at least one discrete action.");
                    }

                    // SAC: no mixing discrete + continuous
                    if (algorithm == RLAlgorithmKind.SAC && firstActionCount > 0 && firstContinuousDims > 0)
                    {
                        validation.Errors.Add($"Group '{groupSummary.GroupId}': SAC does not support mixing discrete and continuous actions.");
                    }

                    // All agents in group must be consistent
                    foreach (var node in groupNodes)
                    {
                        var actionCount = ReadAgentActionCount(node);
                        var isDiscrete = SupportsOnlyDiscreteActions(node);
                        if (isDiscrete != firstIsDiscrete)
                        {
                            validation.Errors.Add($"Group '{groupSummary.GroupId}': {node.Name}: all agents in a group must use the same action type (discrete vs continuous).");
                        }

                        if (algorithm == RLAlgorithmKind.PPO && isDiscrete && actionCount != firstActionCount)
                        {
                            validation.Errors.Add($"Group '{groupSummary.GroupId}': {node.Name}: all agents must share the same discrete action count.");
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
        {
            return agent.GetDiscreteActionCount();
        }

        var actionBinding = ResolveActionBinding(node);
        return actionBinding?.ActionSpace.Length ?? 0;
    }

    private static int ReadAgentContinuousDims(Node node)
    {
        if (node is RLAgent2D agent)
        {
            return agent.GetContinuousActionDimensions();
        }

        var actionBinding = ResolveActionBinding(node);
        return actionBinding?.ContinuousActionDimensions ?? 0;
    }

    private static bool SupportsOnlyDiscreteActions(Node node)
    {
        if (node is RLAgent2D agent)
        {
            return agent.SupportsOnlyDiscreteActions();
        }

        var actionBinding = ResolveActionBinding(node);
        return actionBinding?.SupportsOnlyDiscreteActions ?? true;
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

    private static RLActionBinding? ResolveActionBinding(Node node)
    {
        var managedType = ResolveManagedScriptType(node);
        if (managedType is null || !typeof(RLAgent2D).IsAssignableFrom(managedType))
        {
            return null;
        }

        return RLActionBinding.Create(managedType);
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
