using Godot;

namespace RlAgentPlugin.Editor;

[Tool]
public partial class RLSetupDock : VBoxContainer
{
    private readonly Label _scenePathLabel;
    private readonly Label _validationStatusLabel;
    private readonly Label _validationDetailLabel;
    private readonly Label _configLabel;
    private readonly Label _launchStatusLabel;

    public RLSetupDock()
    {
        Name = "RL Setup";
        CustomMinimumSize = new Vector2(220f, 0f);
        SizeFlagsHorizontal = SizeFlags.ExpandFill;

        var scroll = new ScrollContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical = SizeFlags.ExpandFill,
        };
        AddChild(scroll);

        var outer = new MarginContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        SetMargins(outer, 8);
        scroll.AddChild(outer);

        var vbox = new VBoxContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        outer.AddChild(vbox);

        // Scene
        vbox.AddChild(MakeSectionHeader("Scene"));
        _scenePathLabel = new Label
        {
            Text = "—",
            ClipText = true,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        vbox.AddChild(_scenePathLabel);

        vbox.AddChild(MakeSpacer(4));
        vbox.AddChild(new HSeparator());
        vbox.AddChild(MakeSpacer(4));

        // Validation
        vbox.AddChild(MakeSectionHeader("Validation"));
        _validationStatusLabel = new Label { Text = "—" };
        vbox.AddChild(_validationStatusLabel);

        _validationDetailLabel = new Label
        {
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        vbox.AddChild(_validationDetailLabel);

        vbox.AddChild(MakeSpacer(4));
        vbox.AddChild(new HSeparator());
        vbox.AddChild(MakeSpacer(4));

        // Resources
        vbox.AddChild(MakeSectionHeader("Resources"));
        _configLabel = new Label
        {
            Text = "Configs: not resolved",
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        vbox.AddChild(_configLabel);

        vbox.AddChild(MakeSpacer(6));
        vbox.AddChild(new HSeparator());
        vbox.AddChild(MakeSpacer(4));

        // Start / Stop buttons
        var buttonRow = new HBoxContainer();
        buttonRow.AddThemeConstantOverride("separation", 4);
        vbox.AddChild(buttonRow);

        var startButton = new Button
        {
            Text = "▶  Start Training",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            TooltipText = "Launch a training run using the active or main scene",
            CustomMinimumSize = new Vector2(0f, 32f),
        };
        startButton.Pressed += () => StartTrainingRequested?.Invoke();
        buttonRow.AddChild(startButton);

        var stopButton = new Button
        {
            Text = "■  Stop",
            TooltipText = "Stop the active training run",
            CustomMinimumSize = new Vector2(0f, 32f),
        };
        stopButton.Pressed += () => StopTrainingRequested?.Invoke();
        buttonRow.AddChild(stopButton);

        vbox.AddChild(MakeSpacer(4));

        _launchStatusLabel = new Label
        {
            Text = "Status: idle",
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        vbox.AddChild(_launchStatusLabel);

        vbox.AddChild(MakeSpacer(4));
        vbox.AddChild(new HSeparator());
        vbox.AddChild(MakeSpacer(4));
    }

    public event System.Action? StartTrainingRequested;
    public event System.Action? StopTrainingRequested;

    public void SetScenePath(string scenePath)
    {
        var fileName = System.IO.Path.GetFileNameWithoutExtension(scenePath);
        _scenePathLabel.Text = string.IsNullOrWhiteSpace(fileName) ? "—" : fileName;
        _scenePathLabel.TooltipText = scenePath;
    }

    public void SetValidationSummary(string summary, bool? isValid = null)
    {
        if (isValid == true)
        {
            _validationStatusLabel.Text = "✓  Ready to train";
            _validationStatusLabel.AddThemeColorOverride("font_color", new Color(0.45f, 0.82f, 0.45f));
        }
        else if (isValid == false)
        {
            _validationStatusLabel.Text = "✗  Scene has errors";
            _validationStatusLabel.AddThemeColorOverride("font_color", new Color(0.90f, 0.40f, 0.40f));
        }
        else
        {
            _validationStatusLabel.Text = "—";
            _validationStatusLabel.RemoveThemeColorOverride("font_color");
        }

        _validationDetailLabel.Text = summary;
    }

    public void SetConfigSummary(string trainerPath, string networkPath, string checkpointPath)
    {
        var trainerName = FileName(trainerPath, "(none)");
        var networkName = FileName(networkPath, "(none)");
        var checkpointName = FileName(checkpointPath, "(none)");
        _configLabel.Text = $"Trainer:     {trainerName}\nNetwork:     {networkName}\nCheckpoint: {checkpointName}";
        _configLabel.TooltipText = $"Trainer: {trainerPath}\nNetwork: {networkPath}\nCheckpoint: {checkpointPath}";
    }

    public void SetLaunchStatus(string text)
    {
        _launchStatusLabel.Text = text;
        var lower = text.ToLowerInvariant();
        if (lower.Contains("launch") || lower.Contains("starting"))
        {
            _launchStatusLabel.AddThemeColorOverride("font_color", new Color(0.40f, 0.72f, 0.90f));
        }
        else if (lower.Contains("stop") || lower.Contains("block") || lower.Contains("fail"))
        {
            _launchStatusLabel.AddThemeColorOverride("font_color", new Color(0.90f, 0.50f, 0.40f));
        }
        else
        {
            _launchStatusLabel.RemoveThemeColorOverride("font_color");
        }
    }

    private static Label MakeSectionHeader(string text)
    {
        var label = new Label { Text = text };
        label.AddThemeFontSizeOverride("font_size", 13);
        return label;
    }

    private static Control MakeSpacer(int height)
    {
        return new Control { CustomMinimumSize = new Vector2(0f, height) };
    }

    private static void SetMargins(MarginContainer container, int margin)
    {
        container.AddThemeConstantOverride("margin_left", margin);
        container.AddThemeConstantOverride("margin_right", margin);
        container.AddThemeConstantOverride("margin_top", margin);
        container.AddThemeConstantOverride("margin_bottom", margin);
    }

    private static string FileName(string path, string fallback)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return fallback;
        }

        var name = System.IO.Path.GetFileName(path);
        return string.IsNullOrWhiteSpace(name) ? fallback : name;
    }
}
