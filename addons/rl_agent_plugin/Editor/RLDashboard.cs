using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace RlAgentPlugin.Editor;

/// <summary>
/// Live training dashboard added as a main-screen tab (same row as 2D / 3D / Script).
/// Polls the active run's metrics.jsonl and status.json files every two seconds
/// and updates four charts and a stats bar in real time.
/// </summary>
[Tool]
public partial class RLDashboard : Control
{
    // ── Data model ───────────────────────────────────────────────────────────
    private sealed class Metric
    {
        public float EpisodeReward;
        public int   EpisodeLength;
        public float PolicyLoss;
        public float ValueLoss;
        public float Entropy;
        public long  TotalSteps;
        public long  EpisodeCount;
    }

    private sealed class RunStatus
    {
        public string Status       = "idle";
        public long   TotalSteps;
        public long   EpisodeCount;
        public string Message      = "";
    }

    private sealed record RunMeta(string DisplayName, string[] AgentNames);

    // ── UI handles ───────────────────────────────────────────────────────────
    private OptionButton?   _runDropdown;
    private LineEdit?       _prefixFilter;
    private LineEdit?       _renameEdit;
    private Button?         _renameBtn;
    private Label?          _liveBadge;
    private Label?          _headerStatus;
    private ColorRect?      _statusDot;
    private Label?          _statusLabel;
    private Label?          _statAvgReward;
    private Label?          _statBestReward;
    private Label?          _statTotalSteps;
    private Label?          _statEpisodes;
    private LineChartPanel? _rewardChart;
    private LineChartPanel? _lossChart;
    private LineChartPanel? _entropyChart;
    private LineChartPanel? _lengthChart;
    private FileDialog?     _exportDialog;

    // ── State ────────────────────────────────────────────────────────────────
    private readonly List<Metric> _metrics = new();
    private readonly List<string> _runIds  = new();
    private string _selectedRunId      = "";
    private long   _metricsFileOffset;
    private double _pollTimer;
    private double _livePulseAccum;
    private bool   _isLive;
    private int    _lastKnownRunCount = -1;

    private const double PollInterval      = 2.0;
    private const double LivePulseInterval = 0.8;

    // ── Palette ──────────────────────────────────────────────────────────────
    private static readonly Color CRunning    = new(0.20f, 0.85f, 0.35f);
    private static readonly Color CStopped    = new(0.70f, 0.70f, 0.70f);
    private static readonly Color CIdle       = new(0.45f, 0.45f, 0.45f);
    private static readonly Color CReward     = new(0.22f, 0.82f, 0.42f);
    private static readonly Color CPolicyLoss = new(0.92f, 0.42f, 0.22f);
    private static readonly Color CValueLoss  = new(0.35f, 0.62f, 0.92f);
    private static readonly Color CEntropy    = new(0.88f, 0.68f, 0.22f);
    private static readonly Color CLength     = new(0.72f, 0.42f, 0.92f);

    // ── Godot lifecycle ──────────────────────────────────────────────────────
    public override void _Ready()
    {
        SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);
        SizeFlagsHorizontal = SizeFlags.ExpandFill;
        SizeFlagsVertical   = SizeFlags.ExpandFill;
        BuildUi();
        DiscoverAndSelectLatestRun();
    }

    public override void _Process(double delta)
    {
        if (!IsInsideTree() || !Visible) return;

        _pollTimer += delta;
        if (_pollTimer >= PollInterval)
        {
            _pollTimer = 0;
            PollUpdate();
        }

        // Live badge pulse
        if (_isLive && _liveBadge is not null)
        {
            _livePulseAccum += delta;
            if (_livePulseAccum >= LivePulseInterval)
            {
                _livePulseAccum = 0;
                _liveBadge.Modulate = _liveBadge.Modulate.A > 0.5f
                    ? new Color(1, 1, 1, 0.15f)
                    : new Color(1, 1, 1, 1.0f);
            }
        }
    }

    // ── Public API (called by editor plugin) ─────────────────────────────────

    /// <summary>
    /// Called by <see cref="RLAgentPluginEditor"/> immediately after launching a
    /// training bootstrap scene. Auto-selects the new run and shows the LIVE badge.
    /// </summary>
    public void OnTrainingStarted(string runId)
    {
        DiscoverAndSelectLatestRun();

        if (!_runIds.Contains(runId))
        {
            // Run directory may not exist yet; add the entry manually.
            _runIds.Insert(0, runId);
            RebuildDropdownItems();
        }

        SelectRun(runId);
        _isLive = true;
        ShowLiveBadge();
    }

    // ── UI construction ──────────────────────────────────────────────────────
    private void BuildUi()
    {
        var margin = new MarginContainer();
        margin.SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);
        margin.AddThemeConstantOverride("margin_left",   10);
        margin.AddThemeConstantOverride("margin_right",  10);
        margin.AddThemeConstantOverride("margin_top",    8);
        margin.AddThemeConstantOverride("margin_bottom", 8);
        AddChild(margin);

        var vbox = new VBoxContainer();
        vbox.AddThemeConstantOverride("separation", 6);
        margin.AddChild(vbox);

        vbox.AddChild(BuildHeader());
        vbox.AddChild(new HSeparator());
        vbox.AddChild(BuildStatsBar());
        vbox.AddChild(BuildChartGrid());
    }

    private Control BuildHeader()
    {
        var vbox = new VBoxContainer();
        vbox.AddThemeConstantOverride("separation", 4);

        // ── Row 1: title, filter, run selector, status, live badge ───────────
        var row1 = new HBoxContainer();
        row1.AddThemeConstantOverride("separation", 8);
        row1.CustomMinimumSize = new Vector2(0, 30);

        var title = new Label { Text = "RL Training Dashboard" };
        title.AddThemeFontSizeOverride("font_size", 18);
        title.VerticalAlignment = VerticalAlignment.Center;
        row1.AddChild(title);

        row1.AddChild(new Control { SizeFlagsHorizontal = SizeFlags.ExpandFill });

        // Prefix filter
        var filterLabel = new Label { Text = "Filter:", VerticalAlignment = VerticalAlignment.Center };
        row1.AddChild(filterLabel);

        _prefixFilter = new LineEdit
        {
            PlaceholderText = "by prefix…",
            CustomMinimumSize = new Vector2(120, 0),
        };
        _prefixFilter.TextChanged += _ => DiscoverAndSelectLatestRun();
        row1.AddChild(_prefixFilter);

        // Run selector
        var runLabel = new Label { Text = "Run:", VerticalAlignment = VerticalAlignment.Center };
        row1.AddChild(runLabel);

        _runDropdown = new OptionButton
        {
            CustomMinimumSize = new Vector2(250, 0),
            TooltipText       = "Select a training run to inspect",
        };
        _runDropdown.ItemSelected += OnRunSelected;
        row1.AddChild(_runDropdown);

        var refreshBtn = new Button
        {
            Text        = " ⟳ ",
            TooltipText = "Rescan runs directory for new runs",
        };
        refreshBtn.Pressed += () => DiscoverAndSelectLatestRun();
        row1.AddChild(refreshBtn);

        // Status indicator
        var statusBox = new HBoxContainer();
        statusBox.AddThemeConstantOverride("separation", 5);

        _statusDot = new ColorRect
        {
            Color             = CIdle,
            CustomMinimumSize = new Vector2(10, 10),
        };
        _statusDot.SizeFlagsVertical = SizeFlags.ShrinkCenter;
        statusBox.AddChild(_statusDot);

        _statusLabel = new Label { Text = "Idle", VerticalAlignment = VerticalAlignment.Center };
        statusBox.AddChild(_statusLabel);
        row1.AddChild(statusBox);

        // LIVE badge (hidden until training starts)
        _liveBadge = new Label
        {
            Text                = "● LIVE",
            Visible             = false,
            VerticalAlignment   = VerticalAlignment.Center,
        };
        _liveBadge.AddThemeColorOverride("font_color", CRunning);
        _liveBadge.AddThemeFontSizeOverride("font_size", 12);
        row1.AddChild(_liveBadge);

        vbox.AddChild(row1);

        // ── Row 2: rename + export ────────────────────────────────────────────
        var row2 = new HBoxContainer();
        row2.AddThemeConstantOverride("separation", 6);
        row2.CustomMinimumSize = new Vector2(0, 24);

        var nameLabel = new Label { Text = "Name:", VerticalAlignment = VerticalAlignment.Center };
        row2.AddChild(nameLabel);

        _renameEdit = new LineEdit
        {
            PlaceholderText   = "Display name…",
            CustomMinimumSize = new Vector2(200, 0),
            Editable          = false,
        };
        row2.AddChild(_renameEdit);

        _renameBtn = new Button { Text = "Rename", Visible = false };
        _renameBtn.Pressed += () => SaveDisplayName(_renameEdit?.Text ?? "");
        row2.AddChild(_renameBtn);

        row2.AddChild(new Control { SizeFlagsHorizontal = SizeFlags.ExpandFill });

        var exportBtn = new Button
        {
            Text        = "Export Run",
            TooltipText = "Export trained model weights as .rlmodel file(s)",
        };
        exportBtn.Pressed += ExportRun;
        row2.AddChild(exportBtn);

        _headerStatus = new Label { VerticalAlignment = VerticalAlignment.Center };
        _headerStatus.AddThemeFontSizeOverride("font_size", 11);
        _headerStatus.Modulate = new Color(0.75f, 0.75f, 0.75f);
        row2.AddChild(_headerStatus);

        vbox.AddChild(row2);

        return vbox;
    }

    private Control BuildStatsBar()
    {
        var panel = new PanelContainer();
        panel.CustomMinimumSize = new Vector2(0, 48);

        var hbox = new HBoxContainer();
        hbox.AddThemeConstantOverride("separation", 0);
        panel.AddChild(hbox);

        _statAvgReward  = AddStatCard(hbox, "Avg Reward (last 50)", "—", first: true);
        hbox.AddChild(MakeVSep());
        _statBestReward = AddStatCard(hbox, "Best Episode Reward",  "—", first: false);
        hbox.AddChild(MakeVSep());
        _statTotalSteps = AddStatCard(hbox, "Total Steps",          "—", first: false);
        hbox.AddChild(MakeVSep());
        _statEpisodes   = AddStatCard(hbox, "Episodes",             "—", first: false);

        return panel;
    }

    private static Label AddStatCard(HBoxContainer parent, string title, string dflt, bool first)
    {
        var margin = new MarginContainer();
        margin.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        margin.AddThemeConstantOverride("margin_left",   first ? 10 : 18);
        margin.AddThemeConstantOverride("margin_right",  8);
        margin.AddThemeConstantOverride("margin_top",    5);
        margin.AddThemeConstantOverride("margin_bottom", 5);
        parent.AddChild(margin);

        var vbox = new VBoxContainer();
        vbox.AddThemeConstantOverride("separation", 0);
        margin.AddChild(vbox);

        var lbl = new Label { Text = title };
        lbl.AddThemeFontSizeOverride("font_size", 10);
        lbl.Modulate = new Color(0.60f, 0.60f, 0.60f);
        vbox.AddChild(lbl);

        var value = new Label { Text = dflt };
        value.AddThemeFontSizeOverride("font_size", 15);
        vbox.AddChild(value);

        return value;
    }

    private static VSeparator MakeVSep()
    {
        var sep = new VSeparator();
        sep.SizeFlagsVertical = SizeFlags.ShrinkCenter;
        sep.CustomMinimumSize = new Vector2(1, 26);
        return sep;
    }

    private Control BuildChartGrid()
    {
        var grid = new GridContainer { Columns = 2 };
        grid.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        grid.SizeFlagsVertical   = SizeFlags.ExpandFill;
        grid.AddThemeConstantOverride("h_separation", 4);
        grid.AddThemeConstantOverride("v_separation", 4);

        _rewardChart  = MakeChart("Episode Reward");
        _lossChart    = MakeChart("Policy Loss  /  Value Loss");
        _entropyChart = MakeChart("Entropy");
        _lengthChart  = MakeChart("Episode Length");

        grid.AddChild(_rewardChart);
        grid.AddChild(_lossChart);
        grid.AddChild(_entropyChart);
        grid.AddChild(_lengthChart);

        return grid;
    }

    private static LineChartPanel MakeChart(string title)
    {
        var chart = new LineChartPanel { ChartTitle = title };
        chart.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        chart.SizeFlagsVertical   = SizeFlags.ExpandFill;
        chart.CustomMinimumSize   = new Vector2(200, 160);
        return chart;
    }

    // ── Run discovery ─────────────────────────────────────────────────────────

    private void DiscoverAndSelectLatestRun()
    {
        var absDir    = ProjectSettings.GlobalizePath("res://RL-Agent-Training/runs");
        var prevRunId = _selectedRunId;

        _lastKnownRunCount = System.IO.Directory.Exists(absDir)
            ? System.IO.Directory.GetDirectories(absDir).Length
            : 0;

        _runIds.Clear();
        _runDropdown?.Clear();

        if (!System.IO.Directory.Exists(absDir)) return;

        var filterPrefix = _prefixFilter?.Text.Trim() ?? "";

        var dirs = System.IO.Directory.GetDirectories(absDir)
            .Select(System.IO.Path.GetFileName)
            .OfType<string>()
            .Where(n => !string.IsNullOrEmpty(n))
            .Where(n => string.IsNullOrEmpty(filterPrefix)
                        || ExtractRunPrefix(n).Contains(filterPrefix, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(n => n)
            .ToList();

        foreach (var id in dirs)
        {
            var meta  = ReadMeta(System.IO.Path.Combine(absDir, id));
            var label = BuildRunLabel(id, meta.DisplayName);
            _runIds.Add(id);
            _runDropdown?.AddItem(label);
        }

        // Preserve previous selection if it still appears in the filtered list.
        var existingIdx = _runIds.IndexOf(prevRunId);
        if (existingIdx >= 0)
        {
            _runDropdown?.Select(existingIdx);
        }
        else if (_runIds.Count > 0)
        {
            SelectRun(_runIds[0]);
        }
    }

    /// <summary>Rebuilds dropdown item text from _runIds (preserves order, keeps selection).</summary>
    private void RebuildDropdownItems()
    {
        if (_runDropdown is null) return;
        var absDir = ProjectSettings.GlobalizePath("res://RL-Agent-Training/runs");

        _runDropdown.Clear();
        foreach (var id in _runIds)
        {
            var meta  = ReadMeta(System.IO.Path.Combine(absDir, id));
            var label = BuildRunLabel(id, meta.DisplayName);
            _runDropdown.AddItem(label);
        }

        var selIdx = _runIds.IndexOf(_selectedRunId);
        if (selIdx >= 0) _runDropdown.Select(selIdx);
    }

    private void OnRunSelected(long index)
    {
        if (index < 0 || index >= _runIds.Count) return;
        var runId = _runIds[(int)index];
        SelectRun(runId);

        var meta = ReadMeta(ProjectSettings.GlobalizePath($"res://RL-Agent-Training/runs/{runId}"));
        if (_renameEdit is not null) _renameEdit.Text = meta.DisplayName;
    }

    private void SelectRun(string runId)
    {
        _selectedRunId     = runId;
        _metricsFileOffset = 0;
        _metrics.Clear();

        var idx = _runIds.IndexOf(runId);
        if (idx >= 0) _runDropdown?.Select(idx);

        _rewardChart?.ClearSeries();
        _lossChart?.ClearSeries();
        _entropyChart?.ClearSeries();
        _lengthChart?.ClearSeries();

        if (_renameEdit is not null) _renameEdit.Editable = true;
        if (_renameBtn  is not null) _renameBtn.Visible   = true;

        SetStatusUi(new RunStatus { Status = "loading" });
        PollUpdate();
    }

    // ── Polling & data ────────────────────────────────────────────────────────
    private void PollUpdate()
    {
        // Passive new-run detection.
        var absDir = ProjectSettings.GlobalizePath("res://RL-Agent-Training/runs");
        if (System.IO.Directory.Exists(absDir))
        {
            var currentCount = System.IO.Directory.GetDirectories(absDir).Length;
            if (_lastKnownRunCount >= 0 && currentCount != _lastKnownRunCount)
            {
                _lastKnownRunCount = currentCount;
                DiscoverAndSelectLatestRun();
                return; // DiscoverAndSelectLatestRun → SelectRun → PollUpdate handles the rest.
            }
        }

        if (string.IsNullOrEmpty(_selectedRunId)) return;

        var runDir = $"res://RL-Agent-Training/runs/{_selectedRunId}";
        ReadNewMetrics($"{runDir}/metrics.jsonl");
        var status = ReadStatusFile($"{runDir}/status.json");
        SetStatusUi(status);
        RefreshCharts();
        RefreshStats(status);

        // Clear live badge once training explicitly finishes.
        if (_isLive && status.Status is "done" or "stopped")
        {
            _isLive = false;
            HideLiveBadge();
        }
    }

    private void ReadNewMetrics(string resPath)
    {
        var absPath = ProjectSettings.GlobalizePath(resPath);
        if (!System.IO.File.Exists(absPath)) return;

        try
        {
            using var stream = new System.IO.FileStream(
                absPath, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.ReadWrite);

            if (stream.Length <= _metricsFileOffset) return;

            stream.Seek(_metricsFileOffset, System.IO.SeekOrigin.Begin);
            using var reader = new System.IO.StreamReader(stream, leaveOpen: true);

            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var m = ParseMetricLine(line);
                if (m is not null) _metrics.Add(m);
            }

            _metricsFileOffset = stream.Position;
        }
        catch (Exception ex)
        {
            GD.PushWarning($"[RLDashboard] Failed to read metrics: {ex.Message}");
        }
    }

    private static RunStatus ReadStatusFile(string resPath)
    {
        var absPath = ProjectSettings.GlobalizePath(resPath);
        if (!System.IO.File.Exists(absPath)) return new RunStatus();

        try
        {
            var content = System.IO.File.ReadAllText(absPath);
            var variant = Json.ParseString(content);
            if (variant.VariantType != Variant.Type.Dictionary) return new RunStatus();

            var d = variant.AsGodotDictionary();
            return new RunStatus
            {
                Status       = GetString(d, "status", "unknown"),
                TotalSteps   = GetLong(d, "total_steps"),
                EpisodeCount = GetLong(d, "episode_count"),
                Message      = GetString(d, "message", ""),
            };
        }
        catch
        {
            return new RunStatus();
        }
    }

    private static Metric? ParseMetricLine(string line)
    {
        try
        {
            var variant = Json.ParseString(line);
            if (variant.VariantType != Variant.Type.Dictionary) return null;
            var d = variant.AsGodotDictionary();
            return new Metric
            {
                EpisodeReward = GetFloat(d, "episode_reward"),
                EpisodeLength = (int)GetLong(d, "episode_length"),
                PolicyLoss    = GetFloat(d, "policy_loss"),
                ValueLoss     = GetFloat(d, "value_loss"),
                Entropy       = GetFloat(d, "entropy"),
                TotalSteps    = GetLong(d, "total_steps"),
                EpisodeCount  = GetLong(d, "episode_count"),
            };
        }
        catch
        {
            return null;
        }
    }

    // ── Export ────────────────────────────────────────────────────────────────
    private void ExportRun()
    {
        if (string.IsNullOrEmpty(_selectedRunId))
        {
            SetHeaderStatus("No run selected.");
            return;
        }

        EnsureExportDialog();
        _exportDialog!.PopupCentered(new Vector2I(700, 450));
    }

    private void EnsureExportDialog()
    {
        if (_exportDialog is not null && IsInstanceValid(_exportDialog)) return;

        _exportDialog = new FileDialog
        {
            FileMode = FileDialog.FileModeEnum.OpenDir,
            Access   = FileDialog.AccessEnum.Filesystem,
            Title    = "Select Export Folder",
        };
        _exportDialog.DirSelected += OnExportDirSelected;
        AddChild(_exportDialog);
    }

    private void OnExportDirSelected(string dir)
    {
        if (string.IsNullOrEmpty(_selectedRunId)) return;

        var runDirAbs      = ProjectSettings.GlobalizePath($"res://RL-Agent-Training/runs/{_selectedRunId}");
        var checkpointPath = RLModelExporter.FindCheckpointInRunDir(runDirAbs);

        if (checkpointPath is null)
        {
            SetHeaderStatus("Export failed: no checkpoint found in run directory.");
            return;
        }

        var meta       = ReadMeta(runDirAbs);
        var agentNames = meta.AgentNames.Length > 0 ? meta.AgentNames : new[] { _selectedRunId };

        var failed = new List<string>();
        foreach (var agentName in agentNames)
        {
            var destPath = System.IO.Path.Combine(dir, $"{SanitizeFileName(agentName)}.rlmodel");
            if (RLModelExporter.Export(checkpointPath, destPath) != Error.Ok)
                failed.Add(agentName);
        }

        SetHeaderStatus(failed.Count == 0
            ? $"Exported {agentNames.Length} file(s) → {dir}"
            : $"Export failed for: {string.Join(", ", failed)}");
    }

    // ── Rename / display name ─────────────────────────────────────────────────
    private void SaveDisplayName(string name)
    {
        if (string.IsNullOrEmpty(_selectedRunId)) return;

        var runDirAbs = ProjectSettings.GlobalizePath($"res://RL-Agent-Training/runs/{_selectedRunId}");
        var meta      = ReadMeta(runDirAbs);
        WriteMeta(runDirAbs, meta with { DisplayName = name });

        // Update dropdown label in place.
        var idx = _runIds.IndexOf(_selectedRunId);
        if (idx >= 0)
            _runDropdown?.SetItemText(idx, BuildRunLabel(_selectedRunId, name));

        SetHeaderStatus(string.IsNullOrWhiteSpace(name) ? "Name cleared." : $"Renamed to \"{name}\".");
    }

    // ── Meta sidecar ─────────────────────────────────────────────────────────
    private static RunMeta ReadMeta(string runDirAbsPath)
    {
        var metaPath = System.IO.Path.Combine(runDirAbsPath, "meta.json");
        if (!System.IO.File.Exists(metaPath)) return new RunMeta("", Array.Empty<string>());

        try
        {
            var variant = Json.ParseString(System.IO.File.ReadAllText(metaPath));
            if (variant.VariantType != Variant.Type.Dictionary) return new RunMeta("", Array.Empty<string>());

            var d          = variant.AsGodotDictionary();
            var displayName = GetString(d, "display_name", "");
            var agentArr    = d.ContainsKey("agent_names") ? d["agent_names"].AsGodotArray() : new Godot.Collections.Array();
            var agentNames  = agentArr.Select(v => v.AsString()).ToArray();

            return new RunMeta(displayName, agentNames);
        }
        catch
        {
            return new RunMeta("", Array.Empty<string>());
        }
    }

    private static void WriteMeta(string runDirAbsPath, RunMeta meta)
    {
        try
        {
            System.IO.Directory.CreateDirectory(runDirAbsPath);

            var agentArr = new Godot.Collections.Array();
            foreach (var name in meta.AgentNames) agentArr.Add(Variant.From(name));

            var d = new Godot.Collections.Dictionary
            {
                { "display_name", meta.DisplayName },
                { "agent_names",  agentArr },
            };

            System.IO.File.WriteAllText(
                System.IO.Path.Combine(runDirAbsPath, "meta.json"),
                Json.Stringify(d));
        }
        catch (Exception ex)
        {
            GD.PushWarning($"[RLDashboard] Failed to write meta.json: {ex.Message}");
        }
    }

    // ── UI updates ────────────────────────────────────────────────────────────
    private void SetStatusUi(RunStatus status)
    {
        if (_statusLabel is null || _statusDot is null) return;

        switch (status.Status)
        {
            case "running":
                _statusDot.Color  = CRunning;
                _statusLabel.Text = $"Running  •  ep {status.EpisodeCount:N0}  •  {FormatSteps(status.TotalSteps)} steps";
                break;
            case "done":
            case "stopped":
                _statusDot.Color  = CStopped;
                _statusLabel.Text = $"Stopped  •  {status.EpisodeCount:N0} episodes";
                break;
            case "loading":
                _statusDot.Color  = CIdle;
                _statusLabel.Text = "Loading…";
                break;
            default:
                _statusDot.Color  = CIdle;
                _statusLabel.Text = string.IsNullOrEmpty(_selectedRunId) ? "No run selected" : "Idle";
                break;
        }
    }

    private void RefreshCharts()
    {
        if (_metrics.Count == 0) return;

        _rewardChart?.UpdateSeries("Reward", CReward,
            _metrics.Select(m => m.EpisodeReward));

        _lossChart?.UpdateSeries("Policy", CPolicyLoss,
            _metrics.Select(m => m.PolicyLoss));
        _lossChart?.UpdateSeries("Value",  CValueLoss,
            _metrics.Select(m => m.ValueLoss));

        _entropyChart?.UpdateSeries("Entropy", CEntropy,
            _metrics.Select(m => m.Entropy));

        _lengthChart?.UpdateSeries("Length", CLength,
            _metrics.Select(m => (float)m.EpisodeLength));
    }

    private void RefreshStats(RunStatus status)
    {
        if (_metrics.Count == 0) return;

        var window  = _metrics.TakeLast(50).ToList();
        var avg     = window.Average(m => m.EpisodeReward);
        var best    = _metrics.Max(m => m.EpisodeReward);
        var last    = _metrics[^1];
        var steps   = status.TotalSteps   > 0 ? status.TotalSteps   : last.TotalSteps;
        var eps     = status.EpisodeCount > 0 ? status.EpisodeCount : last.EpisodeCount;

        if (_statAvgReward  is not null) _statAvgReward.Text  = avg.ToString("F3");
        if (_statBestReward is not null) _statBestReward.Text = best.ToString("F3");
        if (_statTotalSteps is not null) _statTotalSteps.Text = FormatSteps(steps);
        if (_statEpisodes   is not null) _statEpisodes.Text   = eps.ToString("N0");
    }

    private void SetHeaderStatus(string message)
    {
        if (_headerStatus is not null) _headerStatus.Text = message;
    }

    private void ShowLiveBadge()
    {
        if (_liveBadge is null) return;
        _liveBadge.Visible  = true;
        _liveBadge.Modulate = Colors.White;
        _livePulseAccum     = 0;
    }

    private void HideLiveBadge()
    {
        if (_liveBadge is null) return;
        _liveBadge.Visible  = false;
        _liveBadge.Modulate = Colors.White;
        _livePulseAccum     = 0;
    }

    // ── Run label helpers ─────────────────────────────────────────────────────

    /// <summary>
    /// Builds a human-readable dropdown label. If a display name is set, shows
    /// "display_name (prefix • Mar 14, 14:32)"; otherwise shows "prefix • Mar 14, 14:32".
    /// </summary>
    private static string BuildRunLabel(string runId, string displayName)
    {
        var timeLabel = ParseRunLabel(runId);
        return string.IsNullOrWhiteSpace(displayName)
            ? timeLabel
            : $"{displayName} ({timeLabel})";
    }

    /// <summary>
    /// Parses a RunId of the form "{prefix}_{unix_timestamp}" into a human-readable string.
    /// Falls back to the raw runId if parsing fails.
    /// </summary>
    private static string ParseRunLabel(string runId)
    {
        var lastUnderscore = runId.LastIndexOf('_');
        if (lastUnderscore <= 0 || lastUnderscore >= runId.Length - 1) return runId;

        var prefix       = runId[..lastUnderscore];
        var timestampStr = runId[(lastUnderscore + 1)..];

        if (!double.TryParse(timestampStr, out var unixSeconds)) return runId;

        var dto   = DateTimeOffset.FromUnixTimeSeconds((long)unixSeconds).ToLocalTime();
        var today = DateTimeOffset.Now.Date;
        var time  = dto.DateTime.Date == today
            ? $"Today, {dto:HH:mm}"
            : $"{dto:MMM d, HH:mm}";

        return $"{prefix} • {time}";
    }

    /// <summary>Extracts the prefix from a RunId (everything before the last '_').</summary>
    private static string ExtractRunPrefix(string runId)
    {
        var last = runId.LastIndexOf('_');
        return last > 0 ? runId[..last] : runId;
    }

    private static string SanitizeFileName(string name)
    {
        var invalid = System.IO.Path.GetInvalidFileNameChars();
        return new string(name.Select(c => Array.IndexOf(invalid, c) >= 0 ? '_' : c).ToArray());
    }

    // ── Variant helpers ───────────────────────────────────────────────────────
    private static float GetFloat(Godot.Collections.Dictionary d, string key)
    {
        if (!d.ContainsKey(key)) return 0f;
        var v = d[key];
        return v.VariantType == Variant.Type.Float ? v.AsSingle()
             : v.VariantType == Variant.Type.Int   ? (float)v.AsInt64()
             : 0f;
    }

    private static long GetLong(Godot.Collections.Dictionary d, string key)
    {
        if (!d.ContainsKey(key)) return 0L;
        var v = d[key];
        return v.VariantType == Variant.Type.Int   ? v.AsInt64()
             : v.VariantType == Variant.Type.Float ? (long)v.AsDouble()
             : 0L;
    }

    private static string GetString(Godot.Collections.Dictionary d, string key, string fallback)
    {
        if (!d.ContainsKey(key)) return fallback;
        var v = d[key];
        return v.VariantType == Variant.Type.String ? v.AsString() : fallback;
    }

    private static string FormatSteps(long n) =>
        n >= 1_000_000 ? $"{n / 1_000_000.0:F2}M"
        : n >= 1_000   ? $"{n / 1_000.0:F1}K"
        : n.ToString();
}
