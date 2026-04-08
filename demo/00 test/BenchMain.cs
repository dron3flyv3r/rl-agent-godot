using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using RlAgentPlugin.Demo.Benchmarks;

namespace RlAgentPlugin.Demo;

public partial class BenchMain : Node
{
    [Export] public AlgorithmBenchRunMode RunMode { get; set; } = AlgorithmBenchRunMode.Catalog;
    [Export] public bool QuitWhenFinished { get; set; } = false;

    private readonly List<string> _logLines = new();
    private RichTextLabel? _outputLabel;

    public override void _Ready()
    {
        CreateOverlay();
        Log("[AlgoBench] Scene initialized. Waiting one frame before starting benchmarks...");
        CallDeferred(MethodName.StartBenchmarks);
    }

    private async void StartBenchmarks()
    {
        await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);

        var catalog = AlgorithmBenchCatalog.CreateDefault();
        var runner = new AlgorithmBenchRunner();
        var smokeCases = catalog.Where(c => c.Suite == AlgorithmBenchSuite.Smoke).ToArray();
        var performanceCases = catalog.Where(c => c.Suite == AlgorithmBenchSuite.Performance && c.IsImplemented).ToArray();

        Log($"[AlgoBench] Scene started. mode={RunMode} quit_when_finished={QuitWhenFinished}");
        Log($"[AlgoBench] Implemented cases: smoke={smokeCases.Length} performance={performanceCases.Length}");
        Log("[AlgoBench] Progress is mirrored both here and in the Output panel.");

        switch (RunMode)
        {
            case AlgorithmBenchRunMode.Catalog:
                Log(runner.DescribeCatalog(catalog));
                Quit(0);
                break;

            case AlgorithmBenchRunMode.Smoke:
                Log($"[AlgoBench] Running smoke suite ({smokeCases.Length} case(s))...");
                var smokeResults = runner.RunSmoke(smokeCases, Log);
                Log(runner.FormatSummary(smokeResults, "Smoke"));
                Quit(smokeResults.All(r => r.Passed) ? 0 : 1);
                break;

            case AlgorithmBenchRunMode.Performance:
                Log($"[AlgoBench] Running performance suite ({performanceCases.Length} case(s))...");
                var performanceRunner = new AlgorithmBenchPerformanceRunner();
                var performanceResults = await performanceRunner.RunPerformanceAsync(performanceCases, Log, YieldFrame);
                Log(runner.FormatSummary(performanceResults, "Performance"));
                Quit(performanceResults.All(r => r.Passed) ? 0 : 1);
                break;

            case AlgorithmBenchRunMode.AllImplemented:
                Log("[AlgoBench] Running all implemented suites...");
                var runnable = smokeCases.Where(c => c.IsImplemented).ToArray();
                var allResults = runner.RunSmoke(runnable, Log);
                var perfResults = await new AlgorithmBenchPerformanceRunner().RunPerformanceAsync(performanceCases, Log, YieldFrame);
                Log(runner.FormatSummary(allResults, "All Implemented / Smoke"));
                Log(runner.FormatSummary(perfResults, "All Implemented / Performance"));
                Quit(allResults.All(r => r.Passed) && perfResults.All(r => r.Passed) ? 0 : 1);
                break;

            default:
                Log($"[AlgoBench] Unsupported run mode: {RunMode}.");
                Quit(1);
                break;
        }
    }

    private void CreateOverlay()
    {
        var canvas = new CanvasLayer();
        AddChild(canvas);

        var panel = new ColorRect
        {
            Color = new Color(0.05f, 0.07f, 0.09f, 0.92f),
            MouseFilter = Control.MouseFilterEnum.Ignore,
        };
        panel.SetAnchorsPreset(Control.LayoutPreset.FullRect);
        canvas.AddChild(panel);

        _outputLabel = new RichTextLabel
        {
            BbcodeEnabled = false,
            FitContent = false,
            ScrollFollowing = true,
            MouseFilter = Control.MouseFilterEnum.Ignore,
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        _outputLabel.SetAnchorsPreset(Control.LayoutPreset.FullRect);
        _outputLabel.OffsetLeft = 16f;
        _outputLabel.OffsetTop = 16f;
        _outputLabel.OffsetRight = -16f;
        _outputLabel.OffsetBottom = -16f;
        canvas.AddChild(_outputLabel);
    }

    private void Log(string message)
    {
        GD.Print(message);

        _logLines.Add(message);
        if (_logLines.Count > 200)
            _logLines.RemoveAt(0);

        if (_outputLabel is not null)
            _outputLabel.Text = string.Join("\n", _logLines);
    }

    private async Task YieldFrame()
    {
        if (!IsInsideTree())
            return;

        await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
    }

    private void Quit(int exitCode)
    {
        if (QuitWhenFinished)
            GetTree().Quit(exitCode);
    }
}
