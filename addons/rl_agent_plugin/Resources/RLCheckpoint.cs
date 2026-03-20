using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
[Tool]
public partial class RLCheckpoint : Resource
{
    public const int CurrentFormatVersion = 3;
    public const string PpoAlgorithm = "PPO";
    public const string SacAlgorithm = "SAC";

    [Export] public int FormatVersion { get; set; } = CurrentFormatVersion;
    [Export] public string RunId { get; set; } = string.Empty;
    [Export] public long TotalSteps { get; set; }
    [Export] public long EpisodeCount { get; set; }
    [Export] public long UpdateCount { get; set; }
    [Export] public float RewardSnapshot { get; set; }
    [Export] public string Algorithm { get; set; } = PpoAlgorithm;
    [Export] public int ObservationSize { get; set; }
    [Export] public int DiscreteActionCount { get; set; }
    [Export] public int ContinuousActionDimensions { get; set; }
    [Export] public int[] GraphLayerSizes { get; set; } = Array.Empty<int>();
    [Export] public int[] GraphLayerActivations { get; set; } = Array.Empty<int>();
    [Export] public int GraphOptimizer { get; set; }
    [Export] public float[] WeightBuffer { get; set; } = Array.Empty<float>();
    [Export] public int[] LayerShapeBuffer { get; set; } = Array.Empty<int>();
    public Dictionary<string, string[]> DiscreteActionLabels { get; set; } = new(StringComparer.Ordinal);
    public Dictionary<string, RLContinuousActionRange> ContinuousActionRanges { get; set; } = new(StringComparer.Ordinal);
    public Dictionary<string, float> Hyperparams { get; set; } = new(StringComparer.Ordinal);

    /// <summary>
    /// Serializes the checkpoint to a JSON file at the given Godot path (supports user://).
    /// More reliable than ResourceSaver.Save() for programmatically-created C# resources.
    /// </summary>
    public static Error SaveToFile(RLCheckpoint checkpoint, string path)
    {
        checkpoint.FormatVersion = CurrentFormatVersion;

        var dir = path.GetBaseDir();
        if (!string.IsNullOrEmpty(dir))
        {
            DirAccess.MakeDirRecursiveAbsolute(ProjectSettings.GlobalizePath(dir));
        }

        using var file = FileAccess.Open(path, FileAccess.ModeFlags.Write);
        if (file is null)
        {
            var err = FileAccess.GetOpenError();
            GD.PushError($"[RLCheckpoint] Failed to open '{path}' for writing: {err}");
            return err;
        }

        var weightArray = new Godot.Collections.Array();
        foreach (var w in checkpoint.WeightBuffer)
        {
            weightArray.Add(Variant.From(w));
        }

        var shapeArray = new Godot.Collections.Array();
        foreach (var s in checkpoint.LayerShapeBuffer)
        {
            shapeArray.Add(Variant.From(s));
        }

        var data = new Godot.Collections.Dictionary
        {
            { "format_version", checkpoint.FormatVersion },
            { "run_id", checkpoint.RunId },
            { "total_steps", checkpoint.TotalSteps },
            { "episode_count", checkpoint.EpisodeCount },
            { "update_count", checkpoint.UpdateCount },
            { "meta", checkpoint.CreateMetadataDictionary() },
            { "weights", weightArray },
            { "shapes", shapeArray },
        };

        file.StoreString(Json.Stringify(data));
        GD.Print($"[RLCheckpoint] Saved {checkpoint.WeightBuffer.Length} weights → {path}");
        return Error.Ok;
    }

    /// <summary>
    /// Loads a checkpoint previously saved with SaveToFile().
    /// </summary>
    public static RLCheckpoint? LoadFromFile(string path)
    {
        var resolvedPath = ResolvePath(path);
        if (!FileAccess.FileExists(resolvedPath))
        {
            GD.PushWarning($"[RLCheckpoint] File not found: {resolvedPath}");
            return null;
        }

        using var file = FileAccess.Open(resolvedPath, FileAccess.ModeFlags.Read);
        if (file is null)
        {
            GD.PushError($"[RLCheckpoint] Failed to open '{resolvedPath}' for reading: {FileAccess.GetOpenError()}");
            return null;
        }

        var parsed = Json.ParseString(file.GetAsText());
        if (parsed.VariantType != Variant.Type.Dictionary)
        {
            GD.PushError($"[RLCheckpoint] Invalid JSON format in '{resolvedPath}'");
            return null;
        }

        var data = parsed.AsGodotDictionary();
        var weightArr = data.ContainsKey("weights") ? data["weights"].AsGodotArray() : new Godot.Collections.Array();
        var shapeArr = data.ContainsKey("shapes") ? data["shapes"].AsGodotArray() : new Godot.Collections.Array();
        var weights = new float[weightArr.Count];
        for (var i = 0; i < weightArr.Count; i++)
        {
            weights[i] = weightArr[i].AsSingle();
        }

        var shapes = new int[shapeArr.Count];
        for (var i = 0; i < shapeArr.Count; i++)
        {
            shapes[i] = (int)shapeArr[i].AsInt64();
        }

        var checkpoint = new RLCheckpoint
        {
            FormatVersion = data.ContainsKey("format_version") ? (int)data["format_version"].AsInt64() : 1,
            RunId = data.ContainsKey("run_id") ? data["run_id"].AsString() : string.Empty,
            TotalSteps = data.ContainsKey("total_steps") ? data["total_steps"].AsInt64() : 0,
            EpisodeCount = data.ContainsKey("episode_count") ? data["episode_count"].AsInt64() : 0,
            UpdateCount = data.ContainsKey("update_count") ? data["update_count"].AsInt64() : 0,
            WeightBuffer = weights,
            LayerShapeBuffer = shapes,
        };

        // Meta was introduced in format v2; v1 checkpoints use the legacy derivation path.
        if (checkpoint.FormatVersion >= 2
            && data.ContainsKey("meta")
            && data["meta"].VariantType == Variant.Type.Dictionary)
        {
            checkpoint.ApplyMetadataDictionary(data["meta"].AsGodotDictionary());
        }
        else
        {
            checkpoint.PopulateLegacyMetadata();
        }

        return checkpoint;
    }

    /// <summary>
    /// Loads only the header and metadata fields from a checkpoint JSON file.
    /// WeightBuffer and LayerShapeBuffer will be empty — suitable for fast
    /// dashboard display without allocating large weight arrays.
    /// Returns null if the file cannot be read or parsed.
    /// </summary>
    public static RLCheckpoint? LoadMetadataOnly(string path)
    {
        var resolvedPath = ResolvePath(path);
        if (!FileAccess.FileExists(resolvedPath)) return null;

        try
        {
            using var file = FileAccess.Open(resolvedPath, FileAccess.ModeFlags.Read);
            if (file is null) return null;

            var parsed = Json.ParseString(file.GetAsText());
            if (parsed.VariantType != Variant.Type.Dictionary) return null;

            var data = parsed.AsGodotDictionary();
            var checkpoint = new RLCheckpoint
            {
                FormatVersion    = data.ContainsKey("format_version") ? (int)data["format_version"].AsInt64() : 1,
                RunId            = data.ContainsKey("run_id")         ? data["run_id"].AsString()             : string.Empty,
                TotalSteps       = data.ContainsKey("total_steps")    ? data["total_steps"].AsInt64()         : 0,
                EpisodeCount     = data.ContainsKey("episode_count")  ? data["episode_count"].AsInt64()       : 0,
                UpdateCount      = data.ContainsKey("update_count")   ? data["update_count"].AsInt64()        : 0,
                WeightBuffer     = Array.Empty<float>(),
                LayerShapeBuffer = Array.Empty<int>(),
            };

            if (checkpoint.FormatVersion >= 2
                && data.ContainsKey("meta")
                && data["meta"].VariantType == Variant.Type.Dictionary)
            {
                checkpoint.ApplyMetadataDictionary(data["meta"].AsGodotDictionary());
            }
            else
            {
                checkpoint.PopulateLegacyMetadata();
            }

            return checkpoint;
        }
        catch
        {
            return null;
        }
    }

    public void PopulateLegacyMetadata()
    {
        FormatVersion = 1;
        Algorithm = PpoAlgorithm;
        ObservationSize = LayerShapeBuffer.Length >= 3 ? LayerShapeBuffer[0] : 0;
        DiscreteActionCount = DeriveLegacyDiscreteActionCount(LayerShapeBuffer);
        ContinuousActionDimensions = 0;
        GraphLayerSizes = DeriveLegacyHiddenLayerSizes(LayerShapeBuffer);
        GraphLayerActivations = Array.Empty<int>();
        GraphOptimizer = 0;
        DiscreteActionLabels = new Dictionary<string, string[]>(StringComparer.Ordinal);
        ContinuousActionRanges = new Dictionary<string, RLContinuousActionRange>(StringComparer.Ordinal);
        Hyperparams = new Dictionary<string, float>(StringComparer.Ordinal);
    }

    internal string CreateMetadataJson()
    {
        return Json.Stringify(CreateMetadataDictionary());
    }

    internal void ApplyMetadataJson(string json)
    {
        var parsed = Json.ParseString(json);
        if (parsed.VariantType != Variant.Type.Dictionary)
        {
            throw new InvalidOperationException("Checkpoint metadata JSON is not a dictionary.");
        }

        ApplyMetadataDictionary(parsed.AsGodotDictionary());
    }

    internal Godot.Collections.Dictionary CreateMetadataDictionary()
    {
        var graphLayerSizes = new Godot.Collections.Array();
        foreach (var size in GraphLayerSizes) graphLayerSizes.Add(size);

        var graphLayerActivations = new Godot.Collections.Array();
        foreach (var activation in GraphLayerActivations) graphLayerActivations.Add(activation);

        var hyperparams = new Godot.Collections.Dictionary();
        foreach (var (key, value) in Hyperparams)
        {
            hyperparams[key] = value;
        }

        var discreteActionLabels = new Godot.Collections.Dictionary();
        foreach (var (key, labels) in DiscreteActionLabels)
        {
            var labelArray = new Godot.Collections.Array();
            foreach (var label in labels)
            {
                labelArray.Add(label);
            }

            discreteActionLabels[key] = labelArray;
        }

        var continuousActionRanges = new Godot.Collections.Dictionary();
        foreach (var (key, range) in ContinuousActionRanges)
        {
            continuousActionRanges[key] = new Godot.Collections.Dictionary
            {
                { "dims", range.Dimensions },
                { "min", range.Min },
                { "max", range.Max },
            };
        }

        return new Godot.Collections.Dictionary
        {
            { "algorithm", Algorithm },
            { "reward_snapshot", RewardSnapshot },
            { "obs_size", ObservationSize },
            { "discrete_action_count", DiscreteActionCount },
            { "continuous_action_dims", ContinuousActionDimensions },
            { "graph_layer_sizes", graphLayerSizes },
            { "graph_layer_activations", graphLayerActivations },
            { "graph_optimizer", GraphOptimizer },
            { "hyperparams", hyperparams },
            { "discrete_action_labels", discreteActionLabels },
            { "continuous_action_ranges", continuousActionRanges },
        };
    }

    internal void ApplyMetadataDictionary(Godot.Collections.Dictionary metadata)
    {
        FormatVersion = CurrentFormatVersion;
        Algorithm = metadata.ContainsKey("algorithm") ? metadata["algorithm"].AsString() : PpoAlgorithm;
        RewardSnapshot = metadata.ContainsKey("reward_snapshot") ? metadata["reward_snapshot"].AsSingle() : 0f;
        ObservationSize = metadata.ContainsKey("obs_size") ? (int)metadata["obs_size"].AsInt64() : 0;
        DiscreteActionCount = metadata.ContainsKey("discrete_action_count") ? (int)metadata["discrete_action_count"].AsInt64() : 0;
        ContinuousActionDimensions = metadata.ContainsKey("continuous_action_dims") ? (int)metadata["continuous_action_dims"].AsInt64() : 0;

        if (metadata.ContainsKey("graph_layer_sizes"))
        {
            GraphLayerSizes = ReadIntArray(metadata["graph_layer_sizes"].AsGodotArray());
            GraphLayerActivations = metadata.ContainsKey("graph_layer_activations")
                ? ReadIntArray(metadata["graph_layer_activations"].AsGodotArray())
                : Array.Empty<int>();
            GraphOptimizer = metadata.ContainsKey("graph_optimizer") ? (int)metadata["graph_optimizer"].AsInt64() : 0;
        }
        else if (metadata.ContainsKey("hidden_layer_sizes"))
        {
            // Legacy checkpoint: sizes only, activations default to Tanh, optimizer defaults to Adam.
            GraphLayerSizes = ReadIntArray(metadata["hidden_layer_sizes"].AsGodotArray());
            GraphLayerActivations = Array.Empty<int>();
            GraphOptimizer = 0;
        }
        else
        {
            GraphLayerSizes = Array.Empty<int>();
            GraphLayerActivations = Array.Empty<int>();
            GraphOptimizer = 0;
        }

        Hyperparams = new Dictionary<string, float>(StringComparer.Ordinal);
        if (metadata.ContainsKey("hyperparams") && metadata["hyperparams"].VariantType == Variant.Type.Dictionary)
        {
            foreach (var key in metadata["hyperparams"].AsGodotDictionary().Keys)
            {
                var name = key.AsString();
                Hyperparams[name] = metadata["hyperparams"].AsGodotDictionary()[key].AsSingle();
            }
        }

        DiscreteActionLabels = new Dictionary<string, string[]>(StringComparer.Ordinal);
        if (metadata.ContainsKey("discrete_action_labels") && metadata["discrete_action_labels"].VariantType == Variant.Type.Dictionary)
        {
            foreach (var key in metadata["discrete_action_labels"].AsGodotDictionary().Keys)
            {
                var name = key.AsString();
                DiscreteActionLabels[name] = ReadStringArray(metadata["discrete_action_labels"].AsGodotDictionary()[key].AsGodotArray());
            }
        }

        ContinuousActionRanges = new Dictionary<string, RLContinuousActionRange>(StringComparer.Ordinal);
        if (metadata.ContainsKey("continuous_action_ranges") && metadata["continuous_action_ranges"].VariantType == Variant.Type.Dictionary)
        {
            foreach (var key in metadata["continuous_action_ranges"].AsGodotDictionary().Keys)
            {
                var name = key.AsString();
                var rangeDict = metadata["continuous_action_ranges"].AsGodotDictionary()[key];
                if (rangeDict.VariantType != Variant.Type.Dictionary)
                {
                    continue;
                }

                var dictionary = rangeDict.AsGodotDictionary();
                ContinuousActionRanges[name] = new RLContinuousActionRange
                {
                    Dimensions = dictionary.ContainsKey("dims") ? (int)dictionary["dims"].AsInt64() : 0,
                    Min = dictionary.ContainsKey("min") ? dictionary["min"].AsSingle() : -1f,
                    Max = dictionary.ContainsKey("max") ? dictionary["max"].AsSingle() : 1f,
                };
            }
        }
    }

    private static string ResolvePath(string path)
    {
        return (path.StartsWith("res://", StringComparison.Ordinal) || path.StartsWith("user://", StringComparison.Ordinal))
            ? ProjectSettings.GlobalizePath(path)
            : path;
    }

    private static int DeriveLegacyDiscreteActionCount(IReadOnlyList<int> layerShapes)
    {
        if (layerShapes.Count < 6 || layerShapes.Count % 3 != 0)
        {
            return 0;
        }

        var layerCount = layerShapes.Count / 3;
        return layerShapes[(layerCount - 2) * 3 + 1];
    }

    private static int[] DeriveLegacyHiddenLayerSizes(IReadOnlyList<int> layerShapes)
    {
        if (layerShapes.Count < 6 || layerShapes.Count % 3 != 0)
        {
            return Array.Empty<int>();
        }

        var layerCount = layerShapes.Count / 3;
        var hiddenLayerCount = Math.Max(0, layerCount - 2);
        var hiddenSizes = new int[hiddenLayerCount];
        for (var index = 0; index < hiddenLayerCount; index++)
        {
            hiddenSizes[index] = layerShapes[index * 3 + 1];
        }

        return hiddenSizes;
    }

    private static int[] ReadIntArray(Godot.Collections.Array values)
    {
        var result = new int[values.Count];
        for (var index = 0; index < values.Count; index++)
        {
            result[index] = (int)values[index].AsInt64();
        }

        return result;
    }

    private static string[] ReadStringArray(Godot.Collections.Array values)
    {
        var result = new string[values.Count];
        for (var index = 0; index < values.Count; index++)
        {
            result[index] = values[index].AsString();
        }

        return result;
    }
}
