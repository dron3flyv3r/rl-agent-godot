using System;
using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
[Tool]
public partial class RLCheckpoint : Resource
{
    [Export] public string RunId { get; set; } = string.Empty;
    [Export] public long TotalSteps { get; set; }
    [Export] public long EpisodeCount { get; set; }
    [Export] public long UpdateCount { get; set; }
    [Export] public float[] WeightBuffer { get; set; } = Array.Empty<float>();
    [Export] public int[] LayerShapeBuffer { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Serializes the checkpoint to a JSON file at the given Godot path (supports user://).
    /// More reliable than ResourceSaver.Save() for programmatically-created C# resources.
    /// </summary>
    public static Error SaveToFile(RLCheckpoint checkpoint, string path)
    {
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
            { "run_id", checkpoint.RunId },
            { "total_steps", checkpoint.TotalSteps },
            { "episode_count", checkpoint.EpisodeCount },
            { "update_count", checkpoint.UpdateCount },
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
        if (!FileAccess.FileExists(path))
        {
            GD.PushWarning($"[RLCheckpoint] File not found: {path}");
            return null;
        }

        using var file = FileAccess.Open(path, FileAccess.ModeFlags.Read);
        if (file is null)
        {
            GD.PushError($"[RLCheckpoint] Failed to open '{path}' for reading: {FileAccess.GetOpenError()}");
            return null;
        }

        var parsed = Json.ParseString(file.GetAsText());
        if (parsed.VariantType != Variant.Type.Dictionary)
        {
            GD.PushError($"[RLCheckpoint] Invalid JSON format in '{path}'");
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

        return new RLCheckpoint
        {
            RunId = data.ContainsKey("run_id") ? data["run_id"].AsString() : string.Empty,
            TotalSteps = data.ContainsKey("total_steps") ? data["total_steps"].AsInt64() : 0,
            EpisodeCount = data.ContainsKey("episode_count") ? data["episode_count"].AsInt64() : 0,
            UpdateCount = data.ContainsKey("update_count") ? data["update_count"].AsInt64() : 0,
            WeightBuffer = weights,
            LayerShapeBuffer = shapes,
        };
    }
}
