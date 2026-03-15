using System;
using System.Collections.Generic;
using System.IO;
using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Editor;

/// <summary>
/// Exports a trained checkpoint to a compact self-describing binary .rlmodel file.
///
/// Binary layout (little-endian):
///   [8  bytes] magic "RLMODEL\0"
///   [2  bytes] uint16 version = 1
///   [4  bytes] int32  obs_size
///   [4  bytes] int32  action_count
///   [4  bytes] int32  layer_count
///   Per layer:
///     [4  bytes] int32  in_size
///     [4  bytes] int32  out_size
///     [4  bytes] int32  activation  (0=linear, 1=Tanh, 2=Relu — matches internal encoding)
///     [n*4 bytes] float32[in_size * out_size]  weights (row-major)
///     [m*4 bytes] float32[out_size]            biases
///
/// Layer order: trunk layers in order, then policy head, then value head.
/// </summary>
public static class RLModelExporter
{
    private static readonly byte[] Magic =
    {
        (byte)'R', (byte)'L', (byte)'M', (byte)'O',
        (byte)'D', (byte)'E', (byte)'L', 0,
    };

    /// <summary>
    /// Reads a checkpoint JSON at <paramref name="checkpointAbsPath"/> and writes a
    /// .rlmodel binary to <paramref name="destAbsPath"/>.
    /// Returns <see cref="Error.Ok"/> on success, <see cref="Error.Failed"/> otherwise.
    /// </summary>
    public static Error Export(string checkpointAbsPath, string destAbsPath)
    {
        var checkpoint = LoadCheckpointJson(checkpointAbsPath);
        if (checkpoint is null) return Error.Failed;

        var shapes  = checkpoint.LayerShapeBuffer;
        var weights = checkpoint.WeightBuffer;

        if (shapes.Length == 0 || shapes.Length % 3 != 0)
        {
            GD.PushError($"[RLModelExporter] Invalid LayerShapeBuffer length {shapes.Length} in {checkpointAbsPath}");
            return Error.Failed;
        }

        var layerCount  = shapes.Length / 3;
        var obsSize     = shapes[0];
        // Policy head is second-to-last layer; its out_size is the action count.
        var actionCount = shapes[(layerCount - 2) * 3 + 1];

        try
        {
            var dir = Path.GetDirectoryName(destAbsPath);
            if (!string.IsNullOrEmpty(dir))
                Directory.CreateDirectory(dir);

            using var stream = File.Open(destAbsPath, FileMode.Create, System.IO.FileAccess.Write);
            using var writer = new BinaryWriter(stream);

            writer.Write(Magic);
            writer.Write((ushort)1); // version
            writer.Write(obsSize);
            writer.Write(actionCount);
            writer.Write(layerCount);

            var weightOffset = 0;
            for (var i = 0; i < layerCount; i++)
            {
                var inSize     = shapes[i * 3];
                var outSize    = shapes[i * 3 + 1];
                var activation = shapes[i * 3 + 2];

                writer.Write(inSize);
                writer.Write(outSize);
                writer.Write(activation);

                var numWeights = inSize * outSize;
                for (var j = 0; j < numWeights; j++)
                    writer.Write(weights[weightOffset++]);
                for (var j = 0; j < outSize; j++)
                    writer.Write(weights[weightOffset++]);
            }

            GD.Print($"[RLModelExporter] Exported {layerCount} layers → {destAbsPath}");
            return Error.Ok;
        }
        catch (Exception ex)
        {
            GD.PushError($"[RLModelExporter] Export failed: {ex.Message}");
            return Error.Failed;
        }
    }

    /// <summary>
    /// Searches <paramref name="runDirAbsPath"/> for a checkpoint JSON file
    /// (any .json that is not status.json or meta.json).
    /// Returns the absolute path, or null if none found.
    /// </summary>
    public static string? FindCheckpointInRunDir(string runDirAbsPath)
    {
        if (!Directory.Exists(runDirAbsPath)) return null;

        foreach (var file in Directory.GetFiles(runDirAbsPath, "*.json"))
        {
            var name = Path.GetFileName(file);
            if (name == "status.json" || name == "meta.json") continue;
            return file;
        }

        return null;
    }

    // ── Checkpoint loading ────────────────────────────────────────────────────

    private static RLCheckpoint? LoadCheckpointJson(string absPath)
    {
        if (!File.Exists(absPath))
        {
            GD.PushError($"[RLModelExporter] Checkpoint not found: {absPath}");
            return null;
        }

        try
        {
            var json    = File.ReadAllText(absPath);
            var variant = Json.ParseString(json);

            if (variant.VariantType != Variant.Type.Dictionary)
            {
                GD.PushError($"[RLModelExporter] Invalid checkpoint JSON at {absPath}");
                return null;
            }

            var data      = variant.AsGodotDictionary();
            var weightArr = data.ContainsKey("weights") ? data["weights"].AsGodotArray() : new Godot.Collections.Array();
            var shapeArr  = data.ContainsKey("shapes")  ? data["shapes"].AsGodotArray()  : new Godot.Collections.Array();

            var w = new float[weightArr.Count];
            for (var i = 0; i < weightArr.Count; i++) w[i] = weightArr[i].AsSingle();

            var s = new int[shapeArr.Count];
            for (var i = 0; i < shapeArr.Count; i++) s[i] = (int)shapeArr[i].AsInt64();

            return new RLCheckpoint { WeightBuffer = w, LayerShapeBuffer = s };
        }
        catch (Exception ex)
        {
            GD.PushError($"[RLModelExporter] Failed to parse checkpoint: {ex.Message}");
            return null;
        }
    }
}
