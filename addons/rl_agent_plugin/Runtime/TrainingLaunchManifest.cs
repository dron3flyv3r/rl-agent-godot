using System;
using Godot;

namespace RlAgentPlugin.Runtime;

public sealed class TrainingLaunchManifest
{
    public const string ActiveManifestPath = "user://rl_agent_plugin/active_manifest.json";

    public string ScenePath { get; set; } = string.Empty;
    public string AcademyNodePath { get; set; } = string.Empty;
    public string RunId { get; set; } = string.Empty;
    public string RunDirectory { get; set; } = string.Empty;
    public string TrainerConfigPath { get; set; } = string.Empty;
    public string NetworkConfigPath { get; set; } = string.Empty;
    public string CheckpointPath { get; set; } = string.Empty;
    public string MetricsPath { get; set; } = string.Empty;
    public string StatusPath { get; set; } = string.Empty;
    public int CheckpointSaveIntervalUpdates { get; set; } = 10;
    public float SimulationSpeed { get; set; } = 1.0f;

    public static TrainingLaunchManifest CreateDefault() => new();

    public Error SaveToUserStorage()
    {
        var directoryError = EnsureParentDirectory(ActiveManifestPath);
        if (directoryError != Error.Ok)
        {
            return directoryError;
        }

        if (!string.IsNullOrWhiteSpace(RunDirectory))
        {
            var runDirectoryError = DirAccess.MakeDirRecursiveAbsolute(ProjectSettings.GlobalizePath(RunDirectory));
            if (runDirectoryError != Error.Ok)
            {
                return runDirectoryError;
            }
        }

        using var file = FileAccess.Open(ActiveManifestPath, FileAccess.ModeFlags.Write);
        if (file is null)
        {
            return FileAccess.GetOpenError();
        }

        file.StoreString(Json.Stringify(ToDictionary(), "\t"));
        return Error.Ok;
    }

    public static TrainingLaunchManifest? LoadFromUserStorage()
    {
        if (!FileAccess.FileExists(ActiveManifestPath))
        {
            return null;
        }

        using var file = FileAccess.Open(ActiveManifestPath, FileAccess.ModeFlags.Read);
        if (file is null)
        {
            return null;
        }

        var parsedManifest = Json.ParseString(file.GetAsText());
        if (parsedManifest.VariantType != Variant.Type.Dictionary)
        {
            return null;
        }

        var data = parsedManifest.AsGodotDictionary();
        return new TrainingLaunchManifest
        {
            ScenePath = ReadString(data, nameof(ScenePath)),
            AcademyNodePath = ReadString(data, nameof(AcademyNodePath)),
            RunId = ReadString(data, nameof(RunId)),
            RunDirectory = ReadString(data, nameof(RunDirectory)),
            TrainerConfigPath = ReadString(data, nameof(TrainerConfigPath)),
            NetworkConfigPath = ReadString(data, nameof(NetworkConfigPath)),
            CheckpointPath = ReadString(data, nameof(CheckpointPath)),
            MetricsPath = ReadString(data, nameof(MetricsPath)),
            StatusPath = ReadString(data, nameof(StatusPath)),
            CheckpointSaveIntervalUpdates = ReadInt(data, nameof(CheckpointSaveIntervalUpdates), 10),
            SimulationSpeed = ReadFloat(data, nameof(SimulationSpeed), 1.0f),
        };
    }

    private Godot.Collections.Dictionary ToDictionary()
    {
        return new Godot.Collections.Dictionary
        {
            { nameof(ScenePath), ScenePath },
            { nameof(AcademyNodePath), AcademyNodePath },
            { nameof(RunId), RunId },
            { nameof(RunDirectory), RunDirectory },
            { nameof(TrainerConfigPath), TrainerConfigPath },
            { nameof(NetworkConfigPath), NetworkConfigPath },
            { nameof(CheckpointPath), CheckpointPath },
            { nameof(MetricsPath), MetricsPath },
            { nameof(StatusPath), StatusPath },
            { nameof(CheckpointSaveIntervalUpdates), CheckpointSaveIntervalUpdates },
            { nameof(SimulationSpeed), SimulationSpeed },
        };
    }

    private static Error EnsureParentDirectory(string filePath)
    {
        var normalizedPath = filePath.Replace('\\', '/');
        var lastSlash = normalizedPath.LastIndexOf('/');
        if (lastSlash < 0)
        {
            return Error.Ok;
        }

        var directoryPath = normalizedPath[..lastSlash];
        return DirAccess.MakeDirRecursiveAbsolute(ProjectSettings.GlobalizePath(directoryPath));
    }

    private static string ReadString(Godot.Collections.Dictionary dictionary, string key)
    {
        return dictionary.ContainsKey(key) ? dictionary[key].ToString() : string.Empty;
    }

    private static int ReadInt(Godot.Collections.Dictionary dictionary, string key, int defaultValue)
    {
        if (!dictionary.ContainsKey(key))
        {
            return defaultValue;
        }

        var value = dictionary[key];
        return value.VariantType == Variant.Type.Int ? (int)value : defaultValue;
    }

    private static float ReadFloat(Godot.Collections.Dictionary dictionary, string key, float defaultValue)
    {
        if (!dictionary.ContainsKey(key))
        {
            return defaultValue;
        }

        var value = dictionary[key];
        return value.VariantType switch
        {
            Variant.Type.Float => (float)(double)value,
            Variant.Type.Int => (int)value,
            _ => defaultValue,
        };
    }
}
