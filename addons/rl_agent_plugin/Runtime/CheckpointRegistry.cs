using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

public static class CheckpointRegistry
{
    private const string RunsRoot = "res://RL-Agent-Training/runs";

    public static List<string> ListCheckpointPaths()
    {
        var results = new List<string>();
        var runsDir = DirAccess.Open(RunsRoot);
        if (runsDir is null)
        {
            return results;
        }

        runsDir.ListDirBegin();
        while (true)
        {
            var name = runsDir.GetNext();
            if (string.IsNullOrEmpty(name))
            {
                break;
            }

            if (!runsDir.CurrentIsDir() || name.StartsWith("."))
            {
                continue;
            }

            foreach (var checkpointPath in ListRunCheckpoints($"{RunsRoot}/{name}"))
            {
                results.Add(checkpointPath);
            }
        }

        runsDir.ListDirEnd();
        results.Sort((left, right) => string.CompareOrdinal(right, left));
        return results;
    }

    private static IEnumerable<string> ListRunCheckpoints(string runDirectory)
    {
        var runDir = DirAccess.Open(runDirectory);
        if (runDir is null)
        {
            yield break;
        }

        runDir.ListDirBegin();
        while (true)
        {
            var entryName = runDir.GetNext();
            if (string.IsNullOrEmpty(entryName))
            {
                break;
            }

            if (runDir.CurrentIsDir() || entryName.StartsWith(".") || !entryName.EndsWith(".json"))
            {
                continue;
            }

            yield return $"{runDirectory}/{entryName}";
        }

        runDir.ListDirEnd();
    }

    public static string GetLatestCheckpointPath()
    {
        var checkpoints = ListCheckpointPaths();
        return checkpoints.Count > 0 ? checkpoints[0] : string.Empty;
    }

    public static string ResolveCheckpointPath(string preferredPath)
    {
        if (!string.IsNullOrWhiteSpace(preferredPath) && FileAccess.FileExists(preferredPath))
        {
            return preferredPath;
        }

        return GetLatestCheckpointPath();
    }
}
