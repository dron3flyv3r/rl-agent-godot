using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Godot;

namespace RlAgentPlugin.Runtime;

// ── Public stats snapshot ─────────────────────────────────────────────────────

/// <summary>
/// Live performance snapshot exposed by <see cref="DistributedMaster"/>.
/// Safe to read from the main thread (e.g. in <c>_Process</c> to update a UI overlay).
/// </summary>
public struct DistributedMasterStats
{
    /// <summary>Background training job is currently in flight.</summary>
    public bool IsTraining;
    /// <summary>Seconds elapsed since the current background training job was scheduled.</summary>
    public float TrainingElapsedSec;
    /// <summary>Wall-clock seconds the most recently completed training update took.</summary>
    public float LastUpdateDurationSec;

    /// <summary>Workers currently connected.</summary>
    public int ConnectedWorkers;
    /// <summary>Workers configured (target count).</summary>
    public int ExpectedWorkers;
    /// <summary>Rollouts received from workers in the current round (resets each update).</summary>
    public int RolloutsThisRound;

    /// <summary>Total training updates completed across all groups.</summary>
    public long TotalUpdates;
    /// <summary>Total simulation steps seen by the master (local + worker).</summary>
    public long TotalSteps;
    /// <summary>Steps contributed by worker processes, cumulative.</summary>
    public long TotalWorkerSteps;
    /// <summary>Steps in the most recently completed training batch.</summary>
    public int LastBatchSteps;

    /// <summary>Rolling-average simulation steps per wall-clock second (last 8 rounds).</summary>
    public float StepsPerSec;

    /// <summary>Losses from the most recent completed update (zeroed until first update).</summary>
    public float LastPolicyLoss;
    public float LastValueLoss;
    public float LastEntropy;
    public float LastClipFraction;
}

// ── Master implementation ─────────────────────────────────────────────────────

/// <summary>
/// Runs on the master process.  Opens a TCP server, accepts worker connections,
/// collects rollout data from each worker, injects it into the local trainers,
/// and broadcasts updated weights after every training update.
///
/// When the trainer implements <see cref="IAsyncTrainer"/> backprop runs on a
/// background thread so the master's game loop never freezes.
/// Read <see cref="GetStats"/> each frame to drive a UI overlay.
/// </summary>
public sealed class DistributedMaster : IDisposable
{
    // ── Worker connection handle ─────────────────────────────────────────────

    private sealed class WorkerConnection
    {
        public TcpClient    Client { get; }
        public BinaryWriter Writer { get; }
        private readonly object _writeLock = new();

        public WorkerConnection(TcpClient client)
        {
            Client = client;
            Writer = new BinaryWriter(client.GetStream());
        }

        public void Send(DistributedMessageType type, string groupId, byte[] payload)
        {
            try   { lock (_writeLock) DistributedProtocol.WriteMessage(Writer, type, groupId, payload); }
            catch (Exception ex) { GD.PushWarning($"[RL Distributed] Send failed: {ex.Message}"); }
        }

        public void Close() => Client.Close();
    }

    // ── State ────────────────────────────────────────────────────────────────

    private readonly int  _port;
    private readonly int  _workerCount;
    private readonly int  _monitorInterval;
    private readonly bool _verbose;
    private readonly Dictionary<string, IDistributedTrainer> _trainers;
    private readonly Dictionary<string, IAsyncTrainer>       _asyncTrainers = new(StringComparer.Ordinal);

    private TcpListener?  _listener;
    private Thread?       _acceptThread;
    private volatile bool _running;

    private readonly List<WorkerConnection> _connections     = new();
    private readonly object                 _connectionsLock = new();

    private readonly Queue<(string groupId, byte[] data)> _pendingRollouts = new();
    private readonly object                               _rolloutsLock    = new();

    // Per-group counters (current round).
    private readonly Dictionary<string, int>  _rolloutsThisRound    = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long> _workerStepsThisRound = new(StringComparer.Ordinal);

    // Async training state.
    private readonly HashSet<string>          _trainingInProgress = new(StringComparer.Ordinal);
    private readonly Dictionary<string, DateTime> _trainStartTime = new(StringComparer.Ordinal);

    // Cumulative stats.
    private readonly Dictionary<string, long>               _totalUpdates     = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long>               _totalWorkerSteps = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long>               _totalStepsAll    = new(StringComparer.Ordinal);
    private readonly Dictionary<string, TrainerUpdateStats> _lastStats        = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float>              _lastUpdateDurSec = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int>                _lastBatchSteps   = new(StringComparer.Ordinal);
    private readonly Dictionary<string, DateTime>           _lastUpdateTime   = new(StringComparer.Ordinal);

    // Rolling steps/sec window (stores (steps, durationSec) for last N rounds).
    private const int StepsWindow = 8;
    private readonly Dictionary<string, Queue<(long steps, float dur)>> _throughputWindow = new(StringComparer.Ordinal);

    // ── Construction / lifecycle ─────────────────────────────────────────────

    public DistributedMaster(
        int  port,
        int  workerCount,
        int  monitorInterval,
        bool verbose,
        Dictionary<string, IDistributedTrainer> trainers)
    {
        _port            = port;
        _workerCount     = workerCount;
        _monitorInterval = monitorInterval;
        _verbose         = verbose;
        _trainers        = trainers;

        foreach (var (g, t) in trainers)
        {
            _rolloutsThisRound[g]    = 0;
            _workerStepsThisRound[g] = 0;
            _totalUpdates[g]         = 0;
            _totalWorkerSteps[g]     = 0;
            _totalStepsAll[g]        = 0;
            _lastUpdateDurSec[g]     = 0f;
            _lastBatchSteps[g]       = 0;
            _lastUpdateTime[g]       = DateTime.UtcNow;
            _throughputWindow[g]     = new Queue<(long, float)>();

            if (t is IAsyncTrainer at) _asyncTrainers[g] = at;
        }
    }

    public int  ConnectedWorkers { get { lock (_connectionsLock) return _connections.Count; } }
    public bool IsTraining       => _trainingInProgress.Count > 0;

    /// <summary>Cumulative episodes completed by all worker processes (counted from Done flags in rollouts).</summary>
    public long TotalWorkerEpisodes { get; private set; }

    /// <summary>Cumulative steps contributed by all worker processes across all groups.</summary>
    public long TotalWorkerSteps
    {
        get
        {
            var total = 0L;
            foreach (var g in _trainers.Keys)
                total += _totalWorkerSteps.GetValueOrDefault(g);
            return total;
        }
    }

    // ── Stats snapshot ────────────────────────────────────────────────────────

    /// <summary>
    /// Returns a combined performance snapshot across all groups.
    /// Safe to call from <c>_Process</c>.
    /// </summary>
    public DistributedMasterStats GetStats(long masterTotalSteps)
    {
        var connected = ConnectedWorkers;

        // Aggregate across groups (typical case: one group).
        var totalUpdates     = 0L;
        var totalWorkerSteps = 0L;
        var lastBatch        = 0;
        var stepsPerSec      = 0f;
        var lastDur          = 0f;
        var policyLoss       = 0f;
        var valueLoss        = 0f;
        var entropy          = 0f;
        var clipFrac         = 0f;
        var groupCount       = Math.Max(1, _trainers.Count);

        foreach (var g in _trainers.Keys)
        {
            totalUpdates     += _totalUpdates.GetValueOrDefault(g);
            totalWorkerSteps += _totalWorkerSteps.GetValueOrDefault(g);
            lastBatch        += _lastBatchSteps.GetValueOrDefault(g);
            stepsPerSec      += CalcStepsPerSec(g);
            lastDur           = Math.Max(lastDur, _lastUpdateDurSec.GetValueOrDefault(g));

            if (_lastStats.TryGetValue(g, out var s))
            {
                policyLoss += s.PolicyLoss;
                valueLoss  += s.ValueLoss;
                entropy    += s.Entropy;
                clipFrac   += s.ClipFraction;
            }
        }

        var trainingSec = 0f;
        foreach (var g in _trainingInProgress)
            if (_trainStartTime.TryGetValue(g, out var t))
                trainingSec = Math.Max(trainingSec, (float)(DateTime.UtcNow - t).TotalSeconds);

        return new DistributedMasterStats
        {
            IsTraining             = IsTraining,
            TrainingElapsedSec     = trainingSec,
            LastUpdateDurationSec  = lastDur,
            ConnectedWorkers       = connected,
            ExpectedWorkers        = _workerCount,
            RolloutsThisRound      = _rolloutsThisRound.Values.Sum(),
            TotalUpdates           = totalUpdates,
            TotalSteps             = masterTotalSteps + totalWorkerSteps,
            TotalWorkerSteps       = totalWorkerSteps,
            LastBatchSteps         = lastBatch,
            StepsPerSec            = stepsPerSec,
            LastPolicyLoss         = policyLoss / groupCount,
            LastValueLoss          = valueLoss  / groupCount,
            LastEntropy            = entropy    / groupCount,
            LastClipFraction       = clipFrac   / groupCount,
        };
    }

    private float CalcStepsPerSec(string groupId)
    {
        if (!_throughputWindow.TryGetValue(groupId, out var window) || window.Count == 0)
            return 0f;
        var totalSteps = 0L;
        var totalDur   = 0f;
        foreach (var (s, d) in window) { totalSteps += s; totalDur += d; }
        return totalDur > 0f ? totalSteps / totalDur : 0f;
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    public void Start()
    {
        _running  = true;
        _listener = new TcpListener(IPAddress.Loopback, _port);
        _listener.Start();
        GD.Print($"[RL Distributed] Master listening on port {_port}, expecting {_workerCount} worker(s).");
        if (_asyncTrainers.Count > 0)
            GD.Print("[RL Distributed] Async training active — no main-thread freeze during backprop.");

        _acceptThread = new Thread(AcceptLoop) { IsBackground = true, Name = "DistMaster-Accept" };
        _acceptThread.Start();
    }

    // ── Background networking ─────────────────────────────────────────────────

    private void AcceptLoop()
    {
        while (_running)
        {
            try
            {
                if (!(_listener?.Pending() ?? false)) { Thread.Sleep(10); continue; }
                var client = _listener!.AcceptTcpClient();
                client.NoDelay = true;
                var conn = new WorkerConnection(client);
                lock (_connectionsLock) _connections.Add(conn);
                GD.Print($"[RL Distributed] Worker connected ({ConnectedWorkers}/{_workerCount}).");
                var reader = new BinaryReader(client.GetStream());
                new Thread(() => ClientReadLoop(conn, reader)) { IsBackground = true, Name = "DistMaster-Client" }.Start();
            }
            catch when (!_running) { break; }
            catch (Exception ex) { GD.PushWarning($"[RL Distributed] Accept error: {ex.Message}"); }
        }
    }

    private void ClientReadLoop(WorkerConnection conn, BinaryReader reader)
    {
        try
        {
            while (_running && conn.Client.Connected)
            {
                var (type, groupId, payload) = DistributedProtocol.ReadMessage(reader);
                switch (type)
                {
                    case DistributedMessageType.Hello:
                        if (_trainers.TryGetValue(groupId, out var t))
                            conn.Send(DistributedMessageType.Weights, groupId, t.ExportWeights());
                        if (_verbose) GD.Print($"[RL Distributed] Worker hello '{groupId}'.");
                        break;
                    case DistributedMessageType.Rollout:
                        lock (_rolloutsLock) _pendingRollouts.Enqueue((groupId, payload));
                        break;
                }
            }
        }
        catch (Exception ex) when (_running) { GD.PushWarning($"[RL Distributed] Worker disconnected: {ex.Message}"); }
        finally
        {
            lock (_connectionsLock) _connections.Remove(conn);
            conn.Close();
            GD.Print($"[RL Distributed] Worker removed ({ConnectedWorkers} remaining).");
        }
    }

    // ── Main-thread API ───────────────────────────────────────────────────────

    public void ProcessIncoming()
    {
        lock (_rolloutsLock)
        {
            while (_pendingRollouts.Count > 0)
            {
                var (groupId, data) = _pendingRollouts.Dequeue();
                if (!_trainers.TryGetValue(groupId, out var trainer)) continue;
                trainer.InjectRollout(data);
                var steps    = data.Length >= 4 ? BitConverter.ToInt32(data, 0) : 0;
                var episodes = DistributedProtocol.CountEpisodesInRollout(data);
                _rolloutsThisRound[groupId]    = _rolloutsThisRound.GetValueOrDefault(groupId) + 1;
                _workerStepsThisRound[groupId] = _workerStepsThisRound.GetValueOrDefault(groupId) + steps;
                TotalWorkerEpisodes           += episodes;
                if (_verbose)
                    GD.Print($"[RL Distributed] Rollout '{groupId}': {steps} steps ({_rolloutsThisRound[groupId]}/{ConnectedWorkers}).");
            }
        }
    }

    public TrainerUpdateStats? TickUpdate(
        IDistributedTrainer trainer,
        string groupId,
        long   totalSteps,
        long   episodeCount)
    {
        ProcessIncoming();

        // Poll running background job.
        if (_trainingInProgress.Contains(groupId))
        {
            if (!_asyncTrainers.TryGetValue(groupId, out var running)) return null;
            var result = running.TryPollResult(groupId, totalSteps, episodeCount);
            if (result is null) return null;
            _trainingInProgress.Remove(groupId);
            OnTrainComplete(groupId, trainer, result, totalSteps);
            return result;
        }

        // Gate on training readiness.
        // On-policy: synchronous barrier — wait for all workers before training.
        // Off-policy: no gate — master trains on its own schedule; workers asynchronously
        //             enrich the replay buffer. TryUpdate/TryScheduleBackgroundUpdate
        //             apply their own internal conditions (warmup, step cadence).
        if (!trainer.IsOffPolicy)
        {
            var connected = ConnectedWorkers;
            if (connected > 0)
            {
                if (_rolloutsThisRound.GetValueOrDefault(groupId) < connected) return null;
            }
            else
            {
                if (!trainer.IsRolloutReady) return null;
            }
        }

        // Prefer async (non-blocking backprop).
        if (_asyncTrainers.TryGetValue(groupId, out var asyncT))
        {
            if (asyncT.TryScheduleBackgroundUpdate(groupId, totalSteps, episodeCount))
            {
                _trainingInProgress.Add(groupId);
                _trainStartTime[groupId] = DateTime.UtcNow;
                if (_verbose) GD.Print($"[RL Distributed] Background training scheduled for '{groupId}'.");
                return null;
            }
        }

        // Synchronous fallback.
        var stats = trainer.TryUpdate(groupId, totalSteps, episodeCount);
        if (stats is not null) OnTrainComplete(groupId, trainer, stats, totalSteps);
        return stats;
    }

    private void OnTrainComplete(
        string groupId,
        IDistributedTrainer trainer,
        TrainerUpdateStats stats,
        long totalSteps)
    {
        var now     = DateTime.UtcNow;
        var durSec  = _trainStartTime.TryGetValue(groupId, out var start)
                      ? (float)(now - start).TotalSeconds : 0f;
        var wSteps  = _workerStepsThisRound.GetValueOrDefault(groupId);
        var bSteps  = (int)wSteps; // local steps were already in the buffer

        // Throughput window.
        if (_throughputWindow.TryGetValue(groupId, out var win))
        {
            win.Enqueue((bSteps, Math.Max(durSec, 0.001f)));
            while (win.Count > StepsWindow) win.Dequeue();
        }

        _totalUpdates[groupId]     = _totalUpdates.GetValueOrDefault(groupId) + 1;
        _totalWorkerSteps[groupId] = _totalWorkerSteps.GetValueOrDefault(groupId) + wSteps;
        _lastStats[groupId]        = stats;
        _lastUpdateDurSec[groupId] = durSec;
        _lastBatchSteps[groupId]   = bSteps;
        _lastUpdateTime[groupId]   = now;

        if (_verbose) GD.Print($"[RL Distributed] Broadcasting weights '{groupId}' to {ConnectedWorkers} worker(s).");
        BroadcastWeights(groupId, trainer);

        if (_monitorInterval > 0 && _totalUpdates[groupId] % _monitorInterval == 0)
            PrintMonitorSummary(groupId, totalSteps, ConnectedWorkers);

        _rolloutsThisRound[groupId]    = 0;
        _workerStepsThisRound[groupId] = 0;
    }

    // ── Monitor console output ────────────────────────────────────────────────

    private void PrintMonitorSummary(string groupId, long totalSteps, int connected)
    {
        var updates  = _totalUpdates.GetValueOrDefault(groupId);
        var wSteps   = _totalWorkerSteps.GetValueOrDefault(groupId);
        var batch    = _lastBatchSteps.GetValueOrDefault(groupId);
        var dur      = _lastUpdateDurSec.GetValueOrDefault(groupId);
        var sps      = CalcStepsPerSec(groupId);
        var stats    = _lastStats.GetValueOrDefault(groupId);
        var sources  = connected + 1;

        var sb = new StringBuilder();
        sb.AppendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        sb.AppendLine($"[RL Distributed]  Update #{updates}  |  {groupId}");
        sb.AppendLine($"  Workers      : {connected}/{_workerCount}  ({sources} sources incl. master)");
        sb.AppendLine($"  Total steps  : {totalSteps + wSteps:N0}  (master: {totalSteps:N0}  workers: {wSteps:N0})");
        sb.AppendLine($"  Batch size   : {batch} worker steps + local rollout");
        sb.AppendLine($"  Steps/sec    : {sps:N0}");
        if (dur > 0.01f) sb.AppendLine($"  Update time  : {dur:F2}s");
        if (stats is not null)
        {
            sb.AppendLine($"  Policy loss  : {stats.PolicyLoss:F4}");
            sb.AppendLine($"  Value loss   : {stats.ValueLoss:F4}");
            sb.AppendLine($"  Entropy      : {stats.Entropy:F4}");
            if (stats.ClipFraction > 0f) sb.AppendLine($"  Clip frac    : {stats.ClipFraction:F4}");
        }
        sb.Append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        GD.Print(sb.ToString());
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void BroadcastWeights(string groupId, IDistributedTrainer trainer)
    {
        var bytes = trainer.ExportWeights();
        List<WorkerConnection> snapshot;
        lock (_connectionsLock) snapshot = new List<WorkerConnection>(_connections);
        foreach (var conn in snapshot)
        {
            var c = conn; var g = groupId; var w = bytes;
            ThreadPool.QueueUserWorkItem(_ => c.Send(DistributedMessageType.Weights, g, w));
        }
    }

    public void Shutdown()
    {
        List<WorkerConnection> snapshot;
        lock (_connectionsLock) snapshot = new List<WorkerConnection>(_connections);
        foreach (var conn in snapshot)
            conn.Send(DistributedMessageType.Shutdown, "", Array.Empty<byte>());
    }

    public void Dispose()
    {
        _running = false;
        _listener?.Stop();
    }
}
