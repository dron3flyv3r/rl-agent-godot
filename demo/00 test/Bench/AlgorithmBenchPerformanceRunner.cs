using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo.Benchmarks;

public sealed class AlgorithmBenchPerformanceRunner
{
    public async Task<AlgorithmBenchResult[]> RunPerformanceAsync(
        IReadOnlyList<AlgorithmBenchCase> cases,
        Action<string>? log = null,
        Func<Task>? yieldFrame = null)
    {
        var results = new List<AlgorithmBenchResult>(cases.Count);
        for (var index = 0; index < cases.Count; index++)
        {
            var benchCase = cases[index];
            log?.Invoke($"[AlgoBench] Perf {index + 1}/{cases.Count} starting: {benchCase.Id} actors={benchCase.Performance.ParallelActors} warmup={benchCase.Performance.WarmupTicks} measure={benchCase.Performance.MeasureTicks}");
            if (yieldFrame is not null)
                await yieldFrame();

            var result = await RunPerformanceCaseAsync(benchCase, log, yieldFrame);
            log?.Invoke($"[AlgoBench] Perf {index + 1}/{cases.Count} finished: {benchCase.Id} passed={result.Passed} env_steps_sec={result.EnvStepsPerSecond.ToString("0.##", CultureInfo.InvariantCulture)} elapsed_ms={result.ElapsedMilliseconds.ToString("0.###", CultureInfo.InvariantCulture)}");
            results.Add(result);
        }
        return results.ToArray();
    }

    private async Task<AlgorithmBenchResult> RunPerformanceCaseAsync(
        AlgorithmBenchCase benchCase,
        Action<string>? log,
        Func<Task>? yieldFrame)
    {
        var config = benchCase.Performance;
        var actorCount = Math.Max(1, config.ParallelActors);
        var environments = new IAlgorithmBenchEnvironment[actorCount];
        var observations = new float[actorCount][];
        var episodeRewards = new float[actorCount];
        var completedEpisodeRewards = new List<float>();
        var decisionDurationsMs = new List<double>(config.MeasureTicks);
        var updateDurationsMs = new List<double>();
        var updates = 0;
        var measuredUpdates = 0;
        var totalSteps = 0;
        var measuredSteps = 0;
        var measuredEpisodes = 0;
        var measuredDecisionCount = 0L;
        var totalDecisionMilliseconds = 0.0;

        try
        {
            for (var index = 0; index < actorCount; index++)
            {
                environments[index] = benchCase.CreateEnvironment();
                observations[index] = environments[index].Reset(benchCase.Seed + index);
            }

            var trainer = AlgorithmBenchRunner.CreateTrainer(benchCase, environments[0]);
            var warmupTicks = Math.Max(0, config.WarmupTicks);
            var measureTicks = Math.Max(1, config.MeasureTicks);
            var totalTicks = warmupTicks + measureTicks;
            var measureStopwatch = new Stopwatch();
            var lastLoggedPercent = -1;

            for (var tick = 0; tick < totalTicks; tick++)
            {
                if (tick == warmupTicks)
                {
                    log?.Invoke($"[AlgoBench] {benchCase.Id}: warmup complete, measurement phase starting.");
                    if (yieldFrame is not null)
                        await yieldFrame();
                    measureStopwatch.Start();
                }

                var batch = new VectorBatch(actorCount, environments[0].ObservationSize);
                for (var actorIndex = 0; actorIndex < actorCount; actorIndex++)
                    batch.SetRow(actorIndex, observations[actorIndex]);

                var decisionWatch = Stopwatch.StartNew();
                var decisions = actorCount == 1
                    ? new[] { trainer.SampleAction(observations[0]) }
                    : trainer.SampleActions(batch);
                decisionWatch.Stop();

                var nextObservations = new float[actorCount][];
                var rewards = new float[actorCount];
                var dones = new bool[actorCount];

                for (var actorIndex = 0; actorIndex < actorCount; actorIndex++)
                {
                    var step = environments[actorIndex].Step(new AlgorithmBenchAction(
                        decisions[actorIndex].DiscreteAction,
                        decisions[actorIndex].ContinuousActions));

                    totalSteps++;
                    episodeRewards[actorIndex] += step.Reward;
                    nextObservations[actorIndex] = step.Observation;
                    rewards[actorIndex] = step.Reward;
                    dones[actorIndex] = step.Done;
                }

                var nextBatch = new VectorBatch(actorCount, environments[0].ObservationSize);
                for (var actorIndex = 0; actorIndex < actorCount; actorIndex++)
                    nextBatch.SetRow(actorIndex, nextObservations[actorIndex]);
                var nextValues = actorCount == 1
                    ? new[] { trainer.EstimateValue(nextObservations[0]) }
                    : trainer.EstimateValues(nextBatch);

                for (var actorIndex = 0; actorIndex < actorCount; actorIndex++)
                {
                    trainer.RecordTransition(new Transition
                    {
                        Observation = observations[actorIndex],
                        DiscreteAction = decisions[actorIndex].DiscreteAction,
                        ContinuousActions = decisions[actorIndex].ContinuousActions,
                        Reward = rewards[actorIndex],
                        Done = dones[actorIndex],
                        NextObservation = nextObservations[actorIndex],
                        OldLogProbability = decisions[actorIndex].LogProbability,
                        Value = decisions[actorIndex].Value,
                        NextValue = dones[actorIndex] ? 0f : nextValues[actorIndex],
                        GroupAgentSlot = actorIndex,
                    });

                    observations[actorIndex] = dones[actorIndex]
                        ? environments[actorIndex].Reset(benchCase.Seed + totalSteps + actorIndex)
                        : nextObservations[actorIndex];

                    if (dones[actorIndex])
                    {
                        if (tick >= warmupTicks)
                        {
                            measuredEpisodes++;
                            completedEpisodeRewards.Add(episodeRewards[actorIndex]);
                        }

                        episodeRewards[actorIndex] = 0f;
                    }
                }

                var updateWatch = Stopwatch.StartNew();
                var updateStats = trainer.TryUpdate(benchCase.Id, totalSteps, measuredEpisodes);
                updateWatch.Stop();

                if (updateStats is not null)
                    updates++;

                if (tick >= warmupTicks)
                {
                    measuredSteps += actorCount;
                    measuredDecisionCount += actorCount;
                    totalDecisionMilliseconds += decisionWatch.Elapsed.TotalMilliseconds;
                    decisionDurationsMs.Add(decisionWatch.Elapsed.TotalMilliseconds);

                    if (updateStats is not null)
                    {
                        measuredUpdates++;
                        updateDurationsMs.Add(updateWatch.Elapsed.TotalMilliseconds);
                    }
                }

                if (tick >= warmupTicks && measureTicks >= 8)
                {
                    var measuredTick = tick - warmupTicks + 1;
                    var percent = (measuredTick * 100) / measureTicks;
                    var bucket = percent / 25;
                    if (bucket > lastLoggedPercent && percent >= 25)
                    {
                        lastLoggedPercent = bucket;
                        log?.Invoke($"[AlgoBench] {benchCase.Id}: measurement {Math.Min(percent, 100)}% complete.");
                        if (yieldFrame is not null)
                            await yieldFrame();
                    }
                }

                if (yieldFrame is not null && ((tick + 1) % 16 == 0))
                    await yieldFrame();
            }

            measureStopwatch.Stop();
            var elapsedMs = Math.Max(0.001, measureStopwatch.Elapsed.TotalMilliseconds);
            var meanReward = completedEpisodeRewards.Count > 0 ? completedEpisodeRewards.Average() : 0f;

            return new AlgorithmBenchResult
            {
                CaseId = benchCase.Id,
                Name = benchCase.Name,
                Algorithm = benchCase.Algorithm,
                Suite = benchCase.Suite,
                Passed = true,
                Episodes = measuredEpisodes,
                Steps = measuredSteps,
                Updates = measuredUpdates,
                MeanEpisodeReward = meanReward,
                ElapsedMilliseconds = elapsedMs,
                EnvStepsPerSecond = measuredSteps / (elapsedMs / 1000.0),
                DecisionsPerSecond = measuredDecisionCount / Math.Max(0.001, totalDecisionMilliseconds / 1000.0),
                UpdatesPerSecond = measuredUpdates / (elapsedMs / 1000.0),
                DecisionMillisecondsP95 = Percentile(decisionDurationsMs, 0.95),
                UpdateMillisecondsP95 = Percentile(updateDurationsMs, 0.95),
                Detail = BuildDetail(benchCase, environments[0], actorCount, updates),
            };
        }
        catch (Exception ex)
        {
            return new AlgorithmBenchResult
            {
                CaseId = benchCase.Id,
                Name = benchCase.Name,
                Algorithm = benchCase.Algorithm,
                Suite = benchCase.Suite,
                Passed = false,
                Episodes = measuredEpisodes,
                Steps = measuredSteps,
                Updates = measuredUpdates,
                MeanEpisodeReward = completedEpisodeRewards.Count > 0 ? completedEpisodeRewards.Average() : 0f,
                ElapsedMilliseconds = 0d,
                Detail = ex.Message,
            };
        }
    }

    private static string BuildDetail(
        AlgorithmBenchCase benchCase,
        IAlgorithmBenchEnvironment environment,
        int actorCount,
        int totalUpdates)
    {
        var networkGraph = benchCase.CreateNetworkGraph();
        var layerSizes = string.Join("x", networkGraph.GetLayerSizes().Where(size => size > 0));
        return string.Create(CultureInfo.InvariantCulture, $"env={environment.Name} actors={actorCount} obs={environment.ObservationSize} discrete={environment.DiscreteActionCount} continuous={environment.ContinuousActionDimensions} net={layerSizes} total_updates={totalUpdates}");
    }

    private static double Percentile(List<double> values, double percentile)
    {
        if (values.Count == 0)
            return 0d;

        values.Sort();
        var index = (int)Math.Ceiling((values.Count - 1) * percentile);
        return values[Math.Clamp(index, 0, values.Count - 1)];
    }
}
