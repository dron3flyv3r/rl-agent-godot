using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo.Benchmarks;

public sealed class AlgorithmBenchRunner
{
    public string DescribeCatalog(IReadOnlyList<AlgorithmBenchCase> cases)
    {
        var sb = new StringBuilder();
        sb.AppendLine("Algorithm benchmark catalog");
        sb.AppendLine("===========================");
        sb.AppendLine("This scene is code-only: one root node, no agent nodes, no academy nodes.");
        sb.AppendLine("Implemented today: direct-trainer smoke cases.");
        sb.AppendLine("Implemented next layer: synthetic throughput benchmarks with batched actors and small/large networks.");
        sb.AppendLine("Planned after that: seeded learning regressions and full TrainingBootstrap system benches.");
        sb.AppendLine();

        foreach (var suiteGroup in cases.GroupBy(c => c.Suite).OrderBy(g => g.Key))
        {
            sb.AppendLine($"{suiteGroup.Key} cases");
            foreach (var benchCase in suiteGroup)
            {
                sb.Append("- ");
                sb.Append(benchCase.Id);
                sb.Append(" | ");
                sb.Append(benchCase.Algorithm);
                sb.Append(" | ");
                sb.Append(benchCase.Task);
                sb.Append(" | ");
                sb.Append(benchCase.IsImplemented ? "implemented" : "planned");
                if (benchCase.Suite == AlgorithmBenchSuite.Performance)
                {
                    sb.Append(" | actors=");
                    sb.Append(benchCase.Performance.ParallelActors);
                    sb.Append(" warmup=");
                    sb.Append(benchCase.Performance.WarmupTicks);
                    sb.Append(" measure=");
                    sb.Append(benchCase.Performance.MeasureTicks);
                }
                if (!string.IsNullOrWhiteSpace(benchCase.Notes))
                {
                    sb.Append(" | ");
                    sb.Append(benchCase.Notes);
                }
                sb.AppendLine();
            }

            sb.AppendLine();
        }

        sb.AppendLine("Known gap: trainer RNG seeding is not externally controllable yet, so learning/perf thresholds are advisory until that is addressed.");
        return sb.ToString().TrimEnd();
    }

    public AlgorithmBenchResult[] RunSmoke(IReadOnlyList<AlgorithmBenchCase> cases, Action<string>? log = null)
    {
        var results = new List<AlgorithmBenchResult>(cases.Count);
        for (var index = 0; index < cases.Count; index++)
        {
            var benchCase = cases[index];
            log?.Invoke($"[AlgoBench] Smoke {index + 1}/{cases.Count} starting: {benchCase.Id}");
            var result = RunSmokeCase(benchCase);
            log?.Invoke($"[AlgoBench] Smoke {index + 1}/{cases.Count} finished: {benchCase.Id} passed={result.Passed} elapsed_ms={result.ElapsedMilliseconds.ToString("0.###", CultureInfo.InvariantCulture)}");
            results.Add(result);
        }
        return results.ToArray();
    }

    public string FormatSummary(IReadOnlyList<AlgorithmBenchResult> results, string label)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"{label} benchmark summary");
        sb.AppendLine("=======================");
        AppendHighlights(sb, results);
        sb.AppendLine();
        foreach (var result in results)
        {
            sb.Append(result.Passed ? "[PASS] " : "[FAIL] ");
            sb.Append(result.CaseId);
            sb.Append(" | steps=");
            sb.Append(result.Steps);
            sb.Append(" episodes=");
            sb.Append(result.Episodes);
            sb.Append(" updates=");
            sb.Append(result.Updates);
            sb.Append(" mean_reward=");
            sb.Append(result.MeanEpisodeReward.ToString("0.###", CultureInfo.InvariantCulture));
            sb.Append(" elapsed_ms=");
            sb.Append(result.ElapsedMilliseconds.ToString("0.###", CultureInfo.InvariantCulture));
            if (result.Suite == AlgorithmBenchSuite.Performance)
            {
                sb.Append(" env_steps_sec=");
                sb.Append(result.EnvStepsPerSecond.ToString("0.##", CultureInfo.InvariantCulture));
                sb.Append(" decisions_sec=");
                sb.Append(result.DecisionsPerSecond.ToString("0.##", CultureInfo.InvariantCulture));
                sb.Append(" updates_sec=");
                sb.Append(result.UpdatesPerSecond.ToString("0.##", CultureInfo.InvariantCulture));
                sb.Append(" decision_p95_ms=");
                sb.Append(result.DecisionMillisecondsP95.ToString("0.###", CultureInfo.InvariantCulture));
                sb.Append(" update_p95_ms=");
                sb.Append(result.UpdateMillisecondsP95.ToString("0.###", CultureInfo.InvariantCulture));
            }
            if (!string.IsNullOrWhiteSpace(result.Detail))
            {
                sb.Append(" | ");
                sb.Append(result.Detail);
            }
            sb.AppendLine();
        }

        var passCount = results.Count(r => r.Passed);
        var failCount = results.Count - passCount;
        sb.AppendLine();
        sb.AppendLine($"Pass={passCount} Fail={failCount}");
        return sb.ToString().TrimEnd();
    }

    private static void AppendHighlights(StringBuilder sb, IReadOnlyList<AlgorithmBenchResult> results)
    {
        var passCount = results.Count(r => r.Passed);
        var failCount = results.Count - passCount;
        var totalElapsedMs = results.Sum(r => r.ElapsedMilliseconds);
        var totalSteps = results.Sum(r => r.Steps);
        var totalUpdates = results.Sum(r => r.Updates);

        sb.AppendLine("Highlights");
        sb.AppendLine($"cases={results.Count} pass={passCount} fail={failCount} total_elapsed_ms={totalElapsedMs.ToString("0.###", CultureInfo.InvariantCulture)} total_steps={totalSteps} total_updates={totalUpdates}");

        var failed = results.Where(r => !r.Passed).Select(r => r.CaseId).ToArray();
        if (failed.Length > 0)
            sb.AppendLine($"failed_cases={string.Join(", ", failed)}");

        var perfResults = results.Where(r => r.Suite == AlgorithmBenchSuite.Performance && r.Passed).ToArray();
        if (perfResults.Length == 0)
            return;

        var fastestEnv = perfResults.MaxBy(r => r.EnvStepsPerSecond);
        var fastestUpdate = perfResults.Where(r => r.Updates > 0).MaxBy(r => r.UpdatesPerSecond);
        var slowestDecision = perfResults.MaxBy(r => r.DecisionMillisecondsP95);
        var slowestUpdate = perfResults.Where(r => r.Updates > 0).MaxBy(r => r.UpdateMillisecondsP95);

        if (fastestEnv is not null)
        {
            sb.AppendLine(
                $"fastest_env={fastestEnv.CaseId} env_steps_sec={fastestEnv.EnvStepsPerSecond.ToString("0.##", CultureInfo.InvariantCulture)} decisions_sec={fastestEnv.DecisionsPerSecond.ToString("0.##", CultureInfo.InvariantCulture)}");
        }

        if (fastestUpdate is not null)
        {
            sb.AppendLine(
                $"fastest_update={fastestUpdate.CaseId} updates_sec={fastestUpdate.UpdatesPerSecond.ToString("0.##", CultureInfo.InvariantCulture)} update_p95_ms={fastestUpdate.UpdateMillisecondsP95.ToString("0.###", CultureInfo.InvariantCulture)}");
        }

        if (slowestDecision is not null)
        {
            sb.AppendLine(
                $"slowest_decision={slowestDecision.CaseId} decision_p95_ms={slowestDecision.DecisionMillisecondsP95.ToString("0.###", CultureInfo.InvariantCulture)}");
        }

        if (slowestUpdate is not null)
        {
            sb.AppendLine(
                $"slowest_update={slowestUpdate.CaseId} update_p95_ms={slowestUpdate.UpdateMillisecondsP95.ToString("0.###", CultureInfo.InvariantCulture)}");
        }

        foreach (var algorithmGroup in perfResults.GroupBy(r => r.Algorithm).OrderBy(g => g.Key))
        {
            var meanEnvSteps = algorithmGroup.Average(r => r.EnvStepsPerSecond);
            var meanDecisionP95 = algorithmGroup.Average(r => r.DecisionMillisecondsP95);
            var meanUpdateP95 = algorithmGroup.Average(r => r.UpdateMillisecondsP95);
            sb.AppendLine(
                $"algo={algorithmGroup.Key} mean_env_steps_sec={meanEnvSteps.ToString("0.##", CultureInfo.InvariantCulture)} mean_decision_p95_ms={meanDecisionP95.ToString("0.###", CultureInfo.InvariantCulture)} mean_update_p95_ms={meanUpdateP95.ToString("0.###", CultureInfo.InvariantCulture)}");
        }
    }

    private AlgorithmBenchResult RunSmokeCase(AlgorithmBenchCase benchCase)
    {
        var stopwatch = Stopwatch.StartNew();
        var episodeRewards = new List<float>(benchCase.EpisodeBudget);
        var updates = 0;
        var steps = 0;

        try
        {
            var environment = benchCase.CreateEnvironment();
            var trainer = CreateTrainer(benchCase, environment);

            for (var episodeIndex = 0; episodeIndex < benchCase.EpisodeBudget && steps < benchCase.StepBudget; episodeIndex++)
            {
                var observation = environment.Reset(benchCase.Seed + episodeIndex);
                var episodeReward = 0f;

                while (steps < benchCase.StepBudget)
                {
                    var decision = trainer.SampleAction(observation);
                    var step = environment.Step(new AlgorithmBenchAction(decision.DiscreteAction, decision.ContinuousActions));
                    steps++;
                    episodeReward += step.Reward;

                    trainer.RecordTransition(BuildTransition(trainer, observation, step, decision));

                    var updateStats = trainer.TryUpdate(benchCase.Id, steps, episodeIndex);
                    if (updateStats is not null)
                        updates++;

                    observation = step.Observation;
                    if (step.Done)
                        break;
                }

                episodeRewards.Add(episodeReward);
            }

            var checkpoint = trainer.CreateCheckpoint(benchCase.Id, steps, episodeRewards.Count, updates);
            var resumeEnv = benchCase.CreateEnvironment();
            var resumedTrainer = CreateTrainer(benchCase, resumeEnv);
            resumedTrainer.LoadFromCheckpoint(checkpoint);
            var probeObservation = resumeEnv.Reset(benchCase.Seed + 10_000);
            var probeDecision = resumedTrainer.SampleAction(probeObservation);
            ValidateDecision(environment, probeDecision);

            stopwatch.Stop();
            return new AlgorithmBenchResult
            {
                CaseId = benchCase.Id,
                Name = benchCase.Name,
                Algorithm = benchCase.Algorithm,
                Suite = benchCase.Suite,
                Passed = true,
                Episodes = episodeRewards.Count,
                Steps = steps,
                Updates = updates,
                MeanEpisodeReward = episodeRewards.Count > 0 ? episodeRewards.Average() : 0f,
                ElapsedMilliseconds = stopwatch.Elapsed.TotalMilliseconds,
                Detail = $"env={environment.Name} checkpoint_resume=ok",
            };
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            return new AlgorithmBenchResult
            {
                CaseId = benchCase.Id,
                Name = benchCase.Name,
                Algorithm = benchCase.Algorithm,
                Suite = benchCase.Suite,
                Passed = false,
                Episodes = episodeRewards.Count,
                Steps = steps,
                Updates = updates,
                MeanEpisodeReward = episodeRewards.Count > 0 ? episodeRewards.Average() : 0f,
                ElapsedMilliseconds = stopwatch.Elapsed.TotalMilliseconds,
                Detail = ex.Message,
            };
        }
    }

    internal static ITrainer CreateTrainer(AlgorithmBenchCase benchCase, IAlgorithmBenchEnvironment environment)
    {
        if (benchCase.Algorithm == RLAlgorithmKind.MCTS)
        {
            if (environment is not IPlanningBenchEnvironment planningEnvironment)
                throw new InvalidOperationException($"Case '{benchCase.Id}' requires an environment implementing IPlanningBenchEnvironment.");
            MctsTrainer.SetEnvironmentModel(planningEnvironment);
        }

        var config = new PolicyGroupConfig
        {
            GroupId = benchCase.Id,
            RunId = benchCase.Id,
            Algorithm = benchCase.Algorithm,
            TrainerConfig = benchCase.CreateTrainerConfig(),
            NetworkGraph = benchCase.CreateNetworkGraph(),
            ActionDefinitions = BuildActionDefinitions(environment),
            ObservationSize = environment.ObservationSize,
            DiscreteActionCount = environment.DiscreteActionCount,
            ContinuousActionDimensions = environment.ContinuousActionDimensions,
        };

        return TrainerFactory.Create(config);
    }

    internal static RLActionDefinition[] BuildActionDefinitions(IAlgorithmBenchEnvironment environment)
    {
        if (environment.DiscreteActionCount > 0)
        {
            var labels = Enumerable.Range(0, environment.DiscreteActionCount)
                .Select(index => $"a{index}")
                .ToArray();
            return new[] { new RLActionDefinition("action", RLActionVariableType.Discrete, labels) };
        }

        return new[]
        {
            new RLActionDefinition(
                "action",
                RLActionVariableType.Continuous,
                dimensions: environment.ContinuousActionDimensions,
                minValue: -1f,
                maxValue: 1f),
        };
    }

    internal static Transition BuildTransition(
        ITrainer trainer,
        float[] observation,
        AlgorithmBenchStep step,
        PolicyDecision decision)
    {
        return new Transition
        {
            Observation = observation,
            DiscreteAction = decision.DiscreteAction,
            ContinuousActions = decision.ContinuousActions,
            Reward = step.Reward,
            Done = step.Done,
            NextObservation = step.Observation,
            OldLogProbability = decision.LogProbability,
            Value = decision.Value,
            NextValue = step.Done ? 0f : trainer.EstimateValue(step.Observation),
        };
    }

    internal static void ValidateDecision(IAlgorithmBenchEnvironment environment, PolicyDecision decision)
    {
        if (environment.DiscreteActionCount > 0)
        {
            if (decision.DiscreteAction < 0 || decision.DiscreteAction >= environment.DiscreteActionCount)
                throw new InvalidOperationException($"Discrete action {decision.DiscreteAction} is outside the expected range.");
            return;
        }

        if (decision.ContinuousActions.Length != environment.ContinuousActionDimensions)
            throw new InvalidOperationException(
                $"Expected {environment.ContinuousActionDimensions} continuous action dimensions, got {decision.ContinuousActions.Length}.");

        foreach (var value in decision.ContinuousActions)
        {
            if (!float.IsFinite(value))
                throw new InvalidOperationException("Continuous action contains non-finite values.");
        }
    }
}
