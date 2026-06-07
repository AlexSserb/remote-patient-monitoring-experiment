"""
Запуск одного алгоритма уведомлений: 10 независимых прогонов по 180 дней.

Использование:
    python -m experiment.runner --algorithm static
    python -m experiment.runner --algorithm empirical --runs 10
    python -m experiment.runner --algorithm ducb --runs 10 --seed 0

Результаты сохраняются в results/<algorithm>.json.
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np

from algorithms.ducb import DUCBAlgorithm
from algorithms.empirical import EmpiricalAlgorithm
from algorithms.static import StaticAlgorithm
from simulation.metrics import RunMetrics, compute_reward
from simulation.user_profile import UserProfile, generate_cohort
from simulation.user_simulator import UserSimulator

N_DAYS = 180
N_RUNS = 10
COHORT_SEED = 0
RESULTS_DIR = "results"

ALGORITHM_REGISTRY: dict[str, type] = {
    "static":    StaticAlgorithm,
    "empirical": EmpiricalAlgorithm,
    "ducb":      DUCBAlgorithm,
}


def run_single(
    algorithm_class: type,
    all_users: list[UserProfile],
    run_seed: int,
    n_days: int = N_DAYS,
) -> RunMetrics:
    """
    Один прогон: инициализирует алгоритм заново, симулирует n_days дней
    для всей когорты и возвращает накопленные метрики.

    run_seed передаётся в simulate_day() и обеспечивает независимость прогонов:
    одни и те же профили пользователей дают разные случайные реализации.
    """
    sims = [UserSimulator(p) for p in all_users]
    alg = algorithm_class(all_users)
    metrics = RunMetrics()

    for day in range(n_days):
        weekday = day % 7
        for uid, (profile, sim) in enumerate(zip(all_users, sims)):
            t = alg.select_slot(uid, weekday)
            result = sim.simulate_day(t, weekday, day, run_seed)
            reward = compute_reward(
                result.opened, result.diary_completed, result.fields_filled_ratio
            )
            metrics.record(result, reward, profile.total_fields)
            alg.update(uid, weekday, t, reward)

    return metrics


def run_experiment(
    algorithm_name: str,
    n_runs: int = N_RUNS,
    n_days: int = N_DAYS,
    cohort_seed: int = COHORT_SEED,
) -> list[RunMetrics]:
    """
    Запускает n_runs независимых прогонов одного алгоритма и сохраняет результаты.
    Возвращает список метрик для каждого прогона.
    """
    print(f"Алгоритм: {algorithm_name.upper()}  |  прогонов: {n_runs}  |  дней: {n_days}")

    algorithm_class = ALGORITHM_REGISTRY[algorithm_name]
    patients, guardians = generate_cohort(base_seed=cohort_seed)
    all_users = patients + guardians
    print(f"Когорта: {len(patients)} пациентов + {len(guardians)} опекунов  (seed={cohort_seed})\n")

    run_results: list[RunMetrics] = []
    t_start = time.time()

    for run_idx in range(n_runs):
        metrics = run_single(algorithm_class, all_users, run_seed=run_idx, n_days=n_days)
        run_results.append(metrics)
        elapsed = time.time() - t_start
        art_str = f"{round(metrics.art)!r} мин" if metrics.art is not None else "—"
        print(
            f"  прогон {run_idx + 1:2d}/{n_runs}"
            f"  R={metrics.mean_reward:.3f}"
            f"  OSR={metrics.osr:.2f}"
            f"  DCR={metrics.dcr:.2f}"
            f"  AFFR={metrics.affr:.2f}"
            f"  ART={art_str}"
            f"  [{elapsed:.0f}с]"
        )

    _save_results(algorithm_name, n_runs, n_days, cohort_seed, run_results)
    _print_summary(algorithm_name, run_results)
    return run_results


def _save_results(
    algorithm_name: str,
    n_runs: int,
    n_days: int,
    cohort_seed: int,
    run_results: list[RunMetrics],
) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "algorithm":   algorithm_name,
        "n_runs":      n_runs,
        "n_days":      n_days,
        "cohort_seed": cohort_seed,
        "timestamp":   datetime.now().isoformat(),
        "runs": [
            {
                "mean_reward": m.mean_reward,
                "osr":         m.osr,
                "dcr":         m.dcr,
                "affr":        m.affr,
                "art":         m.art,
            }
            for m in run_results
        ],
    }
    path = os.path.join(RESULTS_DIR, f"{algorithm_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nСохранено: {path}")


def _print_summary(algorithm_name: str, run_results: list[RunMetrics]) -> None:
    metrics_data: dict[str, list[float]] = {
        "R":         [m.mean_reward for m in run_results],
        "OSR":       [m.osr for m in run_results],
        "DCR":       [m.dcr for m in run_results],
        "AFFR":      [m.affr for m in run_results],
        "ART (мин)": [m.art for m in run_results if m.art is not None],
    }
    sep = "=" * 46
    print(f"\n{sep}")
    print(f"Итоги {algorithm_name.upper()} ({len(run_results)} прогонов):")
    for label, values in metrics_data.items():
        if values:
            print(f"  {label:<10}: {np.mean(values):.3f} ± {np.std(values):.3f}")
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск одного алгоритма уведомлений")
    parser.add_argument(
        "--algorithm", choices=list(ALGORITHM_REGISTRY), required=True,
        help="Название алгоритма"
    )
    parser.add_argument("--runs",  type=int, default=N_RUNS,      help="Число прогонов")
    parser.add_argument("--days",  type=int, default=N_DAYS,       help="Длительность эксперимента (дней)")
    parser.add_argument("--seed",  type=int, default=COHORT_SEED,  help="Seed когорты пользователей")
    args = parser.parse_args()

    run_experiment(args.algorithm, n_runs=args.runs, n_days=args.days, cohort_seed=args.seed)