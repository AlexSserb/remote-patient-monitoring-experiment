"""
Сводный отчёт по результатам всех алгоритмов.

Читает JSON-файлы из results/ и выводит таблицу сравнения (mean ± std).

Использование:
    python -m experiment.report
    python -m experiment.report --dir results/
"""

import argparse
import glob
import json
import os

import numpy as np

RESULTS_DIR = "results"

METRICS: list[tuple[str, str]] = [
    ("mean_reward", "R"),
    ("osr",         "OSR"),
    ("dcr",         "DCR"),
    ("affr",        "AFFR"),
    ("art",         "ART (мин)"),
]

# Ожидаемый порядок алгоритмов в таблице
ALG_ORDER = ["static", "empirical", "ducb"]


def load_results(results_dir: str = RESULTS_DIR) -> dict[str, dict]:
    """Загружает все JSON-файлы с результатами из директории."""
    results: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        results[data["algorithm"]] = data
    return results


def _stats(run_list: list[dict], key: str) -> tuple[float, float] | tuple[None, None]:
    """Среднее и стандартное отклонение по прогонам для одной метрики."""
    values = [r[key] for r in run_list if r[key] is not None]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def print_report(results: dict[str, dict]) -> None:
    if not results:
        print("Нет результатов. Сначала запустите:\n  python -m experiment.runner --algorithm <name>")
        return

    # Упорядочиваем алгоритмы: сначала известные, затем остальные по алфавиту
    known = [a for a in ALG_ORDER if a in results]
    others = sorted(a for a in results if a not in ALG_ORDER)
    algorithms = known + others

    col_w = 20
    label_w = 12

    sep = "=" * (label_w + col_w * len(algorithms))
    header = f"{'Метрика':<{label_w}}" + "".join(
        f"{a.upper():>{col_w}}" for a in algorithms
    )

    print(f"\n{sep}")
    print(header)
    print(sep)

    for key, label in METRICS:
        row = f"{label:<{label_w}}"
        for alg in algorithms:
            mean, std = _stats(results[alg]["runs"], key)
            if mean is None:
                row += f"{'—':>{col_w}}"
            else:
                cell = f"{mean:.3f} ±{std:.3f}"
                row += f"{cell:>{col_w}}"
        print(row)

    print(sep)

    # Краткая сводка по условиям запуска
    print()
    for alg in algorithms:
        d = results[alg]
        ts = d.get("timestamp", "—")[:19].replace("T", " ")
        print(
            f"  {alg.upper():<12}: {d['n_runs']} прогонов × {d['n_days']} дней"
            f"  seed={d['cohort_seed']}  [{ts}]"
        )


def save_csv(results: dict[str, dict], path: str = "results/report.csv") -> None:
    """Сохраняет таблицу метрик в CSV для дальнейшего анализа."""
    known = [a for a in ALG_ORDER if a in results]
    others = sorted(a for a in results if a not in ALG_ORDER)
    algorithms = known + others

    rows = []
    # Шапка: алгоритм, метрика_mean, метрика_std, ...
    header_parts = ["algorithm"]
    for _, label in METRICS:
        header_parts += [f"{label}_mean", f"{label}_std"]
    rows.append(",".join(header_parts))

    for alg in algorithms:
        parts = [alg]
        for key, _ in METRICS:
            mean, std = _stats(results[alg]["runs"], key)
            parts += [
                f"{mean:.4f}" if mean is not None else "",
                f"{std:.4f}" if std is not None else "",
            ]
        rows.append(",".join(parts))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"CSV сохранён: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сводный отчёт по результатам эксперимента")
    parser.add_argument("--dir", default=RESULTS_DIR, help="Директория с JSON-результатами")
    parser.add_argument("--csv", action="store_true", help="Дополнительно сохранить CSV")
    args = parser.parse_args()

    results = load_results(args.dir)
    print_report(results)
    if args.csv:
        save_csv(results)