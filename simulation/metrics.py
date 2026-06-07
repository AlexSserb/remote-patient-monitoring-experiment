"""
Метрики эксперимента.

RunMetrics накапливает результаты по дням и вычисляет итоговые показатели:
OSR, DCR, AFFR, ART и среднюю награду R.
"""

from dataclasses import dataclass, field

import numpy as np

from simulation.user_simulator import DayResult


def compute_reward(opened: bool, diary_completed: bool, fields_filled_ratio: float) -> float:
    """R = 0.2·O + 0.3·D + 0.5·F"""
    return 0.2 * float(opened) + 0.3 * float(diary_completed) + 0.5 * fields_filled_ratio


@dataclass
class RunMetrics:
    """Метрики одного прогона (один алгоритм, 180 дней, вся когорта)."""

    _rewards: list[float] = field(default_factory=list, repr=False)
    _total_opened: int = 0
    _total_diary_completed: int = 0
    _total_fields_filled: int = 0    # числитель AFFR
    _total_fields_possible: int = 0  # знаменатель AFFR (180 * total_fields на пользователя)
    _reaction_times: list[float] = field(default_factory=list, repr=False)

    def record(self, result: DayResult, reward: float, user_total_fields: int) -> None:
        """
        Фиксирует результат одного дня одного пользователя.

        user_total_fields: суммарное число полей дневника пользователя (из профиля).
        Знаменатель AFFR растёт каждый день вне зависимости от того, заполнил ли
        пользователь дневник — учитываются все возможные заполнения.
        """
        self._rewards.append(reward)
        self._total_opened += int(result.opened)
        self._total_diary_completed += int(result.diary_completed)
        self._total_fields_possible += user_total_fields
        self._total_fields_filled += round(result.fields_filled_ratio * user_total_fields)
        if result.reaction_time_min is not None:
            self._reaction_times.append(result.reaction_time_min)

    # --- итоговые метрики ---

    @property
    def n_days(self) -> int:
        return len(self._rewards)

    @property
    def mean_reward(self) -> float:
        """Средняя функция награды R за все дни."""
        return float(np.mean(self._rewards)) if self._rewards else 0.0

    @property
    def osr(self) -> float:
        """Open/Session Rate: доля дней, когда пользователь открыл систему."""
        return self._total_opened / self.n_days if self.n_days else 0.0

    @property
    def dcr(self) -> float:
        """Diary Completion Rate: доля дней с отправленным дневником."""
        return self._total_diary_completed / self.n_days if self.n_days else 0.0

    @property
    def affr(self) -> float:
        """Average Field Fill Rate: доля заполненных полей от всех возможных."""
        return (
            self._total_fields_filled / self._total_fields_possible
            if self._total_fields_possible
            else 0.0
        )

    @property
    def art(self) -> float | None:
        """Average Reaction Time (мин). None если ни одного взаимодействия."""
        return float(np.mean(self._reaction_times)) if self._reaction_times else None

    def summary(self) -> dict[str, float | None]:
        return {
            "mean_reward": self.mean_reward,
            "osr":         self.osr,
            "dcr":         self.dcr,
            "affr":        self.affr,
            "art":         self.art,
        }