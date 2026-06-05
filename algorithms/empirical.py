"""
Эмпирический алгоритм уведомлений.

Стартует с субъективно лучшего времени пользователя и адаптирует время
по следующим правилам:
  1. Скользящее окно 7 дней: если средняя награда < 0.3 → сдвиг на 1 слот.
  2. Если средняя награда ≥ 0.5 → время зафиксировано.
  3. Hill-climbing: после сдвига оцениваем новый слот за 7 дней.
     Если стало хуже — меняем направление.
  4. Каждые 30 дней — принудительная переоценка: выбирается слот
     с наилучшей средней наградой за последние 30 дней.

Алгоритм использует один слот для всех дней недели (адаптация глобальная),
стартуя со среднего субъективно лучшего времени пользователя.
"""

from collections import deque
from dataclasses import dataclass, field

from algorithms.base import NotificationAlgorithm, SLOTS, nearest_slot_idx
from simulation.user_profile import UserProfile

_WINDOW = 7           # размер скользящего окна наград
_SHIFT_THRESHOLD = 0.3  # ниже — сдвигаем слот
_KEEP_THRESHOLD = 0.5   # выше — фиксируем слот
_REEVAL_PERIOD = 30   # дней между принудительными переоценками


@dataclass
class _UserState:
    slot_idx: int                           # текущий слот (индекс в SLOTS)
    direction: int                          # направление сдвига: +1 или -1
    window: deque = field(default_factory=lambda: deque(maxlen=_WINDOW))
    # буфер для принудительной переоценки: слот → список наград
    period_data: dict = field(default_factory=dict)
    day_count: int = 0
    prev_mean: float = 0.0                  # среднее окно до последнего сдвига
    evaluating: bool = False                # ждём оценки после сдвига


class EmpiricalAlgorithm(NotificationAlgorithm):
    def __init__(self, profiles: list[UserProfile]) -> None:
        self._states: dict[int, _UserState] = {}
        for i, p in enumerate(profiles):
            # стартовый слот — ближайший к среднему субъективно лучшему времени
            mean_time = sum(p.subjective_best_time) / 7.0
            self._states[i] = _UserState(
                slot_idx=nearest_slot_idx(mean_time),
                direction=+1,
            )

    def select_slot(self, user_id: int, weekday: int) -> float:
        return SLOTS[self._states[user_id].slot_idx]

    def update(self, user_id: int, weekday: int, selected_slot: float, reward: float) -> None:
        s = self._states[user_id]
        s.day_count += 1
        s.window.append(reward)

        # накапливаем данные для принудительной переоценки
        idx = s.slot_idx
        s.period_data.setdefault(idx, []).append(reward)

        # принудительная переоценка имеет наивысший приоритет
        if s.day_count % _REEVAL_PERIOD == 0:
            self._forced_reeval(s)
            return

        if len(s.window) < _WINDOW:
            return

        mean_r = sum(s.window) / _WINDOW

        if s.evaluating:
            # оцениваем качество нового слота после последнего сдвига
            s.evaluating = False
            if mean_r < s.prev_mean:
                # новый слот хуже — меняем направление и сдвигаемся ещё раз
                s.direction *= -1
                self._shift(s)
            return

        if mean_r >= _KEEP_THRESHOLD:
            return

        if mean_r < _SHIFT_THRESHOLD:
            s.prev_mean = mean_r
            self._shift(s)
            s.evaluating = True

    def _shift(self, s: _UserState) -> None:
        """Сдвигает текущий слот на 1 позицию в направлении direction."""
        new_idx = s.slot_idx + s.direction
        if not (0 <= new_idx < len(SLOTS)):
            # достигли границы диапазона — разворачиваемся
            s.direction *= -1
            new_idx = s.slot_idx + s.direction
            if not (0 <= new_idx < len(SLOTS)):
                return  # диапазон из одного слота, некуда двигаться
        s.slot_idx = new_idx
        s.window.clear()

    def _forced_reeval(self, s: _UserState) -> None:
        """Выбирает слот с наилучшей средней наградой за последние 30 дней."""
        if s.period_data:
            best_idx = max(
                s.period_data,
                key=lambda k: sum(s.period_data[k]) / len(s.period_data[k]),
            )
            s.slot_idx = best_idx
        s.period_data = {}
        s.window.clear()
        s.evaluating = False