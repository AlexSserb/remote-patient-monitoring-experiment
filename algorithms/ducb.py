"""
D-UCB (Discounted Upper Confidence Bound) алгоритм уведомлений.

Реализация многорукого бандита с дисконтированием прошлых наблюдений.
Позволяет адаптироваться к изменениям поведения пользователей во времени.

Параметры:
  gamma       = 0.99  — коэффициент дисконтирования (период полураспада ≈ 69 дней)
  c           = 1.0   — коэффициент исследования в UCB-бонусе
  sigma_prior = 2.0   — ширина (ч) гауссова prior по субъективному времени

Автоматы: 14 временных слотов (08:00–21:00), общие для всех дней недели.

Инициализация (тёплый старт):
  Вместо равномерного prior или форсированного round-robin используется
  гауссов prior с центром в среднем субъективно лучшем времени пользователя.
  Это даёт тот же начальный контекст, что у статического и эмпирического
  алгоритмов, и уменьшает число впустую потраченных дней на плохих слотах.

  q̂_i = exp(-(slot_i - mean_subj)² / (2 * sigma_prior²))   [1 виртуальное наблюдение]
  n̂_i = 1

Формула выбора:
  UCB_i = Q̂_i / n̂_i  +  c * sqrt( ln(Σ visit_count_j) / visit_count_i )

Бонус исследования считается через недисконтируемый счётчик visit_count —
это предотвращает повторное исследование слотов с устойчиво низкой наградой
(их n̂ убывает из-за дисконтирования, но visit_count не убывает).
Адаптивность к нестационарности сохраняется через дисконтированное среднее Q̂/n̂.

Обновление после выбора слота i с наградой r:
  Q̂_j *= γ  для всех j
  n̂_j *= γ  для всех j
  Q̂_i += r
  n̂_i += 1
  visit_count_i += 1
"""

from dataclasses import dataclass, field

import numpy as np

from algorithms.base import NotificationAlgorithm, SLOTS, nearest_slot_idx
from simulation.user_profile import UserProfile

GAMMA = 0.99         # коэффициент дисконтирования
# c=0.1 выбран эмпирически: теоретическое sqrt(2) слишком агрессивно исследует
# плохие слоты в 180-дневном горизонте, снижая итоговое R.
C = 0.1              # коэффициент исследования
SIGMA_PRIOR = 2.0    # ширина гауссова prior (часов)
N_SLOTS = len(SLOTS)


@dataclass
class _DUCBUserState:
    q_hat: np.ndarray        # дисконтированные суммы наград, shape=(N_SLOTS,)
    n_hat: np.ndarray        # дисконтированные счётчики,   shape=(N_SLOTS,)
    visit_count: np.ndarray  # целочисленные счётчики посещений (не дисконтируются)


def _init_state(profile: UserProfile) -> _DUCBUserState:
    """
    Тёплый старт: гауссов prior с центром в среднем субъективно лучшем времени.
    Каждый слот получает 1 виртуальное наблюдение с ожидаемой наградой
    пропорциональной близости к субъективному оптимуму.
    """
    mean_subj = sum(profile.subjective_best_time) / 7.0
    q_hat = np.array([
        np.exp(-((slot - mean_subj) ** 2) / (2.0 * SIGMA_PRIOR ** 2))
        for slot in SLOTS
    ])
    return _DUCBUserState(
        q_hat=q_hat,
        n_hat=np.ones(N_SLOTS, dtype=float),    # 1 виртуальное наблюдение на слот
        visit_count=np.zeros(N_SLOTS, dtype=int),
    )


class DUCBAlgorithm(NotificationAlgorithm):
    def __init__(
        self,
        profiles: list[UserProfile],
        gamma: float = GAMMA,
        c: float = C,
    ) -> None:
        self._gamma = gamma
        self._c = c
        self._states: dict[int, _DUCBUserState] = {
            i: _init_state(profile)
            for i, profile in enumerate(profiles)
        }

    def select_slot(self, user_id: int, weekday: int) -> float:
        s = self._states[user_id]
        n_hat = np.maximum(s.n_hat, 1e-9)
        discounted_mean = s.q_hat / n_hat

        total_visits = int(s.visit_count.sum())
        if total_visits == 0:
            # До первого реального наблюдения: выбираем по prior-среднему
            return SLOTS[int(np.argmax(discounted_mean))]

        visit_safe = np.maximum(s.visit_count.astype(float), 1.0)
        ucb = discounted_mean + self._c * np.sqrt(np.log(total_visits) / visit_safe)
        return SLOTS[int(np.argmax(ucb))]

    def update(self, user_id: int, weekday: int, selected_slot: float, reward: float) -> None:
        s = self._states[user_id]
        i = nearest_slot_idx(selected_slot)

        # Дисконтируем всю историю
        s.q_hat *= self._gamma
        s.n_hat *= self._gamma

        # Добавляем новое наблюдение
        s.q_hat[i] += reward
        s.n_hat[i] += 1.0
        s.visit_count[i] += 1
