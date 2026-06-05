"""
Статический алгоритм уведомлений.

Фиксирует время уведомления по субъективно лучшему времени пользователя
(указанному при регистрации) и никогда его не меняет.
Является нижней границей качества: любой адаптивный алгоритм должен превзойти его.
"""

from algorithms.base import NotificationAlgorithm, nearest_slot
from simulation.user_profile import UserProfile


class StaticAlgorithm(NotificationAlgorithm):
    def __init__(self, profiles: list[UserProfile]) -> None:
        # Предвычисляем слот для каждой пары (пользователь, день недели)
        self._slots: dict[tuple[int, int], float] = {
            (i, weekday): nearest_slot(profile.subjective_best_time[weekday])
            for i, profile in enumerate(profiles)
            for weekday in range(7)
        }

    def select_slot(self, user_id: int, weekday: int) -> float:
        return self._slots[(user_id, weekday)]

    def update(self, user_id: int, weekday: int, selected_slot: float, reward: float) -> None:
        pass  # статический алгоритм не адаптируется