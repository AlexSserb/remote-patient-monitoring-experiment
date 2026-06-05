"""
Базовый класс алгоритма выбора времени уведомлений.
"""

from abc import ABC, abstractmethod

# 14 временных слотов: 08:00, 09:00, …, 21:00
SLOTS: list[float] = [float(h) for h in range(8, 22)]


def nearest_slot(time: float) -> float:
    """Ближайший допустимый слот к заданному времени."""
    return min(SLOTS, key=lambda s: abs(s - time))


def nearest_slot_idx(time: float) -> int:
    """Индекс ближайшего допустимого слота."""
    return min(range(len(SLOTS)), key=lambda i: abs(SLOTS[i] - time))


def slot_idx(slot: float) -> int:
    """Индекс точного слота (slot должен быть элементом SLOTS)."""
    return round(slot - SLOTS[0])


class NotificationAlgorithm(ABC):
    """Интерфейс алгоритма выбора времени уведомления."""

    @abstractmethod
    def select_slot(self, user_id: int, weekday: int) -> float:
        """Возвращает час отправки уведомления для пользователя в данный день недели."""
        ...

    @abstractmethod
    def update(self, user_id: int, weekday: int, selected_slot: float, reward: float) -> None:
        """Обновляет состояние алгоритма после получения награды."""
        ...