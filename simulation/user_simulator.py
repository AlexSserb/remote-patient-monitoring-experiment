"""
Симулятор реакции пользователя на уведомление.

Для каждого дня эксперимента генерирует результат (открыл/заполнил/время реакции)
на основе профиля пользователя и времени отправки уведомления.

Детерминированность по (profile.seed, day_index) обеспечивает справедливое
сравнение алгоритмов: одинаковые условия - разные решения о времени уведомления.
"""

from dataclasses import dataclass

import numpy as np

from simulation.user_profile import UserProfile


@dataclass
class DayResult:
    opened: bool                        # O: пользователь открыл приложение
    diary_completed: bool               # D: все обязательные поля заполнены и форма отправлена
    fields_filled_ratio: float          # F: доля заполненных полей от суммарного числа полей
    reaction_time_min: float | None     # время реакции в минутах; None если уведомление проигнорировано


class UserSimulator:
    def __init__(self, profile: UserProfile) -> None:
        self.profile = profile

    def simulate_day(self, notification_time: float, weekday: int, day_index: int) -> DayResult:
        """
        Симулирует один день эксперимента.

        notification_time: час отправки уведомления (8.0–21.0)
        weekday:           день недели (0=Пн, 6=Вс)
        day_index:         номер дня в эксперименте (0–179)

        RNG привязан к (profile.seed, day_index), поэтому результат зависит только
        от профиля и дня, но не от истории выборов алгоритма. Это гарантирует
        корректность сравнения алгоритмов: один и тот же «бросок монеты» для всех.
        """
        rng = np.random.default_rng([self.profile.seed, day_index])
        p = self.profile

        # Временной фактор: гауссово убывание от объективно лучшего времени
        opt_time = p.objective_best_time[weekday]
        time_diff = notification_time - opt_time
        time_factor = float(np.exp(-time_diff ** 2 / (2.0 * p.time_sensitivity_sigma ** 2)))

        # Вероятность открыть приложение
        weekday_coeff = p.weekday_coefficients[weekday]
        p_open = float(np.clip(p.base_adherence * weekday_coeff * time_factor, 0.0, 1.0))

        if rng.random() > p_open:
            return DayResult(
                opened=False,
                diary_completed=False,
                fields_filled_ratio=0.0,
                reaction_time_min=None,
            )

        reaction_time = self._reaction_time(rng, opt_time, notification_time)

        # Вероятность заполнить обязательные поля: даже в плохое время ~65% открывших завершают форму
        p_mandatory = float(np.clip(0.65 + 0.35 * time_factor, 0.0, 1.0))
        diary_completed = bool(rng.random() < p_mandatory)

        if not diary_completed:
            # Форма не отправлена → F=0 по определению
            return DayResult(
                opened=True,
                diary_completed=False,
                fields_filled_ratio=0.0,
                reaction_time_min=reaction_time,
            )

        # Подсчёт заполненных полей
        total_fields = p.total_fields
        mandatory_count = sum(d.mandatory_count for d in p.diagnoses)
        optional_count = total_fields - mandatory_count

        # Необязательные поля: каждое заполняется независимо с вероятностью time_factor
        filled_optional = int(rng.binomial(optional_count, time_factor)) if optional_count > 0 else 0
        fields_ratio = (mandatory_count + filled_optional) / total_fields if total_fields > 0 else 1.0

        return DayResult(
            opened=True,
            diary_completed=True,
            fields_filled_ratio=float(fields_ratio),
            reaction_time_min=reaction_time,
        )

    def _reaction_time(
        self,
        rng: np.random.Generator,
        opt_time: float,
        notification_time: float,
    ) -> float:
        """
        Время реакции в минутах (логнормальное).

        Чем дальше уведомление от оптимального времени, тем выше mu ->
        медленнее реакция. Каждый час отклонения прибавляет 0.3 к log-mu
        (умножает медиану примерно на 1.35).
        """
        time_diff = abs(notification_time - opt_time)
        mu_adjusted = self.profile.reaction_lognorm_mu + 0.3 * time_diff
        return float(rng.lognormal(mu_adjusted, self.profile.reaction_lognorm_sigma))