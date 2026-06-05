"""
Профиль пользователя и функция его генерации на основе seed.

Профиль полностью определяет поведение симулируемого пользователя:
вероятность реакции на уведомление, скорость ответа, диагнозы.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from data.diagnoses import ALL_DIAGNOSES, Diagnosis

# Допустимое окно отправки уведомлений (часы)
NOTIFICATION_START = 8.0
NOTIFICATION_END = 22.0


class AdherenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Диапазоны базовой вероятности взаимодействия для каждого уровня приверженности
_ADHERENCE_RANGES: dict[AdherenceLevel, tuple[float, float]] = {
    AdherenceLevel.HIGH:   (0.70, 0.90),
    AdherenceLevel.MEDIUM: (0.40, 0.65),
    AdherenceLevel.LOW:    (0.10, 0.35),
}


@dataclass
class UserProfile:
    seed: int
    role: str                          # "patient" | "guardian"
    adherence_level: AdherenceLevel
    base_adherence: float              # базовая вероятность взаимодействия

    # 7 элементов — по одному на каждый день недели (0=Пн, 6=Вс)
    objective_best_time: list[float]   # объективно лучшее время (часы)
    subjective_best_time: list[float]  # субъективно лучшее время (часы)
    weekday_coefficients: list[float]  # мультипликаторы вероятности по дням недели

    # sigma гауссова спада вероятности при отклонении от оптимума (часов)
    time_sensitivity_sigma: float

    # параметры логнормального распределения времени реакции при оптимальном времени
    reaction_lognorm_mu: float
    reaction_lognorm_sigma: float

    diagnoses: list[Diagnosis]

    @property
    def total_fields(self) -> int:
        """Суммарное количество полей дневника по всем диагнозам."""
        return sum(d.total_fields for d in self.diagnoses)


def generate_profile(seed: int, role: str, adherence_level: AdherenceLevel) -> UserProfile:
    """Детерминированная генерация профиля пользователя по seed."""
    rng = np.random.default_rng(seed)

    # Базовая вероятность взаимодействия
    lo, hi = _ADHERENCE_RANGES[adherence_level]
    base_adherence = float(rng.uniform(lo, hi))

    # Объективно лучшее время: общее базовое + небольшие отклонения по дням
    base_time = float(rng.uniform(9.5, 20.0))
    day_offsets = rng.normal(0.0, 0.75, 7)
    objective_best_time = [
        float(np.clip(base_time + off, NOTIFICATION_START + 0.5, NOTIFICATION_END - 0.5))
        for off in day_offsets
    ]

    # Субъективно лучшее время: пользователь знает себя неточно (~±1.5 ч)
    subj_offsets = rng.normal(0.0, 1.5, 7)
    subjective_best_time = [
        float(np.clip(obj + off, NOTIFICATION_START, NOTIFICATION_END - 1.0))
        for obj, off in zip(objective_best_time, subj_offsets)
    ]

    # Коэффициенты дня недели: активность в выходные и будни может отличаться
    weekday_coefficients = rng.uniform(0.6, 1.4, 7).tolist()

    # Ширина «окна доступности» вокруг оптимального времени
    time_sensitivity_sigma = float(rng.uniform(1.5, 3.0))

    # Параметры логнормального распределения времени реакции
    # mu=2.5 → медианная задержка ~12 мин при уведомлении в оптимальный момент
    reaction_lognorm_mu = float(max(1.5, rng.normal(2.5, 0.4)))
    reaction_lognorm_sigma = float(rng.uniform(0.4, 0.8))

    # Назначение диагнозов: 1 диагноз с вероятностью 0.6, 2 - с вероятностью 0.4
    n_diagnoses = int(rng.choice([1, 2], p=[0.6, 0.4]))
    indices = rng.choice(len(ALL_DIAGNOSES), size=n_diagnoses, replace=False)
    diagnoses = [ALL_DIAGNOSES[i] for i in sorted(indices)]

    return UserProfile(
        seed=seed,
        role=role,
        adherence_level=adherence_level,
        base_adherence=base_adherence,
        objective_best_time=objective_best_time,
        subjective_best_time=subjective_best_time,
        weekday_coefficients=weekday_coefficients,
        time_sensitivity_sigma=time_sensitivity_sigma,
        reaction_lognorm_mu=reaction_lognorm_mu,
        reaction_lognorm_sigma=reaction_lognorm_sigma,
        diagnoses=diagnoses,
    )


def generate_cohort(
    n_patients: int = 1000,
    n_guardians: int = 200,
    base_seed: int = 0,
) -> tuple[list[UserProfile], list[UserProfile]]:
    """
    Генерация всей когорты пациентов и опекунов.

    Распределение пациентов по приверженности: 20% / 50% / 30% (выс./ср./низ.)
    Распределение опекунов по приверженности:  30% / 50% / 20% (выс./ср./низ.)

    Seeds пациентов:  base_seed .. base_seed + n_patients - 1
    Seeds опекунов:   base_seed + n_patients .. base_seed + n_patients + n_guardians - 1
    """
    patient_levels = _assign_adherence_levels(
        n_patients,
        high_frac=0.20, medium_frac=0.50,
        rng=np.random.default_rng(base_seed),
    )
    guardian_levels = _assign_adherence_levels(
        n_guardians,
        high_frac=0.30, medium_frac=0.50,
        rng=np.random.default_rng(base_seed + 1),
    )

    patients = [
        generate_profile(seed=base_seed + i, role="patient", adherence_level=lvl)
        for i, lvl in enumerate(patient_levels)
    ]
    guardians = [
        generate_profile(seed=base_seed + n_patients + i, role="guardian", adherence_level=lvl)
        for i, lvl in enumerate(guardian_levels)
    ]

    return patients, guardians


def _assign_adherence_levels(
    n: int,
    high_frac: float,
    medium_frac: float,
    rng: np.random.Generator,
) -> list[AdherenceLevel]:
    """Случайное перемешанное назначение уровней приверженности."""
    n_high = round(n * high_frac)
    n_medium = round(n * medium_frac)
    n_low = n - n_high - n_medium

    levels = (
        [AdherenceLevel.HIGH] * n_high
        + [AdherenceLevel.MEDIUM] * n_medium
        + [AdherenceLevel.LOW] * n_low
    )
    rng.shuffle(levels)
    return levels