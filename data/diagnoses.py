"""
Диагнозы и поля дневника здоровья для симуляции удалённого мониторинга пациентов.

Каждый диагноз определяет:
  - обязательные поля: без них форма не отправляется (D=1 требует заполнения всех)
  - необязательные поля: заполняются при уведомлении в оптимальное время

Типы значений полей приближены к реальной клинической практике:
  numeric  - числовое измерение (например, артериальное давление)
  scale    - целочисленная шкала (например, боль от 0 до 10)
  boolean  - да/нет (чекбокс)
  duration - время в минутах
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DiaryField:
    name: str
    field_type: str   # "numeric" | "scale" | "boolean" | "duration"
    mandatory: bool


@dataclass(frozen=True)
class Diagnosis:
    code: str
    name: str
    fields: tuple[DiaryField, ...]

    @property
    def mandatory_fields(self) -> list[DiaryField]:
        return [f for f in self.fields if f.mandatory]

    @property
    def optional_fields(self) -> list[DiaryField]:
        return [f for f in self.fields if not f.mandatory]

    @property
    def total_fields(self) -> int:
        return len(self.fields)

    @property
    def mandatory_count(self) -> int:
        return len(self.mandatory_fields)

    @property
    def optional_count(self) -> int:
        return len(self.optional_fields)


# ---------------------------------------------------------------------------
# Гипертония (I10)
# Самый распространённый хронический диагноз; ключевые измерения - АД и пульс.
# ---------------------------------------------------------------------------
HYPERTENSION = Diagnosis(
    code="I10",
    name="Гипертония",
    fields=(
        # обязательные: без этих данных запись клинически бессмысленна
        DiaryField("systolic_bp",      "numeric", mandatory=True),   # мм рт. ст.
        DiaryField("diastolic_bp",     "numeric", mandatory=True),   # мм рт. ст.
        DiaryField("pulse",            "numeric", mandatory=True),   # уд/мин
        # необязательные: симптомы и приём лекарств - полезно, но часто пропускается
        DiaryField("headache",         "boolean", mandatory=False),  # головная боль
        DiaryField("dizziness",        "boolean", mandatory=False),  # головокружение
        DiaryField("medication_taken", "boolean", mandatory=False),  # лекарство принято
    ),
)

# ---------------------------------------------------------------------------
# Сахарный диабет 2 типа (E11)
# Контроль гликемии требует нескольких измерений; контекст питания и активности
# ценен для врача, но не всегда фиксируется пациентом.
# ---------------------------------------------------------------------------
DIABETES_T2 = Diagnosis(
    code="E11",
    name="Сахарный диабет 2 типа",
    fields=(
        DiaryField("fasting_glucose",    "numeric",  mandatory=True),  # ммоль/л, натощак
        DiaryField("postmeal_glucose",   "numeric",  mandatory=True),  # ммоль/л, после еды
        DiaryField("insulin_dose",       "numeric",  mandatory=True),  # единиц инсулина
        DiaryField("medication_taken",   "boolean",  mandatory=True),  # лекарство принято
        # необязательные: дополнительный контекст
        DiaryField("physical_activity",  "duration", mandatory=False), # мин физической активности
        DiaryField("diet_compliance",    "boolean",  mandatory=False), # соблюдение диеты
        DiaryField("hypoglycemia_event", "boolean",  mandatory=False), # эпизод гипогликемии
    ),
)

# ---------------------------------------------------------------------------
# Хроническая сердечная недостаточность (I50)
# Ежедневный контроль веса и отёков - клинические опорные точки;
# одышка добавляет нюанс, но требует больше усилий для заполнения.
# ---------------------------------------------------------------------------
CHRONIC_HEART_FAILURE = Diagnosis(
    code="I50",
    name="Хроническая сердечная недостаточность",
    fields=(
        DiaryField("body_weight",         "numeric", mandatory=True),  # кг
        DiaryField("edema_level",         "scale",   mandatory=True),  # 0–3 (нет/слабые/умеренные/выраженные)
        DiaryField("pulse",               "numeric", mandatory=True),  # уд/мин
        DiaryField("medication_taken",    "boolean", mandatory=True),  # лекарство принято
        # необязательные
        DiaryField("shortness_of_breath", "scale",   mandatory=False), # шкала mMRC 0–4
        DiaryField("physical_activity",   "duration",mandatory=False), # мин физической активности
        DiaryField("fatigue_level",       "scale",   mandatory=False), # усталость 0–10
    ),
)

# ---------------------------------------------------------------------------
# ХОБЛ (J44)
# Пикфлоуметрия и шкала одышки - обязательные ежедневные показатели.
# ---------------------------------------------------------------------------
COPD = Diagnosis(
    code="J44",
    name="ХОБЛ",
    fields=(
        DiaryField("peak_flow",         "numeric", mandatory=True),   # л/мин (пикфлоуметр)
        DiaryField("dyspnea_scale",     "scale",   mandatory=True),   # шкала mMRC 0–4
        DiaryField("medication_taken",  "boolean", mandatory=True),   # лекарство принято
        # необязательные
        DiaryField("sputum_amount",     "scale",   mandatory=False),  # объём мокроты 0–3
        DiaryField("physical_activity", "duration",mandatory=False),  # мин физической активности
        DiaryField("exacerbation_sign", "boolean", mandatory=False),  # признак обострения
    ),
)

# ---------------------------------------------------------------------------
# Остеоартрит (M15-M19)
# Боль и скованность - основные показатели, сообщаемые пациентом.
# ---------------------------------------------------------------------------
OSTEOARTHRITIS = Diagnosis(
    code="M15",
    name="Остеоартрит",
    fields=(
        DiaryField("pain_scale",         "scale",   mandatory=True),  # боль по NRS 0–10
        DiaryField("morning_stiffness",  "duration",mandatory=True),  # утренняя скованность, мин
        DiaryField("medication_taken",   "boolean", mandatory=True),  # лекарство принято
        # необязательные
        DiaryField("joint_mobility",     "scale",   mandatory=False), # подвижность сустава 0–3
        DiaryField("physical_activity",  "duration",mandatory=False), # мин физической активности
        DiaryField("swelling_present",   "boolean", mandatory=False), # наличие отёка сустава
    ),
)

# ---------------------------------------------------------------------------
# Реестр диагнозов - используется при назначении диагнозов пользователям
# ---------------------------------------------------------------------------
ALL_DIAGNOSES: list[Diagnosis] = [
    HYPERTENSION,
    DIABETES_T2,
    CHRONIC_HEART_FAILURE,
    COPD,
    OSTEOARTHRITIS,
]

DIAGNOSIS_BY_CODE: dict[str, Diagnosis] = {d.code: d for d in ALL_DIAGNOSES}