"""Deterministic calendar of a target hour. Known at any issue time; uses no data.

French public holidays are computed here (no extra dependency, no network) so the
upkeep cost of the calendar features is visible: this table must be reviewed if the
tool is used outside metropolitan France or if holiday law changes.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

import numpy as np
import pandas as pd


def easter_sunday(year: int) -> dt.date:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


@lru_cache(maxsize=64)
def french_holidays(year: int) -> frozenset[dt.date]:
    e = easter_sunday(year)
    days = {
        dt.date(year, 1, 1),    # Jour de l'an
        e + dt.timedelta(days=1),   # Lundi de Paques
        dt.date(year, 5, 1),    # Fete du travail
        dt.date(year, 5, 8),    # Victoire 1945
        e + dt.timedelta(days=39),  # Ascension
        e + dt.timedelta(days=50),  # Lundi de Pentecote (journee de solidarite 2005-2007)
        dt.date(year, 7, 14),   # Fete nationale
        dt.date(year, 8, 15),   # Assomption
        dt.date(year, 11, 1),   # Toussaint
        dt.date(year, 11, 11),  # Armistice
        dt.date(year, 12, 25),  # Noel
    }
    return frozenset(days)


def is_holiday(d: dt.date) -> bool:
    return d in french_holidays(d.year)


CALENDAR_FEATURES = [
    "hour_sin", "hour_cos",
    "dow_0", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6",
    "is_weekend", "is_holiday",
    "doy_sin", "doy_cos",
]


def day_type(local: pd.Timestamp) -> str:
    if is_holiday(local.date()):
        return "holiday"
    return "weekend" if local.dayofweek >= 5 else "weekday"


def calendar_matrix(targets_utc: pd.DatetimeIndex, local_tz: str) -> np.ndarray:
    """(n, len(CALENDAR_FEATURES)) calendar features of the target hours (local time)."""
    loc = targets_utc.tz_convert(local_tz)
    hour = loc.hour.to_numpy() + loc.minute.to_numpy() / 60.0
    dow = loc.dayofweek.to_numpy()
    doy = loc.dayofyear.to_numpy()
    hol = np.array([is_holiday(d) for d in loc.date], dtype=float)
    out = np.zeros((len(loc), len(CALENDAR_FEATURES)))
    out[:, 0] = np.sin(2 * np.pi * hour / 24)
    out[:, 1] = np.cos(2 * np.pi * hour / 24)
    out[np.arange(len(loc)), 2 + dow] = 1.0
    out[:, 9] = (dow >= 5).astype(float)
    out[:, 10] = hol
    out[:, 11] = np.sin(2 * np.pi * doy / 365.25)
    out[:, 12] = np.cos(2 * np.pi * doy / 365.25)
    return out
