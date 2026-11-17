"""gta6-twitch-ccu-forecast-01 entry: total-first compositional day-curve quantiles.

The forecast is built in two steps, never hour by hour from hour 1 alone:

1. TOTAL. From the case's comparable launches, estimate this launch's TOTAL
   24-hour viewer volume from its hour-1 print with a local log-log fit:
   ln(total) = a + b * ln(hour-1), fitted by least squares on the 12
   comparables whose hour-1 print is closest (in log) to this case's. Local,
   because the total-per-hour-1 multiple drifts with launch size (about 18-24x
   for the biggest launches, far higher for small ones whose unlock ramps
   through the day); a single global fit would understate a big launch's total.

2. SPLIT. Each of the 10 closest comparables contributes its own day curve,
   rescaled to this launch: its hourly viewers multiplied by
   (predicted total for this launch) / (fitted total for that comparable).
   The five quantiles of an hour are the empirical quantiles (linear
   interpolation) of those 10 rescaled curves at that hour. The spread across
   neighbours is the uncertainty: launches of a similar size disagree about
   the shape of the day, and that disagreement is the forecast's width.

Nothing here is per-hour log-ratio quantiles, a normal/hierarchical model, or
a pinball-loss regression. Standard library only, deterministic: every
ordering is fixed (comparables in file order, neighbours by distance with a
stable sort, quantiles by interpolation), so two runs write identical bytes.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P keeps
this file's directory OFF sys.path, so a module you ship beside main.py
imports only after the sys.path line below; -s drops user site-packages.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at a
    time, with only that case's file there): JSON {"schema":
    "makefun/twitch-ccu-case/v1", "id", "game", "day_start", "checkpoint",
    "levels", "known", "forecast", "history", "history_hourly", "streams",
    "categories_hourly", "comparables", "notes"}, where `known` holds hour 1's
    print {hour: 1, from, until, viewers, channels, reads} and `forecast`
    lists the hours to forecast, 2..24;
  - write <id>.out to the output directory: {"hours": [{"hour": 2, "q":
    [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24 each once,
    no other key, each q five finite numbers above 0 and at most 1e9 (viewers;
    fractions allowed), non-decreasing; at most 65,536 bytes. Anything else is
    an invalid case, scored as if every quantile equalled hour 1's print, and
    an entry with an invalid case takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand
in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import re
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (persistence fallback only).
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The spread of an hour's move, in natural-log units per square-root hour
# (persistence fallback only).
SIGMA = 0.15

# How many closest comparables (by |log hour-1| distance) fit the local
# log-log total, and how many closest comparables supply rescaled day curves.
FIT_NEIGHBOURS = 12
CURVE_NEIGHBOURS = 10


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_hourly(hourly) -> list[tuple[str, int]]:
    """A comparable's hourly series as (from, viewers) pairs, in file order.

    The series comes as a list of dicts, or as a string holding that list in
    Python-literal form; both occur in the kit's cases.
    """
    if isinstance(hourly, list):
        return [(row["from"], int(row["viewers"])) for row in hourly]
    if isinstance(hourly, str):
        try:
            parsed = json.loads(hourly)
            if isinstance(parsed, list):
                return [(row["from"], int(row["viewers"])) for row in parsed]
        except (ValueError, KeyError, TypeError):
            pass
        pairs = re.findall(r"'from': '([^']+)'[^}]*?'viewers': (\d+)", hourly)
        return [(when, int(viewers)) for when, viewers in pairs]
    return []


def launch_day(comp: dict) -> list[int] | None:
    """A comparable's 24 launch-day hourly viewer counts, or None.

    The launch day starts at the comparable's day_start; the 24 hourly rows
    from there are hours 1..24. A comparable whose series never reaches its
    day_start, or has fewer than 24 rows after it, cannot be used.
    """
    series = parse_hourly(comp.get("hourly"))
    day_start = comp.get("day_start")
    start = next((i for i, (when, _) in enumerate(series) if when == day_start), None)
    if start is None or len(series) - start < 24:
        return None
    return [viewers for _, viewers in series[start:start + 24]]


def quantile(values: list[float], tau: float) -> float:
    """The tau-quantile of values by linear interpolation (type 7)."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    pos = tau * (n - 1)
    low = int(math.floor(pos))
    high = min(low + 1, n - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def ols_loglog(days: list[tuple[int, int]]) -> tuple[float, float]:
    """Least-squares fit of ln(total) = a + b * ln(hour-1) over the days."""
    xs = [math.log(max(1, h1)) for h1, _ in days]
    ys = [math.log(max(1, total)) for _, total in days]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    sxx = sum((x - mean_x) ** 2 for x in xs)
    slope = sxy / sxx if sxx else 0.0
    return mean_y - slope * mean_x, slope


def persistence_quantiles(p1: float, h: int) -> list[float | int]:
    """The starter's persistence forecast, used only when no comparable is usable."""
    return [round2(p1 * math.exp(z * SIGMA * math.sqrt(h - 1))) for z in Z]


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])

    # The training set: every comparable with a full 24-hour launch day.
    days: list[tuple[int, int, list[int]]] = []
    for comp in case.get("comparables") or []:
        hours = launch_day(comp)
        if hours is None or hours[0] < 1:
            continue
        days.append((hours[0], sum(hours), hours))
    if not days:
        return {"hours": [{"hour": row["hour"], "q": persistence_quantiles(p1, row["hour"])}
                         for row in case["forecast"]]}

    log_p1 = math.log(p1)
    # Closest comparables first, by |log hour-1| distance; the sort is stable,
    # so equal distances keep file order and the run is deterministic.
    ranked = sorted(days, key=lambda day: abs(math.log(day[0]) - log_p1))

    # Step 1: the total. A local log-log fit on the FIT_NEIGHBOURS closest
    # comparables, evaluated at this launch's hour-1 print.
    fit_days = [(h1, total) for h1, total, _ in ranked[:FIT_NEIGHBOURS]]
    intercept, slope = ols_loglog(fit_days)
    total_hat = math.exp(intercept + slope * log_p1)

    # Step 2: the split. Each of the CURVE_NEIGHBOURS closest comparables
    # contributes its day curve rescaled to this launch: its hourly viewers
    # times (predicted total here) / (fitted total there).
    curves: list[list[float]] = []
    for h1, _, hours in ranked[:CURVE_NEIGHBOURS]:
        fitted_here = math.exp(intercept + slope * math.log(h1))
        scale = total_hat / fitted_here if fitted_here > 0 else 1.0
        curves.append([viewers * scale for viewers in hours[1:24]])

    hours_out = []
    for row in case["forecast"]:
        h = row["hour"]
        index = h - 2  # curves hold hours 2..24 in order
        if 0 <= index < 23 and curves:
            column = [curve[index] for curve in curves]
            qs = [quantile(column, tau) for tau in LEVELS]
        else:
            qs = [float(p1)] * len(LEVELS)
        qs = [min(1e9, max(1.0, q)) for q in qs]
        hours_out.append({"hour": h, "q": [round2(q) for q in qs]})
    return {"hours": hours_out}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
