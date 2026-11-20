"""Forecast GTA VI trophy earn rates on Exophase, one week after launch.

Reads every case file in $MAKEFUN_INPUT_DIR (else /input) and writes one
forecast per case to $MAKEFUN_OUTPUT_DIR (else /output).  Each case asks for
the earn rate of every trophy at the case's target, as five quantiles
(5%, 25%, 50%, 75%, 95%).  Python 3.13, standard library only, no network,
fully deterministic: the same input always produces the same bytes.

MODEL (all work in logit space, logit(p) = ln(p / (1 - p)) with p in percent
clamped to 0.01..99.99, because a miss from 1% to 2% costs about what a miss
from 33% to 50% does under the contest's pinball loss on the logit scale):

  * Time is measured as x = ln(1 + hours since the game's launch).  Earn
    rates move fast in the first hours and then flatten as the player base
    matures, so a straight line in x is a better description than a straight
    line in clock time, and it extrapolates gently.
  * Per trophy, an ordinary least-squares line of logit(rate) against x is
    fitted over every read that has the trophy.  The slope is shrunk toward
    the pooled slope of the trophy's tier (platinum and gold are pooled with
    the whole list, since a tier of one or two trophies has no pool of its
    own): a trophy with few distinct reads borrows its tier's trend, one with
    many distinct reads trusts its own.  Slopes are clamped to [-2, 2].
  * The median forecast anchors on the trophy's last observed rate and adds
    the shrunk slope times the distance from the last read to the target in
    x.  Anchoring on the last read keeps the level current when Exophase's
    page moves in steps between refreshes.
  * The spread is built from the fit's residual scatter, the slope's
    standard error times the extrapolation distance, and a floor that grows
    with the horizon (0.30 logit units per square-root day).  It is wider
    upward for rare trophies (their rates can only really grow) and wider
    downward for near-certain ones (a small dip in percent is a large move
    in logit).  A trophy at 100% is pinned at 99.99% with a long downward
    tail, since it can only fall.
  * A trophy whose history is flat (at most two distinct rates, or a logit
    range under 0.05) carries no trend information.  Exophase refreshes its
    counts in jumps: the launch-night page sat frozen at 638 players tracked
    and then jumped to 3,868, and every rate but the one at 100% fell.  A
    flat history therefore forecasts a dilution move instead of a flat line:
    the median shifts down by 0.9 logit units (0.7 for rare trophies, which
    also get a wider upward tail) with a wide spread, and a trophy at 0%
    shifts up slightly with a wide upward tail, since its rate can only
    rise.  Nothing about any trophy is hardcoded: every number above is a
    property of the case at runtime, read from the trophy's own history.

The output for a case is {"trophies": [{"id": n, "q": [q05, q25, q50, q75,
q95]}, ...]}, one row per trophy, quantiles non-decreasing, each rounded to
4 decimals and clamped to 0.01..99.99.  A case that cannot be read still
produces a valid file: a trophy with no usable history is forecast from its
tier's median last rate with a wide spread, and a case with no history at
all is forecast at a flat 5% with a wide spread.  The program never crashes
on a malformed case: it writes a valid file for every *.in it finds.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path: put it back
# first, so a module shipped beside this file would import.  (This entry
# ships no other module; the line is here for the same reason the kit's
# starter has it.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Bounds of the logit scale: the scorer clamps rates to 0.01..99.99 before
# taking logits, so the model works inside the same bounds.
LO = 0.01
HI = 99.99
LOGIT_LO = math.log(LO / (100.0 - LO))
LOGIT_HI = math.log(HI / (100.0 - HI))

# How much a fitted slope is trusted over its tier's pooled slope:
# weight = n_distinct / (n_distinct + SHRINK_K).
SHRINK_K = 6.0
# Slopes are clamped to this range, in logit units per unit of ln(1+hours).
SLOPE_CLAMP = 2.0
# The spread's floor, in logit units per square-root day of horizon.
HORIZON_SIGMA = 0.30
# The smallest residual scatter the model believes, in logit units.
RESID_FLOOR = 0.10


def clamp_rate(pct):
    """A rate kept inside the scorer's logit bounds."""
    return min(HI, max(LO, pct))


def logit_of(pct):
    """logit of a percent rate, with the rate clamped first."""
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    """A logit back to a percent rate."""
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up; a whole number comes back as int."""
    scaled = x * 10000.0
    value = math.floor(scaled + 0.5) / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    """An ISO 8601 timestamp (with a Z or an offset) as an aware datetime."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def launch_time(case):
    """The game's launch instant: release_date at 00:00 UTC.

    Falls back to the earliest history read when the release date is
    missing or unparsable, so the program never fails on a date format.
    """
    game = case.get("game") or {}
    raw = game.get("release_date")
    if isinstance(raw, str) and raw:
        try:
            moment = parse_time(raw)
            return moment.replace(hour=0, minute=0, second=0, microsecond=0)
        except ValueError:
            pass
        try:
            year, month, day = (int(part) for part in raw[:10].split("-"))
            return datetime(year, month, day, tzinfo=timezone.utc)
        except (ValueError, TypeError):
            pass
    history = case.get("history") or []
    if history:
        try:
            return parse_time(history[0]["at"])
        except (KeyError, TypeError, ValueError):
            pass
    return datetime(2026, 11, 19, tzinfo=timezone.utc)


def x_of(moment, launch):
    """ln(1 + hours since launch); reads before launch sit at 0."""
    hours = (moment - launch).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


def fit_line(xs, ys):
    """Least squares y ~ a + b*x.  Returns (a, b, resid_std, sxx, n).

    resid_std is the scatter around the line (0 when the points sit on it,
    or when fewer than 3 points leave nothing to scatter against); sxx is
    the spread of x, which sets the slope's standard error.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 0 else 0.0
    intercept = mean_y - slope * mean_x
    if n > 2:
        resid = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
        resid_std = math.sqrt(resid / (n - 2))
    else:
        resid_std = 0.0
    return intercept, slope, resid_std, sxx, n


def median_of(values):
    """The median of a non-empty list of numbers."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def quantiles_from(mu, sigma_down, sigma_up):
    """Five quantiles around a logit median, with an asymmetric spread.

    The lower quantiles use sigma_down, the upper ones sigma_up; the
    median is untouched.  Results are percent rates, rounded to 4 decimals,
    clamped to the scorer's bounds and forced non-decreasing (rounding can
    otherwise tie or invert neighbours at the extremes).
    """
    out = []
    for z in Z:
        sigma = sigma_down if z < 0 else sigma_up
        value = round4(clamp_rate(inv_logit(mu + z * sigma)))
        if out and value < out[-1]:
            value = out[-1]
        out.append(value)
    return out


def forecast_trophy(points, last_rate, last_x, target_x, horizon_days,
                    tier_slope, overall_slope):
    """Five quantiles for one trophy.

    points: the trophy's (x, logit(rate)) reads, oldest first.
    last_rate / last_x: its last observed rate and that read's x.
    tier_slope / overall_slope: the pooled slopes to shrink toward.
    Returns [q05, q25, q50, q75, q95] in percent.
    """
    if last_rate is None:
        # Never observed: a flat 5% with a wide spread is the honest
        # answer, and it is always a valid row.
        return quantiles_from(logit_of(5.0), 1.2, 1.2)

    distinct = len({round(y, 9) for _, y in points})
    y_values = [y for _, y in points]
    y_range = (max(y_values) - min(y_values)) if y_values else 0.0
    flat = distinct <= 2 or y_range < 0.05
    last_logit = logit_of(last_rate)

    if last_rate >= 99.9:
        # At the ceiling it can only fall, and a small dip in percent is a
        # large move in logit: pin the median at the top and give the
        # downside a long tail.
        return quantiles_from(LOGIT_HI, 2.0, 0.3)

    if flat:
        # No trend to fit: Exophase refreshes in jumps, and a jump dilutes
        # rates as new players arrive, so the median moves down and the
        # spread stays wide.  A trophy at ~0% can only rise, so it moves
        # up slightly instead, with a wide upward tail.
        if last_rate <= 0.02:
            return quantiles_from(last_logit + 0.5, 0.5, 1.3)
        if last_rate < 1.0:
            return quantiles_from(last_logit - 0.7, 0.85, 1.1)
        return quantiles_from(last_logit - 0.9, 0.85, 0.85)

    # A real trend: fit the line, shrink its slope toward the tier pool,
    # and anchor the median on the last observed rate.
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    _, slope, resid_std, sxx, _ = fit_line(xs, ys)
    pool = tier_slope if tier_slope is not None else overall_slope
    weight = distinct / (distinct + SHRINK_K)
    shrunk = weight * slope + (1.0 - weight) * pool
    shrunk = max(-SLOPE_CLAMP, min(SLOPE_CLAMP, shrunk))
    dx = target_x - last_x
    mu = last_logit + shrunk * dx
    mu = max(LOGIT_LO, min(LOGIT_HI, mu))

    resid = max(resid_std, RESID_FLOOR)
    se_slope = resid / math.sqrt(sxx) if sxx > 0 else resid
    sigma = math.sqrt(resid * resid
                      + (se_slope * dx) ** 2
                      + (HORIZON_SIGMA * math.sqrt(horizon_days)) ** 2
                      + RESID_FLOOR ** 2)
    sigma_down, sigma_up = sigma, sigma
    if last_rate < 1.0:
        # Rare trophies: the upside (players reaching the trophy) is the
        # open direction; the downside is bounded by 0%.
        sigma_down *= 0.8
        sigma_up *= 1.4
    elif last_rate > 90.0:
        sigma_down *= 1.6
        sigma_up *= 0.7
    return quantiles_from(mu, sigma_down, sigma_up)


def forecast_case(case):
    """One case's forecast: {"trophies": [{"id": n, "q": [...]}, ...]}."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    launch = launch_time(case)
    horizon_days = max(0.0, (case.get("horizon_minutes") or 0) / 1440.0)
    target_x = None
    try:
        target_x = x_of(parse_time(case["target"]), launch)
    except (KeyError, TypeError, ValueError):
        pass
    if target_x is None:
        # No usable target: the checkpoint plus the horizon, else the
        # last read plus the horizon, so the extrapolation distance is
        # never a wild guess from a missing timestamp.
        from datetime import timedelta
        horizon = timedelta(minutes=case.get("horizon_minutes") or 0)
        anchor = None
        for key in ("at",):
            try:
                anchor = parse_time(case[key])
                break
            except (KeyError, TypeError, ValueError):
                continue
        if anchor is None and history:
            try:
                anchor = parse_time(history[-1]["at"])
            except (KeyError, TypeError, ValueError):
                anchor = None
        target_x = x_of(anchor + horizon, launch) if anchor is not None else 0.0

    # Each trophy's reads, in (x, logit(rate)) form, oldest first.
    per_trophy = []
    for index in range(len(trophies)):
        points = []
        last_rate = None
        last_x = None
        for read in history:
            rates = read.get("rates") or []
            if index >= len(rates):
                continue
            rate = rates[index]
            if rate is None:
                continue
            try:
                moment = parse_time(read["at"])
            except (KeyError, TypeError, ValueError):
                continue
            read_x = x_of(moment, launch)
            points.append((read_x, logit_of(float(rate))))
            last_rate = float(rate)
            last_x = read_x
        per_trophy.append((points, last_rate, last_x))

    # Pooled slopes: one per tier (tiers with fewer than three trophies
    # borrow the overall pool), and one overall, from the trophies that
    # have enough distinct reads to fit a line at all.
    slopes_by_tier = {}
    all_slopes = []
    for (points, _, _), trophy in zip(per_trophy, trophies):
        if len({round(y, 9) for _, y in points}) < 3:
            continue
        _, slope, _, _, _ = fit_line([x for x, _ in points],
                                     [y for _, y in points])
        slope = max(-SLOPE_CLAMP, min(SLOPE_CLAMP, slope))
        all_slopes.append(slope)
        slopes_by_tier.setdefault(trophy.get("tier"), []).append(slope)
    overall_slope = median_of(all_slopes) if all_slopes else 0.0
    tier_slopes = {}
    for tier, slopes in slopes_by_tier.items():
        tier_slopes[tier] = median_of(slopes) if len(slopes) >= 3 else None

    # A fallback anchor for a trophy never observed: its tier's median
    # last rate, else the overall median last rate, else 5%.
    observed = [last for (_, last, _) in per_trophy if last is not None]
    overall_fallback = median_of(observed) if observed else 5.0
    tier_fallback = {}
    for tier in {t.get("tier") for t in trophies}:
        rates = [last for (_, last, _), t in zip(per_trophy, trophies)
                 if last is not None and t.get("tier") == tier]
        tier_fallback[tier] = median_of(rates) if rates else overall_fallback

    rows = []
    for trophy, (points, last_rate, last_x) in zip(trophies, per_trophy):
        tier = trophy.get("tier")
        if last_rate is None:
            anchor = tier_fallback.get(tier, overall_fallback)
            rows.append({"id": trophy["id"],
                         "q": quantiles_from(logit_of(anchor), 1.2, 1.2)})
            continue
        rows.append({
            "id": trophy["id"],
            "q": forecast_trophy(points, last_rate, last_x, target_x,
                                 horizon_days, tier_slopes.get(tier),
                                 overall_slope),
        })
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def fallback_rows(case):
    """A valid forecast from a case whose full model failed: each trophy
    at its last observed rate (or its tier's median last rate) with a wide
    spread.  Only used when forecast_case raises, so a run never dies
    without writing a file the grader can score.
    """
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    observed = []
    last_by_trophy = []
    for index in range(len(trophies)):
        last = None
        for read in history:
            rates = read.get("rates") or []
            if index < len(rates) and rates[index] is not None:
                last = float(rates[index])
        last_by_trophy.append(last)
        if last is not None:
            observed.append(last)
    overall = median_of(observed) if observed else 5.0
    rows = []
    for trophy, last in zip(trophies, last_by_trophy):
        anchor = last if last is not None else overall
        rows.append({"id": trophy["id"],
                     "q": quantiles_from(logit_of(anchor), 1.2, 1.2)})
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        forecast = None
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            case = None
        if case is not None:
            try:
                forecast = forecast_case(case)
            except Exception:
                forecast = fallback_rows(case)
        if forecast is None:
            # The file itself is unreadable: no trophy list exists to
            # forecast, so the smallest JSON shape is written rather than
            # crashing.  (House cases always parse; this is a safety net.)
            forecast = {"trophies": []}
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
