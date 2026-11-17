# Total-first compositional day-curve quantiles

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Spasar1/gta6-twitch-forecast-4/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Spasar1/gta6-twitch-forecast-4/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's launch-day Twitch viewers, hours 2–24, as five quantiles from the hour-1 print — for the [make.fun](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) bounty `gta6-twitch-ccu-forecast-01`.

## Method — total first, then split

1. **Total.** Estimate the launch's total 24-hour viewer volume from hour 1 with a *local* log-log fit, `ln(total) = a + b·ln(hour-1)`, least squares on the 12 comparables whose hour-1 print is closest (in log) to this launch's. Local, because the total-per-hour-1 multiple drifts with launch size — about 18–24× for the biggest launches, far higher for small ones whose unlock ramps through the day.
2. **Split.** Each of the 20 closest comparables contributes its own day curve, rescaled to this launch. The rescale is the geometric mean of two estimates of the same scale factor — the fitted-total ratio (predicted total here / fitted total there) and the direct ratio (hour-1 here / hour-1 there) — which hedges both biases: the fitted ratio can understate a big launch when the local fit is noisy, the direct ratio assumes the total is exactly proportional to hour 1. The five quantiles of an hour are the *distance-weighted* quantiles of those 20 rescaled curves at that hour: a neighbour counts by a Gaussian kernel on |log hour-1| distance (bandwidth 1.0 in log units), so nearer launches count more. The spread across similar-size launches is the forecast's uncertainty.

Python 3.13, standard library only, deterministic (identical output bytes across runs).

## Leave-one-out check (kit comparables, 37 launches)

Mean pinball loss on the log scale, vs the house persistence baseline:

| | Overall | Big hour-1 (≥100k) |
|---|---|---|
| This entry | **0.3954** | **0.2167** |
| House baseline | 0.9297 | 0.2611 |

Beats the house on both headline numbers. The split step was revised once (the fitted-total rescale alone scored 0.4236 / 0.2218): blending in the direct hour-1 ratio and weighting neighbours by distance improved the overall loss to 0.3954 and big-h1 to 0.2167. The worst cases remain the small, spiky launches (Black Myth: Wukong, Marathon), whose day curves are hardest to pin down — the gain comes from the broad middle, not from any single launch.
