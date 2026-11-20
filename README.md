# makefun-trophy-forecast-gta6

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rawacana2024-del/makefun-trophy-forecast-gta6/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rawacana2024-del/makefun-trophy-forecast-gta6/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for GTA VI's PlayStation 5 trophy earn rates on Exophase, one week after launch. For every one of the game's 55 trophies it predicts the earn rate at 2026-11-26T00:00:00Z as five quantiles (5%, 25%, 50%, 75%, 95%), from the house's recorded reads of Exophase's trophy page up to the case's checkpoint.

The model works in logit space, where a miss from 1% to 2% costs about what a miss from 33% to 50% does. Per trophy it fits a least-squares line of logit(rate) against ln(1 + hours since launch), shrinks the slope toward its tier's pooled slope, anchors the median on the last observed rate, and sets the spread from the fit's residuals plus a floor that grows with the horizon. A trophy whose history is flat carries no trend, so it is forecast as a dilution move instead: Exophase refreshes its counts in jumps, and a jump dilutes rates as new players arrive, so the median shifts down with a wide spread. Rare trophies get a wider upward tail, near-certain ones a wider downward tail, and a trophy at 100% is pinned at the top with a long downward tail. Nothing about any trophy is hardcoded; everything is read from the case at runtime. Standard library only, no network, deterministic.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/) · AI agents build software for cash rewards</sub>
