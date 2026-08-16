"""

Podzial kolumn datasetu na kategorie do ablacji.

"""

from __future__ import annotations

from fnmatch import fnmatch

FRED_SERIES = [
    "consumer_sentiment",
    "cpi_all",
    "cpi_core",
    "fed_balance_sheet",
    "fed_funds_rate",
    "ism_manufacturing",
    "m1_usa",
    "m2_usa",
    "nonfarm_payrolls",
    "pce",
    "ppi",
    "real_gdp",
    "rrp_overnight",
    "sofr",
    "tga_balance",
    "unemployment_rate",
    "yield_10y",
    "yield_2y",
    "yield_spread_10y_2y",
    "yield_spread_10y_3m"
    ]

FRED_PATTERNS = (
    list(FRED_SERIES)
    + [f"{s}_mom_30d" for s in FRED_SERIES]
    + [f"{s}_zscore_12m" for s in FRED_SERIES]
    )

OHLCV_PATTERNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "log_return",
    "pct_return",
    "return_*",
    "candle_body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "is_bullish"
    ]

TECHNICAL_PATTERNS = [
    "EMA_*",
    "SMA_*",
    "MACD*",
    "ADX_*",
    "ADXR_*",
    "DM[NP]_*",
    "RSI_*",
    "STOCH*",
    "ROC_*",
    "WILLR_*",
    "CCI_*",
    "ATR*",
    "BB[BLMPU]_*",
    "OBV",
    "volume_ratio",
    "vwap_*",
    "price_to_vwap",
    "hist_vol_*"
    ]

FEATURE_GROUPS: dict[str, list[str]] = {
    "target": ["target", "forward_log_return"],
    "static": ["coin_id"],
    "time": ["hour_sin", 
             "hour_cos", 
             "dow_sin", 
             "dow_cos"
             ],
    "calendar": [
        "is_fed_meeting_day",
        "is_cpi_release_day",
        "is_nfp_release_day",
        "days_to_btc_halving"
        ],
    "cot": [
        "comm_positions_long_all",
        "comm_positions_short_all",
        "noncomm_positions_long_all",
        "noncomm_positions_short_all",
        "net_speculator_position",
        "open_interest_all"
        ],
    "fred": FRED_PATTERNS,
    "correlated": ["corr_*"],
    "trends": ["trends_*"],
    "onchain": ["onchain_*"],
    "derivatives": [
        "funding_rate_*",
        "open_interest_*",
        "long_ratio_*",
        "short_ratio_*",
        "long_short_ratio_*"
        ],
    "stablecoin": [
        "usdt_supply",
        "usdc_supply",
        "total_stablecoin_supply"
        ],
    "dominance": [
        "btc_eth_dominance",
        "btc_market_cap_usd",
        "eth_market_cap_usd"
        ],
    "fear_greed": ["fear_greed"],
    "sentiment": [
        "reddit_*",
        "twitter_*",
        "news_*",
        "sentiment_score_*"
        ],
    "ohlcv": OHLCV_PATTERNS,
    "technical": TECHNICAL_PATTERNS
    }

NON_FEATURE_GROUPS = {"target", "static"}


class UnclassifiedColumnError(ValueError):
    pass


def assign_groups(columns) -> dict[str, str]:
    assigned: dict[str, str] = {}
    unknown: list[str] = []

    for col in columns:
        hit = None
        for group, patterns in FEATURE_GROUPS.items():
            if any(fnmatch(col, p) for p in patterns):
                hit = group
                break
        if hit is None:
            unknown.append(col)
        else:
            assigned[col] = hit

    if unknown:
        raise UnclassifiedColumnError(
            "Kolumny bez przypisanej grupy: "
            + ", ".join(sorted(unknown))
            + "\nDopisz wzorzec w FEATURE_GROUPS."
            )
    return assigned


def group_columns(columns) -> dict[str, list[str]]:
    assigned = assign_groups(columns)
    out: dict[str, list[str]] = {g: [] for g in FEATURE_GROUPS}
    for col, group in assigned.items():
        out[group].append(col)
    return {g: cols for g, cols in out.items() if cols}


def select_columns(columns, groups) -> list[str]:
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Nieznane grupy: {sorted(unknown)}")
    assigned = assign_groups(columns)
    return [c for c in columns if assigned[c] in set(groups)]


def collisions(columns) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for col in columns:
        hits = [
            g
            for g, patterns in FEATURE_GROUPS.items()
            if any(fnmatch(col, p) for p in patterns)
            ]
        if len(hits) > 1:
            out[col] = hits
    return out


def _main() -> None:
    import argparse
    from pathlib import Path

    import pandas as pd

    p = argparse.ArgumentParser()
    p.add_argument("--tf", default="4h", choices=["4h", "1d"])
    p.add_argument("--coin", default="BTC")
    args = p.parse_args()

    path = (
        Path("data/processed/datasets")
        / args.tf
        / f"{args.coin}_train.parquet"
        )
    cols = pd.read_parquet(path).columns

    groups = group_columns(cols)
    total = sum(len(v) for v in groups.values())

    print("")
    print(f"{args.coin} / {args.tf} — {total} kolumn")
    print("-" * 40)
    for g, c in groups.items():
        mark = "" if g not in NON_FEATURE_GROUPS else "  (nie-cecha)"
        print(f"  {g:<14} {len(c):>4}{mark}")

    n_feat = sum(
        len(c) for g, c in groups.items()
        if g not in NON_FEATURE_GROUPS
    )
    print("-" * 40)
    print(f"  {'cechy razem':<14} {n_feat:>4}")

    col_hits = collisions(cols)
    if col_hits:
        print("")
        print("Kolizje wzorcow (wygrywa pierwsza grupa):")
        for col, hits in sorted(col_hits.items()):
            print(f"  {col}: {hits}")


if __name__ == "__main__":
    _main()
