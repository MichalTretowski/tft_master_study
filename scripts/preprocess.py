"""

Skrypt odpowiedzialny za cały preprocessing danych
od raw data do gotowych datasetow train/val/test.

Pipeline:
  1. OHLCV 1h  -> czyszczenie, fill luk, resample 4h/1d
  2. FeatureEngineer -> wskazniki techniczne na 4h/1d
  3. Zrodla makro/correlated/onchain -> resample do 4h/1d (ffill)
  4. TimeAligner -> join wszystkich zrodel na master index 4h/1d
  5. Target -> kierunek za 6 swiec (24h) z dead zone 0.3%
  6. Zapis -> data/processed/datasets/{coin}_{split}.parquet

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.alignment.time_aligner import TimeAligner, SPLIT_DATES
from src.preprocessing.macro.macro_preprocessor import MacroPreprocessor
from src.preprocessing.market.feature_engineer import FeatureEngineer
from src.preprocessing.market.ohlcv_preprocessor import OHLCVPreprocessor
from src.utils.logger import get_logger

logger = get_logger("preprocess")

TIMEFRAMES = {
    "4h": ("4h", 6),
    "1d": ("1D", 1)
    }

FORECAST_HORIZON_BARS = 1

_SENT_SCORES = ["reddit_sentiment_score",
                "sentiment_score_combined",
                "twitter_sentiment_score",
                "news_sentiment_score"
                ]
_SENT_COUNTS = ["reddit_n_texts",
                "twitter_n_texts",
                "news_n_texts"
                ]
_SENT_BASES = ["reddit_sentiment_score",
               "sentiment_score_combined"
               ]

RAW        = Path("data/raw")
PROCESSED  = Path("data/processed")
FEATURES   = PROCESSED / "features"
DATASETS   = PROCESSED / "datasets"


def step_ohlcv(coins: list[str],
               tf: str,
               bpd: int
               ) -> dict[str, pd.DataFrame]:
    """Funkcja czyści zbiór OHLCV 1h i buduje wskazniki techniczne 4h lub 1d"""
    prep = OHLCVPreprocessor(FEATURES,
                             timeframe=tf
                             )
    eng = FeatureEngineer(FEATURES,
                          bars_per_day=bpd,
                          timeframe=tf
                          )
    result: dict[str, pd.DataFrame] = {}

    for coin in coins:
        raw_path = RAW / "ohlcv" / f"{coin}_USDT_1h.parquet"
        if not raw_path.exists():
            logger.warning(f"Brak {raw_path} - pomijam {coin}")
            continue

        logger.info(f"[OHLCV] {coin}: czyszczenie i resample 1h->{tf}...")
        df_raw = pd.read_parquet(raw_path)
        dfs = prep.process(df_raw, 
                           coin
                           )
        df_tf = dfs[tf]

        logger.info(f"[OHLCV] {coin}: feature engineering ({len(df_tf)} świec {tf}...")
        df_feat = eng.build_features(df_tf, 
                                     coin
                                     )
        result[coin] = df_feat

    return result



def _fix_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Funkcja sprowadza każde źródło do tej samej postaci: 
    indeks czasowy, w UTC, posortowany rosnąco
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["timestamp", "date", "Date", "Timestamp"]:
            if col in df.columns:
                df = df.set_index(col)
                break
    df.index = pd.to_datetime(df.index, 
                              utc=True
                              )
    return df.sort_index()


def step_macro(master_index: pd.DatetimeIndex,
               tf: str,
               bpd: int
               ) -> dict[str, pd.DataFrame]:
    """Funkcja przetwarza zrodla makro do 4h/1d"""
    mp = MacroPreprocessor(FEATURES,
                           bars_per_day=bpd,
                           timeframe=tf
                           )
    sources: dict[str, pd.DataFrame] = {}


    fred_path = RAW / "macro" / "fred_macro_raw.parquet"
    if fred_path.exists():
        logger.info(f"[Macro] FRED: ffill do {tf}...")
        df = _fix_index(pd.read_parquet(fred_path))
        df = mp.process_and_save(df, 
                                 "fred_macro", 
                                 target_index=master_index,
                                 method="ffill", 
                                 add_derived=True
                                 )
        sources["fred"] = df
    else:
        logger.warning("Brak FRED macro - pomijam")


    fg_path = RAW / "macro" / "fear_greed_index.parquet"
    if fg_path.exists():
        logger.info(f"[Macro] Fear&Greed: ffill do {tf}...")
        df = _fix_index(pd.read_parquet(fg_path))
        df = df[["fear_greed_value"]].rename(columns={"fear_greed_value": "fear_greed"})
        df_tf = mp.resample_to_target(df, 
                                  method="ffill", 
                                  target_index=master_index
                                  )
        df_tf["fear_greed"] = df_tf["fear_greed"].fillna(50.0)
        FEATURES.mkdir(parents=True, 
                       exist_ok=True
                       )
        df_tf.to_parquet(FEATURES / f"fear_greed_{tf}.parquet")
        logger.info(f"[Macro] Fear&Greed: {df_tf['fear_greed'].notna().sum()} wierszy {tf}")
        sources["fear_greed"] = df_tf
    else:
        logger.warning("Brak Fear&Greed - pomijam")


    cot_path = RAW / "institutional" / "cot_btc_futures.parquet"
    if cot_path.exists():
        logger.info(f"[Macro] COT: ffill do {tf}...")
        df = _fix_index(pd.read_parquet(cot_path))
        df = mp.process_and_save(df, 
                                 "cot", 
                                 target_index=master_index,
                                 method="ffill", 
                                 add_derived=False
                                 )
        sources["cot"] = df
    else:
        logger.warning("Brak COT - pomijam")


    trends_files = {
        "google_trends_btc.parquet": {
            "bitcoin": "trends_bitcoin",
            "buy bitcoin": "trends_buy_bitcoin",
            "bitcoin price": "trends_bitcoin_price",
            "bitcoin crash": "trends_bitcoin_crash"
            },
        "google_trends_eth.parquet": {
            "ethereum": "trends_ethereum",
            "buy ethereum": "trends_buy_ethereum",
            "ethereum price": "trends_ethereum_price"
            },
        "google_trends_general.parquet": {
            "crypto": "trends_crypto",
            "cryptocurrency": "trends_cryptocurrency",
            "blockchain": "trends_blockchain"
            }}
    trends_frames: list[pd.DataFrame] = []
    for fname, col_map in trends_files.items():
        path = RAW / "macro" / fname
        if path.exists():
            df = _fix_index(pd.read_parquet(path))
            df = df.rename(columns=col_map)[list(col_map.values())]
            trends_frames.append(df)
        else:
            logger.warning(f"Brak Google Trends {fname} - pomijam")

    if trends_frames:
        logger.info(f"[Macro] Google Trends: ffill do {tf}...")
        combined = pd.concat(trends_frames, 
                             axis=1,
                             sort=False
                             ).sort_index()
        df_tf = mp.resample_to_target(combined, 
                                  method="ffill", 
                                  target_index=master_index
                                  )
        df_tf = df_tf.fillna(50.0)
        df_tf.to_parquet(FEATURES / f"google_trends_{tf}.parquet")
        logger.info(f"[Macro] Google Trends: {len(df_tf)} wierszy {tf}, {len(df_tf.columns)} kolumn")
        sources["google_trends"] = df_tf

    return sources


def step_correlated(master_index: pd.DatetimeIndex,
                    tf: str,
                    bpd: int
                    ) -> dict[str, pd.DataFrame]:
    """Funkcja przetwarza aktywa skorelowane"""
    mp = MacroPreprocessor(FEATURES,
                           bars_per_day=bpd,
                           timeframe=tf
                           )
    corr_dir = RAW / "correlated"
    sources: dict[str, pd.DataFrame] = {}

    tickers = [p.stem.replace("_1d", "") for p in corr_dir.glob("*_1d.parquet")]
    if not tickers:
        logger.warning("Brak plikow correlated - pomijam")
        return sources

    logger.info(f"[Correlated] {len(tickers)} instrumentow -> {tf}...")
    frames: list[pd.DataFrame] = []

    for name in tickers:
        path = corr_dir / f"{name}_1d.parquet"
        df = _fix_index(pd.read_parquet(path))
        close_col = next((c for c in df.columns if c.lower() == "close"), None)
        if close_col is None:
            continue
        df = df[[close_col]].rename(columns={close_col: f"corr_{name}_close"})
        df[f"corr_{name}_ret1d"] = df[f"corr_{name}_close"].pct_change()
        frames.append(df)

    if not frames:
        return sources

    combined = pd.concat(frames, 
                         axis=1,
                         sort=False
                         ).sort_index()
    df_tf = mp.resample_to_target(combined, 
                              method="ffill", 
                              target_index=master_index
                              )
    df_tf.to_parquet(FEATURES / f"correlated_{tf}.parquet")
    logger.info(f"[Correlated] {len(frames)} instrumentow, {len(df_tf)} wierszy {tf}")
    sources["correlated"] = df_tf
    return sources


def step_onchain(master_index: pd.DatetimeIndex,
                 tf: str,
                 bpd:int
                 ) -> dict[str, pd.DataFrame]:
    """Funkcja przetwarza dane on-chain"""
    mp = MacroPreprocessor(FEATURES,
                           bars_per_day=bpd,
                           timeframe=tf
                           )
    sources: dict[str, pd.DataFrame] = {}

    for coin in ["BTC", "ETH"]:
        path = RAW / "onchain" / f"{coin}_onchain_daily.parquet"
        if not path.exists():
            logger.warning(f"Brak on-chain {coin} ({path}) - pomijam")
            continue

        logger.info(f"[OnChain] {coin}: ffill do {tf}...")
        df = _fix_index(pd.read_parquet(path))
        df.columns = [f"onchain_{coin.lower()}_{c}" for c in df.columns]
        df_tf = mp.resample_to_target(df, 
                                  method="ffill", 
                                  target_index=master_index
                                  )
        df_tf.to_parquet(FEATURES / f"onchain_{coin}_{tf}.parquet")
        logger.info(f"[OnChain] {coin}: {len(df_tf)} wierszy {tf}")
        sources[f"onchain_{coin}"] = df_tf

    return sources


def step_derivatives(master_index: pd.DatetimeIndex,
                     tf: str,
                     bpd: int
                     ) -> dict[str, pd.DataFrame]:
    """Funkcja przetwarza derywaty"""
    mp = MacroPreprocessor(FEATURES,
                           bars_per_day=bpd,
                           timeframe=tf
                           )
    deriv_dir = RAW / "derivatives"
    sources: dict[str, pd.DataFrame] = {}

    for coin in ["BTC", "ETH"]:
        frames: list[pd.DataFrame] = []

        fr_path = deriv_dir / f"{coin}_funding_rates_binance.parquet"
        if fr_path.exists():
            df = _fix_index(pd.read_parquet(fr_path))
            df = df.rename(columns={"funding_rate": f"funding_rate_{coin.lower()}"})
            frames.append(df)

        oi_path = deriv_dir / f"{coin}_open_interest_1h_bybit.parquet"
        if oi_path.exists():
            df = _fix_index(pd.read_parquet(oi_path))
            df = df.rename(columns={"open_interest": f"open_interest_{coin.lower()}"})
            frames.append(df)

        ls_path = deriv_dir / f"{coin}_long_short_ratio_1h_bybit.parquet"
        if ls_path.exists():
            df = _fix_index(pd.read_parquet(ls_path))
            df.columns = [f"{c}_{coin.lower()}" for c in df.columns]
            frames.append(df)

        if not frames:
            continue

        combined = pd.concat(frames, 
                             axis=1,
                             sort=False
                             ).sort_index()
        df_tf = mp.resample_to_target(combined, 
                                  method="ffill", 
                                  target_index=master_index
                                  )

        df_tf.to_parquet(FEATURES / f"derivatives_{coin}_{tf}.parquet")
        logger.info(f"[Derivatives] {coin}: {len(frames)} zrodel, {len(df_tf)} wierszy {tf}")
        sources[f"derivatives_{coin}"] = df_tf

    return sources


# Daty ogloszen FOMC (data konca posiedzenia = dzien decyzji)
_FOMC_DATES = [
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05",
    "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10"
    ]

# Znane daty halvingów BTC (i prognoza kolejnego)
_HALVING_DATES = [
    pd.Timestamp("2012-11-28", tz="UTC"),
    pd.Timestamp("2016-07-09", tz="UTC"),
    pd.Timestamp("2020-05-11", tz="UTC"),
    pd.Timestamp("2024-04-19", tz="UTC"),
    pd.Timestamp("2028-04-18", tz="UTC")  # prognoza
    ]


def step_event_calendar(master_index: pd.DatetimeIndex,
                        tf: str
                        ) -> pd.DataFrame:
    """
    Funkcja buduje kolumny kalendarza eventów makro na master_index 4h:
      - is_fed_meeting_day: 1 jezeli dzien posiedzenia FOMC
      - is_cpi_release_day: 1 jezeli w tym dniu CPI zostalo opublikowane
      - is_nfp_release_day: 1 jezeli w tym dniu NFP zostalo opublikowane
      - days_to_btc_halving: liczba dni do nastepnego halvingu BTC
    """

    df = pd.DataFrame(index=master_index)

    fomc_days = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="UTC") for d in _FOMC_DATES]
        ).normalize()
    df["is_fed_meeting_day"] = master_index.normalize().isin(fomc_days).astype(float)
    logger.info(f"[Calendar] FOMC: {int(df['is_fed_meeting_day'].sum())} swiec {tf} oznaczonych")


    fred_path = RAW / "macro" / "fred_macro_raw.parquet"
    for col_name, flag_col in [("cpi_all", "is_cpi_release_day"),
                               ("nonfarm_payrolls", "is_nfp_release_day")]:
        df[flag_col] = 0.0
        if fred_path.exists():
            fred = _fix_index(pd.read_parquet(fred_path))
            if col_name in fred.columns:
                series = fred[col_name].dropna()
                release_days = series[series.diff().abs() > 0].index.normalize()
                if len(series) > 0:
                    release_days = release_days.union(series.index[:1].normalize())
                df[flag_col] = master_index.normalize().isin(release_days).astype(float)
                n = int(df[flag_col].sum())
                logger.info(f"[Calendar] {col_name}: {n} swiec {tf} oznaczonych (release days)")


    def _days_to_next_halving(ts: pd.Timestamp) -> float:
        future = [h for h in _HALVING_DATES if h > ts]
        if not future:
            return 0.0
        return (future[0] - ts).days

    days_arr = np.array([_days_to_next_halving(ts) for ts in master_index])
    df["days_to_btc_halving"] = days_arr
    logger.info(
        f"[Calendar] days_to_btc_halving: "
        f"min={days_arr.min():.0f}, max={days_arr.max():.0f}"
        )

    FEATURES.mkdir(parents=True, 
                   exist_ok=True
                   )
    df.to_parquet(FEATURES / f"event_calendar_{tf}.parquet")
    logger.info(f"[Calendar] Zapisano event_calendar_{tf}.parquet ({len(df)} wierszy)")
    return df



def step_sentiment(master_index: pd.DatetimeIndex,
                   freq: str
                   ) -> pd.DataFrame | None:
    """
    Funkcja wczytuje wyniki z CryptoBERT i wyrównuje do master_index.
    Plik generuje scripts/run_sentiment.py - uruchom go przed preprocessingiem.
    Jeśli plik nie istnieje, pipeline kontynuuje z kolumnami sentymentu = 0
    (ColumnMapper w dataset.py uzupełnia brakujące kolumny zerami).
    """
    sentiment_path = PROCESSED / "sentiment_embeddings" / "sentiment_combined_4h.parquet"
    if not sentiment_path.exists():
        logger.warning(
            "Brak pliku sentymentu - uruchom najpierw: python scripts/run_sentiment.py\n"
            "Pipeline kontynuuje bez sentymentu (kolumny = 0 w datasecie)."
        )
        return None

    df = pd.read_parquet(sentiment_path)
    df.index = pd.to_datetime(df.index, 
                              utc=True
                              )

    scores = [c for c in _SENT_SCORES if c in df.columns]
    counts = [c for c in _SENT_COUNTS if c in df.columns]

    out = pd.concat([
        df[scores].resample(freq).mean(),
        df[counts].resample(freq).sum()
        ], axis=1
        )

    bpd = 6 if freq == "4h" else 1
    for base in _SENT_BASES:
        if base not in out.columns:
            continue
        for lag in [1, 6, 24, 42]:
            out[f"{base}_lag{lag}"] = out[base].shift(lag)
        for days, label in [(1, "1d"),
                            (7, "7d"),
                            (30, "30d")
                            ]:
            out[f"{base}_roll_{label}"] = out[base].rolling(days * bpd,
                                                            min_periods=1
                                                            ).mean()
    out = out.reindex(master_index).ffill().fillna(0.0)
    logger.info(f"[Sentiment] {len(out)} wierszy, {len(out.columns)} kolumn sentymentu")
    return out
    

def step_market_supplement(master_index: pd.DatetimeIndex,
                           tf: str,
                           bpd: int
                           ) -> dict[str, pd.DataFrame]:
    """Funkcja przetwarza stablecoin supply i market dominance"""
    mp = MacroPreprocessor(FEATURES,
                           bars_per_day=bpd,
                           timeframe=tf
                           )
    sources: dict[str, pd.DataFrame] = {}

    stablecoin_path = RAW / "onchain" / "stablecoin_supply_daily.parquet"
    if stablecoin_path.exists():
        logger.info(f"[Market] Stablecoin supply: ffill do {tf}...")
        df = _fix_index(pd.read_parquet(stablecoin_path))
        df_tf = mp.resample_to_target(df, 
                                  method="ffill", 
                                  target_index=master_index
                                  )
        df_tf = df_tf.fillna(0.0)
        df_tf.to_parquet(FEATURES / f"stablecoin_supply_{tf}.parquet")
        logger.info(f"[Market] Stablecoin supply: {len(df_tf)} wierszy {tf}")
        sources["stablecoin_supply"] = df_tf
    else:
        logger.warning("Brak stablecoin_supply_daily.parquet - pomijam")

    dominance_path = RAW / "onchain" / "market_dominance_daily.parquet"
    if dominance_path.exists():
        logger.info(f"[Market] Market dominance: ffill do {tf}...")
        df = _fix_index(pd.read_parquet(dominance_path))
        df_tf = mp.resample_to_target(df, 
                                  method="ffill", 
                                  target_index=master_index
                                  )
        df_tf = df_tf.ffill()
        df_tf.to_parquet(FEATURES / f"market_dominance_{tf}.parquet")
        logger.info(f"[Market] Market dominance: {len(df_tf)} wierszy {tf}")
        sources["market_dominance"] = df_tf
    else:
        logger.warning("Brak market_dominance_daily.parquet - pomijam")

    return sources



def step_build_datasets(
    ohlcv_features: dict[str, pd.DataFrame],
    macro_sources: dict[str, pd.DataFrame],
    corr_sources: dict[str, pd.DataFrame],
    onchain_sources: dict[str, pd.DataFrame],
    deriv_sources: dict[str, pd.DataFrame],
    coins: list[str],
    event_calendar: pd.DataFrame | None = None,
    market_sources: dict[str, pd.DataFrame] | None = None,
    sentiment_source: pd.DataFrame | None = None,
    master_index: pd.DatetimeIndex | None = None,
    datasets_dir: Path | None = None,
    bpd: int = 6,
    horizon_bars: int = FORECAST_HORIZON_BARS
    ) -> None:
    """Funkcja łączy wszystkie zrodla, dodaje target, zapisuje splity"""
    aligner = TimeAligner(datasets_dir if datasets_dir is not None else DATASETS)
    if master_index is None:
        master_index = aligner.build_master_index()


    shared: dict[str, pd.DataFrame] = {**macro_sources, 
                                       **corr_sources
                                       }
    if event_calendar is not None:
        shared["event_calendar"] = event_calendar
    if market_sources:
        shared.update(market_sources)
    if sentiment_source is not None:
        shared["sentiment"] = sentiment_source

    for coin in coins:
        if coin not in ohlcv_features:
            logger.warning(f"Brak features OHLCV dla {coin} - pomijam dataset")
            continue

        logger.info(f"[Dataset] Buduje {coin}...")

        coin_sources: dict[str, pd.DataFrame] = {
            "ohlcv": ohlcv_features[coin],
            **shared,
            **{k: v for k, v in onchain_sources.items() if coin in k},
            **{k: v for k, v in deriv_sources.items() if coin in k}
            }

        aligned = aligner.align_sources(coin_sources, 
                                        master_index
                                        )

        train_slice = aligned[SPLIT_DATES["train_start"]:SPLIT_DATES["train_end"]]
        dead = [c for c in aligned.columns if train_slice[c].isna().all()]
        if dead:
            logger.warning(f"[Dataset] {coin}: kolumny bez danych w treningu - usuwam {dead}")
            aligned = aligned.drop(columns=dead)

        aligned = aligner.add_target(
            aligned,
            price_col="close",
            horizon_bars=horizon_bars,
            dead_zone_pct=0.003
            )

        n_before = len(aligned)
        aligned = aligned.dropna(subset=["close"])

        logger.info(f"[Dataset] {coin}: {n_before} -> {len(aligned)} wierszy po dropna(close, target)")

        dataset = aligner.build_coin_dataset(
            aligned,
            coin=coin,
            target_col="target",
            purge_bars=horizon_bars
            )
        logger.info(
            f"[Dataset] {coin}: train={len(dataset.train)} | "
            f"val={len(dataset.val)} | test={len(dataset.test)} | "
            f"features={len(dataset.feature_cols)}"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--coins", 
                   default="BTC,ETH"
                   )
    p.add_argument("--no-derivatives", 
                   action="store_true",
                   help="Pomin derywaty"
                   )
    p.add_argument("--only-features", 
                   action="store_true",
                   help="Tylko OHLCV features"
                   )
    p.add_argument("--freq",
                   default="4h",
                   choices=["4h", "1d"]
                   )
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    coins = args.coins.split(",")
    tf = args.freq
    pandas_freq, bpd = TIMEFRAMES[tf]

    datasets_dir = DATASETS / tf

    FEATURES.mkdir(parents=True, 
                   exist_ok=True
                   )
    DATASETS.mkdir(parents=True, 
                   exist_ok=True
                   )

    logger.info(f"=== Interwał docelowy: {tf} ({bpd} barów na dobę) ===")

    aligner = TimeAligner(datasets_dir)
    master_index = aligner.build_master_index(freq=pandas_freq)

    logger.info("=== OHLCV preprocessing + feature engineering ===")
    ohlcv_features = step_ohlcv(coins,
                                tf,
                                bpd
                                )

    if args.only_features:
        logger.info("Tryb --only-features")
        return

    logger.info("=== Macro preprocessing ===")
    macro_sources = step_macro(master_index,
                               tf,
                               bpd
                               )

    logger.info("=== Correlated assets ===")
    corr_sources = step_correlated(master_index,
                                   tf,
                                   bpd
                                   )

    logger.info("=== On-chain ===")
    onchain_sources = step_onchain(master_index,
                                   tf,
                                   bpd
                                   )

    deriv_sources: dict = {}
    if not args.no_derivatives:
        logger.info("=== Derivatives ===")
        deriv_sources = step_derivatives(master_index,
                                         tf,
                                         bpd
                                         )
    else:
        logger.info("===  Derivatives - pominiete (--no-derivatives) ===")

    logger.info("=== Event calendar ===")
    event_calendar = step_event_calendar(master_index,
                                         tf
                                         )

    logger.info("=== Market supplement (stablecoins, dominance) ===")
    market_sources = step_market_supplement(master_index,
                                            tf,
                                            bpd
                                            )

    logger.info("=== Sentiment ===")
    sentiment_source = step_sentiment(master_index,
                                      pandas_freq
                                      )

    logger.info("=== Align + target + train/val/test split ===")
    step_build_datasets(
        ohlcv_features, 
        macro_sources, 
        corr_sources,
        onchain_sources, 
        deriv_sources, 
        coins,
        event_calendar=event_calendar,
        market_sources=market_sources,
        sentiment_source=sentiment_source,
        master_index=master_index,
        datasets_dir=datasets_dir,
        bpd=bpd,
        horizon_bars=FORECAST_HORIZON_BARS
        )

    logger.info(f"Preprocessing gotowy. Pliki w {datasets_dir}")


if __name__ == "__main__":
    main()
