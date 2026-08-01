"""

Skrypt pobiera wszystkie surowe dane do data/raw/

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger("collect_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pobierz dane")
    parser.add_argument("--sources", 
                        default="all",
                        help="Źródła do pobrania: all | ohlcv,macro,fear_greed,... (przecinek)"
                        )
    parser.add_argument("--coins", 
                        default="BTC,ETH"
                        )
    parser.add_argument("--start", 
                        default=None, 
                        help="Override daty startowej (YYYY-MM-DD)"
                        )
    parser.add_argument("--end",   
                        default=None
                        )
    return parser.parse_args()


def collect_ohlcv(coins: list[str], 
                  start: str, 
                  end: str
                  ) -> None:
    from src.collection.market.ohlcv_collector import OHLCVCollector
    collector = OHLCVCollector()
    symbols = [f"{c}/USDT" for c in coins]
    for symbol in symbols:
        collector.collect_symbol(symbol, 
                                 since=f"{start}T00:00:00Z", 
                                 until=f"{end}T00:00:00Z"
                                 )


def collect_correlated(start: str, 
                       end: str
                       ) -> None:
    from src.collection.market.correlated_collector import CorrelatedCollector
    CorrelatedCollector().collect_all(start=start, 
                                      end=end
                                      )


def collect_derivatives(coins: list[str]) -> None:
    from src.collection.market.derivatives_collector import DerivativesCollector
    DerivativesCollector().collect_all(coins=coins)


def collect_macro() -> None:
    from src.collection.macro.fred_collector import FREDCollector
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        logger.warning("Brak FRED_API_KEY - pomijam FRED")
        return
    FREDCollector(api_key=api_key).collect_all()


def collect_fear_greed() -> None:
    from src.collection.sentiment_indices.fear_greed_collector import FearGreedCollector
    FearGreedCollector().collect()


def collect_google_trends() -> None:
    from src.collection.sentiment_indices.google_trends_collector import GoogleTrendsCollector
    GoogleTrendsCollector().collect_all()


def collect_onchain(coins,
                    start,
                    end
                    ) -> None:
    from src.collection.onchain.coinmetrics_collector import CoinMetricsCollector
    c = CoinMetricsCollector()
    c.collect_all(coins=coins,
                  start=start,
                  end=end
                  )
    c.collect_stablecoin_supply(start=start,
                                end=end
                                )
    c.collect_market_dominance(start=start,
                               end=end
                               )

def collect_cot() -> None:
    from src.collection.macro.institutional_collector import COTCollector
    COTCollector().collect()

def collect_reddit(start: str = "2017-08-01", 
                   end: str = "2026-01-01"
                   ) -> None:
    from src.collection.text.reddit_collector import ArcticShiftCollector
    ArcticShiftCollector().collect_all(start=start, 
                                       end=end
                                       )


def main() -> None:
    args = parse_args()
    cfg = load_config("configs/sources.yaml")

    start = args.start or cfg["data_range"]["start"]
    end = args.end or cfg["data_range"]["end"]
    coins = args.coins.split(",")

    sources_all = {
        "ohlcv": lambda: collect_ohlcv(coins, start, end),
        "correlated": lambda: collect_correlated(start, end),
        "derivatives": lambda: collect_derivatives(coins),
        "macro": collect_macro,
        "cot": collect_cot,
        "fear_greed": collect_fear_greed,
        "google_trends": collect_google_trends,
        "onchain": lambda: collect_onchain(coins, start, end),
        "reddit": lambda: collect_reddit(start, end)
        }

    requested = list(sources_all) if args.sources == "all" else args.sources.split(",")

    for name in requested:
        if name not in sources_all:
            logger.warning(f"Nieznane źródło: {name}")
            continue
        logger.info(f"Pobieram: {name}...")
        try:
            sources_all[name]()
            logger.info(f"{name}: OK")
        except Exception as e:
            logger.error(f"{name}: BŁĄD — {e}")


if __name__ == "__main__":
    main()
