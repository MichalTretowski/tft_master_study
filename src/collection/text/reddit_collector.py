"""
Reddit Collector - Arctic Shift API

Arctic Shift: https://arctic-shift.photon-reddit.com/api
  GET /posts/search?subreddit=Bitcoin&after=<unix>&before=<unix>&limit=100
  GET /comments/search?subreddit=Bitcoin&after=<unix>&before=<unix>&limit=100

API zwraca newest-first, paginacja przez before = min_ts - 1.

"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

RAW_DIR   = Path("data/raw/text/reddit")
START = "2017-08-01"
END = "2026-01-01"

ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com/api"

# Subreddity do zbierania postow i komentarzy
POST_SUBREDDITS    = ["Bitcoin", "CryptoCurrency", "ethereum", "CryptoMarkets"]
COMMENT_SUBREDDITS = ["Bitcoin", "CryptoCurrency"]


class ArcticShiftCollector:
    """
    Klasa pobiera historyczne posty i komentarze Reddit przez Arctic Shift API.

    """

    def __init__(
        self,
        raw_dir: Path = RAW_DIR,
        post_subreddits: list[str] | None = None,
        comment_subreddits: list[str] | None = None,
        request_delay: float = 0.1,
        checkpoint_every: int = 500,
        ) -> None:
        if post_subreddits is None:
            post_subreddits = POST_SUBREDDITS
        if comment_subreddits is None:
            comment_subreddits = COMMENT_SUBREDDITS
        self.raw_dir = raw_dir
        self.post_subreddits = post_subreddits
        self.comment_subreddits = comment_subreddits
        self.request_delay = request_delay
        self.checkpoint_every = checkpoint_every
        self.raw_dir.mkdir(parents=True, 
                           exist_ok=True
                           )
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "tft-crypto-research/1.0"

    def collect_all(
        self,
        start: str = START,
        end:   str = END,
        post_min_score:    int = 5,
        comment_min_score: int = 3,
        comments_top_level_only: bool = True
        ) -> None:
        """Metoda pobiera posty i komentarze ze wszystkich skonfigurowanych subredditow"""
        print("[Reddit] === POSTY ===")
        for sub in self.post_subreddits:
            df = self.collect_posts(sub, 
                                    start=start, 
                                    end=end, 
                                    min_score=post_min_score
                                    )
            if df.empty:
                continue
            out = self.raw_dir / f"posts_{sub.lower()}.parquet"
            df.to_parquet(out, 
                          index=False
                          )
            print(f"[Reddit] {sub} posty: {len(df):,} -> {out}")

        print("\n[Reddit] === KOMENTARZE ===")
        for sub in self.comment_subreddits:
            df = self.collect_comments(
                sub, 
                start=start, 
                end=end,
                min_score=comment_min_score,
                top_level_only=comments_top_level_only
                )
            if df.empty:
                continue
            out = self.raw_dir / f"comments_{sub.lower()}.parquet"
            df.to_parquet(out, 
                          index=False
                          )
            print(f"[Reddit] {sub} komentarze: {len(df):,} -> {out}")

    def collect_posts(
        self,
        subreddit: str,
        start: str = START,
        end:   str = END,
        min_score: int = 5
        ) -> pd.DataFrame:
        """Metoda pobiera posty (submissions) z jednego subredditu"""
        ckpt_key = f"posts_{subreddit.lower()}"
        records = self._paginate(
            endpoint="posts",
            subreddit=subreddit,
            start=start,
            end=end,
            extract_fn=self._extract_post,
            min_score=min_score,
            label=f"{subreddit}/posts",
            checkpoint_key=ckpt_key
            )
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["created_utc"], 
                                         unit="s", 
                                         utc=True
                                         )
        df = df.drop(columns=["created_utc"])
        df["selftext"] = df["selftext"].replace("[deleted]", "").replace("[removed]", "")
        df = df.drop_duplicates(subset="id").sort_values("timestamp").reset_index(drop=True)

        print(f"[Arctic Shift] {subreddit} posty: {len(df):,} "
              f"({df['timestamp'].min().date()} -> {df['timestamp'].max().date()}) "
              f"[score>={min_score}]")
        self._delete_checkpoint(ckpt_key)
        return df

    def collect_comments(
        self,
        subreddit: str,
        start: str = START,
        end:   str = END,
        min_score: int = 3,
        top_level_only: bool = True,
        ) -> pd.DataFrame:
        """
        Metoda pobiera komentarze z jednego subredditu.
        top_level_only=True: tylko bezposrednie odpowiedzi na posty
        """
        ckpt_key = f"comments_{subreddit.lower()}"
        records = self._paginate(
            endpoint="comments",
            subreddit=subreddit,
            start=start,
            end=end,
            extract_fn=lambda c: self._extract_comment(c, top_level_only),
            min_score=min_score,
            label=f"{subreddit}/comments{'(top)' if top_level_only else ''}",
            checkpoint_key=ckpt_key
            )
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["created_utc"], 
                                         unit="s", 
                                         utc=True
                                         )
        df = df.drop(columns=["created_utc"])
        df = df.drop_duplicates(subset="id").sort_values("timestamp").reset_index(drop=True)

        print(f"[Arctic Shift] {subreddit} komentarze: {len(df):,} "
              f"({df['timestamp'].min().date()} -> {df['timestamp'].max().date()}) "
              f"[score>={min_score}, top_level={top_level_only}]")
        self._delete_checkpoint(ckpt_key)
        return df

    def load_all(self, 
                 kind: str = "both"
                 ) -> pd.DataFrame:
        """
        Metoda wczytuje zapisane parquety.
        kind: 'posts' | 'comments' | 'both'
        """
        patterns = []
        if kind in ("posts", "both"):
            patterns.append("posts_*.parquet")
        if kind in ("comments", "both"):
            patterns.append("comments_*.parquet")

        frames = [
            pd.read_parquet(p)
            for pat in patterns
            for p in sorted(self.raw_dir.glob(pat))
        ]
        if not frames:
            return pd.DataFrame()
        return (
            pd.concat(frames, 
                      ignore_index=True
                      )
            .drop_duplicates(subset="id")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def _ckpt_paths(self, 
                    key: str
                    ) -> tuple[Path, Path]:
        return (
            self.raw_dir / f".ckpt_{key}.parquet",
            self.raw_dir / f".ckpt_{key}.json",
        )

    def _save_checkpoint(self, 
                         key: str, 
                         records: list[dict], 
                         current_before: int, 
                         fetched_total: int
                         ) -> None:
        ckpt_parquet, ckpt_json = self._ckpt_paths(key)
        pd.DataFrame(records).to_parquet(ckpt_parquet, index=False)
        ckpt_json.write_text(
            json.dumps({"current_before": current_before, 
                        "fetched_total": fetched_total}),
            encoding="utf-8",
        )

    def _load_checkpoint(self, 
                         key: str
                         ) -> tuple[list[dict], int, int] | None:
        ckpt_parquet, ckpt_json = self._ckpt_paths(key)
        if not ckpt_parquet.exists() or not ckpt_json.exists():
            return None
        state = json.loads(ckpt_json.read_text(encoding="utf-8"))
        records = pd.read_parquet(ckpt_parquet).to_dict("records")
        print(f"  [Checkpoint] Wznawiam od {datetime.fromtimestamp(state['current_before'], tz=timezone.utc).strftime('%Y-%m-%d')} "
              f"({len(records):,} rekordów już pobranych)")
        return records, state["current_before"], state["fetched_total"]

    def _delete_checkpoint(self, key: str) -> None:
        for p in self._ckpt_paths(key):
            p.unlink(missing_ok=True)

    def _paginate(
        self,
        endpoint: str,
        subreddit: str,
        start: str,
        end: str,
        extract_fn,
        min_score: int,
        label: str,
        checkpoint_key: str = ""
        ) -> list[dict]:
        """
        Metoda paginuje wyniki newest-first przez before = min_ts - 1.
        Filtruje score klientem. Zwraca liste surowych rekordow.
        """
        after_ts  = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
        before_ts = int(datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp())

        records: list[dict] = []
        fetched_total = 0
        current_before = before_ts
        page = 0

        if checkpoint_key:
            resumed = self._load_checkpoint(checkpoint_key)
            if resumed:
                records, current_before, fetched_total = resumed

        while True:
            batch_raw = self._fetch_page(endpoint, 
                                         subreddit, 
                                         after_ts, 
                                         current_before
                                         )
            if not batch_raw:
                break

            fetched_total += len(batch_raw)
            min_ts = min(item["created_utc"] for item in batch_raw)

            for item in batch_raw:
                if item.get("score", 0) >= min_score:
                    record = extract_fn(item)
                    if record is not None:
                        records.append(record)

            page += 1
            if page % 100 == 0:
                pct = 100 * (before_ts - min_ts) / max(before_ts - after_ts, 1)
                dt  = datetime.fromtimestamp(min_ts, tz=timezone.utc).strftime("%Y-%m")
                kept_pct = 100 * len(records) / max(fetched_total, 1)
                print(f"  [{label}] {len(records):,} zachowanych / {fetched_total:,} pobranych "
                      f"({kept_pct:.0f}% pass) | {pct:.0f}% zakresu | {dt}")

            if checkpoint_key and page % self.checkpoint_every == 0:
                self._save_checkpoint(checkpoint_key, 
                                      records, 
                                      current_before, 
                                      fetched_total
                                      )
                print(f"  [Checkpoint] Zapisano ({len(records):,} rekordów, before={current_before})")

            if min_ts <= after_ts:
                break
            current_before = min_ts - 1
            time.sleep(self.request_delay)

        return records

    def _fetch_page(
        self,
        endpoint: str,
        subreddit: str,
        after: int,
        before: int,
        limit: int = 100
        ) -> list[dict]:
        """Jedno zapytanie HTTP do Arctic Shift"""
        for attempt in range(3):
            try:
                resp = self._session.get(
                    f"{ARCTIC_SHIFT_BASE}/{endpoint}/search",
                    params={
                        "subreddit": subreddit,
                        "after": after,
                        "before": before,
                        "limit": limit
                        },
                    timeout=30
                    )
                if resp.status_code == 429:
                    wait = 30 * (attempt + 1)
                    print(f"  [Arctic Shift] rate limit, czekam {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if data.get("error"):
                    print(f"  [Arctic Shift] API error: {data['error']}")
                    return []
                return data.get("data") or []
            except requests.RequestException as e:
                print(f"  [Arctic Shift] request error (prob {attempt+1}/3): {e}")
                time.sleep(5 * (attempt + 1))
        return []


    @staticmethod
    def _extract_post(p: dict) -> dict:
        return {
            "id": p.get("id", ""),
            "subreddit": p.get("subreddit", ""),
            "created_utc": p.get("created_utc", 0),
            "title": p.get("title", ""),
            "selftext": p.get("selftext", ""),
            "score": p.get("score", 0),
            "upvote_ratio": p.get("upvote_ratio"),
            "num_comments": p.get("num_comments", 0)
            }

    @staticmethod
    def _extract_comment(c: dict, 
                         top_level_only: bool
                         ) -> dict | None:
        if top_level_only:
            parent_id = c.get("parent_id", "")
            # t3_ = post, t1_ = inny komentarz
            if not parent_id.startswith("t3_"):
                return None
        body = c.get("body", "")
        if body in ("[deleted]", "[removed]", ""):
            return None
        return {
            "id": c.get("id", ""),
            "subreddit": c.get("subreddit", ""),
            "created_utc": c.get("created_utc", 0),
            "post_id": c.get("link_id", ""),
            "body": body,
            "score": c.get("score", 0)
            }

