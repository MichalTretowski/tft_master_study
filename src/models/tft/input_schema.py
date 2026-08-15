"""
Podzial kolumn datasetu na wejscia modelu.

TFT rozroznia trzy typy zmiennych:
  STATIC - stałe dla całej sekwencji (coin_id)
  KNOWN - znane z wyprzedzeniem (czas, kalendarz eventow makro)
  OBSERVED - nieznane w przyszlosci (ceny, makro, sentyment, on-chain)

"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.data.feature_groups import FEATURE_GROUPS, group_columns

STATIC_GROUPS = {"static"}
KNOWN_GROUPS = {"time", "calendar"}
DROP_GROUPS = {"target"}

ABLATABLE_GROUPS = [
    g for g in FEATURE_GROUPS
    if g not in STATIC_GROUPS | KNOWN_GROUPS | DROP_GROUPS
    ]

@dataclass
class TFTInputSchema:
    static_categoricals: list[str] = field(default_factory=list)
    static_reals: list[str] = field(default_factory=list)
    known_categoricals: list[str] = field(default_factory=list)
    known_reals: list[str] = field(default_factory=list)
    observed_reals: list[str] = field(default_factory=list)

    target: str = "target"

    # Interwal danych — 6 barow na dobe przy 4h, 1 przy 1d
    bars_per_day: int = 6

    # Dlugosc kontekstu podawana w DNIACH, nie w barach.
    encoder_days: int = 30
    decoder_length: int = 1

    source_groups: list[str] = field(default_factory=list)


    @classmethod
    def from_columns(
        cls,
        columns,
        bars_per_day: int = 6,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        encoder_days: int = 30
        ) -> "TFTInputSchema":
        """
        Metoda buduje schemat z kolumn datasetu.
        """
        grouped = group_columns(columns)

        unknown = set(include or []) | set(exclude or [])
        unknown -= set(FEATURE_GROUPS)
        if unknown:
            raise ValueError(f"Nieznane grupy: {sorted(unknown)}")

        chosen = [g for g in ABLATABLE_GROUPS if g in grouped]
        if include is not None:
            chosen = [g for g in chosen if g in set(include)]
        if exclude is not None:
            chosen = [g for g in chosen if g not in set(exclude)]

        static = [c for g in STATIC_GROUPS for c in grouped.get(g, [])]
        known = [c for g in KNOWN_GROUPS for c in grouped.get(g, [])]
        observed = [c for g in chosen for c in grouped.get(g, [])]

        target_cols = grouped.get("target", [])
        if not target_cols:
            raise ValueError("Brak kolumny 'target' w datasecie")

        return cls(
            static_categoricals=sorted(static),
            known_reals=sorted(known),
            observed_reals=sorted(observed),
            target=target_cols[0],
            bars_per_day=bars_per_day,
            encoder_days=encoder_days,
            source_groups=chosen
            )


    @property
    def encoder_length(self) -> int:
        """Dlugosc kontekstu w barach"""
        return self.encoder_days * self.bars_per_day

    @property
    def n_features(self) -> int:
        return len(self.get_all_input_cols())

    def get_all_input_cols(self) -> list[str]:
        return (
            self.static_categoricals
            + self.static_reals
            + self.known_categoricals
            + self.known_reals
            + self.observed_reals
            )

    def validate(self, columns) -> None:
        """
        Metoda sprawdza czy schemat i dataset sa zgodne.
        """
        have = set(columns)
        want = self.get_all_input_cols()

        missing = [c for c in want if c not in have]
        if missing:
            raise ValueError(
                "Schemat wymaga kolumn nieobecnych w danych: "
                + ", ".join(missing)
            )
        if self.target not in have:
            raise ValueError(f"Brak kolumny targetu '{self.target}'")

        dupes = [c for c in want if want.count(c) > 1]
        if dupes:
            raise ValueError(f"Kolumny zduplikowane: {sorted(set(dupes))}")


    def summary(self) -> None:
        days = self.encoder_days
        print("=== TFT Input Schema ===")
        print(f"  Static:        {len(self.static_categoricals):>4}")
        print(f"  Known reals:   {len(self.known_reals):>4}")
        print(f"  Observed:      {len(self.observed_reals):>4}")
        print(f"  Razem cech:    {self.n_features:>4}")
        print(f"  Encoder:       {self.encoder_length} barow = {days} dni")
        print(f"  Decoder:       {self.decoder_length} bar")
        print(f"  Target:        {self.target}")
        print(f"  Grupy:         {', '.join(self.source_groups)}")


def _main() -> None:
    import argparse
    from pathlib import Path

    import pandas as pd

    p = argparse.ArgumentParser()
    p.add_argument("--tf", 
                   default="4h", 
                   choices=["4h", "1d"]
                   )
    p.add_argument("--coin", 
                   default="BTC"
                   )
    p.add_argument("--include", 
                   default=None,
                   help="grupy rozdzielone przecinkiem"
                   )
    p.add_argument("--exclude", 
                   default=None
                   )
    args = p.parse_args()

    bpd = 6 if args.tf == "4h" else 1
    path = (Path("data/processed/datasets") / args.tf
            / f"{args.coin}_train.parquet")
    cols = pd.read_parquet(path).columns

    schema = TFTInputSchema.from_columns(
        cols,
        bars_per_day=bpd,
        include=args.include.split(",") if args.include else None,
        exclude=args.exclude.split(",") if args.exclude else None,
    )
    schema.summary()
    schema.validate(cols)
    print("")
    print("  validate(): OK")


if __name__ == "__main__":
    _main()
