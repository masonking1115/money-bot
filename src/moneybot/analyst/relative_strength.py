"""Pure relative-strength math: a name's excess trailing return vs the benchmark.

No I/O and no pandas here — callers pass plain close-price sequences so this is
trivially unit-testable. Higher excess return = firmer vs the benchmark, which
the strategy's ranker uses as a tiebreaker ("don't fight the tape").
"""

from __future__ import annotations

from collections.abc import Sequence


def _trailing_return(closes: Sequence[float | None]) -> float | None:
    """Total return from first to last valid close, or None if not computable."""
    vals = [c for c in closes if c is not None]
    if len(vals) < 2 or vals[0] == 0:
        return None
    return (vals[-1] / vals[0]) - 1.0


def excess_return(
    closes: Sequence[float | None],
    benchmark_closes: Sequence[float | None],
) -> float:
    """Name's trailing return minus the benchmark's. 0.0 if either is uncomputable."""
    name = _trailing_return(closes)
    bench = _trailing_return(benchmark_closes)
    if name is None or bench is None:
        return 0.0
    return name - bench
