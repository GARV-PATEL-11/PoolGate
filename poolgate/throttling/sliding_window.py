"""SlidingWindowBucket — re-export from poolgate.utils."""

from poolgate.utils import SlidingWindowCounter

SlidingWindowBucket = SlidingWindowCounter

__all__ = ["SlidingWindowBucket", "SlidingWindowCounter"]
