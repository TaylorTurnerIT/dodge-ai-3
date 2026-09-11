"""A metric series that always covers a whole training run at fixed cost.

A fixed-length deque answers "what is happening now". Tracking improvement
needs the opposite: the x-axis must stay pinned to the start of the run so a
rising curve has something to rise away from.

RunSeries keeps at most CAPACITY points spanning every sample ever appended.
When it fills, adjacent points are merged pairwise and the sampling interval
doubles, so old history loses resolution instead of falling off the left edge.
Memory is flat whether a run lasts 500 updates or 500,000.

Each point carries both the raw value and the smoothed trend at that moment.
The trend is an exponential moving average maintained over raw appends, before
any merging, so compaction cannot distort it.
"""

CAPACITY = 240
TREND_SMOOTHING = 0.12
BASELINE_SAMPLES = 5


class RunSeries:
    def __init__(self, capacity=CAPACITY, smoothing=TREND_SMOOTHING):
        if capacity < 2 or capacity % 2:
            raise ValueError("capacity must be an even number of at least 2")
        self.capacity = capacity
        self.smoothing = smoothing
        self.values = []
        self.trend = []
        self.interval = 1  # raw samples represented by each stored point
        self.count = 0  # raw samples appended
        self._bucket = []  # raw samples not yet flushed to a point
        self._trend_bucket = []
        self._ema = None
        self._baseline_samples = []

    # -- writing ----------------------------------------------------------
    def append(self, value):
        value = float(value)
        self.count += 1

        self._ema = (
            value
            if self._ema is None
            else (self._ema + self.smoothing * (value - self._ema))
        )
        if len(self._baseline_samples) < BASELINE_SAMPLES:
            self._baseline_samples.append(value)

        self._bucket.append(value)
        self._trend_bucket.append(self._ema)
        if len(self._bucket) >= self.interval:
            self._flush()

    def _flush(self):
        self.values.append(sum(self._bucket) / len(self._bucket))
        self.trend.append(sum(self._trend_bucket) / len(self._trend_bucket))
        self._bucket.clear()
        self._trend_bucket.clear()
        if len(self.values) >= self.capacity:
            self._compact()

    def _compact(self):
        """Halve the resolution: merge adjacent pairs, double the interval."""
        self.values = [(a + b) / 2 for a, b in zip(self.values[::2], self.values[1::2])]
        self.trend = [(a + b) / 2 for a, b in zip(self.trend[::2], self.trend[1::2])]
        self.interval *= 2

    def extend(self, values):
        for value in values:
            self.append(value)

    # -- reading ----------------------------------------------------------
    def __len__(self):
        return len(self.values)

    def __bool__(self):
        return bool(self.values)

    def __iter__(self):
        return iter(self.values)

    @property
    def latest(self):
        """The most recent raw sample, not the most recent bucket average."""
        if self._bucket:
            return self._bucket[-1]
        return self.values[-1] if self.values else 0.0

    @property
    def baseline(self):
        """Mean of the run's opening samples — what improvement is measured from."""
        if not self._baseline_samples:
            return 0.0
        return sum(self._baseline_samples) / len(self._baseline_samples)

    @property
    def smoothed(self):
        """Current trend value, which is what 'improved to' should be read from."""
        if self._trend_bucket:
            return self._trend_bucket[-1]
        return self.trend[-1] if self.trend else 0.0

    @property
    def delta(self):
        return self.smoothed - self.baseline

    def bounds(self):
        """Low and high across the whole run, padded, never zero-width.

        Both the raw values and the trend are included so neither can be drawn
        outside the plot area.
        """
        if not self.values:
            return 0.0, 1.0
        low = min(min(self.values), min(self.trend))
        high = max(max(self.values), max(self.trend))
        margin = max((high - low) * 0.22, 0.5)
        return low - margin, high + margin

    def as_list(self):
        return list(self.values)

    def copy(self):
        """A detached copy, so a reader can work without holding the writer's lock.

        Every mutable field is rebuilt; the scalars carry over as they are. The
        cost is bounded by CAPACITY, which is the whole point of this class.
        """
        clone = RunSeries.__new__(RunSeries)
        clone.__dict__.update(self.__dict__)
        for name in (
            "values",
            "trend",
            "_bucket",
            "_trend_bucket",
            "_baseline_samples",
        ):
            setattr(clone, name, list(getattr(self, name)))
        return clone
