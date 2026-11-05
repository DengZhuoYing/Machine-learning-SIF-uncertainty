# Machine-learning-SIF-uncertainty
Contains code and models for testing the reliability of machine learning-derived SIF products.

## Phenology

`code/calculate_phenology.py` fits annual curves with Savitzky-Golay, splines
and HANTS. Run the file in PyCharm to view its example.

POS is the day of the highest peak supported by the original observations.
Each candidate must be within 15 calendar days of a valid observation.
This excludes peaks that HANTS extrapolates into long winter gaps. A real
late-season peak remains eligible when observations support it.

Pass the original series to `extract_phenology` when processing a smoothed
curve. Keep missing and invalid observations as `np.nan` in this series.
For example, after computing `fitted` from the original `observations`:

```python
from calculate_phenology import extract_phenology

sos, pos, eos = extract_phenology(
    fitted,
    thresholds,
    dates=dates,
    observed=observations,
    max_gap_days=15,
)
```

The `observed` parameter is optional for extraction directly from observations.
`max_gap_days` uses actual days rather than sample counts, including four-day
and eight-day data. The default is set by `POS_SUPPORT_DAYS`.
Flat curves and years without a supported peak return `np.nan` for POS.
SOS and EOS retain their existing threshold calculations. POS is independent
of the SOS/EOS threshold, so the 10%, 20% and 30% outputs share the same POS.

Regenerate annual phenology rasters with the corrected peak extraction before
running `code/compare_phenology.py`, which reads the saved raster values.

Run `tests/test_phenology.py` in PyCharm to check peak selection, observation
support, threshold independence and calendar dates.

