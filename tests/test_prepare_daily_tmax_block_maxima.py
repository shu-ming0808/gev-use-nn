from pathlib import Path

import pandas as pd

from src.prepare_daily_tmax_block_maxima import aggregate_file_to_monthly


def test_monthly_maximum_keeps_earliest_and_all_tied_dates(tmp_path: Path):
    path = tmp_path / "daily.csv"
    pd.DataFrame(
        {
            "LON": [121.0],
            "LAT": [24.0],
            "20000101": [30.0],
            "20000102": [35.0],
            "20000103": [35.0],
        }
    ).to_csv(path, index=False)

    _, monthly, coverage, events = aggregate_file_to_monthly(
        path,
        min_daily_coverage=3 / 31,
    )

    assert monthly.iloc[0, 0] == 35.0
    assert monthly.columns.tolist() == ["G121.00_24.00"]
    assert coverage.loc[0, "valid_days"] == 3
    assert events.loc[0, "max_date"] == "2000-01-02"
    assert events.loc[0, "all_tied_max_dates"] == "2000-01-02;2000-01-03"
    assert events.loc[0, "n_tied_max_dates"] == 2
