"""Result export for /export/.

NEEDS REVIEW -- see the boundary marker in workflow_demo.py.

Collects every computed JumpParameters record into one flat results
table and writes it to disk. Deliberately simple: no figures, no
per-condition statistics (Bland-Altman, ICC, CV%, RMSE) -- those
belong with the validation-analysis work already tracked separately
in Jumpstart_data_entry_v2.xlsx, not in this script.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from jumpstart.calc.parameters import JumpParameters


def parameters_to_dataframe(
    all_parameters: List[JumpParameters],
) -> pd.DataFrame:
    """Collect JumpParameters records into a flat results table."""
    return pd.DataFrame([vars(p) for p in all_parameters])


def _excel_safe(table: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of table with timezone-aware datetime columns
    made naive, so it can be written with .to_excel().

    recorded_at (see jumpstart.io.video_metadata.get_recording_datetime)
    is timezone-aware (UTC) by construction, but Excel's datetime type
    has no timezone concept -- pandas raises ValueError otherwise.
    The CSV path doesn't need this: to_csv() just writes the ISO
    string, tzinfo and all.
    """
    table = table.copy()
    for column in table.columns:
        if pd.api.types.is_datetime64tz_dtype(table[column]):
            table[column] = table[column].dt.tz_localize(None)
    return table


def export_results(
    all_parameters: List[JumpParameters], output_path: Path
) -> Path:
    """Export all computed jump parameters to a spreadsheet.

    Args:
        all_parameters: One JumpParameters per detected jump, across
            every athlete and video in the session.
        output_path: Where to write the results (.xlsx or .csv).

    Returns:
        The path results were written to.
    """
    output_path = Path(output_path)
    table = parameters_to_dataframe(all_parameters)
    if output_path.suffix.lower() == ".csv":
        table.to_csv(output_path, index=False)
    else:
        _excel_safe(table).to_excel(output_path, index=False)
    return output_path
