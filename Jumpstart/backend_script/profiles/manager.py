"""Profile management: template creation and profile construction.

Workflow this module supports (see project workflow notes):
    FIRST TIME ONLY:
        1. Download an empty participant info template
           (create_participant_template).
    EVERY USE:
        2. Fill in the template with real athlete data, then load it
           and build Participant profiles (load_participant_table,
           build_profiles).

This module intentionally has no knowledge of videos, tracking, or
pose estimation -- it only produces validated Participant objects
for later modules to consume.
"""

from pathlib import Path
from typing import List

import pandas as pd

from backend_script.profiles.models import Participant

TEMPLATE_COLUMNS = [
    "participant_id",
    "name",
    "height_cm",
    "leg_length_cm",
    "weight_kg",
]

_EXAMPLE_ROW = {
    "participant_id": "P001",
    "name": "Jane Doe",
    "height_cm": 178.0,
    "leg_length_cm": 92.0,
    "weight_kg": 68.0,
}


def create_participant_template(output_path: Path) -> Path:
    """Create an empty participant info template as an .xlsx file.

    Args:
        output_path: Destination path for the template, e.g.
            Path("participant_template.xlsx").

    Returns:
        The path the template was written to.
    """
    output_path = Path(output_path)
    template_df = pd.DataFrame([_EXAMPLE_ROW], columns=TEMPLATE_COLUMNS)
    template_df.to_excel(output_path, index=False)
    return output_path


def load_participant_table(input_path: Path) -> pd.DataFrame:
    """Load a filled-in participant template from disk.

    Accepts .xlsx or .csv files. Validates that all required columns
    are present before returning.

    Args:
        input_path: Path to the filled-in participant file.

    Returns:
        A DataFrame with one row per participant.

    Raises:
        ValueError: If required columns are missing.
        FileNotFoundError: If input_path does not exist.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Participant file not found: {input_path}")

    if input_path.suffix.lower() == ".csv":
        table = pd.read_csv(input_path)
    else:
        table = pd.read_excel(input_path)

    required = set(TEMPLATE_COLUMNS) - {"weight_kg", "leg_length_cm"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"Participant file is missing required columns: "
            f"{sorted(missing)}"
        )
    return table


def build_profiles(table: pd.DataFrame) -> List[Participant]:
    """Convert a validated participant table into Participant objects.

    Args:
        table: DataFrame as returned by load_participant_table.

    Returns:
        List of Participant instances, one per row.
    """
    profiles = []
    for _, row in table.iterrows():
        weight = row.get("weight_kg")
        leg_length = row.get("leg_length_cm")
        profiles.append(
            Participant(
                participant_id=str(row["participant_id"]),
                name=str(row["name"]),
                height_cm=float(row["height_cm"]),
                leg_length_cm=float(leg_length) if pd.notna(leg_length) else None,
                weight_kg=float(weight) if pd.notna(weight) else None,
            )
        )
    return profiles
