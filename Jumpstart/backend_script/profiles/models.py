"""Data models for athlete profiles used across the Jumpstart pipeline.

This module defines the minimal, immutable representation of a
participant that every downstream module (/who/, /identify/, /calc/,
etc.) relies on. Keeping this model small and stable avoids coupling
unrelated modules to each other's implementation details.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Participant:
    """A single athlete taking part in a recording session.

    Attributes:
        participant_id: Stable unique identifier (e.g. "P001").
        name: Full name of the athlete.
        height_cm: Standing height in centimeters.
        leg_length_cm: Leg length in centimeters. Currently not used
            in any calculation -- kept for possible future
            biomechanical scaling, optional.
        weight_kg: Body mass in kilograms. Not required for jump
            height itself, but needed for parameters such as peak
            power (Sayers equation).
    """

    participant_id: str
    name: str
    height_cm: float
    leg_length_cm: Optional[float] = None
    weight_kg: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate that required numeric fields are physically sane."""
        if self.height_cm <= 0:
            raise ValueError(
                f"height_cm must be positive, got {self.height_cm!r} "
                f"for participant {self.participant_id!r}"
            )
        if self.leg_length_cm is not None and self.leg_length_cm <= 0:
            raise ValueError(
                f"leg_length_cm must be positive if provided, got "
                f"{self.leg_length_cm!r} for participant "
                f"{self.participant_id!r}"
            )
        if self.weight_kg is not None and self.weight_kg <= 0:
            raise ValueError(
                f"weight_kg must be positive if provided, got "
                f"{self.weight_kg!r} for participant "
                f"{self.participant_id!r}"
            )
