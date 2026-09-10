"""Epistemic Spatial Memory for ASTRO Social Gaze.

Maintains:
  1. Persistent Person Spatial Presence (remembers where humans are in the room for 15-20s)
  2. Negative Acoustic Evidence (marks empty sectors where sound orientation found zero faces as Reverb Zones)
  3. Epistemic Cross-Modal Grounding (associates speech with known humans, not wall reflections)
"""

from dataclasses import dataclass, field
import math
import time
from typing import Dict, List, Optional

from astro_base.gaze.angle_math import circular_distance_deg, wrap_deg


@dataclass
class SpatialPersonRecord:
    """Represents a spatially grounded human in the robot's physical environment."""
    person_id: str
    last_seen_bearing_deg: float
    last_seen_time: float
    confidence: float
    distance_m: float = 1.5
    person_name: Optional[str] = None
    is_confirmed: bool = True
    observations_count: int = 1


@dataclass
class AcousticReverbSector:
    """Represents an angle sector proven to be an empty acoustic reflection / false alarm."""
    center_deg: float
    tolerance_deg: float
    expiry_time: float
    registered_time: float
    reason: str = "NO_FACE_IN_ACQUIRE"
    hit_count: int = 1


class EpistemicSpatialMemory:
    """Maintains short-to-medium term situational awareness of people and acoustic false alarms."""

    def __init__(
        self,
        person_memory_ttl_s: float = 20.0,
        reverb_suppression_ttl_s: float = 15.0,
        reverb_sector_width_deg: float = 25.0,
    ):
        self.person_memory_ttl_s = person_memory_ttl_s
        self.reverb_suppression_ttl_s = reverb_suppression_ttl_s
        self.reverb_sector_width_deg = reverb_sector_width_deg

        self._people_memory: Dict[str, SpatialPersonRecord] = {}
        self._reverb_sectors: List[AcousticReverbSector] = []

    def clear(self) -> None:
        self._people_memory.clear()
        self._reverb_sectors.clear()

    # --- 1. Person Spatial Presence Memory ---

    def register_person_observation(
        self,
        person_id: str,
        bearing_deg: float,
        confidence: float,
        timestamp: float,
        distance_m: float = 1.5,
        person_name: Optional[str] = None,
    ) -> None:
        """Records or updates a confirmed human location in spatial memory."""
        bearing = wrap_deg(bearing_deg)
        if person_id in self._people_memory:
            rec = self._people_memory[person_id]
            rec.last_seen_bearing_deg = bearing
            rec.last_seen_time = timestamp
            rec.confidence = max(rec.confidence, confidence)
            rec.distance_m = distance_m
            rec.observations_count += 1
            if person_name:
                rec.person_name = person_name
        else:
            self._people_memory[person_id] = SpatialPersonRecord(
                person_id=person_id,
                last_seen_bearing_deg=bearing,
                last_seen_time=timestamp,
                confidence=confidence,
                distance_m=distance_m,
                person_name=person_name,
                is_confirmed=True,
                observations_count=1,
            )

        # Seeing a real person at this angle removes any conflicting reverb suppression
        self.clear_reverb_near(bearing, tolerance_deg=15.0)

    def get_known_people(self, timestamp: float) -> List[SpatialPersonRecord]:
        """Returns all currently active (non-expired) people in spatial memory."""
        active = []
        for pid, rec in list(self._people_memory.items()):
            age = timestamp - rec.last_seen_time
            if age <= self.person_memory_ttl_s:
                active.append(rec)
        active.sort(key=lambda r: (r.confidence, - (timestamp - r.last_seen_time)), reverse=True)
        return active

    def get_most_likely_person_location(self, timestamp: float, max_age_s: Optional[float] = None) -> Optional[float]:
        """Returns the best available bearing (in degrees) of a known human in the room."""
        max_age = max_age_s if max_age_s is not None else self.person_memory_ttl_s
        people = [p for p in self.get_known_people(timestamp) if (timestamp - p.last_seen_time) <= max_age]
        if people:
            return people[0].last_seen_bearing_deg
        return None

    def get_person_record(self, person_id: str, timestamp: float) -> Optional[SpatialPersonRecord]:
        if person_id in self._people_memory:
            rec = self._people_memory[person_id]
            if (timestamp - rec.last_seen_time) <= self.person_memory_ttl_s:
                return rec
        return None

    # --- 2. Negative Acoustic Evidence & Reverb Suppression Map ---

    def register_negative_acoustic_evidence(
        self,
        bearing_deg: float,
        timestamp: float,
        reason: str = "NO_FACE_IN_ACQUIRE",
        duration_s: Optional[float] = None,
    ) -> None:
        """Registers an angle sector as proven empty space / acoustic reflection."""
        ttl = duration_s if duration_s is not None else self.reverb_suppression_ttl_s
        bearing = wrap_deg(bearing_deg)
        expiry = timestamp + ttl

        # Check if already registered
        for sec in self._reverb_sectors:
            if circular_distance_deg(bearing, sec.center_deg) <= sec.tolerance_deg:
                sec.expiry_time = max(sec.expiry_time, expiry)
                sec.hit_count += 1
                sec.registered_time = timestamp
                return

        self._reverb_sectors.append(
            AcousticReverbSector(
                center_deg=bearing,
                tolerance_deg=self.reverb_sector_width_deg / 2.0,
                expiry_time=expiry,
                registered_time=timestamp,
                reason=reason,
                hit_count=1,
            )
        )

    def is_acoustic_reverb_zone(self, bearing_deg: float, timestamp: float) -> bool:
        """Returns True if the bearing is currently marked as a verified empty acoustic reflection."""
        bearing = wrap_deg(bearing_deg)
        # Prune expired sectors
        self._reverb_sectors = [s for s in self._reverb_sectors if s.expiry_time > timestamp]

        for sec in self._reverb_sectors:
            if circular_distance_deg(bearing, sec.center_deg) <= sec.tolerance_deg:
                return True
        return False

    def clear_reverb_near(self, bearing_deg: float, tolerance_deg: float = 15.0) -> None:
        """Removes reverb suppression around an angle where a real visual target is now confirmed."""
        bearing = wrap_deg(bearing_deg)
        self._reverb_sectors = [
            s for s in self._reverb_sectors
            if circular_distance_deg(bearing, s.center_deg) > (s.tolerance_deg + tolerance_deg)
        ]
