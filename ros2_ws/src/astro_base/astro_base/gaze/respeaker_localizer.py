#!/usr/bin/env python3
"""ReSpeaker XVF3000 DSP (VOICEACTIVITY + DOAANGLE) -> ASTRO Head Yaw Localizer.

Sector-based architecture (V2):
    ReSpeaker (VOICEACTIVITY + DOAANGLE)
                ↓
    DOA -> 3 sector mapping (LEFT / CENTER / RIGHT)
                ↓
    sector confirmation (N consecutive readings to switch)
                ↓
    sector -> coarse target yaw (-60° / 0° / +60°)
                ↓
    target_yaw_deg -> head.send_angle()

Why sectors instead of continuous angles:
    ReSpeaker XVF3000 DOA output is unreliable for precise angular mapping.
    Live testing showed DOA ~148° (calibrated RIGHT 83°) when the user was
    physically in FRONT (visual confirmed ~-10° to +27°). The previous
    piecewise linear calibration amplified DOA noise into 80°+ wrong-direction
    turns. Sectors limit the worst-case error to ~45° and let visual tracking
    refine the final heading.
"""

import math
from typing import Optional, Sequence


class ReSpeakerAudioLocalizer:
    """ReSpeaker XVF3000 DSP (VOICEACTIVITY + DOAANGLE) -> ASTRO Head Yaw Localizer."""

    # ── Sector boundaries ──────────────────────────────────────────
    # DOA ranges for the front operating hemisphere [20°, 155°].
    # Generous margins prevent jitter at boundaries.
    VALID_DOA_MIN = 20.0
    VALID_DOA_MAX = 155.0

    SECTOR_LEFT_MAX = 55.0     # DOA < 55° = LEFT
    SECTOR_RIGHT_MIN = 100.0   # DOA >= 100° = RIGHT
    #                            55°..100° = CENTER

    SECTOR_TARGETS = {"LEFT": 60.0, "CENTER": 0.0, "RIGHT": -60.0}
    # NOTE: LEFT/RIGHT targets are SWAPPED from the original calibration labels.
    # Live testing proved calibration labels were from user's perspective (facing robot):
    #   DOA ~32  ("LEFT" in calib)  = robot's physical RIGHT  → motor +60°
    #   DOA ~148 ("RIGHT" in calib) = robot's physical LEFT   → motor -60°
    # Evidence: visual tracking found user at -70.2° when DOA read 148;
    # +-60° puts both 45° and 70°-80° speakers squarely inside the camera FOV (69°).
    SECTOR_CONFIRM_COUNT = 2   # consecutive readings to switch sector while tracking

    # ── Legacy calibration (kept for backward-compat tests) ────────
    CALIBRATION_POINTS = (
        (25.0, -90.0),
        (32.0, -45.0),
        (78.0, 0.0),
        (142.0, 45.0),
        (149.0, 90.0),
    )

    def __init__(
        self,
        hid=None,
        hold_timeout_s: float = 5.0,
        deadband_deg: float = 5.0,
        outlier_threshold_deg: float = 30.0,
        filter_window_size: int = 5,
    ):
        self.hid = hid
        self.hold_timeout_s = float(hold_timeout_s)
        # Legacy params kept for CLI compat; unused in sector mode.
        self.deadband_deg = float(deadband_deg)
        self.outlier_threshold_deg = float(outlier_threshold_deg)
        self.filter_window_size = int(filter_window_size)

        # Sector state
        self._confirmed_sector: Optional[str] = None
        self._pending_sector: Optional[str] = None
        self._pending_count: int = 0

        # Target
        self.active_target_yaw: float = 0.0
        self._last_voice_activity_time: float = 0.0
        self._last_valid_target_time: float = 0.0
        self._tracking_active: bool = False

        # Diagnostics / logging
        self.last_raw_doa: Optional[float] = None

        # Polling cache
        self._last_poll_time: float = 0.0
        self._cached_vad: Optional[bool] = None
        self._cached_doa: Optional[float] = None
        self._vad_warning_emitted: bool = False
        self._last_vad_warning_time: float = 0.0

        # Legacy filter state (kept for backward-compat static methods/tests)
        self.filtered_doa: Optional[float] = None
        self._history: list[float] = []
        self._outlier_candidate: Optional[float] = None
        self._outlier_count: int = 0

    @staticmethod
    def circular_dist(a: float, b: float) -> float:
        """Angular distance between two angles in degrees [0, 180]."""
        diff = (float(a) - float(b) + 180.0) % 360.0 - 180.0
        return abs(diff)

    @staticmethod
    def circular_mean(angles: Sequence[float]) -> float:
        """Circular mean of angles in degrees, properly handling wrap-around."""
        if not angles:
            return 0.0
        s = sum(math.sin(math.radians(float(a))) for a in angles)
        c = sum(math.cos(math.radians(float(a))) for a in angles)
        if abs(s) < 1e-9 and abs(c) < 1e-9:
            return float(angles[-1]) % 360.0
        return math.degrees(math.atan2(s, c)) % 360.0

    @classmethod
    def is_valid_doa(cls, doa_deg: Optional[float]) -> bool:
        """Returns True if doa_deg is within the valid front operating workspace.

        Rear and ambiguous DOA regions (e.g. ~285-330°, 155-360°, 0-20°) are invalid.
        """
        if doa_deg is None:
            return False
        raw = float(doa_deg) % 360.0
        return cls.VALID_DOA_MIN <= raw <= cls.VALID_DOA_MAX

    @classmethod
    def doa_to_sector(cls, doa_deg: float) -> Optional[str]:
        """Maps a raw DOA reading to a coarse sector.

        Returns:
            "LEFT", "CENTER", or "RIGHT" for valid DOA in front hemisphere.
            None for rear / invalid DOA.
        """
        raw = float(doa_deg) % 360.0
        if not cls.is_valid_doa(raw):
            return None
        if raw < cls.SECTOR_LEFT_MAX:
            return "LEFT"
        elif raw >= cls.SECTOR_RIGHT_MIN:
            return "RIGHT"
        else:
            return "CENTER"

    @property
    def confirmed_sector(self) -> Optional[str]:
        """Currently confirmed sector, or None if not tracking."""
        return self._confirmed_sector if self._tracking_active else None

    @classmethod
    def calibrated_yaw(cls, doa_deg: float) -> Optional[float]:
        """Monotonic piecewise linear calibration from ReSpeaker DOA to ASTRO Head Yaw.

        Operating workspace:
            LEFT <= 90° (yaw >= -90.0°)
            FRONT = 0° (yaw = 0.0°)
            RIGHT <= 90° (yaw <= +90.0°)

        Physical calibration measurements on robot:
            DOA 32.0°  -> Yaw -45.0°
            DOA 78.0°  -> Yaw   0.0°
            DOA 142.0° -> Yaw +45.0°
            DOA 149.0° -> Yaw +90.0°

        Saturation within valid workspace [20.0°, 155.0°]:
            DOA 20.0°..25.0°   -> -90.0°
            DOA 25.0°..32.0°   -> -90.0°..-45.0° interpolation
            DOA 32.0°..78.0°   -> -45.0°..0.0° interpolation
            DOA 78.0°..142.0°  -> 0.0°..+45.0° interpolation
            DOA 142.0°..149.0° -> +45.0°..+90.0° interpolation
            DOA 149.0°..155.0° -> +90.0°

        Returns:
            Calibrated yaw in [-90.0°, +90.0°], or None if DOA is in the rear/invalid region.
        """
        raw = float(doa_deg) % 360.0
        if not cls.is_valid_doa(raw):
            return None

        if raw <= 25.0:
            return -90.0
        elif raw <= 32.0:
            t = (raw - 25.0) / (32.0 - 25.0)
            return -90.0 + t * 45.0
        elif raw <= 78.0:
            t = (raw - 32.0) / (78.0 - 32.0)
            return -45.0 + t * 45.0
        elif raw <= 142.0:
            t = (raw - 78.0) / (142.0 - 78.0)
            return 0.0 + t * 45.0
        elif raw <= 149.0:
            t = (raw - 142.0) / (149.0 - 142.0)
            return 45.0 + t * 45.0
        else:  # 149.0 < raw <= 155.0
            return 90.0

    def reject_outlier(self, raw_doa: float) -> bool:
        """Rejects single transient spikes in DOA angle.

        Returns True if raw_doa is rejected as an outlier, False if accepted.
        If 2 consecutive samples are at the new angle, it is accepted as a step change.
        """
        raw = float(raw_doa) % 360.0
        if self.filtered_doa is None:
            self.filtered_doa = raw
            self._history = [raw]
            self._outlier_candidate = None
            self._outlier_count = 0
            return False

        dist = self.circular_dist(raw, self.filtered_doa)
        if dist <= self.outlier_threshold_deg:
            self._outlier_candidate = None
            self._outlier_count = 0
            self._history.append(raw)
            if len(self._history) > self.filter_window_size:
                self._history.pop(0)
            self.filtered_doa = self.circular_mean(self._history)
            return False

        # dist > outlier_threshold_deg
        if (
            self._outlier_candidate is not None
            and self.circular_dist(raw, self._outlier_candidate) <= self.outlier_threshold_deg
        ):
            # 2nd consecutive sample at new angle -> accept step change
            self.filtered_doa = raw
            self._history = [raw]
            self._outlier_candidate = None
            self._outlier_count = 0
            return False

        self._outlier_candidate = raw
        self._outlier_count = 1
        return True

    def apply_deadband(self, candidate_yaw: float) -> float:
        """Applies deadband to prevent servo jitter on small yaw changes."""
        if abs(candidate_yaw - self.active_target_yaw) >= self.deadband_deg:
            self.active_target_yaw = candidate_yaw
        return self.active_target_yaw

    def target_yaw(self, candidate_yaw: float) -> float:
        """Alias for apply_deadband."""
        return self.apply_deadband(candidate_yaw)

    # ── Tracking state update ─────────────────────────────────────────

    def update(
        self,
        doa_raw: Optional[float],
        voice_activity: bool,
        timestamp: float,
    ) -> float:
        """Updates localizer state with new DOA and VAD readings.

        Sector-based logic:
        1. When idle (or first detection ever) → immediately accept sector (fast response)
        2. While actively tracking, switching to a different sector requires
           SECTOR_CONFIRM_COUNT consecutive readings (prevents phantom noise flip)
        3. Within a sector, target stays at sector target (no jitter)
        """
        is_valid = self.is_valid_doa(doa_raw)

        if voice_activity and is_valid:
            assert doa_raw is not None
            self.last_raw_doa = float(doa_raw)
            self._last_voice_activity_time = timestamp
            self._last_valid_target_time = timestamp
            was_tracking = self._tracking_active
            self._tracking_active = True

            sector = self.doa_to_sector(doa_raw)
            if sector is not None:
                if not was_tracking or self._confirmed_sector is None:
                    # Idle or first detection → accept immediately to react fast
                    self._confirmed_sector = sector
                    self._pending_sector = None
                    self._pending_count = 0
                    self.active_target_yaw = self.SECTOR_TARGETS[sector]
                elif sector == self._confirmed_sector:
                    # Same sector → refresh/reinforce, restore target, clear pending
                    self.active_target_yaw = self.SECTOR_TARGETS[sector]
                    self._pending_sector = None
                    self._pending_count = 0
                elif sector == self._pending_sector:
                    # Consecutive reading in different sector while actively tracking → count up
                    self._pending_count += 1
                    if self._pending_count >= self.SECTOR_CONFIRM_COUNT:
                        self._confirmed_sector = sector
                        self.active_target_yaw = self.SECTOR_TARGETS[sector]
                        self._pending_sector = None
                        self._pending_count = 0
                else:
                    # New pending sector
                    self._pending_sector = sector
                    self._pending_count = 1

        elif voice_activity and not is_valid:
            # VAD true but DOA is rear/invalid: hold current target, don't generate rear target
            if self._tracking_active:
                if timestamp - self._last_valid_target_time > self.hold_timeout_s:
                    self._tracking_active = False
                    self.active_target_yaw = 0.0
                    # Keep _confirmed_sector so phantom DOA after timeout needs confirmation
                    self._pending_sector = None
                    self._pending_count = 0
        else:
            # VOICEACTIVITY is False: hold current target for grace period
            if self._tracking_active:
                if timestamp - self._last_valid_target_time > self.hold_timeout_s:
                    self._tracking_active = False
                    self.active_target_yaw = 0.0
                    # Keep _confirmed_sector so phantom DOA after timeout needs confirmation
                    self._pending_sector = None
                    self._pending_count = 0

        return self.target_yaw_deg

    @property
    def target_yaw_deg(self) -> float:
        """Authoritative audio target yaw. Returns 0.0 when not tracking."""
        if not self._tracking_active:
            return 0.0
        return self.active_target_yaw

    @property
    def motor_yaw_deg(self) -> float:
        """Motor command yaw matching target_yaw_deg."""
        return self.target_yaw_deg

    def is_tracking(self, now: Optional[float] = None) -> bool:
        """Returns True if localizer is currently actively tracking speech."""
        if not self._tracking_active:
            return False
        if now is not None and (now - self._last_valid_target_time > self.hold_timeout_s):
            self._tracking_active = False
            self.active_target_yaw = 0.0
            self._pending_sector = None
            self._pending_count = 0
            return False
        return True

    def _clear_sector_state(self) -> None:
        """Clears sector confirmation state."""
        self._confirmed_sector = None
        self._pending_sector = None
        self._pending_count = 0

    def reset_filter(self) -> None:
        """Clears circular filter history and outlier state (legacy compat)."""
        self.filtered_doa = None
        self._history.clear()
        self._outlier_candidate = None
        self._outlier_count = 0

    def reset(self) -> None:
        """Full reset of localizer state."""
        self._tracking_active = False
        self.active_target_yaw = 0.0
        self._last_voice_activity_time = 0.0
        self._last_valid_target_time = 0.0
        self._clear_sector_state()
        self.last_raw_doa = None
        self.reset_filter()

    def on_vision_active(self) -> None:
        """Called when visual tracking is active; immediately drops audio tracking."""
        self.reset()

    def read_voice_activity(self) -> Optional[bool]:
        """Reads hardware VAD from ReSpeaker XVF3000.

        Uses public ReSpeakerHID methods:
        - voice_activity() if defined
        - speech_detected() from respeaker_usb.py
        Does NOT touch private _read_param.
        """
        if self.hid is None:
            return None
        if hasattr(self.hid, "voice_activity") and callable(self.hid.voice_activity):
            val = self.hid.voice_activity()
            return bool(val) if val is not None else None
        if hasattr(self.hid, "speech_detected") and callable(self.hid.speech_detected):
            val = self.hid.speech_detected()
            return bool(val) if val is not None else None
        return None

    def read_doa_angle(self) -> Optional[float]:
        """Reads hardware DOA angle from ReSpeaker XVF3000.

        Uses public ReSpeakerHID method:
        - doa_angle()
        Does NOT touch private _read_param.
        """
        if self.hid is None:
            return None
        if hasattr(self.hid, "doa_angle") and callable(self.hid.doa_angle):
            val = self.hid.doa_angle()
            return float(val) if val is not None and 0 <= val <= 359 else None
        return None

    def read_and_update(
        self,
        now: float,
        poll_interval_s: float = 0.05,
    ) -> float:
        """Polls ReSpeaker hardware registers and updates localizer.

        Authoritative VAD and DOA come strictly from hardware.
        If hardware VAD cannot be read, tracking is disabled and a warning is logged.
        No software VAD fallback is used for audio tracking.
        """
        if self.hid is not None:
            if now - self._last_poll_time >= poll_interval_s or self._last_poll_time == 0.0:
                self._cached_vad = self.read_voice_activity()
                self._cached_doa = self.read_doa_angle()
                self._last_poll_time = now

        if self._cached_vad is None:
            if not self._vad_warning_emitted or (now - self._last_vad_warning_time > 5.0):
                print("⚠️  ReSpeaker Hardware VAD okunamıyor (donanım yok veya yanıt vermiyor) — ses takibi devre dışı")
                self._vad_warning_emitted = True
                self._last_vad_warning_time = now
            self.reset()
            return 0.0

        vad = bool(self._cached_vad)
        doa = self._cached_doa

        return self.update(doa_raw=doa, voice_activity=vad, timestamp=now)
