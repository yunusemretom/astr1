"""ASTRO V1 — Office Concierge Manager.

Handles lobby/entrance presence gating (LiDAR 1.5 - 2.0m),
face/biometric identification, head nod greeting gesture triggering,
and personalized guest welcome dialogue.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from .calendar_service import CalendarService
from .slack_service import SlackService


class OfficeConciergeManager:
    def __init__(
        self,
        calendar_service: Optional[CalendarService] = None,
        slack_service: Optional[SlackService] = None,
        cooldown_seconds: float = 40.0
    ):
        self.calendar = calendar_service or CalendarService()
        self.slack = slack_service or SlackService()
        self.cooldown_seconds = cooldown_seconds

        self._last_greeting_time = 0.0
        self._last_greeted_person = ""
        self._waiting_visitor_response = False
        self._detected_visitor_host = ""

    def evaluate_entrance_presence(
        self,
        lidar_ranges: List[float],
        recognized_identity: Optional[Dict[str, Any]] = None,
        is_speaking: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Evaluates whether someone just entered the door (1.5 - 2.0m) and returns greeting action."""
        if is_speaking:
            return None

        now = time.monotonic()
        if (now - self._last_greeting_time) < self.cooldown_seconds:
            return None

        # Check forward door sector (middle forward angles of LiDAR)
        # Typically forward angles are around 0 deg (or indices near 0 and -1)
        if not lidar_ranges:
            return None

        n = len(lidar_ranges)
        # Inspect forward 60-degree sector
        door_samples = lidar_ranges[: max(1, n // 12)] + lidar_ranges[- max(1, n // 12) :]
        # Target distance: 1.5 - 2.0 meters (with small margin 1.4 - 2.1m)
        presence_hits = [r for r in door_samples if 1.40 <= r <= 2.10]

        # Require sustained cluster of at least 3 points
        if len(presence_hits) < 3:
            return None

        # Distance match confirmed! Now check identity
        identity = recognized_identity or {}
        is_known = identity.get("is_known", False)
        name = identity.get("name", "").strip()
        formal_title = identity.get("formal_title", name or "Misafirimiz")

        self._last_greeting_time = now
        self._last_greeted_person = name or "Misafir"

        if is_known and name:
            # --- Scenario B: Known Partner or Employee ---
            # Check host (Baran) meeting status
            in_meeting, meeting_info = self.calendar.is_employee_in_meeting("Baran")
            if in_meeting:
                meeting_title = meeting_info.get("title", "toplantı")
                speech_text = (
                    f"Tekrar hoş geldiniz {formal_title}! "
                    f"Baran Bey'in birazdan toplantısı bitiyor, haber vereyim mi?"
                )
            else:
                speech_text = (
                    f"Tekrar hoş geldiniz {formal_title}! "
                    f"Baran Bey şu an masasında, kendisine geldiğinizi haber vereyim mi?"
                )

            # Send background notice to Slack
            self.slack.send_message(
                text=f"👋 *[Lobi Karşılama]* {formal_title} ({name}) ofise giriş yaptı.",
                channel="#ofis-giris"
            )

            return {
                "type": "known_person_welcome",
                "name": name,
                "formal_title": formal_title,
                "gesture": "nod",
                "speech_text": speech_text
            }
        else:
            # --- Scenario A: Unknown Visitor ---
            self._waiting_visitor_response = True
            speech_text = "Hoş geldiniz! Kimi aramıştınız?"

            return {
                "type": "unknown_visitor_welcome",
                "gesture": "nod",
                "speech_text": speech_text
            }

    def process_visitor_answer(self, visitor_transcript: str) -> Optional[Dict[str, Any]]:
        """If robot was waiting for unknown visitor's answer, process who they came to see."""
        if not self._waiting_visitor_response:
            return None

        self._waiting_visitor_response = False
        text_low = visitor_transcript.lower()

        host_name = "Baran"  # Default office host
        for possible_host in ["baran", "ahmet", "mehmet", "selin", "zeynep", "müdür"]:
            if possible_host in text_low:
                host_name = possible_host.capitalize()
                break

        # Notify via Slack
        slack_res = self.slack.notify_visitor_arrival(
            employee_name=host_name,
            visitor_name="Ziyaretçi",
            note=visitor_transcript
        )

        return {
            "status": "notified",
            "host": host_name,
            "slack_delivered": slack_res.get("delivered", False),
            "response_text": f"{host_name} Bey'e geldiğinizi hemen Slack üzerinden haber verdim. Lütfen lobide biraz dinlenin, birazdan yanınıza gelecektir."
        }
