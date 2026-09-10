"""ASTRO V1 — Office Calendar Service.

Provides seamless calendar querying, Google Calendar REST integration,
and proactive pre-meeting reminder detection.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple


class CalendarService:
    def __init__(self, storage_path: Optional[str] = None):
        if storage_path:
            self.storage_path = storage_path
        else:
            astro_dir = os.path.expanduser(os.path.join("~", ".astro"))
            os.makedirs(astro_dir, exist_ok=True)
            self.storage_path = os.path.join(astro_dir, "office_calendar.json")

        self.google_api_key = os.environ.get("GOOGLE_CALENDAR_API_KEY", "")
        self.google_calendar_id = os.environ.get("GOOGLE_CALENDAR_ID", "")
        self.google_ical_url = os.environ.get("GOOGLE_CALENDAR_ICAL_URL", "")
        self._reminded_event_ids = set()

        self._ensure_storage_initialized()

    def _ensure_storage_initialized(self):
        """Seeds default office events if storage file is missing or empty."""
        if not os.path.exists(self.storage_path) or os.path.getsize(self.storage_path) == 0:
            now = datetime.now()
            # Seed 2 realistic events for today
            seed_events = [
                {
                    "id": "evt_sprint_review",
                    "title": "Haftalık Sprint Değerlendirmesi",
                    "start_time": (now + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M"),
                    "duration_minutes": 45,
                    "location": "Toplantı Odası A",
                    "organizer": "Baran",
                    "attendees": ["Baran", "Selin", "Ahmet"],
                    "description": "Yeni robotik ve arayüz geliştirmelerinin değerlendirilmesi."
                },
                {
                    "id": "evt_arch_sync",
                    "title": "Astro Sistem Mimarisi İncelemesi",
                    "start_time": (now + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M"),
                    "duration_minutes": 60,
                    "location": "Lobi / Ar-Ge Alanı",
                    "organizer": "Baran",
                    "attendees": ["Baran", "Yapay Zeka Ekibi"],
                    "description": "ROS2 ve LLM gerçek zamanlı gecikme optimizasyonları."
                }
            ]
            try:
                with open(self.storage_path, "w", encoding="utf-8") as f:
                    json.dump({"events": seed_events}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    def _load_local_events(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.storage_path):
            return []
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("events", [])
        except Exception:
            return []

    def _fetch_google_rest_events(self) -> List[Dict[str, Any]]:
        """Fetches upcoming events via Google Calendar v3 REST API (if key and calendar ID provided)."""
        if not self.google_api_key or not self.google_calendar_id:
            return []
        import urllib.request
        import urllib.parse
        now_iso = datetime.now(timezone.utc).isoformat()
        cal_id = urllib.parse.quote(self.google_calendar_id)
        url = (
            f"https://www.googleapis.com/calendar/v3/calendars/{cal_id}/events"
            f"?key={self.google_api_key}&timeMin={now_iso}&singleEvents=true&orderBy=startTime&maxResults=15"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AstroV1-OfficeBot"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                items = data.get("items", [])
                events = []
                for it in items:
                    st_val = it.get("start", {}).get("dateTime", it.get("start", {}).get("date", ""))
                    title = it.get("summary", "Toplantı")
                    loc = it.get("location", "Ofis")
                    attendees = [a.get("displayName", a.get("email", "")) for a in it.get("attendees", [])]
                    events.append({
                        "id": it.get("id"),
                        "title": title,
                        "start_time": st_val,
                        "duration_minutes": 30,
                        "location": loc,
                        "organizer": it.get("organizer", {}).get("displayName", "Ekip"),
                        "attendees": attendees
                    })
                return events
        except Exception:
            return []

    def _fetch_google_ical_events(self) -> List[Dict[str, Any]]:
        """Fetches upcoming events via Google Calendar Secret iCal Feed URL (Zero API Key needed!)."""
        if not self.google_ical_url:
            return []
        import urllib.request
        try:
            req = urllib.request.Request(self.google_ical_url, headers={"User-Agent": "AstroV1-OfficeBot"})
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                content = resp.read().decode("utf-8", errors="ignore")

            events = []
            cur_event = None
            for line in content.splitlines():
                line = line.strip()
                if line == "BEGIN:VEVENT":
                    cur_event = {}
                elif line == "END:VEVENT" and cur_event is not None:
                    if "title" in cur_event and "start_time" in cur_event:
                        events.append(cur_event)
                    cur_event = None
                elif cur_event is not None:
                    if line.startswith("SUMMARY:"):
                        cur_event["title"] = line[8:]
                    elif line.startswith("LOCATION:"):
                        cur_event["location"] = line[9:]
                    elif line.startswith("DTSTART"):
                        val = line.split(":")[-1].replace("Z", "")
                        try:
                            if len(val) == 8 and val.isdigit():
                                dt = datetime.strptime(val, "%Y%m%d")
                                cur_event["start_time"] = dt.strftime("%Y-%m-%d 09:00")
                            else:
                                dt = datetime.strptime(val[:15], "%Y%m%dT%H%M%S")
                                cur_event["start_time"] = dt.strftime("%Y-%m-%d %H:%M")
                            cur_event["duration_minutes"] = 45
                            cur_event["organizer"] = "Google Takvim"
                            cur_event["id"] = f"ical_{val[:15]}_{abs(hash(cur_event.get('title', '')))}"
                        except Exception:
                            pass
                    elif line.startswith("DTEND"):
                        val = line.split(":")[-1].replace("Z", "")
                        try:
                            if len(val) >= 15 and "start_time" in cur_event:
                                dt_end = datetime.strptime(val[:15], "%Y%m%dT%H%M%S")
                                dt_start = datetime.strptime(cur_event["start_time"], "%Y-%m-%d %H:%M")
                                dur = int((dt_end - dt_start).total_seconds() / 60.0)
                                if dur > 0:
                                    cur_event["duration_minutes"] = dur
                        except Exception:
                            pass
            return events
        except Exception:
            return []

    def get_upcoming_events(self, hours: float = 12.0) -> List[Dict[str, Any]]:
        """Returns sorted upcoming events merged across Google Calendar (REST or iCal) and local storage."""
        # 1. Collect from Google Calendar (if configured)
        google_events = self._fetch_google_ical_events() or self._fetch_google_rest_events()

        # 2. Collect from local JSON storage
        local_events = self._load_local_events()

        # Merge with deduplication (by event id or title+start_time)
        all_events = []
        seen_keys = set()

        for ev in (google_events + local_events):
            t_key = f"{ev.get('title')}_{ev.get('start_time')}"
            if t_key not in seen_keys:
                seen_keys.add(t_key)
                all_events.append(ev)

        now = datetime.now()
        cutoff = now + timedelta(hours=hours)

        upcoming = []
        for ev in all_events:
            try:
                st_str = ev.get("start_time", "")
                if "T" in st_str:
                    st = datetime.fromisoformat(st_str.replace("Z", "+00:00")).astimezone().replace(tzinfo=None)
                else:
                    st = datetime.strptime(st_str, "%Y-%m-%d %H:%M")

                if now - timedelta(minutes=15) <= st <= cutoff:
                    ev_copy = dict(ev)
                    ev_copy["dt_start"] = st
                    upcoming.append(ev_copy)
            except Exception:
                continue

        upcoming.sort(key=lambda x: x["dt_start"])
        return upcoming

    def get_today_summary(self) -> str:
        """Returns concise Turkish conversational summary of today's schedule."""
        events = self.get_upcoming_events(hours=14.0)
        if not events:
            return "Bugün için planlanmış herhangi bir toplantı veya etkinlik bulunmuyor."

        lines = [f"Bugün toplam {len(events)} etkinlik bulunuyor:"]
        for ev in events:
            time_part = ev["dt_start"].strftime("%H:%M")
            title = ev.get("title", "Toplantı")
            loc = ev.get("location", "")
            loc_str = f" ({loc})" if loc else ""
            lines.append(f"- Saat {time_part}: {title}{loc_str}")

        return "\n".join(lines)

    def get_events_summary(self, days: int = 7, query: str = "") -> str:
        """Returns structured Turkish summary of events for specified days (e.g. 7 days / this week)."""
        q_low = (query or "").lower().strip()
        if "bugün" in q_low or "bugun" in q_low:
            return self.get_today_summary()

        hours = float(max(1, days) * 24)
        events = self.get_upcoming_events(hours=hours)

        if q_low and q_low not in ("bu hafta", "hafta", "önümüzdeki hafta", "onumuzdeki hafta"):
            filtered = [
                e for e in events
                if q_low in e.get("title", "").lower() or q_low in e.get("location", "").lower()
            ]
            if filtered:
                events = filtered

        if not events:
            return f"Önümüzdeki {days} gün için planlanmış herhangi bir etkinlik veya toplantı bulunmuyor."

        tr_days = {
            "Monday": "Pazartesi", "Tuesday": "Salı", "Wednesday": "Çarşamba",
            "Thursday": "Perşembe", "Friday": "Cuma", "Saturday": "Cumartesi", "Sunday": "Pazar"
        }
        tr_months = {
            1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
            7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
        }

        grouped = {}
        for ev in events:
            dt = ev["dt_start"]
            d_key = dt.strftime("%Y-%m-%d")
            if d_key not in grouped:
                day_name = tr_days.get(dt.strftime("%A"), dt.strftime("%A"))
                m_name = tr_months.get(dt.month, "")
                label = f"{day_name} ({dt.day} {m_name})"
                grouped[d_key] = {"label": label, "items": []}
            grouped[d_key]["items"].append(ev)

        lines = [f"Önümüzdeki {days} gün için toplam {len(events)} etkinlik bulunuyor:"]
        for d_key, grp in grouped.items():
            lines.append(f"\n📅 {grp['label']}:")
            for ev in grp["items"]:
                t_str = ev["dt_start"].strftime("%H:%M")
                title = ev.get("title", "Toplantı")
                loc = ev.get("location", "")
                loc_str = f" ({loc})" if loc else ""
                lines.append(f"  • {t_str} - {title}{loc_str}")

        return "\n".join(lines)

    def is_employee_in_meeting(self, employee_name: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Checks if employee is currently in a meeting or one is ending soon."""
        now = datetime.now()
        events = self._load_local_events()
        low_name = employee_name.lower()

        for ev in events:
            try:
                st = datetime.strptime(ev["start_time"], "%Y-%m-%d %H:%M")
                dur = ev.get("duration_minutes", 30)
                end = st + timedelta(minutes=dur)

                attendees = [a.lower() for a in ev.get("attendees", [])]
                organizer = ev.get("organizer", "").lower()

                if low_name in organizer or any(low_name in a for a in attendees):
                    # Currently in meeting or ending within 10 minutes
                    if st <= now <= end:
                        return True, ev
                    # If meeting starts in <= 15 minutes:
                    if timedelta(0) <= (st - now) <= timedelta(minutes=15):
                        return True, ev
            except Exception:
                continue

        return False, None

    def check_meeting_reminders(self, lead_minutes: int = 10) -> List[Dict[str, Any]]:
        """Returns meetings starting within lead_minutes that haven't been reminded yet."""
        now = datetime.now()
        events = self.get_upcoming_events(hours=2.0)
        reminders_due = []

        for ev in events:
            ev_id = ev.get("id", ev.get("title", ""))
            if ev_id in self._reminded_event_ids:
                continue

            dt_start = ev.get("dt_start")
            if not dt_start:
                continue

            diff = (dt_start - now).total_seconds() / 60.0
            # Trigger reminder if within lead window, e.g. between 0 and lead_minutes + 1
            if 0.0 <= diff <= (lead_minutes + 1.0):
                self._reminded_event_ids.add(ev_id)
                reminders_due.append({
                    "event_id": ev_id,
                    "title": ev.get("title", "Toplantı"),
                    "minutes_left": int(round(max(1.0, diff))),
                    "location": ev.get("location", "Toplantı Odası"),
                    "attendees": ev.get("attendees", []),
                    "organizer": ev.get("organizer", "Ekip")
                })

        return reminders_due

    def add_event(
        self,
        title: str,
        start_time_str: str,
        duration_minutes: int = 30,
        location: str = "Ofis",
        organizer: str = "Baran",
        attendees: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Adds an event to local storage."""
        events = self._load_local_events()
        new_event = {
            "id": f"evt_{int(time.time())}",
            "title": title,
            "start_time": start_time_str,
            "duration_minutes": duration_minutes,
            "location": location,
            "organizer": organizer,
            "attendees": attendees or [organizer]
        }
        events.append(new_event)
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({"events": events}, f, ensure_ascii=False, indent=2)
            return {"status": "success", "event": new_event}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def delete_event(self, query: str) -> Dict[str, Any]:
        """Deletes an event matching the query title or keyword from local storage."""
        q_low = query.lower().strip()
        if not q_low:
            return {"status": "error", "message": "Silinecek etkinlik adı belirtilmedi."}

        events = self._load_local_events()
        to_keep = []
        deleted = []

        for ev in events:
            title = ev.get("title", "").lower()
            ev_id = str(ev.get("id", "")).lower()
            if q_low in title or q_low in ev_id or (len(q_low) > 3 and any(w in title for w in q_low.split())):
                deleted.append(ev)
            else:
                to_keep.append(ev)

        if not deleted:
            return {
                "status": "not_found",
                "message": f"'{query}' ile eşleşen bir etkinlik bulunamadı."
            }

        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({"events": to_keep}, f, ensure_ascii=False, indent=2)
            del_title = deleted[0].get("title", query)
            return {
                "status": "success",
                "deleted_title": del_title,
                "count": len(deleted),
                "message": f"'{del_title}' takvimden başarıyla kaldırıldı."
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def add_event_smart(
        self,
        title: str,
        date_str: str = "bugün",
        time_str: str = "10:00",
        duration_minutes: int = 45,
        location: str = "Ofis",
        organizer: str = "Baran",
        attendees: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Smart event creator that parses relative Turkish date words and saves event."""
        now = datetime.now()
        target_date = now

        d_low = (date_str or "").lower().strip()
        tr_weekdays = {
            "pazartesi": 0, "salı": 1, "sali": 1, "çarşamba": 2, "carsamba": 2,
            "perşembe": 3, "persembe": 3, "cuma": 4, "cumartesi": 5, "pazar": 6
        }

        if "yarın" in d_low or "yarin" in d_low:
            target_date = now + timedelta(days=1)
        elif "öbür gün" in d_low or "obur gun" in d_low:
            target_date = now + timedelta(days=2)
        elif any(w in d_low for w in tr_weekdays):
            for w_name, w_idx in tr_weekdays.items():
                if w_name in d_low:
                    cur_w = now.weekday()
                    days_ahead = (w_idx - cur_w) % 7
                    if days_ahead == 0 or "gelecek" in d_low or "önümüzdeki" in d_low or "onumuzdeki" in d_low:
                        days_ahead += 7
                    target_date = now + timedelta(days=days_ahead)
                    break
        elif len(d_low) == 10 and "-" in d_low:
            try:
                target_date = datetime.strptime(d_low, "%Y-%m-%d")
            except Exception:
                pass

        # Parse time string: e.g. "14:00", "15.30", "14"
        clean_time = (time_str or "10:00").replace(".", ":").strip()
        if ":" not in clean_time and clean_time.isdigit():
            clean_time = f"{int(clean_time):02d}:00"

        try:
            t_parts = [int(p) for p in clean_time.split(":")[:2]]
            full_dt = target_date.replace(hour=t_parts[0], minute=t_parts[1], second=0, microsecond=0)
        except Exception:
            full_dt = target_date.replace(hour=10, minute=0, second=0, microsecond=0)

        start_time_str = full_dt.strftime("%Y-%m-%d %H:%M")
        return self.add_event(
            title=title,
            start_time_str=start_time_str,
            duration_minutes=duration_minutes,
            location=location,
            organizer=organizer,
            attendees=attendees
        )
