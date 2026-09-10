"""ASTRO V1 — Office Slack Service.

Handles real-time employee notifications (visitor arrivals, meeting alerts)
and parses incoming office commands.
"""

import json
import os
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional


class SlackService:
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL", "")
        self.bot_token = os.environ.get("SLACK_BOT_TOKEN", "")
        self.sent_messages_history: List[Dict[str, Any]] = []

    def send_message(self, text: str, channel: str = "#ofis-giris") -> Dict[str, Any]:
        """Sends message to Slack via Webhook or Bot API. Fails gracefully in offline/test mode."""
        msg_record = {
            "timestamp": time.time(),
            "channel": channel,
            "text": text,
            "delivered": False
        }

        if not self.webhook_url and not self.bot_token:
            # Offline / Mock mode
            msg_record["mode"] = "local_mock"
            msg_record["delivered"] = True
            self.sent_messages_history.append(msg_record)
            return {"status": "success", "mode": "local_mock", "delivered": True, "message": text}

        try:
            if self.webhook_url:
                payload = json.dumps({"text": text, "channel": channel}).encode("utf-8")
                req = urllib.request.Request(
                    self.webhook_url,
                    data=payload,
                    headers={"Content-Type": "application/json", "User-Agent": "AstroV1-OfficeBot"}
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    if resp.status in (200, 204):
                        msg_record["delivered"] = True
            elif self.bot_token:
                payload = json.dumps({"channel": channel, "text": text}).encode("utf-8")
                req = urllib.request.Request(
                    "https://slack.com/api/chat.postMessage",
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.bot_token}",
                        "User-Agent": "AstroV1-OfficeBot"
                    }
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    if res_json.get("ok"):
                        msg_record["delivered"] = True

            self.sent_messages_history.append(msg_record)
            return {"status": "success", "delivered": msg_record["delivered"], "text": text}
        except Exception as e:
            msg_record["error"] = str(e)
            self.sent_messages_history.append(msg_record)
            return {"status": "error", "message": str(e)}

    def notify_visitor_arrival(
        self,
        employee_name: str,
        visitor_name: str,
        note: str = ""
    ) -> Dict[str, Any]:
        """Formats and sends a high-priority visitor check-in card to Slack."""
        now_str = time.strftime("%H:%M")
        text = (
            f"👋 *[ASTRO Ofis Lobisi]* Yeni Misafir Geldi!\n"
            f"• *Misafir:* {visitor_name}\n"
            f"• *Aranan Kişi:* {employee_name}\n"
            f"• *Giriş Saati:* {now_str}\n"
        )
        if note:
            text += f"• *Not:* {note}\n"
        text += "_ASTRO lobide misafire eşlik ediyor veya bekletiyor._"

        return self.send_message(text=text, channel="#ofis-giris")

    def parse_incoming_command(self, raw_command: str) -> Dict[str, Any]:
        """Parses slash commands received from Slack or ROS 2 /office/slack_command."""
        cmd = raw_command.strip()
        if not cmd:
            return {"status": "empty"}

        # Handle json payloads: {"command": "come_to_desk", "target": "baran_masa"}
        if cmd.startswith("{") and cmd.endswith("}"):
            try:
                data = json.loads(cmd)
                return {"status": "success", "action": data.get("command"), "target": data.get("target", "")}
            except Exception:
                pass

        # Handle slash text: /astro gel baran_masa or /astro durum
        parts = cmd.split()
        if parts[0] == "/astro" and len(parts) > 1:
            action = parts[1].lower()
            if action in ("gel", "come"):
                target = parts[2] if len(parts) > 2 else "baran_masa"
                return {"status": "success", "action": "navigate_to", "target": target}
            elif action in ("durum", "status"):
                return {"status": "success", "action": "report_status"}
            elif action in ("duyur", "announce"):
                announcement = " ".join(parts[2:])
                return {"status": "success", "action": "announce", "text": announcement}

        return {"status": "unknown_command", "raw": cmd}
