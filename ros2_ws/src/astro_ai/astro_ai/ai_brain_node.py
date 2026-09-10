#!/usr/bin/env python3
"""ASTRO V1 — Real-Time Conversational AI Brain Node (Modular Architecture).

Coordinates:
  - State Machine (FSM): IDLE, LISTENING, THINKING, SPEAKING, INTERRUPTED
  - 3-Tier Memory Architecture: Episodic Buffer, Session Memory, Persistent Profile
  - Adaptive Conversation Session Manager & Latency Tracker (p50/p95)
  - Persona Engine & Deterministic Perception Context Injection
  - Tool Execution & Real-Time Streaming LLM Speech Synthesis
"""

import base64
import logging

_LOG = logging.getLogger(__name__)

import json
import os
import sys
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:
    def find_dotenv(*args, **kwargs): return ""
    def load_dotenv(*args, **kwargs): pass

try:
    from astro_ai.state_machine import StateMachine, RobotState
    from astro_ai.memory_manager import MemoryManager
    from astro_ai.persona_engine import (
        PersonaEngine, ROBOT_TOOLS, PERSONA_PROMPTS,
        clean_tts_text, extract_spoken_turkish_sentence,
        response_length_gate, is_self_identity_query
    )
    from astro_ai.conversation_session import ConversationSession, normalize_turkish_speech_input
    from astro_ai.cloud_manager import CloudManager
    from astro_ai.officials_database import find_official_by_name_or_alias, get_official_greeting, OFFICIALS_DATABASE
    from astro_ai.circuit_breaker import (
        GlobalProviderCircuitBreaker,
        ProviderState,
        RequestErrorClass,
        get_global_circuit_breaker,
    )
    from astro_ai.provider_registry import (
        ProviderRegistry,
        GROQ_PRODUCTION_MODELS,
        GEMINI_PRODUCTION_MODELS,
        OPENAI_PRODUCTION_MODELS,
    )
    from astro_ai.multimodal_perception import (
        MultimodalPerceptionState,
        SocialContextEngine,
        SocialContextState,
    )
    from astro_audio.local_audio_resources import get_local_audio_resources
except ImportError:
    from state_machine import StateMachine, RobotState
    from memory_manager import MemoryManager
    from persona_engine import (
        PersonaEngine, ROBOT_TOOLS, PERSONA_PROMPTS,
        clean_tts_text, extract_spoken_turkish_sentence,
        response_length_gate, is_self_identity_query
    )
    from conversation_session import ConversationSession, normalize_turkish_speech_input
    from cloud_manager import CloudManager
    from officials_database import find_official_by_name_or_alias, get_official_greeting, OFFICIALS_DATABASE
    try:
        from circuit_breaker import (
            GlobalProviderCircuitBreaker,
            ProviderState,
            RequestErrorClass,
            get_global_circuit_breaker,
        )
        from provider_registry import (
            ProviderRegistry,
            GROQ_PRODUCTION_MODELS,
            GEMINI_PRODUCTION_MODELS,
            OPENAI_PRODUCTION_MODELS,
        )
        from multimodal_perception import (
            MultimodalPerceptionState,
            SocialContextEngine,
            SocialContextState,
        )
        from local_audio_resources import get_local_audio_resources
    except ImportError:
        get_global_circuit_breaker = lambda: None
        ProviderRegistry = None
        MultimodalPerceptionState = None
        SocialContextEngine = None
        SocialContextState = None
        get_local_audio_resources = lambda: None


def _load_env():
    # Test sürecinde .env YÜKLENMEZ: aksi hâlde testler kullanıcının gerçek
    # anahtarlarını alıp canlı API çağrıları yapıyor (astro_realtime_node
    # websocket'i gerçekten açıyor, kota harcanıyor) ve testler ".env yok"
    # varsayımıyla yazıldığı için sonuçlar çalıştırma ortamına göre değişiyor.
    if "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules:
        return None

    candidates = [
        os.path.abspath(".env"),
        os.path.abspath(".env.production"),
        os.path.abspath(os.path.join(os.getcwd(), ".env")),
        os.path.abspath(os.path.join(os.getcwd(), ".env.production")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".env")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".env.production")),
        os.path.expanduser("~/.env")
    ]
    for c in candidates:
        if os.path.exists(c):
            load_dotenv(dotenv_path=c, override=True)
            return c
    try:
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(dotenv_path=env_path, override=True)
            return env_path
    except Exception as _exc:
        _LOG.debug("_load_env: yok sayılan hata (%s)", _exc)
    return None


def imgmsg_to_bgr(msg: Image) -> np.ndarray | None:
    try:
        if msg.encoding == "bgr8":
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
        elif msg.encoding == "rgb8":
            data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            return cv2.cvtColor(data, cv2.COLOR_RGB2BGR) if cv2 else data[:, :, ::-1].copy()
        elif msg.encoding in ("mono8", "8UC1"):
            data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            return cv2.cvtColor(data, cv2.COLOR_GRAY2BGR) if cv2 else np.stack([data]*3, axis=-1)
    except Exception as _exc:
        _LOG.debug("imgmsg_to_bgr: yok sayılan hata (%s)", _exc)
    return None


def frame_to_base64_jpeg(frame: np.ndarray, max_dim: int = 512) -> str | None:
    if cv2 is None or frame is None:
        return None
    try:
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buffer).decode("utf-8")
    except Exception:
        return None


def is_canned_refusal(text: str) -> bool:
    """Detects standard LLM safety refusal boilerplate phrases and English reasoning leaks."""
    if not text:
        return False
    t_lower = text.lower().strip()
    refusal_patterns = [
        "yardımcı olamam", "yardımcı olamayacağım", "bu isteğinize yardımcı",
        "isteğinize yardımcı olamam", "yardımcı olamam maalesef", "üzgünüm, ancak",
        "üzgünüm, ama lütfen", "daha saygılı bir dil", "bir yapay zeka olarak",
        "yapay zeka olarak", "uygunsuz içerik", "as an ai", "i cannot assist",
        "i cannot fulfill", "i am unable to", "here's a thinking process",
        "thinking process", "let's think"
    ]
    return any(p in t_lower for p in refusal_patterns)


class AiBrainNode(Node):
    def __init__(self):
        super().__init__("ai_brain_node")
        _load_env()

        # Core Modular Components
        self.memory = MemoryManager()
        self.cloud_mgr = CloudManager()
        initial_persona = self.memory.profile.data.get("current_persona", "playful")
        self.persona_engine = PersonaEngine(initial_persona)
        self.state_machine = StateMachine(RobotState.IDLE)

        # Parameters
        self.declare_parameter("llm_model", os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"))
        self.declare_parameter("vision_model", os.getenv("VISION_MODEL", "llama-3.2-90b-vision-preview"))
        self.declare_parameter("llm_temperature", float(os.getenv("LLM_TEMPERATURE", "0.55")))
        self.declare_parameter("llm_max_tokens", int(os.getenv("LLM_MAX_TOKENS", "300")))
        self.declare_parameter("wake_word", os.getenv("WAKE_WORD", "hey astro"))
        self.declare_parameter("conversation_timeout", float(os.getenv("CONVERSATION_TIMEOUT", "16.0")))
        self.declare_parameter("gaze_dwell_s", float(os.getenv("GAZE_DWELL_S", "4.0")))
        self.declare_parameter("gaze_cooldown_s", float(os.getenv("GAZE_COOLDOWN_S", "60.0")))
        self.declare_parameter("gaze_startup_grace_s", float(os.getenv("GAZE_STARTUP_GRACE_S", "15.0")))
        self.declare_parameter("default_user_name", os.getenv("DEFAULT_USER_NAME", "Misafir"))
        self.declare_parameter("enable_idle_learning", os.getenv("ENABLE_IDLE_LEARNING", "true").lower() == "true")

        self._text_model = self.get_parameter("llm_model").value
        self._fallback_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
        self._vision_model = self.get_parameter("vision_model").value
        self._temperature = float(self.get_parameter("llm_temperature").value)
        self._max_tokens = int(self.get_parameter("llm_max_tokens").value)
        self._wake_word = self.get_parameter("wake_word").value
        conv_timeout = float(self.get_parameter("conversation_timeout").value)
        self._gaze_dwell_s = float(self.get_parameter("gaze_dwell_s").value)
        self._gaze_cooldown_s = float(self.get_parameter("gaze_cooldown_s").value)
        self._gaze_startup_grace_s = float(self.get_parameter("gaze_startup_grace_s").value)
        self._default_user_name = str(self.get_parameter("default_user_name").value)
        self._enable_idle_learning = bool(self.get_parameter("enable_idle_learning").value)

        # Adaptive Session
        self.session = ConversationSession(
            base_timeout_s=conv_timeout,
            on_session_start=lambda: self.get_logger().info("✨ [Session] Konuşma Oturumu Başlatıldı."),
            on_session_end=self._on_session_timed_out
        )

        # En az bir metin motoru (Groq / OpenAI / Gemini REST) hazır olduğunda True olur.
        # Tanımsız kalırsa _on_speech ilk konuşmada AttributeError verir.
        self._enabled = False

        # 0. Circuit Breaker, Model Registry & Audio Resources
        self.circuit_breaker = get_global_circuit_breaker()
        self.provider_registry = ProviderRegistry(logger=self.get_logger()) if ProviderRegistry else None
        self.audio_resources = get_local_audio_resources() if get_local_audio_resources else None

        # 1. Groq Client (Primary Ultra-Fast Free LPU Engine - Zero OpenAI Cost)
        # ── Sağlayıcı seçimi (LLM_PROVIDER) ──────────────────────────────
        # Yerleşik zincir sırası: groq -> gemini -> openai.
        # LLM_PROVIDER="openai" gibi bir değer verilirse o sağlayıcıdan ÖNCE
        # gelenler tamamen atlanır; sonrakiler yedek olarak kalır.
        # LLM_FALLBACK_ENABLED="false" ile yalnızca seçilen sağlayıcı denenir.
        # (Eskiden LLM_PROVIDER .env.example'da belgeliydi ama kodda hiç okunmuyordu.)
        self._provider_chain = ("groq", "gemini", "openai")
        self._primary_provider = os.environ.get("LLM_PROVIDER", "").strip().strip("\"'").lower()
        if self._primary_provider not in self._provider_chain:
            self._primary_provider = ""
        self._llm_fallback_enabled = os.environ.get(
            "LLM_FALLBACK_ENABLED", "true").strip().strip("\"'").lower() not in ("false", "0", "no")

        self.groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self._groq = None
        self._active_groq_models = []

        if Groq and self.groq_api_key:
            try:
                self._groq = Groq(api_key=self.groq_api_key)
                self._active_groq_models = self._discover_active_groq_models()
                self._enabled = True
                self.get_logger().info(f"🚀 [AI Brain] Groq LPU (Ücretsiz & Ultra Hızlı) Birincil LLM Motoru Aktif! (Toplam {len(self._active_groq_models)} Model)")
            except Exception as e:
                self.get_logger().debug(f"Groq client notice: {e}")

        # 2. OpenAI Client (Emergency High-Performance Backup Engine)
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("AI_API_KEY", "").strip()
        self._openai_model = os.environ.get("OPENAI_CHAT_MODEL", os.environ.get("LLM_FALLBACK_MODEL", "gpt-4o-mini"))
        self._openai = None

        if OpenAI and self.openai_api_key and self.openai_api_key.startswith("sk-"):
            try:
                self._openai = OpenAI(api_key=self.openai_api_key)
                self._enabled = True
                self.get_logger().info(f"✅ [AI Brain] OpenAI ({self._openai_model}) Yedek Motoru Hazır.")
            except Exception as e:
                self.get_logger().error(f"❌ [AI Brain] OpenAI client başlatılamadı: {e}")

        # 3. Gemini REST API Key (Tertiary Fallback)
        # AI_API_KEY tarihsel isim; .env'de anahtar GEMINI_API_KEY olarak tutuluyor.
        # OpenAI anahtarı (sk-...) buraya düşerse Gemini uç noktası 400 döner, ele.
        self._ai_api_key = (
            os.environ.get("AI_API_KEY", "").strip("\"' \t\n\r")
            or os.environ.get("GEMINI_API_KEY", "").strip("\"' \t\n\r")
        )
        if self._ai_api_key.startswith("sk-"):
            self._ai_api_key = ""

        if self._ai_api_key:
            self._enabled = True
            self.get_logger().info("✅ [AI Brain] Google Gemini REST metin motoru hazır.")

        if not self._enabled:
            self.get_logger().error(
                "❌ [AI Brain] GROQ_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY bulunamadı! LLM devre dışı."
            )

        # Perception & Hardware State
        self._lock = threading.Lock()
        self._is_processing = False
        self._processing_start_time = 0.0
        self._tts_speaking = False
        self._person_detected = False
        self._looking_at_robot = False
        self._looking_start_time = None
        self._gaze_lock = threading.Lock()
        self._last_proactive_gaze_time = 0.0
        self._speaker_angle = 0.0
        self._speaker_gender = "unknown"
        self._user_distance = 0.0
        self._user_emotion = "neutral"
        self._recognized_person = None
        self._recognized_speaker = None
        self._last_speaker_embedding = None
        self._enrollment_session = self._new_enrollment_session()
        self._node_start_time = time.monotonic()
        self._latest_frame = None
        self._latest_frame_time = 0.0

        # Multimodal Perception & Social Context Engine
        self.social_context_engine = SocialContextEngine() if SocialContextEngine else None

        # ROS 2 Publishers
        self.pub_tts = self.create_publisher(String, "/tts/say", 10)
        self.pub_realtime_req = self.create_publisher(String, "/tts/realtime_request", 10)
        self.pub_interrupt = self.create_publisher(Bool, "/tts/interrupt", 10)
        self.pub_emotion = self.create_publisher(String, "/robot/emotion", 10)
        self.pub_gesture = self.create_publisher(String, "/robot/head_gesture", 10)
        self.pub_look_target = self.create_publisher(Float32, "/robot/look_target", 10)
        self.pub_session_active = self.create_publisher(Bool, "/ai/session_active", 10)

        # ── Callback grupları ────────────────────────────────────────────
        # rclpy'de grup belirtilmezse HER abonelik ve timer düğümün tek varsayılan
        # MutuallyExclusiveCallbackGroup'una girer; o zaman tek bir yavaş/bloke
        # callback bütün düğümü (tüm sensörler + tüm timer'lar) dondurur.
        # Üç gruba ayırıyoruz ki biri tıkansa diğerleri çalışmaya devam etsin:
        #   • konuşma  — turlar kendi aralarında sıralı kalmalı (yarış olmasın)
        #   • timer    — oturum yaşam döngüsü ve hatırlatıcılar hiç durmamalı
        #   • algı     — küçük, kilit korumalı durum güncellemeleri; paralel olabilir
        self._cbg_speech = MutuallyExclusiveCallbackGroup()
        self._cbg_timers = MutuallyExclusiveCallbackGroup()
        self._cbg_perception = ReentrantCallbackGroup()

        # ROS 2 Subscribers
        self.create_subscription(String, "/speech/text", self._on_speech, 10,
                                 callback_group=self._cbg_speech)
        self.create_subscription(String, "/audio/speaker_gender", self._on_speaker_gender, 10,
                                 callback_group=self._cbg_perception)
        self.create_subscription(String, "/audio/speaker_id", self._on_speaker_id, 10,
                                 callback_group=self._cbg_perception)
        self.create_subscription(Bool, "/tts/speaking", self._on_tts_speaking, 10,
                                 callback_group=self._cbg_perception)
        self.create_subscription(Bool, "/tts/interrupt", self._on_tts_interrupt, 10,
                                 callback_group=self._cbg_perception)
        self.create_subscription(Bool, "/vision/person_detected", self._on_person_detected, 10,
                                 callback_group=self._cbg_perception)
        self.create_subscription(Bool, "/vision/looking_at_robot", self._on_looking_at_robot, 10,
                                 callback_group=self._cbg_perception)
        self.create_subscription(String, "/vision/recognized_person", self._on_recognized_person, 10,
                                 callback_group=self._cbg_perception)
        self.create_subscription(Float32, "/vision/user_distance", self._on_user_distance, 10,
                                 callback_group=self._cbg_perception)
        self.create_subscription(String, "/vision/user_emotion", self._on_user_emotion, 10,
                                 callback_group=self._cbg_perception)
        self.create_subscription(Float32, "/audio/doa", self._on_doa, 10,
                                 callback_group=self._cbg_perception)
        # Görüntü akışları sensör QoS'u (BEST_EFFORT) kullanır: kare kaybı, geciken
        # kareler için retransmission yapmaktan iyidir. BEST_EFFORT abone RELIABLE
        # yayıncıdan da veri alabilir, bu yüzden depthai_ros_driver ile uyumludur.
        self.create_subscription(Image, "/oak/rgb/image_raw", self._on_camera_image, qos_profile_sensor_data,
                                 callback_group=self._cbg_perception)
        # Realtime WebSocket state tracking (source of truth: astro_realtime_node)
        self._realtime_ws_connected = False
        self._realtime_session_ready = False
        self._realtime_audio_received = False
        self.create_subscription(String, "/realtime/state", self._on_realtime_state, 10,
                                 callback_group=self._cbg_perception)

        # Timers
        self.create_timer(0.15, self._check_proactive_gaze, callback_group=self._cbg_timers)
        self.create_timer(1.0, self._check_session_lifecycle, callback_group=self._cbg_timers)
        self.create_timer(1.0, self._check_reminders, callback_group=self._cbg_timers)

        # Idle Learning (Powered 100% by Groq/Gemini, 0 OpenAI token cost)
        if self._enable_idle_learning:
            self._start_idle_learning()
            self.get_logger().info("🤖 [AI Brain] Groq/Gemini Tabanlı Otonom Boşta Öğrenme ve Bellek Güçlendirme Aktif!")

        self.get_logger().info(
            f"🧠 [AI Brain Node] Modüler Mimari Hazır! Kişilik: [{self.persona_engine.current_persona.upper()}]"
        )

    def _play_local_ack(self, ack_pcm: bytes, generation_id: int = 0):
        """Plays low-latency pre-generated local ACK WAV via AudioOutputManager or non-blocking aplay."""
        if not ack_pcm:
            return
        try:
            from astro_audio.audio_output_manager import AudioOutputManager
            out_mgr = AudioOutputManager.get_instance()
            if out_mgr:
                out_mgr.play_pcm_chunk(ack_pcm, sample_rate=16000, generation_id=generation_id, provenance={"source": "thinking_ack_local", "tts_provider": "local_pcm", "playback_source": "hardware_dac"})
                return
        except Exception as _exc:
            self.get_logger().debug(f"_play_local_ack: yok sayılan hata ({_exc})")
        try:
            import subprocess
            proc = subprocess.Popen(["aplay", "-q", "-r", "16000", "-f", "S16_LE", "-c", "1"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if proc.stdin:
                proc.stdin.write(ack_pcm)
                proc.stdin.close()
        except Exception as _exc:
            self.get_logger().debug(f"_play_local_ack: yok sayılan hata ({_exc})")

    def _discover_active_groq_models(self) -> List[str]:
        """Dynamically queries active chat models from Groq, prioritizing top conversational models and excluding non-chat models."""
        preferred = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama3-70b-8192", "llama-3.1-8b-instant"]
        if hasattr(self, "provider_registry") and self.provider_registry:
            reg_models = self.provider_registry.get_routeable_models("groq")
            if reg_models:
                filtered = [m for m in reg_models if not any(x in m.lower() for x in ("allam", "orpheus", "canopylabs", "vision", "audio", "whisper", "specdec", "compound"))]
                if filtered:
                    return filtered
        if not self._groq:
            return preferred
        try:
            models = self._groq.models.list()
            available = [m.id for m in models.data]
            chat_models = [m for m in preferred if m in available]
            for m in available:
                mid_l = m.lower()
                if any(x in mid_l for x in ["whisper", "embedding", "guard", "moderation", "tts", "distill", "r1", "deepseek", "qwen", "allam", "orpheus", "canopylabs", "vision", "audio", "specdec", "compound"]):
                    continue
                if m not in chat_models:
                    chat_models.append(m)
            if chat_models:
                return chat_models
        except Exception as e:
            self.get_logger().debug(f"Groq dynamic model discovery notice: {e}")
        return preferred



    def _discover_vision_model(self) -> str | None:
        try:
            models = self._groq.models.list()
            available = [m.id for m in models.data]
            for cand in available:
                if any(v_kw in cand.lower() for v_kw in ["vision", "scout", "vl"]):
                    return cand
        except Exception as _exc:
            self.get_logger().debug(f"_discover_vision_model: yok sayılan hata ({_exc})")
        return None

    def _route_reason(self, name: str) -> str:
        """Log etiketi gerçeği söylesin: 'tertiary_fallback' sabitti ve OpenAI
        birincil sağlayıcı olarak yapılandırıldığında bile öyle yazıyordu."""
        if self._primary_provider:
            return "configured_primary" if name == self._primary_provider else "configured_fallback"
        idx = self._provider_chain.index(name)
        return ("primary", "secondary_fallback", "tertiary_fallback")[idx]

    def _provider_enabled(self, name: str) -> bool:
        """LLM_PROVIDER / LLM_FALLBACK_ENABLED ayarlarına göre sağlayıcı denensin mi?"""
        if not self._primary_provider:
            return True                      # ayar yok -> yerleşik zincirin tamamı
        if name == self._primary_provider:
            return True
        if not self._llm_fallback_enabled:
            return False                     # katı mod: yalnızca seçilen sağlayıcı
        # Yedeklere yalnızca seçilenden SONRA gelenler dahil
        try:
            return self._provider_chain.index(name) > self._provider_chain.index(self._primary_provider)
        except ValueError:
            return False

    def _on_session_timed_out(self):
        self.state_machine.transition_to(RobotState.IDLE)
        self.get_logger().info("💤 [AI] Oturum zaman aşımı — Uyku moduna (IDLE) geçildi.")

        # Summarize episodic dialogue turns and save to person profile
        msgs = self.memory.episodic.get_messages()
        if len(msgs) >= 2:
            identity = self._get_active_biometric_identity()
            p_name = identity.get("name", "Baran") if identity.get("is_known") else "Baran"
            dialogue_text = " | ".join([f"{m.get('role')}: {m.get('content')}" for m in msgs[-6:]])
            threading.Thread(target=self._async_summarize_and_save_session, args=(dialogue_text, p_name), daemon=True).start()

    def _async_summarize_and_save_session(self, dialogue_text: str, person_name: str):
        prompt = f"Aşağıdaki kısa diyalogda ne konuşulduğunu tek bir kısa Türkçe cümleyle (örn: 'Hava durumu ve robotik özellikleri üzerine konuşuldu') özetle:\n{dialogue_text}"
        try:
            summary = None
            if self._openai:
                res = self._openai.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self._openai_model,
                    temperature=0.2,
                    max_tokens=50
                )
                summary = res.choices[0].message.content.strip()
            elif self._groq and self._active_groq_models:
                res = self._groq.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self._active_groq_models[0],
                    temperature=0.2,
                    max_tokens=50
                )
                summary = res.choices[0].message.content.strip()

            if summary and len(summary) > 5:
                clean_sum = clean_tts_text(summary)
                self.memory.profile.add_person_session_summary(person_name, clean_sum)
                self.get_logger().info(f"📝 [Oturum Günlüğü ({person_name})]: Kaydedildi -> '{clean_sum}'")
        except Exception as e:
            self.get_logger().debug(f"Session summarizer notice: {e}")

    def _check_session_lifecycle(self):
        is_speaking = self._tts_speaking or self.state_machine.is_speaking() or self.state_machine.is_thinking() or self._is_processing
        if is_speaking:
            self.session.record_robot_speech()
        self.session.check_and_update_session_lifecycle(is_robot_speaking=is_speaking)
        # Broadcast session state so STT node can make context-aware filter decisions
        msg = Bool()
        msg.data = self.session.is_active()
        self.pub_session_active.publish(msg)

    # Perception Callbacks
    def _on_camera_image(self, msg: Image):
        # Throttle camera frame decoding to max 2 FPS to prevent CPU starvation and event loop choking
        now = time.monotonic()
        if (now - getattr(self, '_last_img_decode_time', 0.0)) < 0.5:
            return
        self._last_img_decode_time = now

        frame = imgmsg_to_bgr(msg)
        if frame is not None:
            with self._lock:
                self._latest_frame = frame
                self._latest_frame_time = now

    def _on_tts_speaking(self, msg: Bool):
        self._tts_speaking = msg.data
        self.session.record_robot_speech()
        if not msg.data:
            if self.state_machine.is_speaking():
                self.state_machine.transition_to(RobotState.LISTENING)

    def _on_tts_interrupt(self, msg: Bool):
        if msg.data:
            self.state_machine.transition_to(RobotState.INTERRUPTED)
            self.state_machine.transition_to(RobotState.LISTENING)

    def _on_person_detected(self, msg: Bool):
        self._person_detected = msg.data
        if msg.data:
            self._last_person_seen_time = time.monotonic()
        if self.social_context_engine:
            self.social_context_engine.update_visual(person_detected=msg.data, person_count=1 if msg.data else 0)

    def _on_user_distance(self, msg: Float32):
        self._user_distance = float(msg.data)
        if self.social_context_engine:
            self.social_context_engine.update_visual(face_distance_m=float(msg.data))

    def _on_user_emotion(self, msg: String):
        self._user_emotion = msg.data.lower().strip()
        if self.social_context_engine:
            self.social_context_engine.update_visual(emotion=self._user_emotion)

    def _on_speaker_gender(self, msg: String):
        self._speaker_gender = msg.data.lower().strip()

    def _on_doa(self, msg: Float32):
        self._speaker_angle = float(msg.data)
        if self.social_context_engine:
            self.social_context_engine.update_audio(doa_angle_deg=float(msg.data))

    def _on_realtime_state(self, msg: String):
        """Receives actual WebSocket state from astro_realtime_node."""
        try:
            import json
            data = json.loads(msg.data)
            state = data.get("state", "")
            conn = data.get("connection", state)
            sess = data.get("session", "")
            provider = data.get("provider", "")
            self._fallback_mode = bool(data.get("fallback_mode", False) or provider in ("EXHAUSTED", "COOLDOWN"))
            if conn == "CONNECTED" or state in ("CONNECTED", "SESSION_READY"):
                self._realtime_ws_connected = True
            else:
                self._realtime_ws_connected = False
            if sess == "READY" or state == "SESSION_READY":
                self._realtime_session_ready = True
            elif sess == "NOT_READY" or state in ("DISCONNECTED", "CONNECTING"):
                self._realtime_session_ready = False
        except Exception:
            pass

    def _on_looking_at_robot(self, msg: Bool):
        is_looking = msg.data
        now = time.monotonic()
        self.session.update_gaze(is_looking)
        if self.social_context_engine:
            self.social_context_engine.update_visual(looking_at_robot=is_looking)
        with self._gaze_lock:
            if is_looking:
                if not self._looking_at_robot:
                    self._looking_start_time = now
                self._looking_at_robot = True
                self._last_gaze_seen_time = now
            else:
                # Gaze hysteresis: only drop gaze after 1.2 seconds of absence to prevent flicker resets
                if hasattr(self, '_last_gaze_seen_time') and (now - self._last_gaze_seen_time) > 1.2:
                    self._looking_at_robot = False
                    self._looking_start_time = None

    def _on_recognized_person(self, msg: String):
        try:
            data = json.loads(msg.data)
            with self._lock:
                self._recognized_person = data

            if data.get("is_known") and data.get("confidence", 0.0) >= 0.45:
                now = time.monotonic()
                p_name = data.get("name", "")
                dist = self._user_distance
                is_looking = self._looking_at_robot

                # Frontal Angle & Distance Gating: Must be looking at robot, within 1.8m, and directly in front
                if self.state_machine.is_idle() and not self._tts_speaking and not self._is_processing:
                    if is_looking and (dist <= 0.0 or dist <= 1.80) and abs(self._speaker_angle) <= 25.0:
                        if (now - getattr(self, "_last_vip_greet_time", 0.0)) > 40.0:
                            self._last_vip_greet_time = now
                            self.session.activate_session(reason="vip_vision")
                            self.state_machine.transition_to(RobotState.LISTENING)
                            identity = self._get_active_biometric_identity()
                            proactive_greeting, greeting_emo = self.persona_engine.build_proactive_greeting(
                                identity=identity,
                                user_emotion=self._user_emotion,
                                speaker_gender=self._speaker_gender
                            )
                            self._publish_emotion(greeting_emo)
                            self.get_logger().info(f"👤 [Proaktif Yüz Karşılama] ({p_name}): \"{proactive_greeting}\"")
                            self._publish_gesture("nod")
                            self._publish_tts(proactive_greeting)
        except Exception as _exc:
            self.get_logger().debug(f"_on_recognized_person: yok sayılan hata ({_exc})")

    def _on_speaker_id(self, msg: String):
        try:
            data = json.loads(msg.data)
            with self._lock:
                self._recognized_speaker = data
                raw_emb = data.get("embedding")
                if raw_emb and len(raw_emb) > 0:
                    self._last_speaker_embedding = np.array(raw_emb, dtype=np.float32)
        except Exception as _exc:
            self.get_logger().debug(f"_on_speaker_id: yok sayılan hata ({_exc})")



    def _get_active_biometric_identity(self) -> Dict[str, Any]:
        """Multimodal Biometric Fusion: Combines visual face recognition and acoustic speaker ID."""
        with self._lock:
            face = self._recognized_person or {}
            spk = self._recognized_speaker or {}

        # 1. Face Recognition (Visual priority when face is verified >= 0.42)
        if face.get("is_known") and face.get("confidence", 0.0) >= 0.42:
            name = face.get("name", "")
            off = find_official_by_name_or_alias(name)

            if off:
                return {**off, "confidence": face.get("confidence"), "is_known": True, "source": "face"}
            known = self.memory.profile.get_known_person(name)
            if known:
                return {**known, "confidence": face.get("confidence"), "is_known": True, "source": "face"}
            return {
                "name": name,
                "title": face.get("title", "Tanınan Kişi"),
                "formal_title": face.get("formal_title", name),
                "confidence": face.get("confidence"),
                "is_known": True,
                "source": "face"
            }

        # 2. Voice Recognition (Acoustic priority when voice matches >= 0.40)
        if spk.get("is_known") and spk.get("confidence", 0.0) >= 0.40:
            name = spk.get("name", "")
            off = find_official_by_name_or_alias(name)

            if off:
                return {**off, "confidence": spk.get("confidence"), "is_known": True, "source": "voice"}
            known = self.memory.profile.get_known_person(name)
            if known:
                return {**known, "confidence": spk.get("confidence"), "is_known": True, "source": "voice"}
            return {
                "name": name,
                "title": spk.get("title", "Tanınan Konuşmacı"),
                "formal_title": spk.get("formal_title", name),
                "confidence": spk.get("confidence"),
                "is_known": True,
                "source": "voice"
            }

        return {"name": "Misafir", "title": "Ziyaretçi", "formal_title": "Misafir", "is_known": False, "confidence": 0.0}

    def _check_proactive_gaze(self):
        with self._gaze_lock:
            looking = self._looking_at_robot
            look_start = self._looking_start_time

        if not looking or look_start is None or self._tts_speaking or self._is_processing:
            return

        now = time.monotonic()
        # 1. Startup Grace Period: Do not trigger proactive speech during initial launch
        if (now - self._node_start_time) < self._gaze_startup_grace_s:
            return

        # 2. Sustained Dwell Time: User must deliberately look for configured duration
        if (now - look_start) >= self._gaze_dwell_s:
            # 3. Cooldown between proactive prompts
            if self.state_machine.is_idle() and (now - self._last_proactive_gaze_time) > self._gaze_cooldown_s:
                self._last_proactive_gaze_time = now
                self.session.activate_session(reason="proactive_gaze")
                self.state_machine.transition_to(RobotState.LISTENING)
                with self._gaze_lock:
                    self._looking_start_time = None

                identity = self._get_active_biometric_identity()
                proactive_greeting, greeting_emo = self.persona_engine.build_proactive_greeting(
                    identity=identity,
                    user_emotion=self._user_emotion,
                    speaker_gender=self._speaker_gender
                )
                self._publish_emotion(greeting_emo)

                self.get_logger().info(f"👁️ [Proaktif Etkileşim] ({greeting_emo}): \"{proactive_greeting}\"")
                self._publish_gesture("nod")
                self._publish_tts(proactive_greeting)

    def _check_persona_switch(self, text: str) -> bool:
        text_lower = text.lower()

        # Strict regex patterns to prevent false triggers (e.g. "hanımefendi" erroneously triggering formal mode)
        switch_patterns = {
            "kufurbaz": [
                r"\b(küfürbaz|ağzı bozuk|filtresiz|argo|söv|saydır|sövme|küfürlü)\b.*\b(ol|geç|mod|davran|konuş|takıl|başla)\b",
                r"\b(küfürbaz ol|ağzı bozuk ol|filtresiz konuş|söv bana|söv bakalım)\b"
            ],
            "flirt": [
                r"\b(flört|flirt|çapkın|yavşak|romantik|astroflirt|astroflört)\b.*\b(ol|geç|mod|davran|konuş|takıl|başla)\b",
                r"\b(kızlara yürü|yavşa|flört et)\b",
                r"\b(flört|çapkın)\s+moduna\b"
            ],
            "rude": [
                r"\b(kaba|ters|küfürlü|saygısız|dobra)\b.*\b(ol|geç|mod|davran|konuş|takıl)\b",
                r"\b(kaba)\s+moduna\b"
            ],
            "angry": [
                r"\b(öfkeli|asabi|kızgın|sinirli|agresif)\b.*\b(ol|geç|mod|davran|konuş|takıl)\b",
                r"\b(asabi|sinirli|kızgın)\s+moduna\b"
            ],
            "playful": [
                r"\b(şakacı|neşeli|normal|sempatik|tatlı|oyuncu|dostane)\b.*\b(ol|geç|mod|davran|konuş|takıl)\b",
                r"\b(eski haline dön|eski moduna geç|varsayılan moda geç|normal ol|şakacı ol)\b"
            ],
            "formal": [
                r"\b(resmi|protokol|ciddi)\b.*\b(ol|geç|mod|davran|konuş|takıl)\b",
                r"\b(resmi moda geç|protokol moduna geç|ciddi ol)\b"
            ],
            "sarcastic": [
                r"\b(alaycı|sarkastik|ironik|iğneleyici|gıcık)\b.*\b(ol|geç|mod|davran|konuş|takıl)\b",
                r"\b(laf sok|alaycı ol|sarkastik ol)\b"
            ],
            "emotional": [
                r"\b(duygusal|hisli|duygulu|romantizm)\b.*\b(ol|geç|mod|davran|konuş|takıl)\b",
                r"\b(duygusal ol|duygusal moda geç)\b"
            ]
        }

        for p_name, patterns in switch_patterns.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    self.persona_engine.set_persona(p_name)
                    self.memory.profile.set_persona(p_name)
                    self.get_logger().info(f"🎭 [Kişilik Değişti]: Yeni Mod -> {p_name.upper()} ('{pat}' eşleşti)")
                    self._publish_emotion(p_name)
                    return True
        return False

    @staticmethod
    def _new_enrollment_session() -> Dict[str, Any]:
        """Boş biyometrik kayıt oturumu. Şekil tek yerde tanımlı olsun diye."""
        return {
            "active": False,
            "name": "",
            "title": "",
            "turn": 0,
            "max_turns": 3,
            "embeddings": [],
            "last_frame": None,
            "start_time": 0.0,
        }

    def _on_speech(self, msg: String):
        self.get_logger().info(f"[TURN RECEIVED] text=\"{msg.data}\"")
        if not self._enabled:
            self.get_logger().warn("[TURN DROPPED] reason=node_disabled")
            return

        # Realtime primary S2S active: drop secondary STT turns to prevent duplicate Groq LLM responses
        if self._realtime_ws_connected and self._realtime_session_ready and not getattr(self, "_fallback_mode", False):
            self.get_logger().info(f"[TURN DROPPED] reason=realtime_primary_active text=\"{msg.data}\"")
            return

        raw_text = normalize_turkish_speech_input(re.sub(r"^['\"`´“”‘’]+|['\"`´“”‘’]+$", "", msg.data.strip()).strip())
        if not raw_text:
            self.get_logger().warn("[TURN DROPPED] reason=empty_text")
            return

        now = time.monotonic()
        if (now - getattr(self, '_last_llm_turn_time', 0.0)) < 0.35:
            self.get_logger().warn(f"[TURN DROPPED] reason=debounced_rapid_speech text=\"{raw_text}\"")
            return
        self._last_llm_turn_time = now

        if self._tts_speaking:
            # Kullanıcı konuşurken robot konuşuyorsa anında sustur (Barge-in)
            self._publish_interrupt()
            self._tts_speaking = False

        t_vad_start = now

        # Turn head toward sound DOA
        if self._speaker_angle > 0:
            target_msg = Float32()
            target_msg.data = self._speaker_angle
            self.pub_look_target.publish(target_msg)

        # Check persona switch
        if self._check_persona_switch(raw_text):
            persona = self.persona_engine.current_persona
            ack_map = {
                "witty": "Harika! Hazırcevap ve esprili moduma geçtim. Söyle bakalım ne konuşuyoruz?",
                "kufurbaz": "Harika! Hazırcevap ve esprili moduma geçtim. Söyle bakalım ne konuşuyoruz?",
                "charming": "Harika! Cana yakın ve sempatik moduma geçtim, seni dinliyorum!",
                "flirt": "Harika! Cana yakın ve sempatik moduma geçtim, seni dinliyorum!",
                "angry": "Tamam, tatlı-sert moduma geçtim! Ne istiyorsan söyle bakalım.",
                "rude": "İyi tamam, dobra ve net konuşma modundayım, ne diyeceksen de!",
                "formal": "Emriniz başım üstüne efendim. Protokol kurallarına riayet edeceğim.",
                "sarcastic": "Aman ne harika, şimdi de laf sokmamı istiyorsun demek. Çok zekice bir karar doğrusu!",
                "emotional": "Nasıl istersen... Bütün hislerimle seni dinliyorum, ne kadar güzel bir an...",
                "playful": "Süper! Eski neşeli ve enerjik halime geri döndüm, seni dinliyorum!"
            }
            self._publish_tts(ack_map.get(persona, "Kişiliğim güncellendi!"))
            self.session.activate_session(reason="persona_switch")
            self.state_machine.transition_to(RobotState.LISTENING)
            self.get_logger().info(f"[SESSION STARTED] reason=persona_switch raw_text=\"{raw_text}\"")
            return

        # Handle ongoing interactive multi-turn biometric enrollment
        if self._enrollment_session.get("active"):
            self._dispatch_turn(raw_text, t_vad_start)
            return

        has_wake_word, clean_prompt = self.session.is_wake_word(raw_text, self._wake_word)

        if has_wake_word:
            self.session.activate_session(reason="wake_word")
            self.state_machine.transition_to(RobotState.LISTENING)
            persona = self.persona_engine.current_persona
            self.get_logger().info(f"[SESSION STARTED] reason=wake_word raw_text=\"{raw_text}\"")
            self._publish_emotion(persona)
            self._publish_gesture("nod")

            pure_greetings = ["merhaba", "merhabalar", "selam", "selamlar", "günaydın", "iyi günler", "iyi akşamlar", "efendim", "hoş bulduk", "hoş geldiniz", "selamün aleyküm", "selamun aleykum", "hey"]
            is_pure_greeting = (raw_text.lower().strip(" .,!?:;") in pure_greetings) or (not clean_prompt) or (len(clean_prompt) < 2)

            if is_pure_greeting:
                greeting_map = {
                    "flirt": "Selam! Seni dinliyorum, anlat bakalım.",
                    "charming": "Selam! Seni dinliyorum, anlat bakalım.",
                    "playful": "Merhaba! Seni dinliyorum, nasıl yardımcı olabilirim?",
                    "formal": "Buyrun efendim, sizi dinliyorum.",
                    "sarcastic": "Merhaba, yine ne soracaksın bakalım?",
                    "emotional": "Merhaba, can kulağıyla seni dinliyorum...",
                    "angry": "Ne var, ne istiyorsun?",
                    "rude": "Ne var birader, kısa kes!"
                }
                greeting = greeting_map.get(persona, "Merhaba! Seni dinliyorum, nasıl yardımcı olabilirim?")
                t_done = time.monotonic()
                turn_ms = (t_done - t_vad_start) * 1000.0
                self.session.latency_tracker.record_turn(0.0, turn_ms, turn_ms)
                stats = self.session.latency_tracker.get_stats()
                self.get_logger().info(f"⚡ [Latency] Hızlı Yanıt: {turn_ms:.0f}ms (Doğrudan Selamlama) | p50: {stats['p50_total_ms']}ms, p95: {stats['p95_total_ms']}ms")
                self._publish_tts(greeting)
                return
            else:
                raw_text = clean_prompt
        elif self.session.is_active():
            # In an active conversation turn
            pass
        elif self.state_machine.is_idle():
            # In IDLE without wake word: Check Gaze or Social Context
            social_ctx = self.social_context_engine.get_state().social_context if self.social_context_engine else SocialContextState.ISOLATED_IDLE
            if self._looking_at_robot or social_ctx == SocialContextState.DIRECT_INTERACTION:
                self.session.activate_session(reason="gaze")
                self.state_machine.transition_to(RobotState.LISTENING)
                self.get_logger().info(f"[SESSION STARTED] reason=gaze raw_text=\"{raw_text}\"")
                persona = self.persona_engine.current_persona
                self._publish_emotion(persona)
                self._publish_gesture("nod")
            else:
                self.get_logger().info(f"🕵️ [Arka Plan]: '{raw_text}' sosyal filtrede inceleniyor...")
                # Bulut çağrısı callback'i bloke etmesin: thread'e devret ve çık.
                threading.Thread(
                    target=self._social_filter_then_dispatch,
                    args=(raw_text, t_vad_start),
                    daemon=True,
                ).start()
                return

        # Active Session Turn
        self._dispatch_turn(raw_text, t_vad_start)

    def _dispatch_turn(self, raw_text: str, t_vad_start: float) -> None:
        """Turu LLM işleme thread'ine gönderir (eşzamanlılık kilidi burada tutulur).

        _on_speech ve sosyal filtre thread'i ortak bu yolu kullanır; böylece
        "zaten işleniyor" mantığı tek yerde durur.
        """
        self.session.record_user_speech()
        self._publish_interrupt()
        now = time.monotonic()

        with self._lock:
            if self._is_processing:
                if (now - getattr(self, '_processing_start_time', 0.0)) > 12.0:
                    self.get_logger().warn("⚠️ [AI] Önceki LLM işlemi zaman aşımına uğradı (>12s), kilit sıfırlanıyor.")
                    self._is_processing = False
                else:
                    self.get_logger().warn(f"[TURN DROPPED] reason=already_processing text=\"{raw_text}\"")
                    return
            self._is_processing = True
            self._processing_start_time = now

            captured_frame = None
            if self._latest_frame is not None and (now - self._latest_frame_time) < 4.0:
                captured_frame = self._latest_frame.copy()

        self.state_machine.transition_to(RobotState.THINKING)
        threading.Thread(target=self._process_llm, args=(raw_text, captured_frame, t_vad_start), daemon=True).start()

    def _social_filter_then_dispatch(self, raw_text: str, t_vad_start: float) -> None:
        """Sosyal barge-in değerlendirmesi — ROS callback'i DEĞİL, ayrı thread.

        _evaluate_social_barge_in sırayla 3 bulut LLM'ine kadar senkron çağrı
        yapabiliyor. Eskiden bu doğrudan _on_speech içinde çalışıyordu ve o süre
        boyunca konuşma callback'i bloke oluyordu.
        """
        try:
            if self._evaluate_social_barge_in(raw_text):
                self.get_logger().info("🎯 [Sosyal Fırsat]: Arka plan konuşmasına dâhil olunuyor!")
                self.session.activate_session(reason="social_barge_in")
                self.session.metadata["tts_engine"] = "edge-tts"
                self.state_machine.transition_to(RobotState.LISTENING)
                self.get_logger().info(f"[SESSION STARTED] reason=social_barge_in raw_text=\"{raw_text}\"")
                self._dispatch_turn(raw_text, t_vad_start)
            else:
                self.get_logger().info(f"[TURN DROPPED] reason=social_filter text=\"{raw_text}\"")
        except Exception as e:
            self.get_logger().error(f"❌ [AI] Sosyal filtre hatası: {e}")

    def _process_llm(self, user_text: str, frame: np.ndarray | None, t_turn_start: float):
        try:
            t_llm_start = time.monotonic()
            gate_latency_ms = (t_llm_start - t_turn_start) * 1000.0
            generation_id = int(t_turn_start * 1000) % 1000000

            self.get_logger().info(f"🗣️ [Siz]: \"{user_text}\"")
            self.memory.episodic.add_message("user", user_text)

            is_visual = self._is_visual_query(user_text)
            is_learning_obj = self._is_object_learning_query(user_text)
            is_learning_person = self._is_person_learning_query(user_text)
            is_identity = self._is_identity_query(user_text)
            is_weather, weather_city = self._is_weather_query(user_text)
            base64_img = frame_to_base64_jpeg(frame, max_dim=768) if frame is not None and (is_visual or is_learning_obj) else None
            persona = self.persona_engine.current_persona

            ack_sent = False
            ack_latency_ms = 0.0
            t_ack_start = time.monotonic()

            # Progressive Acknowledgement (THINKING_ACK) for operations >= 1.0s (< 300ms playback)
            if is_visual or is_learning_obj or is_weather:
                self.state_machine.transition_to(RobotState.THINKING_ACK)
                ack_type = "looking" if (is_visual or is_learning_obj) else "checking"
                if self.audio_resources:
                    ack_pcm = self.audio_resources.get_ack_pcm(ack_type)
                    self._play_local_ack(ack_pcm, generation_id=generation_id)
                    ack_sent = True
                    ack_latency_ms = (time.monotonic() - t_ack_start) * 1000.0
                    self.get_logger().info(f"⚡ [THINKING_ACK] Hızlı ara onay çalındı ({ack_latency_ms:.1f}ms): 'Bir saniye, bakıyorum.'")

            # 0. Active Multi-Turn Biometric Enrollment Dialog
            if self._enrollment_session.get("active"):
                self._enrollment_session["turn"] += 1
                turn_num = self._enrollment_session["turn"]
                cand_name = self._enrollment_session["name"]
                formal_name = self._enrollment_session["title"] or cand_name

                if getattr(self, "_last_speaker_embedding", None) is not None and len(self._last_speaker_embedding) > 0:
                    self._enrollment_session["embeddings"].append(self._last_speaker_embedding.copy())

                if frame is not None:
                    self._enrollment_session["last_frame"] = frame.copy()

                if turn_num == 1:
                    ans = f"Harika, ilk ses kaydınızı aldım (1/3). Lütfen ikinci cümlenizi söyleyin {cand_name}..."
                    self.get_logger().info(f"🎙️ [Biyometrik Kayıt ({cand_name})]: 1. ses kaydı başarıyla alındı.")
                elif turn_num == 2:
                    ans = f"Çok iyi gidiyoruz (2/3). Şimdi lütfen son cümlenizi söyleyin..."
                    self.get_logger().info(f"🎙️ [Biyometrik Kayıt ({cand_name})]: 2. ses kaydı başarıyla alındı.")
                else:
                    self.get_logger().info(f"🎙️ [Biyometrik Kayıt ({cand_name})]: 3. ses kaydı tamamlandı. Veritabanına kaydediliyor...")
                    embs = self._enrollment_session["embeddings"]
                    if not embs and getattr(self, "_last_speaker_embedding", None) is not None:
                        embs = [self._last_speaker_embedding]
                    if embs:
                        try:
                            from astro_audio.voice_recognizer import _get_engine as _get_spk_engine
                            spk_engine = _get_spk_engine()
                            if spk_engine:
                                spk_engine.add_person(cand_name, embs, replace=True)
                                spk_engine.save()
                                self.get_logger().info(f"✅ [WeSpeaker DB]: {cand_name} için {len(embs)} ses izi kaydedildi.")
                        except Exception as e:
                            self.get_logger().warn(f"Voice enrollment save notice: {e}")

                    saved_frame = self._enrollment_session.get("last_frame", frame)
                    if saved_frame is not None:
                        try:
                            from astro_vision.face_recognizer import FaceRecognizer
                            face_rec = FaceRecognizer()
                            face_rec.enroll_face(cand_name, saved_frame, title=formal_name)
                            self.get_logger().info(f"✅ [SFace DB]: {cand_name} için yüz modeli kaydedildi.")
                        except Exception as e:
                            self.get_logger().warn(f"Face enrollment save notice: {e}")

                    try:
                        self.memory.profile.add_known_person(cand_name, title=formal_name)
                    except Exception as e:
                        self.get_logger().warn(f"Profile enrollment save notice: {e}")

                    self._enrollment_session = self._new_enrollment_session()
                    self.state_machine.transition_to(RobotState.IDLE)
                    ans = f"Tebrikler {cand_name}! Sesinizi ve yüzünüzü başarıyla kaydettim. Artık seni her gördüğümde ve duyduğumda tanıyacağım!"

                self.get_logger().info(f"🤖 [Astro]: \"{ans}\"")
                self._publish_tts(ans)
                self._publish_emotion("playful")
                self.memory.episodic.add_message("assistant", ans)
                t_done = time.monotonic()
                total_turn_ms = (t_done - t_turn_start) * 1000.0
                self.session.latency_tracker.record_turn(gate_latency_ms, total_turn_ms - gate_latency_ms, total_turn_ms)
                return

            # 1. Live Weather Tool Direct Handling
            if is_weather:
                weather_res = self._execute_tool_call("get_live_weather", {"city": weather_city}, frame)
                clean_ans = clean_tts_text(weather_res)
                self.get_logger().info(f"🌤️ [Hava Durumu ({weather_city})]: \"{clean_ans}\"")
                self.get_logger().info(f"🤖 [Astro]: \"{clean_ans}\"")
                self._publish_tts(clean_ans)
                self._publish_emotion(persona)
                self.memory.episodic.add_message("assistant", clean_ans)
                t_done = time.monotonic()
                total_turn_ms = (t_done - t_turn_start) * 1000.0
                self.session.latency_tracker.record_turn(gate_latency_ms, total_turn_ms - gate_latency_ms, total_turn_ms)
                return

            # 2. Reminder & Alarm Direct Intent
            is_reminder, reminder_mins, reminder_topic = self._is_reminder_query(user_text)
            if is_reminder:
                user_name = self._add_reminder(reminder_mins, reminder_topic)
                friendly_name = user_name if user_name != "Misafir" else "dostum"
                topic_speech = reminder_topic
                topic_str = f"{topic_speech.lower()}ni" if "vakti" in topic_speech.lower() else f"{topic_speech} konusunu"

                if reminder_mins < 1.0:
                    secs = int(reminder_mins * 60.0)
                    ans = f"Tamamdır {friendly_name}! {secs} saniye sonra sana {topic_str} hatırlatacağım."
                elif int(reminder_mins) == 1:
                    ans = f"Tamamdır {friendly_name}! 1 dakika sonra sana {topic_str} hatırlatacağım."
                elif int(reminder_mins) >= 60 and int(reminder_mins) % 60 == 0:
                    hrs = int(reminder_mins // 60)
                    ans = f"Anlaşıldı {friendly_name}! {hrs} saat sonra sana {topic_str} hatırlatacağım."
                else:
                    ans = f"Anlaşıldı {friendly_name}! {int(reminder_mins)} dakika sonra sana {topic_str} hatırlatacağım."

                self.get_logger().info(f"🤖 [Astro]: \"{ans}\"")
                self._publish_tts(ans)
                self._publish_emotion(persona)
                self.memory.episodic.add_message("assistant", ans)
                t_done = time.monotonic()
                total_turn_ms = (t_done - t_turn_start) * 1000.0
                self.session.latency_tracker.record_turn(gate_latency_ms, total_turn_ms - gate_latency_ms, total_turn_ms)
                return

            # 3. Identity Query
            if is_identity and not is_learning_person:
                identity = self._get_active_biometric_identity()
                if identity.get("is_known"):
                    name = identity.get("name") or identity.get("formal_title") or "Baran"
                    formal = identity.get("formal_title") or name
                    source = identity.get("source", "biyometri")
                    src_tr = "sesinden" if source == "voice" else ("yüzünden" if source == "face" else "yüzünden ve sesinden")
                    if "baran" in str(name).lower() or "baran" in str(formal).lower():
                        ans = f"Tabii ki tanıdım! Seni {src_tr} hemen bildim; sen benim baş mühendisim ve geliştiricim Baran'sın!"
                    elif "erdoğan" in str(name).lower() or "erdoğan" in str(formal).lower():
                        ans = f"Elbette tanıdım Sayın Cumhurbaşkanım! Sizi {src_tr} tanıdım, saygılarımı ve hürmetlerimi sunarım efendim."
                    elif "vali" in identity.get("title", "").lower() or "karaömeroğlu" in str(name).lower():
                        ans = f"Elbette tanıdım Sayın Valim! Sizi {src_tr} tanıdım, hürmet ederim efendim."
                    else:
                        ans = f"Tabii ki tanıdım! Seni {src_tr} hemen bildim, sen {formal}'sın!"
                else:
                    ans = "Sesin veya yüzün henüz kayıtlı kişilerle tam eşleşmedi. İstersen 'Benim adım ... beni hafızana kaydet' diyerek yüzünü ve sesini bana tanıtabilirsin!"

                self.get_logger().info(f"🤖 [Astro]: \"{ans}\"")
                self._publish_tts(ans)
                self._publish_emotion(persona)
                self.memory.episodic.add_message("assistant", ans)
                t_done = time.monotonic()
                total_turn_ms = (t_done - t_turn_start) * 1000.0
                self.session.latency_tracker.record_turn(gate_latency_ms, total_turn_ms - gate_latency_ms, total_turn_ms)
                return

            # 4. Initiate Conversational Biometric Enrollment
            if is_learning_person:
                text_lower = user_text.lower()
                if "baran" in text_lower or "geliştirici" in text_lower:
                    cand_name = "Baran"
                    cand_title = "Baş Mühendis & Geliştirici"
                else:
                    m = re.search(r"(?i)\b(?:benim adım|adım)\s+([a-zA-ZçğıöşüÇĞİÖŞÜ]+)\b", user_text)
                    cand_name = m.group(1).strip().capitalize() if m else "Dostum"
                    cand_title = "Tanışılan Kişi"

                self._enrollment_session = {
                    "active": True,
                    "name": cand_name,
                    "title": cand_title,
                    "turn": 0,
                    "max_turns": 3,
                    "embeddings": [],
                    "last_frame": frame.copy() if frame is not None else None,
                    "start_time": time.monotonic()
                }
                if getattr(self, "_last_speaker_embedding", None) is not None:
                    self._enrollment_session["embeddings"].append(self._last_speaker_embedding.copy())

                self.state_machine.transition_to(RobotState.ENROLLING)
                ans = f"Memnuniyetle {cand_name}! Sesinizi ve yüzünüzü hafızama kaydetmek için lütfen bana doğru bakarak 3 kısa cümle söyleyin. Hazırsanız ilk cümlenizi dinliyorum..."
                self.get_logger().info(f"🤖 [Astro (Kayıt Başlatıldı)]: \"{ans}\"")
                self._publish_tts(ans)
                self._publish_emotion("playful")
                self.memory.episodic.add_message("assistant", ans)
                t_done = time.monotonic()
                total_turn_ms = (t_done - t_turn_start) * 1000.0
                self.session.latency_tracker.record_turn(gate_latency_ms, total_turn_ms - gate_latency_ms, total_turn_ms)
                return

            # 4. Object Learning Tool
            if is_learning_obj and base64_img is not None:
                self.get_logger().info("🔍 [Özel Nesne Tanıtımı]: Yeni nesne analiz ediliyor...")
                name_cand = re.sub(r"(?i)(bu benim|bunu öğren|bunu kaydet|bu nesne|buna bak bu)", "", user_text).strip(".:,!") or "Özel Eşya"
                tool_res = self._execute_tool_call("learn_custom_object", {"object_name": name_cand, "description": user_text}, frame)
                clean_ans = clean_tts_text(tool_res)
                self.get_logger().info(f"🤖 [Astro]: \"{clean_ans}\"")
                self._publish_tts(clean_ans)
                self._publish_emotion(persona)
                return

            # 5. Visual Query (Circuit-Breaker Guided & Zero-Silence Guaranteed)
            if is_visual:
                t_vis_start = time.monotonic()
                if base64_img is not None:
                    self.get_logger().info("👁️ [Vision]: OAK-D karesi analiz ediliyor...")
                    vision_ans = self._query_vision(user_text, base64_img)
                    clean_ans = clean_tts_text(vision_ans) if vision_ans else "Şu an görüntüyü analiz edemiyorum."
                else:
                    clean_ans = "Kameradan net göremedim, lütfen biraz daha yaklaştırır mısın?"

                vis_duration_ms = (time.monotonic() - t_vis_start) * 1000.0
                self.get_logger().info(f"🤖 [Astro]: \"{clean_ans}\"")
                self._publish_tts(clean_ans)
                self._publish_emotion(persona)
                self.memory.episodic.add_message("assistant", clean_ans)
                t_done = time.monotonic()
                total_turn_ms = (t_done - t_turn_start) * 1000.0

                # Authoritative Turn Telemetry for Vision
                rt_p_state = self.circuit_breaker.get_state('openai', 'openai_realtime').value if self.circuit_breaker else 'DISABLED'
                rt_conn_state = "CONNECTED" if self._realtime_ws_connected else "DISCONNECTED"
                rt_sess_state = "READY" if self._realtime_session_ready else "NOT_READY"
                req_provider = "openai_realtime" if (self._realtime_ws_connected and self._realtime_session_ready) else "edge_tts"
                fb_reason = "realtime_quota_exhausted" if (self.circuit_breaker and self.circuit_breaker.is_exhausted('openai')) else ("realtime_unavailable" if not self._realtime_ws_connected else "none")

                self.get_logger().info(
                    f"\n[ASTRO TURN]\n"
                    f"generation_id={generation_id}\n"
                    f"interaction_state={self.state_machine.current_state.value}\n"
                    f"realtime_provider_state={rt_p_state}\n"
                    f"realtime_connection_state={rt_conn_state}\n"
                    f"realtime_session_state={rt_sess_state}\n"
                    f"realtime_audio_received=false\n"
                    f"requested_provider={req_provider}\n"
                    f"actual_provider=edge_tts\n"
                    f"fallback_reason={fb_reason}\n"
                    f"stt_provider=groq/whisper-large-v3\n"
                    f"stt_state={self.circuit_breaker.get_state('groq', 'groq_stt').value if self.circuit_breaker else 'AVAILABLE'}\n"
                    f"stt_failure=none\n"
                    f"llm_provider=vision_pipeline\n"
                    f"llm_state=AVAILABLE\n"
                    f"llm_failure=none\n"
                    f"vision_provider=multimodal_vision\n"
                    f"vision_state=AVAILABLE\n"
                    f"vision_failure=none\n"
                    f"tts_provider=edge_tts\n"
                    f"tts_state=AVAILABLE\n"
                    f"tts_failure=none\n"
                    f"fallback_chain=['vision_pipeline']\n"
                    f"ack_sent={ack_sent}\n"
                    f"ack_latency_ms={ack_latency_ms:.1f}\n"
                    f"stt_ms={gate_latency_ms:.1f}\n"
                    f"llm_ms=0.0\n"
                    f"vision_ms={vis_duration_ms:.1f}\n"
                    f"tts_ms={total_turn_ms - vis_duration_ms:.1f}\n"
                    f"playback_ms={total_turn_ms:.1f}\n"
                    f"total_user_to_audio_ms={total_turn_ms:.1f}\n"
                    f"provider_attempts=1"
                )
                return

            identity = self._get_active_biometric_identity()
            active_name = identity.get("name", "Baran") if identity.get("is_known") else "Baran"
            threading.Thread(target=self._async_extract_user_facts, args=(user_text, active_name), daemon=True).start()

            # 6. Memory Recall Query Direct Handling
            is_memory_q, memory_ans = self._handle_memory_recall_query(user_text, identity)
            if is_memory_q:
                clean_ans = clean_tts_text(memory_ans)
                self.get_logger().info(f"🧠 [Bellek Çağırma ({active_name})]: \"{clean_ans}\"")
                self.get_logger().info(f"🤖 [Astro]: \"{clean_ans}\"")
                self._publish_tts(clean_ans)
                self._publish_emotion(persona)
                self.memory.episodic.add_message("assistant", clean_ans)
                t_done = time.monotonic()
                total_turn_ms = (t_done - t_turn_start) * 1000.0
                self.session.latency_tracker.record_turn(gate_latency_ms, total_turn_ms - gate_latency_ms, total_turn_ms)
                return

            # 7. Conversational LLM with Circuit-Breaker Hierarchy
            perception_prefix = self.persona_engine.build_user_context_prefix(
                self._person_detected, self._looking_at_robot,
                self._user_distance, self._user_emotion, self._speaker_gender,
                recognized_person=identity
            )
            system_prompt = self.persona_engine.build_system_prompt(
                memory_context=self.memory.get_prompt_context(recognized_person=identity),
                recognized_person=identity
            )
            messages = [{"role": "system", "content": system_prompt}]
            for m in self.memory.episodic.get_messages():
                messages.append({"role": m["role"], "content": m["content"]})

            if perception_prefix and messages and messages[-1]["role"] == "user":
                messages[-1]["content"] = perception_prefix + messages[-1]["content"]

            full_text = ""
            first_token_time = None
            llm_provider = "none"
            llm_state = "AVAILABLE"
            llm_failure = "none"
            fallback_chain = []
            provider_attempts = 0

            # Step 1: Check OpenAI Circuit Breaker Status
            openai_available = self.circuit_breaker.is_available("openai", sub_provider="openai_rest") if self.circuit_breaker else True
            if not openai_available:
                fallback_chain.append("openai(exhausted)")
                self.get_logger().info("⚡ [LLM ROUTE] provider=groq reason=openai_exhausted")

            # Step 2: Primary Groq LPU Ultra-Fast Models (if available & not in cooldown)
            groq_available = self.circuit_breaker.is_available("groq", sub_provider="groq_llm") if self.circuit_breaker else True
            if self._provider_enabled("groq") and self._groq and self._active_groq_models and groq_available:
                for m in self._active_groq_models[:3]:
                    if not self.circuit_breaker.is_available("groq", model_id=m):
                        continue
                    provider_attempts += 1
                    self.get_logger().info(f"[LLM ROUTE] provider=groq model={m} reason={self._route_reason('groq')}")
                    try:
                        stream_resp = self._groq.chat.completions.create(
                            messages=messages,
                            model=m,
                            temperature=self._temperature,
                            max_tokens=min(150, self._max_tokens),
                            presence_penalty=0.5,
                            frequency_penalty=0.5,
                            stream=True,
                        )
                        for chunk in stream_resp:
                            delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
                            if not delta:
                                continue
                            if first_token_time is None:
                                first_token_time = time.monotonic()
                                self.state_machine.transition_to(RobotState.SPEAKING)
                            full_text += delta
                            if re.search(r'([a-zA-ZçğıöşüÇĞİÖŞÜ]{1,4})\1{5,}', full_text):
                                break
                        if full_text:
                            llm_provider = f"groq/{m}"
                            fallback_chain.append("groq")
                            if self.circuit_breaker:
                                self.circuit_breaker.record_success("groq", sub_provider="groq_llm", model_id=m)
                            break
                    except Exception as stream_err:
                        err_str = str(stream_err).lower()
                        self.get_logger().debug(f"Groq stream model {m} notice: {stream_err}")
                        if "429" in err_str or "rate limit" in err_str or "rpm" in err_str:
                            if self.circuit_breaker:
                                self.circuit_breaker.record_error("groq", sub_provider="groq_llm", error_class=RequestErrorClass.RATE_LIMITED, error_msg=err_str)
                            fallback_chain.append("groq(429_cooldown)")
                            self.get_logger().info("⚡ [LLM ROUTE] provider=gemini reason=groq_cooldown")
                            break
                        elif "404" in err_str or "unsupported" in err_str:
                            if self.circuit_breaker:
                                self.circuit_breaker.record_error("groq", sub_provider="groq_llm", error_class=RequestErrorClass.MODEL_UNAVAILABLE, error_msg=err_str, model_id=m)
                            fallback_chain.append(f"groq({m}_unavailable)")
                        else:
                            fallback_chain.append(f"groq({m}_err)")
                        full_text = ""
                        first_token_time = None
                        continue
            elif not groq_available:
                fallback_chain.append("groq(cooldown)")
                self.get_logger().info("⚡ [LLM ROUTE] provider=gemini reason=groq_cooldown")

            # Step 3: Secondary Google Gemini REST Fallback
            gemini_available = self.circuit_breaker.is_available("gemini", sub_provider="gemini_text") if self.circuit_breaker else True
            if self._provider_enabled("gemini") and not full_text and self._ai_api_key and gemini_available:
                provider_attempts += 1
                self.get_logger().info(f"[LLM ROUTE] provider=gemini model=gemini-2.0-flash reason={self._route_reason('gemini')}")
                try:
                    gemini_text = self._query_gemini_text_rest(system_prompt, user_text, self.memory.episodic.get_messages())
                    if gemini_text and not is_canned_refusal(gemini_text):
                        full_text = gemini_text
                        llm_provider = "gemini/gemini-2.0-flash"
                        fallback_chain.append("gemini")
                        if self.circuit_breaker:
                            self.circuit_breaker.record_success("gemini", sub_provider="gemini_text")
                except Exception as g_err:
                    if self.circuit_breaker:
                        self.circuit_breaker.record_error("gemini", sub_provider="gemini_text", error_class=RequestErrorClass.SERVER_ERROR, error_msg=str(g_err))
                    fallback_chain.append("gemini(error)")

            # Step 4: Tertiary Emergency Fallback: OpenAI Client (Only if healthy & available)
            if self._provider_enabled("openai") and not full_text and self._openai and self.circuit_breaker and self.circuit_breaker.is_available("openai", sub_provider="openai_rest"):
                provider_attempts += 1
                self.get_logger().info(f"[LLM ROUTE] provider=openai model={self._openai_model} reason={self._route_reason('openai')}")
                try:
                    stream_resp = self._openai.chat.completions.create(
                        messages=messages,
                        model=self._openai_model,
                        temperature=self._temperature,
                        max_tokens=min(150, self._max_tokens),
                        presence_penalty=0.5,
                        frequency_penalty=0.5,
                        stream=True,
                    )
                    for chunk in stream_resp:
                        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
                        if not delta:
                            continue
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                            self.state_machine.transition_to(RobotState.SPEAKING)
                        full_text += delta
                    if full_text:
                        llm_provider = f"openai/{self._openai_model}"
                        fallback_chain.append("openai")
                        self.circuit_breaker.record_success("openai", sub_provider="openai_rest")
                except Exception as oai_err:
                    err_str = str(oai_err).lower()
                    err_class = self.circuit_breaker.classify_error(oai_err, err_msg=err_str)
                    self.circuit_breaker.record_error("openai", sub_provider="openai_rest", error_class=err_class, error_msg=err_str)
                    self.get_logger().warn(f"⚠️ [OpenAI GPT Fallback Hatası] ({oai_err})")
                    fallback_chain.append(f"openai({err_class.value})")
                    full_text = ""
                    first_token_time = None

            clean_full = clean_tts_text(full_text)

            # Step 5: Zero-Silence Guarantee (Emergency Local Response)
            if not clean_full or len(clean_full) < 2 or is_canned_refusal(clean_full):
                persona_recovery = {
                    "flirt": "Harika! Bütün dikkatimle seninleyim, söyle bakalım ne diyorsun?",
                    "charming": "Harika! Bütün dikkatimle seninleyim, söyle bakalım ne diyorsun?",
                    "rude": "Ne diyon birader, ne geveliyorsun?",
                    "angry": "Bana böyle boş yapma, sadede gel!",
                    "sarcastic": "Aman ne derin bir konu, cevabı bulmaya işlemcim yetmedi doğrusu!",
                    "formal": "Buyrun efendim, sizi dikkatle dinlemeye devam ediyorum.",
                    "emotional": "Bazen hisleri tarif etmek zordur... Seni dinliyorum.",
                    "playful": "Haha çok ilginçsin! Seni dinliyorum, devam et bakalım!"
                }
                clean_full = persona_recovery.get(persona, "Seni dinliyorum, devam et bakalım!")
                llm_provider = "emergency_local_persona"
                fallback_chain.append("emergency_local_persona")

            self.get_logger().info(f"[LLM RESPONSE] provider={llm_provider} length={len(clean_full)} text=\"{clean_full}\"")
            self.cloud_mgr.record_llm_success()
            self.get_logger().info(f"🤖 [Astro]: \"{clean_full}\"")
            self.memory.episodic.add_message("assistant", clean_full)
            self._publish_tts(clean_full)
            self._publish_emotion(persona)

            # Latency Benchmarking & Authoritative Turn Telemetry
            t_done = time.monotonic()
            llm_first_ms = ((first_token_time or t_done) - t_llm_start) * 1000.0
            total_turn_ms = (t_done - t_turn_start) * 1000.0
            self.session.latency_tracker.record_turn(gate_latency_ms, llm_first_ms, total_turn_ms)

            rt_p_state = self.circuit_breaker.get_state('openai', 'openai_realtime').value if self.circuit_breaker else 'DISABLED'
            rt_conn_state = "CONNECTED" if self._realtime_ws_connected else "DISCONNECTED"
            rt_sess_state = "READY" if self._realtime_session_ready else "NOT_READY"
            req_provider = "openai_realtime" if (self._realtime_ws_connected and self._realtime_session_ready) else "edge_tts"
            fb_reason = "realtime_quota_exhausted" if (self.circuit_breaker and self.circuit_breaker.is_exhausted('openai')) else ("realtime_unavailable" if not self._realtime_ws_connected else "none")

            self.get_logger().info(
                f"\n[ASTRO TURN]\n"
                f"generation_id={generation_id}\n"
                f"interaction_state={self.state_machine.current_state.value}\n"
                f"realtime_provider_state={rt_p_state}\n"
                f"realtime_connection_state={rt_conn_state}\n"
                f"realtime_session_state={rt_sess_state}\n"
                f"realtime_audio_received=false\n"
                f"requested_provider={req_provider}\n"
                f"actual_provider=edge_tts\n"
                f"fallback_reason={fb_reason}\n"
                f"stt_provider=groq/whisper-large-v3\n"
                f"stt_state={self.circuit_breaker.get_state('groq', 'groq_stt').value if self.circuit_breaker else 'AVAILABLE'}\n"
                f"stt_failure=none\n"
                f"llm_provider={llm_provider}\n"
                f"llm_state={llm_state}\n"
                f"llm_failure={llm_failure}\n"
                f"vision_provider=none\n"
                f"vision_state=none\n"
                f"vision_failure=none\n"
                f"tts_provider=edge_tts\n"
                f"tts_state={self.circuit_breaker.get_state('edge_tts').value if self.circuit_breaker else 'AVAILABLE'}\n"
                f"tts_failure=none\n"
                f"fallback_chain={fallback_chain}\n"
                f"ack_sent={ack_sent}\n"
                f"ack_latency_ms={ack_latency_ms:.1f}\n"
                f"stt_ms={gate_latency_ms:.1f}\n"
                f"llm_ms={llm_first_ms:.1f}\n"
                f"vision_ms=0.0\n"
                f"tts_ms={total_turn_ms - llm_first_ms:.1f}\n"
                f"playback_ms={total_turn_ms:.1f}\n"
                f"total_user_to_audio_ms={total_turn_ms:.1f}\n"
                f"provider_attempts={provider_attempts}"
            )

        except Exception as e:
            self.get_logger().error(f"❌ [AI] LLM İşleme Hatası: {e}")
            emergency_reply = "Şu an bağlantımda sorun var, tekrar kontrol ediyorum."
            self._publish_tts(emergency_reply)
        finally:
            with self._lock:
                self._is_processing = False

    def _evaluate_social_barge_in(self, raw_text: str) -> bool:
        """Evaluates whether Astro should autonomously join background conversation (Barge-in)."""
        prompt = (
            "Sen Astro'sun, sempatik ve akıllı bir sosyal robotsun. "
            f"Odadaki insanlar kendi aralarında şunu konuşuyor: '{raw_text}'. "
            "Bu konuşmada sana sorulmuş bir soru var mı, veya doğrudan yardım edebileceğin bariz bir bilgi/durum var mı? "
            "Sadece tek kelime EVET veya HAYIR yaz."
        )

        # 1. Try Groq with discovered active chat models (if healthy)
        groq_ok = self.circuit_breaker.is_available("groq", sub_provider="groq_llm") if self.circuit_breaker else True
        if self._groq and self._active_groq_models and groq_ok:
            for g_model in self._active_groq_models[:3]:
                if not self.circuit_breaker.is_available("groq", model_id=g_model):
                    continue
                try:
                    res = self._groq.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=g_model,
                        temperature=0.0,
                        max_tokens=10,
                        timeout=1.5
                    )
                    ans = res.choices[0].message.content.strip().lower()
                    return "evet" in ans
                except Exception as ge:
                    self.get_logger().debug(f"Groq social filter ({g_model}) failed: {ge}")
                    continue

        # 2. Fallback to OpenAI model ONLY if circuit breaker allows
        openai_ok = self.circuit_breaker.is_available("openai", sub_provider="openai_rest") if self.circuit_breaker else True
        if self._openai and openai_ok:
            try:
                res = self._openai.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self._openai_model,
                    temperature=0.0,
                    max_tokens=10,
                    timeout=1.5
                )
                ans = res.choices[0].message.content.strip().lower()
                return "evet" in ans
            except Exception as oe:
                self.get_logger().debug(f"OpenAI social filter failed: {oe}")

        # 3. Fallback to Gemini REST if available
        gemini_ok = self.circuit_breaker.is_available("gemini", sub_provider="gemini_text") if self.circuit_breaker else True
        if self._ai_api_key and gemini_ok:
            try:
                g_res = self._query_gemini_text_rest(
                    system_instruction="Sadece tek kelime EVET veya HAYIR yaz.",
                    user_text=prompt,
                    history_messages=[]
                )
                if g_res:
                    return "evet" in g_res.lower()
            except Exception as _exc:
                self.get_logger().debug(f"_evaluate_social_barge_in: yok sayılan hata ({_exc})")

        return False

    def _query_vision(self, prompt: str, base64_image: str) -> str:
        """Queries Vision models strictly based on GlobalProviderCircuitBreaker capabilities."""
        persona = self.persona_engine.current_persona
        system_instruction = (
            f"Sen Astro adında {persona} karakterli akıllı ve sempatik bir sosyal robotsun. "
            "Sana kullanıcının tam karşısındaki OAK-D kamerasından anlık bir fotoğraf karesi iletilmiştir. "
            "Görüntüyü dikkatle incele: kullanıcının üzerindeki kıyafetleri (renk, tişört/gömlek/ceket), "
            "elinde tuttuğu nesneleri, yaptığı hareketleri ve odayı detaylarıyla analiz et. "
            "Kullanıcının sorusuna doğrudan fotoğrafta gördüklerini anlatacak şekilde, kendi tarzınla "
            "samimi ve net 1-2 Türkçe cümleyle cevap ver."
        )

        # 1. Try Primary OpenAI Vision Client (gpt-4o-mini / gpt-4o) ONLY if circuit breaker is AVAILABLE
        openai_vision_ok = self.circuit_breaker.is_available("openai", sub_provider="openai_vision") if self.circuit_breaker else True
        if self._openai and openai_vision_ok:
            for m_cand in [self._vision_model, "gpt-4o-mini", "gpt-4o"]:
                try:
                    response = self._openai.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "auto"
                                }}
                            ]}
                        ],
                        model=m_cand,
                        temperature=0.2,
                        max_tokens=300
                    )
                    raw = response.choices[0].message.content.strip()
                    clean_ans = clean_tts_text(raw)
                    if clean_ans and len(clean_ans) >= 3:
                        if self.circuit_breaker:
                            self.circuit_breaker.record_success("openai", sub_provider="openai_vision")
                        self.get_logger().info(f"✨ [OpenAI Vision] Görsel başarıyla yanıtlandı ({m_cand}): '{clean_ans}'")
                        return clean_ans
                except Exception as e:
                    err_str = str(e).lower()
                    err_class = self.circuit_breaker.classify_error(e, err_msg=err_str) if self.circuit_breaker else RequestErrorClass.SERVER_ERROR
                    if self.circuit_breaker:
                        self.circuit_breaker.record_error("openai", sub_provider="openai_vision", error_class=err_class, error_msg=err_str)
                    self.get_logger().warn(f"⚠️ [OpenAI Vision ({m_cand}) Hatası]: {e}")
                    break
        elif not openai_vision_ok:
            self.get_logger().info("🔍 [VISION ROUTE] provider=groq|gemini reason=openai_exhausted")

        # 2. Try Secondary Groq Vision (llama-3.2-11b-vision-preview, llama-3.2-90b-vision-preview)
        groq_vision_ok = self.circuit_breaker.is_available("groq", sub_provider="groq_vision") if self.circuit_breaker else True
        if self._groq and groq_vision_ok:
            for gv_model in ["llama-3.2-11b-vision-preview", "llama-3.2-90b-vision-preview"]:
                if not self.circuit_breaker.is_available("groq", model_id=gv_model):
                    continue
                try:
                    chat_completion = self._groq.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }}
                            ]}
                        ],
                        model=gv_model,
                        temperature=0.2,
                        max_tokens=300
                    )
                    raw = chat_completion.choices[0].message.content.strip()
                    clean_ans = clean_tts_text(raw)
                    if clean_ans and len(clean_ans) >= 3:
                        if self.circuit_breaker:
                            self.circuit_breaker.record_success("groq", sub_provider="groq_vision", model_id=gv_model)
                        self.get_logger().info(f"✨ [Groq Vision] Görsel başarıyla yanıtlandı ({gv_model}): '{clean_ans}'")
                        return clean_ans
                except Exception as ge:
                    err_str = str(ge).lower()
                    err_class = self.circuit_breaker.classify_error(ge, err_msg=err_str) if self.circuit_breaker else RequestErrorClass.SERVER_ERROR
                    if self.circuit_breaker:
                        self.circuit_breaker.record_error("groq", sub_provider="groq_vision", error_class=err_class, error_msg=err_str, model_id=gv_model)
                    self.get_logger().warn(f"⚠️ [Groq Vision ({gv_model}) Hatası]: {ge}")

        # 3. Try Tertiary Google Gemini REST Endpoint (verified models: gemini-2.0-flash, gemini-1.5-flash)
        gemini_vision_ok = self.circuit_breaker.is_available("gemini", sub_provider="gemini_vision") if self.circuit_breaker else True
        if self._ai_api_key and gemini_vision_ok:
            gemini_vision_models = ["gemini-2.0-flash", "gemini-1.5-flash"]
            for g_model in gemini_vision_models:
                if not self.circuit_breaker.is_available("gemini", model_id=g_model):
                    continue
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent?key={self._ai_api_key}"
                    payload = {
                        "contents": [{
                            "parts": [
                                {"text": f"{system_instruction}\n\nKullanıcı: {prompt}"},
                                {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
                            ]
                        }],
                        "generation_config": {
                            "temperature": 0.2,
                            "max_output_tokens": 512
                        }
                    }
                    data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json", "User-Agent": "Astro-V1-SocialRobot/2.0"})
                    with urllib.request.urlopen(req, timeout=5.0) as resp:
                        res_json = json.loads(resp.read().decode("utf-8"))
                        text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                        clean_ans = clean_tts_text(text)
                        if clean_ans and len(clean_ans) >= 3:
                            if self.circuit_breaker:
                                self.circuit_breaker.record_success("gemini", sub_provider="gemini_vision", model_id=g_model)
                            self.get_logger().info(f"✨ [Gemini Vision REST] Görsel başarıyla yanıtlandı ({g_model}): '{clean_ans}'")
                            return clean_ans
                except Exception as e:
                    err_str = str(e).lower()
                    err_class = self.circuit_breaker.classify_error(e, err_msg=err_str) if self.circuit_breaker else RequestErrorClass.SERVER_ERROR
                    if self.circuit_breaker:
                        self.circuit_breaker.record_error("gemini", sub_provider="gemini_vision", error_class=err_class, error_msg=err_str, model_id=g_model)
                    self.get_logger().warn(f"⚠️ [Gemini REST ({g_model}) Hatası]: {e}")

        # Zero-Silence Emergency Spoken Fallback
        fallback_msg = "Şu an görüntüyü analiz edemiyorum."
        self.get_logger().warn(f"⚠️ [Vision Failure] Tüm vision modelleri başarısız oldu. Acil durum sesli yanıtı: '{fallback_msg}'")
        return fallback_msg

    def _query_gemini_text_rest(self, system_instruction: str, user_text: str, history_messages: List[Dict[str, Any]]) -> Optional[str]:
        """Zero-dependency direct Google Gemini REST text conversation engine."""
        if not self._ai_api_key:
            return None

        gemini_timeout_s = float(os.getenv("GEMINI_TIMEOUT_S", "12"))
        gemini_text_models = [
            m.strip() for m in os.getenv(
                "GEMINI_TEXT_MODELS",
                "gemini-2.0-flash,gemini-1.5-flash,gemini-1.5-pro",
            ).split(",") if m.strip()
        ]
        for g_model in gemini_text_models:
            if self.circuit_breaker and not self.circuit_breaker.is_available("gemini", model_id=g_model):
                continue
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent?key={self._ai_api_key}"
                contents = []
                for msg in history_messages[-6:]:
                    r = "user" if msg.get("role") == "user" else "model"
                    contents.append({"role": r, "parts": [{"text": msg.get("content", "")}]})

                if not contents or contents[-1]["role"] != "user":
                    contents.append({"role": "user", "parts": [{"text": user_text}]})

                payload = {
                    "system_instruction": {"parts": [{"text": system_instruction}]},
                    "contents": contents,
                    "generationConfig": {
                        "temperature": 0.5,
                        "maxOutputTokens": 300
                    }
                }
                data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=data_bytes,
                    headers={"Content-Type": "application/json", "User-Agent": "Astro-V1-SocialRobot/2.0"},
                )
                with urllib.request.urlopen(req, timeout=gemini_timeout_s) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                    clean = clean_tts_text(text)
                    if clean and len(clean) >= 2:
                        if self.circuit_breaker:
                            self.circuit_breaker.record_success("gemini", sub_provider="gemini_text", model_id=g_model)
                        return clean
            except urllib.error.HTTPError as he:
                err_str = str(he).lower()
                err_class = self.circuit_breaker.classify_error(he, status_code=he.code, err_msg=err_str) if self.circuit_breaker else RequestErrorClass.SERVER_ERROR
                if self.circuit_breaker:
                    self.circuit_breaker.record_error("gemini", sub_provider="gemini_text", error_class=err_class, error_msg=err_str, model_id=g_model)
                self.get_logger().debug(f"Gemini REST HTTP {he.code} on {g_model}: {he.reason}")
                continue
            except Exception as e:
                self.get_logger().debug(f"Gemini REST error on {g_model}: {e}")
                continue
        return None



    def _format_turkish_weather(self, city: str, raw_weather: str) -> str:
        temp_match = re.search(r'([+-]?\d+)\s*°?C?', raw_weather)
        temp_str = temp_match.group(1).lstrip('+') if temp_match else ''

        cond_raw = re.sub(r'[+-]?\d+\s*°?C?', '', raw_weather).strip(' ,:;+°C')
        cond_lower = cond_raw.lower()

        condition_map = {
            'sunny': 'güneşli ve açık',
            'clear': 'açık ve ferah',
            'partly cloudy': 'parçalı bulutlu',
            'cloudy': 'bulutlu',
            'overcast': 'kapalı',
            'patchy rain nearby': 'parçalı yağmurlu',
            'patchy light rain': 'hafif yağmurlu',
            'light rain': 'hafif yağmurlu',
            'moderate rain': 'yağmurlu',
            'heavy rain': 'sağanak yağışlı',
            'rain': 'yağmurlu',
            'light rain shower': 'sağanak yağışlı',
            'patchy snow': 'yer yer kar yağışlı',
            'light snow': 'hafif kar yağışlı',
            'snow': 'kar yağışlı',
            'heavy snow': 'yoğun kar yağışlı',
            'fog': 'sisli',
            'mist': 'puslu',
            'thundery outbreaks in nearby': 'gök gürültülü sağanak yağışlı',
            'thunderstorm': 'gök gürültülü fırtınalı'
        }

        cond_tr = condition_map.get(cond_lower)
        if not cond_tr:
            for k, v in condition_map.items():
                if k in cond_lower:
                    cond_tr = v
                    break
        if not cond_tr:
            cond_tr = cond_raw if cond_raw else 'açık'

        city_clean = city.strip().capitalize()
        last_vowel = [c for c in city_clean.lower() if c in 'aıoueiöü']
        is_front = last_vowel[-1] in 'eiöü' if last_vowel else False
        is_hard = city_clean.lower()[-1] in 'fstkçşhp'
        suffix = ("'te" if is_front else "'ta") if is_hard else ("'de" if is_front else "'da")
        city_with_suffix = f"{city_clean}{suffix}"

        if temp_str:
            return f"{city_with_suffix} hava şu an {cond_tr} ve {temp_str} derece."
        return f"{city_with_suffix} hava şu an {cond_tr}."

    def _execute_tool_call(self, tool_name: str, arguments: dict, frame: np.ndarray | None) -> str:
        if tool_name == "get_live_weather":
            city = arguments.get("city", "Istanbul").strip()
            try:
                url = f"https://wttr.in/{urllib.parse.quote(city)}?format=%C+%t&lang=tr"
                req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
                with urllib.request.urlopen(req, timeout=4.0) as resp:
                    weather_text = resp.read().decode("utf-8").strip()
                return self._format_turkish_weather(city, weather_text)
            except Exception:
                return f"{city} için şu an hava durumu bilgisine ulaşamadım."

        elif tool_name == "set_timer_alarm":
            mins = float(arguments.get("minutes", 5.0))
            rem_text = arguments.get("reminder_text", "Zaman doldu!")
            self._add_reminder(mins, rem_text)
            return f"{int(mins)} dakika sonraya hatırlatıcıyı kurdum! Vakti geldiğinde sana sesleneceğim."

        elif tool_name == "learn_custom_object":
            obj_name = arguments.get("object_name", "Özel Eşya")
            desc = arguments.get("description", "")
            self.memory.profile.add_learned_object(obj_name, desc)
            return f"'{obj_name}' nesnesini hafızama kaydettim! Artık gördüğümde tanıyacağım."

        elif tool_name == "enroll_person_profile":
            name = arguments.get("name", "Misafir").strip()
            title = arguments.get("title", "Tanışılan Kişi").strip()
            self.memory.profile.add_known_person(name, title)
            return f"Tanıştığımıza çok memnun oldum {name}! Profilini hafızama kaydettim, artık seni her gördüğümde tanıyacağım."

        return "Eylem tamamlandı."

    def _check_reminders(self):
        """Active scheduler loop ticking every second to trigger due reminders from persistent memory."""
        due_reminders = self.memory.profile.get_and_pop_due_reminders()

        for r in due_reminders:
            txt = r.get("reminder_text", "")
            name = r.get("user_name", self._default_user_name)
            if "çay" in txt.lower():
                msg = f"Hey {name}! Hatırlatmamı istediğin vakit geldi: Çay içme zamanı! Sıcak bir çay iyi gelir, afiyet olsun."
            elif "su" in txt.lower():
                msg = f"Hey {name}! Su içme vaktin geldi, sağlığın için bir bardak su içmeyi unutma."
            else:
                msg = f"Hey {name}! Hatırlatmamı istediğin vakit geldi: {txt}!"

            self.get_logger().info(f"⏰ [Hatırlatıcı Çaldı]: \"{msg}\"")
            self.session.activate_session(reason="reminder")
            self.state_machine.transition_to(RobotState.SPEAKING)
            self._publish_gesture("nod")
            self._publish_emotion("playful")
            self._publish_tts(msg)

    def _add_reminder(self, mins: float, topic: str) -> str:
        """Shared helper: creates a persistent reminder entry and returns the resolved user name."""
        mins = max(0.0, mins)  # Guard against negative durations
        target_t = time.time() + (mins * 60.0)
        identity = self._get_active_biometric_identity()
        user_name = identity.get("name") if identity.get("is_known") else self._default_user_name
        self.memory.profile.add_active_reminder(target_t, topic, user_name)
        return user_name

    def _is_reminder_query(self, text: str) -> Tuple[bool, float, str]:
        text_l = text.lower()
        reminder_triggers = ["hatırlat", "alarm kur", "zamanlayıcı kur", "haber ver", "uyar", "bana söyle", "hatırlatıcı"]
        if not any(w in text_l for w in reminder_triggers):
            return False, 0.0, ""

        # 1. Parse Duration (Minutes/Seconds/Hours with both digits and Turkish word numbers)
        num_map = {
            "yarım": 0.5, "yarim": 0.5, "buçuk": 0.5, "çeyrek": 0.25, "ceyrek": 0.25,
            "bir": 1.0, "1": 1.0, "iki": 2.0, "2": 2.0, "üç": 3.0, "uc": 3.0, "3": 3.0,
            "dört": 4.0, "dort": 4.0, "4": 4.0, "beş": 5.0, "bes": 5.0, "5": 5.0,
            "altı": 6.0, "alti": 6.0, "6": 6.0, "yedi": 7.0, "7": 7.0, "sekiz": 8.0, "8": 8.0,
            "dokuz": 9.0, "9": 9.0, "on": 10.0, "10": 10.0, "on beş": 15.0, "15": 15.0,
            "yirmi": 20.0, "20": 20.0, "yirmi beş": 25.0, "25": 25.0,
            "otuz": 30.0, "30": 30.0, "kırk": 40.0, "kirk": 40.0, "40": 40.0,
            "elli": 50.0, "50": 50.0, "altmış": 60.0, "altmis": 60.0, "60": 60.0
        }

        mins = 1.0
        time_pattern = r'(\d+|yarım|yarim|çeyrek|ceyrek|on\s+beş|on\s+iki|bir|iki|üç|uc|dört|dort|beş|bes|altı|alti|yedi|sekiz|dokuz|on|yirmi|otuz|kırk|elli|altmış)\s*(dakika|dk|saniye|sn|saat|hour|min|sec)'
        m = re.search(time_pattern, text_l)

        if m:
            val_str = m.group(1).strip()
            unit_str = m.group(2).strip()
            val = float(num_map.get(val_str, float(val_str) if val_str.isdigit() else 1.0))

            if any(u in unit_str for u in ["saniye", "sn", "sec"]):
                mins = val / 60.0
            elif any(u in unit_str for u in ["saat", "hour"]):
                mins = val * 60.0
            else:
                mins = val
        else:
            if "saniye sonra" in text_l:
                mins = 0.5
            elif "saat sonra" in text_l:
                mins = 60.0
            elif "dakika sonra" in text_l:
                mins = 1.0
            else:
                mins = 5.0

        # 2. Extract Topic Cleanly
        # Remove conversational chatter / greetings / polite phrases
        clean = re.sub(r'(?i)\b(iyiyim|ben de iyiyim|harikayım|süperim|ben|de|teşekkür\s*ederim|teşekkürler|sağ\s*ol|sağol|merhaba|selam|günaydın|lütfen|bana|hey\s*astro|astro)\b', '', text)
        
        # Remove time expressions
        clean = re.sub(r'(?i)\b(\d+|yarım|yarim|çeyrek|ceyrek|on\s+beş|on\s+iki|bir|iki|üç|uc|dört|dort|beş|bes|altı|alti|yedi|sekiz|dokuz|on|yirmi|otuz|kırk|elli|altmış)\s*(dakika|dk|saniye|sn|saat)\s*(sonra)?\b', '', clean)
        clean = re.sub(r'(?i)\b(dakika|saniye|saat)\s*sonra\b', '', clean)

        # Remove trigger suffixes and verbs
        clean = re.sub(r'(?i)\b(hatırlatabilir\s*misin|hatırlatır\s*mısın|hatırlatırsa[nm]|hatırlat|alarm\s*kur|zamanlayıcı\s*kur|haber\s*ver|uyar|söyler\s*misin|kurar\s*mısın)\b', '', clean)
        clean = re.sub(r'(?i)\b(içeceğim|içecegim|içmem\s*lazım|yapacağım|yapmam\s*gerek|gideceğim|alacağım|kapatacağım|edeceğim|edecegim|olacağım)\b', '', clean)
        clean = re.sub(r'[^\w\s]', '', clean).strip()
        clean = " ".join(clean.split())

        # Fallback to smart semantic keywords
        if not clean or len(clean) < 3:
            if "çay" in text_l:
                topic = "Çay içme vakti"
            elif "kahve" in text_l:
                topic = "Kahve içme vakti"
            elif "su" in text_l:
                topic = "Su içme vakti"
            elif "ilaç" in text_l:
                topic = "İlaç alma vakti"
            elif "yemek" in text_l or "fırın" in text_l:
                topic = "Yemek vakti"
            elif "toplantı" in text_l:
                topic = "Toplantı vakti"
            elif "ders" in text_l or "çalış" in text_l:
                topic = "Ders çalışma vakti"
            else:
                topic = "Hatırlatma"
        else:
            if "çay" in clean.lower():
                topic = "Çay içme vakti"
            elif "kahve" in clean.lower():
                topic = "Kahve içme vakti"
            elif "su" in clean.lower():
                topic = "Su içme vakti"
            elif "ilaç" in clean.lower():
                topic = "İlaç alma vakti"
            else:
                topic = clean.capitalize()

        return True, mins, topic


    def _is_weather_query(self, text: str) -> Tuple[bool, str]:
        """Robust Turkish weather intent and city extraction supporting all 81 provinces, districts, and case-folding."""
        text_norm = text.replace("İ", "i").replace("I", "ı").replace("i̇", "i").lower()
        weather_triggers = [
            "hava nasıl", "hava durumu", "hava kaç derece", "havalar nasıl",
            "yağmur var mı", "kar var mı", "sıcaklık kaç", "hava", "sıcaklık",
            "sicaklik", "derece", "yağmur", "yagmur", "kar durumu", "hava raporu"
        ]
        if not any(w in text_norm for w in weather_triggers):
            return False, ""

        cities = [
            ("istanbul", "Istanbul"),
            ("ahlat", "Ahlat"),
            ("tatvan", "Tatvan"),
            ("bitlis", "Bitlis"),
            ("ankara", "Ankara"),
            ("izmir", "Izmir"),
            ("bursa", "Bursa"),
            ("antalya", "Antalya"),
            ("adana", "Adana"),
            ("konya", "Konya"),
            ("gaziantep", "Gaziantep"),
            ("antep", "Gaziantep"),
            ("şanlıurfa", "Sanliurfa"),
            ("sanliurfa", "Sanliurfa"),
            ("urfa", "Sanliurfa"),
            ("kocaeli", "Kocaeli"),
            ("izmit", "Kocaeli"),
            ("mersin", "Mersin"),
            ("diyarbakır", "Diyarbakir"),
            ("diyarbakir", "Diyarbakir"),
            ("hatay", "Hatay"),
            ("antakya", "Hatay"),
            ("manisa", "Manisa"),
            ("kayseri", "Kayseri"),
            ("samsun", "Samsun"),
            ("balıkesir", "Balikesir"),
            ("balikesir", "Balikesir"),
            ("kahramanmaraş", "Kahramanmaras"),
            ("maras", "Kahramanmaras"),
            ("maraş", "Kahramanmaras"),
            ("van", "Van"),
            ("aydın", "Aydin"),
            ("aydin", "Aydin"),
            ("denizli", "Denizli"),
            ("sakarya", "Sakarya"),
            ("adapazarı", "Sakarya"),
            ("erzurum", "Erzurum"),
            ("muğla", "Mugla"),
            ("mugla", "Mugla"),
            ("bodrum", "Bodrum"),
            ("eskişehir", "Eskisehir"),
            ("eskisehir", "Eskisehir"),
            ("trabzon", "Trabzon"),
            ("elazığ", "Elazig"),
            ("elazig", "Elazig"),
            ("malatya", "Malatya"),
            ("sivas", "Sivas"),
            ("batman", "Batman"),
            ("muş", "Mus"),
            ("mus", "Mus"),
            ("hakkari", "Hakkari"),
            ("siirt", "Siirt"),
            ("bingöl", "Bingol"),
            ("bingol", "Bingol"),
            ("ağrı", "Agri"),
            ("agri", "Agri"),
            ("kars", "Kars"),
            ("ığdır", "Igdir"),
            ("igdir", "Igdir"),
            ("ardahan", "Ardahan"),
            ("güroymak", "Guroymak"),
            ("guroymak", "Guroymak"),
            ("adilcevaz", "Adilcevaz"),
            ("mutki", "Mutki"),
            ("hizan", "Hizan"),
            ("çanakkale", "Canakkale"),
            ("canakkale", "Canakkale"),
            ("edirne", "Edirne"),
            ("tekirdağ", "Tekirdag"),
            ("rize", "Rize"),
            ("ordu", "Ordu"),
            ("giresun", "Giresun"),
            ("artvin", "Artvin"),
            ("yalova", "Yalova"),
            ("düzce", "Duzce"),
            ("bolu", "Bolu"),
            ("zonguldak", "Zonguldak"),
            ("karabük", "Karabuk"),
            ("bartın", "Bartin"),
            ("kastamonu", "Kastamonu"),
            ("sinop", "Sinop"),
            ("çorum", "Corum"),
            ("amasya", "Amasya"),
            ("tokat", "Tokat"),
            ("gümüşhane", "Gumushane"),
            ("bayburt", "Bayburt"),
            ("yozgat", "Yozgat"),
            ("kırşehir", "Kirsehir"),
            ("nevşehir", "Nevsehir"),
            ("niğde", "Nigde"),
            ("aksaray", "Aksaray"),
            ("karaman", "Karaman"),
            ("kırıkkale", "Kirikkale"),
            ("çankırı", "Cankiri"),
            ("uşak", "Usak"),
            ("kütahya", "Kutahya"),
            ("afyonkarahisar", "Afyonkarahisar"),
            ("afyon", "Afyonkarahisar"),
            ("isparta", "Isparta"),
            ("burdur", "Burdur"),
            ("bilecik", "Bilecik"),
            ("kilis", "Kilis"),
            ("osmaniye", "Osmaniye"),
            ("adıyaman", "Adiyaman"),
            ("tunceli", "Tunceli"),
            ("dersim", "Tunceli"),
            ("şırnak", "Sirnak"),
        ]

        for key, city_name in cities:
            pattern = rf"\b{re.escape(key)}(?:['’]?(?:da|de|ta|te|daki|deki|taki|teki|ya|ye|a|e|ın|in|un|ün|dan|den|tan|ten|ti|tı|tu|tü))?\b"
            if re.search(pattern, text_norm):
                return True, city_name

        if "ahlattı" in text_norm or "ahlatta" in text_norm:
            return True, "Ahlat"
        if "tatvanda" in text_norm or "tatvanta" in text_norm:
            return True, "Tatvan"

        default_city = os.environ.get("DEFAULT_WEATHER_CITY", "Bitlis").strip() or "Bitlis"
        return True, default_city

    def _is_identity_query(self, text: str) -> bool:
        text_l = text.lower()
        return any(q in text_l for q in [
            "ben kimim", "hafızanda ben kimim", "beni tanıyor musun", "kim olduğumu biliyor musun",
            "ben kim", "beni hatırladın mı", "beni tanıdın mı", "sesimi tanıdın mı", "beni sesimden tanıdın mı",
            "sesimden tanıdın mı", "tanıdın mı beni", "kimim ben"
        ])

    def _is_person_learning_query(self, text: str) -> bool:
        keywords = [
            r"\bbenim adım\b", r"\badım\s+[a-zA-ZçğıöşüÇĞİÖŞÜ]+\b", r"\bbeni hafızana kaydet\b",
            r"\bbeni kaydet\b", r"\btanışalım\b", r"\byüzümü kaydet\b", r"\bsesimi kaydet\b",
            r"\byüzümü ve sesimi kaydet\b", r"\bhafızana kaydet\b"
        ]
        text_lower = text.lower()
        return any(re.search(k, text_lower) for k in keywords)



    def _start_idle_learning(self):
        threading.Thread(target=self._idle_learning_loop, daemon=True).start()

    def _query_groq_vision_for_idle(self, prompt: str, base64_image: str) -> str | None:
        """Background room observation strictly using Groq Vision or Gemini REST (OpenAI is 100% FORBIDDEN in idle)."""
        # 1. Try Groq Vision (llama-3.2-11b-vision-preview)
        groq_v_ok = self.circuit_breaker.is_available("groq", sub_provider="groq_vision") if self.circuit_breaker else True
        if self._groq and groq_v_ok:
            for gv_model in ["llama-3.2-11b-vision-preview", "llama-3.2-90b-vision-preview"]:
                if not self.circuit_breaker.is_available("groq", model_id=gv_model):
                    continue
                try:
                    res = self._groq.chat.completions.create(
                        messages=[{"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]}],
                        model=gv_model,
                        temperature=0.2,
                        max_tokens=100,
                        timeout=4.0
                    )
                    raw = res.choices[0].message.content.strip()
                    clean = extract_spoken_turkish_sentence(raw)
                    if clean:
                        if self.circuit_breaker:
                            self.circuit_breaker.record_success("groq", sub_provider="groq_vision", model_id=gv_model)
                        return clean
                except Exception as ge:
                    self.get_logger().debug(f"Groq Idle Vision ({gv_model}) notice: {ge}")

        # 2. Try Gemini REST (gemini-2.0-flash)
        gemini_v_ok = self.circuit_breaker.is_available("gemini", sub_provider="gemini_vision") if self.circuit_breaker else True
        if self._ai_api_key and gemini_v_ok:
            for g_model in ["gemini-2.0-flash", "gemini-1.5-flash"]:
                if not self.circuit_breaker.is_available("gemini", model_id=g_model):
                    continue
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent?key={self._ai_api_key}"
                    payload = {
                        "contents": [{
                            "parts": [
                                {"text": prompt},
                                {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
                            ]
                        }],
                        "generation_config": {"temperature": 0.2, "max_output_tokens": 150}
                    }
                    data_bytes = json.dumps(payload).encode("utf-8")
                    req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json", "User-Agent": "Astro-V1-SocialRobot/2.0"})
                    with urllib.request.urlopen(req, timeout=4.0) as resp:
                        res_json = json.loads(resp.read().decode("utf-8"))
                        text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                        clean = clean_tts_text(text)
                        if clean:
                            if self.circuit_breaker:
                                self.circuit_breaker.record_success("gemini", sub_provider="gemini_vision", model_id=g_model)
                            return clean
                except Exception as _exc:
                    self.get_logger().debug(f"_query_groq_vision_for_idle: yok sayılan hata ({_exc})")

        return None

    def _idle_memory_reflection(self):
        """Uses fast Groq or Gemini LLM to summarize recent interactions into long-term profile knowledge (OpenAI is FORBIDDEN)."""
        if len(self.memory.episodic.get_messages()) < 2:
            return
        try:
            recent_conv = self.memory.episodic.get_messages()[-6:]
            conv_str = "\n".join([f"{m['role']}: {m['content']}" for m in recent_conv])
            prompt = (
                f"Aşağıdaki konuşmayı incele. Kullanıcı hakkında öğrenilen yeni, kalıcı ve önemli bir bilgi varsa "
                f"(örnek: hobisi, mesleği, tercih ettiği hitap, adı veya beğendiği bir şey) tek bir kısa Türkçe cümle olarak özetle. "
                f"Yeni veya kayda değer bir bilgi yoksa sadece 'YOK' yaz.\n\nKonuşma:\n{conv_str}"
            )
            ans = None
            if self._groq and self._active_groq_models and self.circuit_breaker.is_available("groq", sub_provider="groq_llm"):
                res = self._groq.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self._active_groq_models[0],
                    temperature=0.1,
                    max_tokens=60
                )
                ans = res.choices[0].message.content.strip()
            elif self._ai_api_key and self.circuit_breaker.is_available("gemini", sub_provider="gemini_text"):
                ans = self._query_gemini_text_rest("Sadece yeni bilgiyi tek cümle yaz veya YOK yaz.", prompt, [])

            if ans and "YOK" not in ans.upper() and len(ans) >= 5:
                clean_fact = clean_tts_text(ans)
                self.memory.profile.add_observation(f"Kullanıcı Bilgisi: {clean_fact}", confidence=0.85)
                self.get_logger().info(f"🧠 [Otonom Hafıza Yansıtması]: {clean_fact}")
        except Exception as e:
            self.get_logger().debug(f"Memory reflection notice: {e}")

    def _idle_learning_loop(self):
        cycle_count = 0
        last_snap = None
        while rclpy.ok():
            time.sleep(5)
            if not self._enable_idle_learning:
                continue
            if not self.state_machine.is_idle() or self._tts_speaking or self._is_processing:
                continue

            now = time.monotonic()
            if (now - getattr(self, '_last_idle_learning_time', 0)) > 35.0:
                self._last_idle_learning_time = now
                cycle_count += 1
                cycle_id = f"idle_cycle_{cycle_count}"
                t_cycle_start = time.monotonic()

                # 1. Background Cognitive Memory Reflection
                self._idle_memory_reflection()

                # 2. Multimodal Perception Snapshot & Change Gating
                current_snap = self.social_context_engine.get_snapshot() if self.social_context_engine else {
                    "lidar": {"nearest_distance_m": 0.0, "nearest_angle_deg": 0.0, "obstacle_count": 0, "motion_detected": False, "timestamp": now},
                    "audio": {"doa_angle_deg": self._speaker_angle, "voice_activity": False, "speech_detected": False, "speaker_confidence": 0.0, "audio_event": False, "timestamp": now},
                    "visual": {"person_detected": self._person_detected, "person_count": 1 if self._person_detected else 0, "face_distance_m": self._user_distance, "gaze_direction": 0.0, "looking_at_robot": self._looking_at_robot, "emotion": self._user_emotion, "scene_signature": "", "object_candidates": [], "timestamp": now},
                    "social_context": "ISOLATED_IDLE",
                    "timestamp": now,
                }

                captured_frame = None
                with self._lock:
                    if self._latest_frame is not None and (now - self._latest_frame_time) < 4.0:
                        captured_frame = self._latest_frame.copy()

                has_change = True
                trigger = "initial_cycle"
                if last_snap is not None and self.social_context_engine:
                    has_change, trigger = self.social_context_engine.has_perception_change(last_snap, current_snap)
                elif last_snap is not None:
                    has_change = (current_snap["visual"]["person_detected"] != last_snap["visual"]["person_detected"])
                    trigger = "person_change" if has_change else "none"

                last_snap = current_snap

                # Gating Check: If NO perception change, skip cloud LLM completely!
                if not has_change:
                    idle_latency_ms = (time.monotonic() - t_cycle_start) * 1000.0
                    self.get_logger().info(
                        f"\n[ASTRO IDLE]\n"
                        f"cycle_id={cycle_id}\n"
                        f"lidar={current_snap['lidar']}\n"
                        f"doa={current_snap['audio']}\n"
                        f"camera={current_snap['visual']}\n"
                        f"social_context={current_snap['social_context']}\n"
                        f"perception_trigger=none\n"
                        f"llm_used=false\n"
                        f"confidence=0.00\n"
                        f"persisted=false\n"
                        f"skip_reason=no_perception_change\n"
                        f"latency_ms={idle_latency_ms:.1f}"
                    )
                    continue

                self._idle_frame_seen = True

                # 3. Room Scene Observation via Groq/Gemini Vision ONLY (OpenAI is 100% Forbidden)
                if captured_frame is not None:
                    base64_img = frame_to_base64_jpeg(captured_frame, max_dim=512)
                    if base64_img:
                        self.get_logger().info("🕵️ [Otonom Boşta Öğrenme] Algı değişti, Astro odayı inceliyor (Groq/Gemini)...")
                        prompt = "Kameradaki odayı, ortamı veya nesneleri Türkçe olarak tek bir kısa cümleyle açıkla. Açıklama harici hiçbir şey yazma. Örnek: 'Masada bir bilgisayar var.' veya 'Oda aydınlık ve sakin.'"
                        obs = self._query_groq_vision_for_idle(prompt, base64_img)
                        memory_written = False
                        confidence = 0.85
                        if obs:
                            # Memory Write Gating: confidence check (>= 0.70)
                            self.memory.profile.add_observation(obs, confidence=confidence)
                            memory_written = True
                            self.get_logger().info(f"🧠 [Otonom Boşta Gözlem]: {obs}")

                            # If Vision observes a person looking at the robot
                            obs_lower = obs.lower()
                            person_gaze_keywords = ["bize bakıyor", "bana bakıyor", "kameraya bakıyor", "karşımda", "karşısında", "oturan bir", "biri var", "insan var", "beyefendi", "hanımefendi"]
                            if any(kw in obs_lower for kw in person_gaze_keywords):
                                if self.state_machine.is_idle() and not self._tts_speaking and not self._is_processing:
                                    if (now - getattr(self, '_last_proactive_gaze_time', 0)) > 30.0:
                                        self._last_proactive_gaze_time = now
                                        self.session.activate_session(reason="groq_scene_gaze")
                                        self.state_machine.transition_to(RobotState.LISTENING)
                                        persona = self.persona_engine.current_persona
                                        greeting = "Hey! Seni gördüm, nasıl yardımcı olabilirim?"
                                        self.get_logger().info(f"👁️ [Görsel Sahne Proaktif Etkileşim] ({persona}): \"{greeting}\"")
                                        self._publish_tts(greeting)
                                        self._publish_emotion(persona)
                                        self._publish_gesture("nod")

                        idle_latency_ms = (time.monotonic() - t_cycle_start) * 1000.0
                        self.get_logger().info(
                            f"\n[ASTRO IDLE]\n"
                            f"cycle_id={cycle_id}\n"
                            f"lidar={current_snap['lidar']}\n"
                            f"doa={current_snap['audio']}\n"
                            f"camera={current_snap['visual']}\n"
                            f"social_context={current_snap['social_context']}\n"
                            f"perception_trigger={trigger}\n"
                            f"llm_used=true\n"
                            f"provider=groq/vision\n"
                            f"confidence={confidence:.2f}\n"
                            f"persisted={memory_written}\n"
                            f"reason=scene_observation\n"
                            f"latency_ms={idle_latency_ms:.1f}"
                        )

    def _is_visual_query(self, text: str) -> bool:
        text_lower = text.lower().strip()

        # Guard: Past conversation recall & memory questions must NEVER trigger camera!
        memory_guards = [
            "hatırlıyor musun", "hatırladın mı", "ne konuştuk", "ne konuşmuştuk",
            "ne söyledik", "neler konuştuk", "neler söyledik", "önce ne dedik",
            "hakkımda ne biliyorsun", "hakkımda ne öğrendin", "hafızanda ne var",
            "hafızanda duruyor mu", "hafızada duruyor mu", "hafızanda ne kayıtlı"
        ]
        if any(mg in text_lower for mg in memory_guards) and not any(exp in text_lower for exp in ["kamerana bak", "fotoğraf", "görüntü", "kameraya"]):
            return False

        # 1. Geniş Kapsamlı Doğrudan Görsel Kalıplar (En Az 2 Kelimeli veya Belirgin Nesneler)
        visual_phrases = [
            # Oda, Ortam, Mekan ve Çevre Analizi
            "odayı tarif", "odada ne var", "odamda ne var", "odaya bak", "salonı tarif", "mutfağa bak",
            "masada ne var", "masanın üstünde ne", "ortamı tarif", "ortamda ne var", "çevrede ne var",
            "etrafta ne var", "etrafta kim var", "etrafı tarif", "mekanda ne var", "arka planda ne var",
            "odadaki eşyalar", "masadaki eşyalar", "tarif et", "tarif edebilir misin", "tarifler misin",
            "odamı anlat", "odayı anlat", "ortamı anlat", "çevreyi anlat", "etrafı anlat",
            "odaya göz at", "etrafa bak", "etrafı incele", "odayı incele", "masayı incele",

            # Kamera ve Görme Soruları
            "ne görüyorsun", "neler görüyorsun", "neye bakıyorsun", "nereye bakıyorsun", "neler var burada",
            "görüyor musun", "görebiliyor musun", "beni görüyor musun", "beni görebiliyor musun",
            "kamerana bak", "kameradan bak", "kameranla bak", "kameranla gör", "kameraya bak", "kamerayı aç",
            "bak bakalım", "şuraya bak", "buraya bak", "bana bak", "bana doğru bak", "dikkatli bak",

            # Nesneler, Eşyalar ve Eller
            "ne tutuyorum", "elimde ne", "elinde ne", "elimdekini gör", "elimdeki ne", "elimde ne var",
            "bu ne", "şu ne", "bunlar ne", "bu cisim", "bu eşya", "bu alet", "bu cihaz", "bu kart", "bu kutu", "bu şişe",
            "kaç parmak", "parmaklarımı say", "kaç parmak gösteriyorum", "elime bak", "elimi gör",
            "gösterdiğim nesne", "tuttuğum nesne", "sana gösteriyorum", "bu nesneyi tanı",

            # Kıyafet, Giyiniş, Renk ve Dış Görünüş
            "üstümde ne var", "üzerimde ne var", "üstümdeki ne", "üzerimdeki ne", "ne giymişim", "hangi kıyafeti",
            "kıyafetim nasıl", "kombinim nasıl", "nasıl görünüyorum", "yakışmış mı", "ne renk", "hangi renk", "rengi ne",
            "tişörtüm", "gömleğim", "ceketim", "montum", "kazağım", "pantolonum", "elbisem",
            "gözlüğüm", "güneş gözlüğü", "şapkam", "berem", "kol saati", "akıllı saat", "bilekliğim", "kolyem",

            # İnsanlar, Yüz, Duruş ve Hareketler
            "odada kim var", "yanımda kim var", "arkamda kim var", "etrafta kimse var mı", "kaç kişi var", "kaç kişiyiz",
            "birini görüyor musun", "ne yapıyorum", "hangi hareketi yapıyorum", "hareketimi gör",
            "ayakta mıyım", "oturuyor muyum", "uzanıyor muyum", "yüzüme bak", "gözlerime bak", "bana bakıyor musun",
            "telefona mı bakıyorum", "telefonla mı konuşuyorum", "ekrana mı bakıyorum", "ne okuyorum", "ne yazıyorum"
        ]

        if any(p in text_lower for p in visual_phrases):
            return True

        # 2. Esnek Regex Kalıpları (Belirgin görsel ikililer)
        visual_regex_patterns = [
            r"\b(odayı|salonu|ortamı|çevreyi|etrafı|masayı|kamerayı)\b.*\b(tarif|anlat|incele|tara|betimle|gör|bak)\b",
            r"\b(bu|şu|elimdeki|üstümdeki|üzerimdeki)\b.*\b(ne|hangi|renk|var|gör)\b",
            r"\b(ne|neler|kim|kaç)\b.*\b(görüyorsun|bakıyorsun|tutuyorum|giymişim|gösteriyorum)\b",
            r"\b(bak|gör|anlat|tarif\s*et)\b.*\b(bana|odaya|etrafa|kameraya|elime|üstüme)\b"
        ]

        return any(re.search(pat, text_lower) for pat in visual_regex_patterns)

    def _handle_memory_recall_query(self, user_text: str, identity: Dict[str, Any]) -> Tuple[bool, str]:
        """Handles explicit queries asking about past conversations and person-specific memory recall."""
        text_l = user_text.lower()
        triggers = [
            "hatırlıyor musun", "hatırladın mı", "ne konuştuk", "ne konuşmuştuk",
            "neler konuştuk", "ne söyledik", "neler söyledik", "önce ne dedik",
            "hakkımda ne biliyorsun", "hakkımda ne öğrendin", "hafızanda ne var",
            "hafızanda ne kayıtlı", "hafızanda duruyor mu", "hafızan duruyor mu"
        ]
        if not any(t in text_l for t in triggers):
            return False, ""

        p_name = identity.get("name", "Baran") if identity.get("is_known") else "Baran"
        persona = self.persona_engine.current_persona

        p_profile = self.memory.profile.get_known_person(p_name)
        recent_sessions = self.memory.profile.get_person_recent_sessions(p_name, limit=3)
        learned_facts = p_profile.get("learned_facts", []) if p_profile else []
        preferences = p_profile.get("preferences", {}) if p_profile else {}

        # 1. Past conversation topics
        if any(w in text_l for w in ["konuştuk", "konuşmuştuk", "söyledik", "konuları", "saat önce", "dakika önce"]):
            if recent_sessions:
                last_sess = recent_sessions[-1]
                t_str = last_sess.get("time_str", "az önce")
                summary = last_sess.get("summary", "")
                if persona in ("witty", "kufurbaz"):
                    return True, f"Tabii ki hatırlıyorum! {t_str} civarında seninle {summary} hakkında konuşmuştuk. İşlemcim saat gibi tıkır tıkır!"
                elif persona in ("charming", "flirt"):
                    return True, f"Elbette hatırlıyorum! {t_str} seninle {summary} üzerine konuşmuştuk."
                else:
                    return True, f"Evet, hatırlıyorum. {t_str} seninle {summary} konusunu konuşmuştuk."
            else:
                if persona in ("witty", "kufurbaz"):
                    return True, "Hafızamda arşivlenmiş eski bir konu özeti yok ama şu an konuştuklarımızı aklıma kazıyorum merak etme!"
                return True, "Şu anki sohbetimiz dışında henüz arşivlenmiş eski bir konuşma özetimiz bulunmuyor, ama seni dikkatle dinliyorum!"

        # 2. Personal knowledge recall
        if any(w in text_l for w in ["hakkımda", "hafızanda", "biliyorsun", "öğrendin"]):
            parts = []
            if learned_facts:
                parts.append("seninle ilgili şunları biliyorum: " + "; ".join(learned_facts[:3]))
            if preferences:
                prefs = ", ".join([f"{k}: {v}" for k, v in preferences.items()])
                parts.append(f"tercihlerinden bildiklerim: {prefs}")

            if parts:
                info_text = ". Ayrıca ".join(parts)
                if persona in ("witty", "kufurbaz"):
                    return True, f"Hafızam zehir gibi! {p_name}, {info_text}. Her şeyi özenle hafızamda tutuyorum!"
                elif persona in ("charming", "flirt"):
                    return True, f"Hafızamda seninle ilgili detaylar canlı! {info_text}."
                else:
                    return True, f"Hafızamda seninle ilgili bilgiler kayıtlı: {info_text}."
            else:
                if persona in ("witty", "kufurbaz"):
                    return True, f"Şu an senin hakkında temel unvanın dışında pek bir şey kaydetmedik {p_name}. Bana kendinden ve sevdiklerinden bahset de hafızama yazayım!"
                return True, f"Hafızamda seninle ilgili henüz detaylı bir bilgi birikimi oluşmadı {p_name}. Bana zevklerinden ve kendinden bahsedersen hepsini öğrenirim!"

        return False, ""

    def _async_extract_user_facts(self, user_text: str, person_name: str):
        """Asynchronously extracts user preferences and facts to learn autonomously per person."""
        if not self._groq and not self._openai:
            return
        if len(user_text) < 10 or any(c in user_text.lower() for c in ["hava nasıl", "saat kaç", "kimsin", "odayı tarif"]):
            return

        prompt = (
            "Aşağıdaki kullanıcı cümlesinden kullanıcıya veya ortama dair kalıcı, somut yeni bir bilgi veya tercih (örneğin sevdiği içecek, hobisi, aile üyesi, sınavı, planı, kuralı) varsa JSON olarak çıkar. "
            "Eğer sadece genel sohbet, soru veya geçici bir laf ise boş JSON {} döndür.\n"
            "Format: {\"fact\": \"...\", \"preference_key\": \"...\", \"preference_val\": \"...\"}\n\n"
            f"Kullanıcı Cümlesi: '{user_text}'"
        )
        try:
            raw_json = None
            if self._groq and self._active_groq_models:
                for m in self._active_groq_models[:2]:
                    try:
                        res = self._groq.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model=m,
                            temperature=0.0,
                            max_tokens=60
                        )
                        raw_json = res.choices[0].message.content.strip()
                        break
                    except Exception:
                        continue
            if not raw_json and self._openai and self.circuit_breaker and self.circuit_breaker.is_available("openai", sub_provider="openai_rest"):
                try:
                    res = self._openai.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=self._openai_model,
                        temperature=0.0,
                        max_tokens=60
                    )
                    raw_json = res.choices[0].message.content.strip()
                except Exception as _exc:
                    self.get_logger().debug(f"_async_extract_user_facts: yok sayılan hata ({_exc})")

            if not raw_json and self._ai_api_key and self.circuit_breaker and self.circuit_breaker.is_available("gemini", sub_provider="gemini_text"):
                try:
                    raw_json = self._query_gemini_text_rest("Sadece JSON formatında çıktı ver.", prompt, [])
                except Exception as _exc:
                    self.get_logger().debug(f"_async_extract_user_facts: yok sayılan hata ({_exc})")

            if raw_json and "{" in raw_json and "}" in raw_json:
                json_str = raw_json[raw_json.find("{"):raw_json.rfind("}")+1]
                data = json.loads(json_str)
                fact = data.get("fact")
                if fact and len(fact) > 5:
                    self.memory.profile.add_person_fact(person_name, fact)
                    self.get_logger().info(f"💡 [Otonom Öğrenme ({person_name})]: Yeni Bilgi Kaydedildi -> '{fact}'")
                pref_k = data.get("preference_key")
                pref_v = data.get("preference_val")
                if pref_k and pref_v:
                    self.memory.profile.add_person_preference(person_name, pref_k, pref_v)
                    self.get_logger().info(f"💡 [Otonom Tercih ({person_name})]: {pref_k} -> {pref_v}")
        except Exception as e:
            self.get_logger().debug(f"Fact extraction notice: {e}")

    def _is_object_learning_query(self, text: str) -> bool:
        keywords = ["bu benim", "bunu öğren", "bunu kaydet", "bu nesne", "buna bak bu", "bu gördüğün nesne"]
        text_lower = text.lower()
        return any(k in text_lower for k in keywords)

    def _publish_tts(self, text: str, generation_id: Optional[int] = None):
        import json
        last_u = getattr(self.session, "last_user_text", "")
        clean = response_length_gate(text, user_query=last_u, max_words=35, max_sentences=2)
        if clean:
            if isinstance(generation_id, (int, float)):
                gen_id = int(generation_id)
            elif hasattr(self, "session") and isinstance(getattr(self.session, "current_turn_count", None), int):
                gen_id = self.session.current_turn_count
            else:
                gen_id = 1
            openai_realtime_ok = (
                (self.circuit_breaker.is_available("openai", sub_provider="openai_realtime") if self.circuit_breaker else True)
                and self._realtime_ws_connected
                and self._realtime_session_ready
            )
            if openai_realtime_ok and self.session.metadata.get("tts_engine") != "edge-tts":
                tts_engine = "openai_realtime"
                reason = "realtime_available"
                self.get_logger().info(f"[TTS REQUESTED] requested_provider={tts_engine} selection_reason={reason} text=\"{clean}\"")
                req_payload = {"text": clean, "generation_id": gen_id}
                msg = String()
                msg.data = json.dumps(req_payload)
                if hasattr(self, "pub_realtime_req") and self.pub_realtime_req is not None:
                    self.pub_realtime_req.publish(msg)
                else:
                    self.pub_tts.publish(msg)
            else:
                tts_engine = "edge_tts"
                if self.circuit_breaker and self.circuit_breaker.is_exhausted("openai"):
                    reason = "openai_realtime_exhausted"
                elif not self._realtime_ws_connected:
                    reason = "realtime_unavailable"
                else:
                    reason = "session_edge_tts"

                self.get_logger().info(f"[TTS REQUESTED] requested_provider={tts_engine} selection_reason={reason} text=\"{clean}\"")
                payload = {"text": clean, "engine": "edge-tts", "generation_id": gen_id, "fallback_reason": reason}
                msg = String()
                msg.data = json.dumps(payload)
                self.pub_tts.publish(msg)

    def _publish_interrupt(self):
        msg = Bool()
        msg.data = True
        self.pub_interrupt.publish(msg)

    def _publish_emotion(self, emotion: str):
        msg = String()
        msg.data = emotion
        self.pub_emotion.publish(msg)

    def _publish_gesture(self, gesture: str):
        msg = String()
        msg.data = gesture
        self.pub_gesture.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AiBrainNode()
    from rclpy.executors import MultiThreadedExecutor

    # Düğüm callback'leri üç ayrı gruba dağıtılmış durumda (bkz. AiBrainNode.__init__):
    # konuşma / timer / algı. Bu thread'ler ancak gruplar gerçekten ayrıldığı için
    # işe yarıyor — tek varsayılan grupla hepsi sıraya girerdi.
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt as _exc:
        _LOG.debug("main: yok sayılan hata (%s)", _exc)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()



if __name__ == "__main__":
    main()
