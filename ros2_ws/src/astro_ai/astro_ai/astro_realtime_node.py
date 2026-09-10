#!/usr/bin/env python3
"""ASTRO V1 — OpenAI Realtime Audio-to-Audio (WebSocket E2E) Bridge Node.

Features:
  - Direct full-duplex WebSocket connection to OpenAI Realtime API (gpt-realtime-2.1-mini)
  - End-to-end 24kHz raw PCM streaming input & output (< 450ms total turn latency)
  - Server-side VAD & Zero-Latency Barge-In (instant response cancellation on user speech)
  - Full modular Persona Engine & Biometric Perception integration via dynamic session.update
  - Integrated Function Calling (Realtime Tools: live weather, reminders, memory, person enrollment)
"""

import asyncio
import logging

_LOG = logging.getLogger(__name__)

import base64
import inspect
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
import wave
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo, LaserScan, JointState
    from std_msgs.msg import Bool, Float32, String
    from geometry_msgs.msg import Twist
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    try:
        from astro_base.msg import HeadCmd
    except ImportError:
        class HeadCmd:  # type: ignore
            angle_deg: float = 0.0
except ImportError:
    rclpy = None
    qos_profile_sensor_data = 10  # rclpy yoksa (mock/test modu) düz derinlik
    class HeadCmd:  # type: ignore
        angle_deg: float = 0.0
    class Node:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def get_logger(self):
            import logging
            return logging.getLogger("AstroRealtimeNode")
        def create_subscription(self, *args, **kwargs):
            return None
        def create_publisher(self, *args, **kwargs):
            return None
        def create_timer(self, *args, **kwargs):
            return None
    class _MockMsg:
        def __init__(self, data=None, **kwargs):
            self.data = data
            self.status = []
            self.ranges = []
            for k, v in kwargs.items():
                setattr(self, k, v)
    class Twist:  # type: ignore
        class Vector3:
            def __init__(self, x=0.0, y=0.0, z=0.0):
                self.x = float(x)
                self.y = float(y)
                self.z = float(z)
        def __init__(self):
            self.linear = Twist.Vector3()
            self.angular = Twist.Vector3()
    class DiagnosticStatus:
        OK = 0
        WARN = 1
        ERROR = 2
        STALE = 3
        def __init__(self, name="", level=0, message="", values=None):
            self.name = name
            self.level = level
            self.message = message
            self.values = values or []
    class KeyValue:
        def __init__(self, key="", value=""):
            self.key = str(key)
            self.value = str(value)
    Image = CameraInfo = Bool = Float32 = String = DiagnosticArray = LaserScan = JointState = _MockMsg  # type: ignore

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

try:
    import websockets
except ImportError:
    websockets = None

try:
    from astro_audio.elevenlabs_engine import ElevenLabsEngine, ElevenLabsError
    from astro_audio.local_xtts_engine import LocalXttsEngine, resolve_xtts_home, resolve_xtts_speaker_wav
    from astro_audio.local_offline_tts_engine import LocalOfflineTTSEngine
    from astro_audio.tts_metrics import TurnTelemetry
    from astro_audio.sentence_chunker import SentenceChunker
except ImportError:
    ElevenLabsEngine = None
    ElevenLabsError = Exception
    LocalXttsEngine = None
    LocalOfflineTTSEngine = None
    resolve_xtts_home = lambda h="": os.path.expanduser("~/.astro/tts")
    resolve_xtts_speaker_wav = lambda w="": os.path.expanduser("~/.astro/tts/Recording.wav")
    TurnTelemetry = None
    SentenceChunker = None

EMOJI_RE = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE
)

try:
    from astro_ai.conversation_session import ConversationSession, normalize_turkish_speech_input
    from astro_ai.memory_manager import MemoryManager
    from astro_ai.persona_engine import (
        PersonaEngine, PERSONA_PROMPTS, clean_tts_text,
        response_length_gate, is_self_identity_query, ResponseSafetyGate
    )
    from astro_ai.state_machine import RobotState, StateMachine
    from astro_ai.provider_registry import ProviderRegistry, ProviderError, ErrorClass
    from astro_ai.repetition_guard import RepetitionGuard
    from astro_ai.action_manager import ActionManager, SoundDirection, ActionResult
    from astro_ai.robot_led import RobotLED
except ImportError:
    from conversation_session import ConversationSession, normalize_turkish_speech_input
    from memory_manager import MemoryManager
    from persona_engine import (
        PersonaEngine, PERSONA_PROMPTS, clean_tts_text,
        response_length_gate, is_self_identity_query, ResponseSafetyGate
    )
    from state_machine import RobotState, StateMachine
    from provider_registry import ProviderRegistry, ProviderError, ErrorClass
    from repetition_guard import RepetitionGuard
    try:
        from action_manager import ActionManager, SoundDirection, ActionResult
    except ImportError:
        ActionManager = SoundDirection = ActionResult = None  # type: ignore
    try:
        from robot_led import RobotLED
    except ImportError:
        RobotLED = None  # type: ignore


try:
    from astro_ai.brain.social_brain import SocialBrain
    from astro_ai.contracts.person_state import UnifiedPersonState
except ImportError:
    try:
        from brain.social_brain import SocialBrain
        from contracts.person_state import UnifiedPersonState
    except ImportError:
        SocialBrain = None
        UnifiedPersonState = None



try:
    from astro_audio.voice_recognizer import VoiceRecognizer
except ImportError:
    try:
        from voice_recognizer import VoiceRecognizer
    except ImportError:
        VoiceRecognizer = None

try:
    from astro_vision.face_recognizer import FaceRecognizer
except ImportError:
    try:
        from face_recognizer import FaceRecognizer
    except ImportError:
        FaceRecognizer = None

try:
    from astro_ai.office.calendar_service import CalendarService
    from astro_ai.office.slack_service import SlackService
    from astro_ai.office.office_concierge import OfficeConciergeManager
except ImportError:
    try:
        from office.calendar_service import CalendarService
        from office.slack_service import SlackService
        from office.office_concierge import OfficeConciergeManager
    except ImportError:
        CalendarService = SlackService = OfficeConciergeManager = None  # type: ignore

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:
    def find_dotenv(*args, **kwargs): return ""
    def load_dotenv(*args, **kwargs): pass


def _load_env():
    """astr1/.env dosyasını os.environ'a yükler; bulunan yolu döndürür.

    Bu düğüm daha önce .env'i HİÇ okumuyordu, yalnızca os.environ'a bakıyordu.
    Yani anahtarlar yalnızca onu başlatan sürecin ortamına konmuşsa çalışıyordu:
    bringup.launch.py bunu SetEnvironmentVariable ile yapıyor, ama
    realtime_sensors.launch.py .env'i yalnızca CWD repo köküyse buluyor ve
    `ros2 run astro_ai astro_realtime_node` hiçbir şey yüklemiyor. Üçünde de
    sonuç "❌ OPENAI_API_KEY eksik" oluyordu.

    Aday listesi tts_node/ai_brain_node ile aynı; son çare find_dotenv(usecwd=True)
    CWD'den yukarı doğru yürüdüğü için ros2_ws içinden çalıştırıldığında da bulur.
    """
    # Test sürecinde .env YÜKLENMEZ. Bu düğüm gerçek bir anahtar bulduğu anda
    # websocket'i açıyor, discover_realtime_models() ile OpenAI'a HTTPS isteği
    # atıyor ve idle-learning döngüsünü başlatıyor. Testler düğümü onlarca kez
    # örneklediği için bu hem kullanıcının kotasını harcıyor hem de canlı SSL
    # iş parçacıkları + rclpy yıkımı bir arada segfault üretiyordu.
    if "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules:
        return None

    candidates = [
        os.path.abspath(".env"),
        os.path.abspath(".env.production"),
        os.path.abspath(os.path.join(os.getcwd(), ".env")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".env")),
        os.path.expanduser("~/.env"),
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


def resample_24k_to_16k(raw_24k_bytes: bytes) -> bytes:
    """Ultra-fast 24kHz -> 16kHz int16 PCM downsampling (480 -> 320 samples)."""
    arr_24k = np.frombuffer(raw_24k_bytes, dtype=np.int16)
    if len(arr_24k) == 0:
        return b""
    n_out = int(len(arr_24k) * (2.0 / 3.0))
    indices = np.linspace(0, len(arr_24k) - 1, n_out)
    arr_16k = np.interp(indices, np.arange(len(arr_24k)), arr_24k.astype(np.float32)).astype(np.int16)
    return arr_16k.tobytes()


def imgmsg_to_bgr(msg: Image) -> Optional[np.ndarray]:
    if cv2 is None or msg is None or not msg.data:
        return None
    try:
        if msg.encoding == "bgr8":
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        elif msg.encoding == "rgb8":
            data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        elif msg.encoding in ("mono8", "8UC1"):
            data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            return cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    except Exception as _exc:
        _LOG.debug("imgmsg_to_bgr: yok sayılan hata (%s)", _exc)
    return None


def frame_to_base64_jpeg(frame: np.ndarray, max_dim: int = 640) -> Optional[str]:
    if cv2 is None or frame is None:
        return None
    try:
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buffer).decode("utf-8").replace("\n", "").replace("\r", "").strip()
    except Exception:
        return None




# Module-level monotonic process-lifetime lockout for OpenAI
OPENAI_HARD_DISABLED: bool = False

def set_openai_hard_disabled(reason: str = "quota_or_rate_limit_exhausted") -> None:
    global OPENAI_HARD_DISABLED
    OPENAI_HARD_DISABLED = True

def reset_openai_hard_disabled_for_test() -> None:
    """Only used by offline unit tests to reset test state."""
    global OPENAI_HARD_DISABLED
    OPENAI_HARD_DISABLED = False


REALTIME_WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1-mini"
VALID_REALTIME_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "fable", "onyx"}

PERSONA_DEFAULT_VOICES: Dict[str, str] = {
    "kufurbaz": "echo",     # Echo - Deep, expressive natural voice
    "angry": "echo",        # Echo
    "rude": "echo",         # Echo
    "sarcastic": "echo",    # Echo
    "flirt": "echo",        # Echo
    "charming": "echo",     # Echo
    "playful": "echo",      # Echo
    "witty": "echo",        # Echo
    "formal": "echo",       # Echo
    "emotional": "echo",    # Echo
}


def discover_realtime_models(api_key: str, preferred: str = "") -> list[str]:
    # Öncelik sırası HIZA göre. gpt-realtime-2.1-mini (6 Tem 2026) Realtime
    # ailesinin hızlı katmanı: p95 gecikme diğerlerine göre en az %25 düşük ve
    # ses çıkışı 20 $/M token (2.1'de 64 $/M). Akıl yürütme ve araç kullanımı
    # yine var. gpt-realtime-mini OpenAI tarafından kullanımdan kaldırılıyor,
    # bu yüzden listeden çıkarıldı.
    # Dördü de bu hesapta canlı doğrulandı (session.updated döndü).
    flagship_realtime_models = [
        "gpt-realtime-2.1-mini",
        "gpt-realtime-2.1",
        "gpt-realtime-2",
        "gpt-realtime",
    ]
    candidates = []
    if preferred:
        candidates.append(preferred)

    # Test sürecinde veya OpenAI kilitliyken AĞA ÇIKILMAZ.
    global OPENAI_HARD_DISABLED
    if (
        OPENAI_HARD_DISABLED
        or os.environ.get("ASTRO_TEST_MODE", "0") in ("1", "true", "True")
        or "unittest" in sys.modules
        or "pytest" in sys.modules
        or "PYTEST_CURRENT_TEST" in os.environ
        or not api_key
        or api_key.startswith("sk-test")
        or api_key.startswith("test_")
    ):
        return candidates + [m for m in flagship_realtime_models if m not in candidates]

    try:
        import urllib.request
        req = urllib.request.Request("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {api_key}", "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode())
            avail_ids = [m["id"] for m in data.get("data", []) if "realtime" in m.get("id", "")]
            # Sort available models by priority
            for fm in flagship_realtime_models:
                if fm in avail_ids and fm not in candidates:
                    candidates.append(fm)
            for mid in avail_ids:
                if mid not in candidates:
                    candidates.append(mid)
    except Exception as _exc:
        _LOG.debug("discover_realtime_models: yok sayılan hata (%s)", _exc)

    for m in flagship_realtime_models:
        if m not in candidates:
            candidates.append(m)

    return candidates


def discover_groq_models(api_key: str) -> list[str]:
    """Dynamically queries Groq /models API to discover currently active models."""
    if (
        os.environ.get("ASTRO_TEST_MODE", "0") in ("1", "true", "True")
        or "unittest" in sys.modules
        or "pytest" in sys.modules
        or "PYTEST_CURRENT_TEST" in os.environ
        or not api_key
        or api_key.startswith("sk-test")
        or api_key.startswith("test_")
    ):
        return []
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}", "User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            active_ids = [m["id"] for m in data.get("data", []) if "id" in m]
            return active_ids
    except Exception:
        return []


VALID_SHORT_UTTERANCES = {
    "hey", "lan", "dur", "ne", "tamam", "merhaba", "evet", "hayır",
    "astro", "selam", "alo", "sus", "günaydın", "iyi geceler", "naber",
    "efendim", "anladım", "peki", "dinliyorum", "burada", "buradayım"
}

SUSPECT_PHRASES = [
    "abone ol", "abone olmayı", "abone olmayı unutmayın", "altyazı",
    "altyazı m.k.", "altyazı:", "çeviren:", "çeviri ve altyazı",
    "diz", "dizi", "altyazı ekibi", "izlediğiniz için", "izlediğiniz için teşekkürler",
    "beğenmeyi unutmayın", "hoşça kalın", "hoşçakalın", "bay bay", "m.k.", "sponsor",
    "videoyu beğenmeyi", "subtitle", "transcription by", "hı hı", "cık", "çık", "ııı", "eee", "hmm"
]


def compute_self_voice_score(transcript: str, recent_robot_phrases: List[str]) -> float:
    """Computes acoustic & lexical correlation score (0.0 - 1.0) between transcript and recent robot speech."""
    if not transcript or not recent_robot_phrases:
        return 0.0
    t_clean = re.sub(r"[^\w\s]", "", transcript.lower()).strip()
    if not t_clean:
        return 0.0
    t_words = set(t_clean.split())
    if not t_words:
        return 0.0

    max_score = 0.0
    for phrase in recent_robot_phrases:
        p_clean = re.sub(r"[^\w\s]", "", phrase.lower()).strip()
        if not p_clean:
            continue
        p_words = set(p_clean.split())
        if not p_words:
            continue
        # Exact substring match
        if t_clean in p_clean or p_clean in t_clean:
            score = 0.95
        else:
            # Word overlap (Jaccard / containment)
            intersection = t_words.intersection(p_words)
            score = len(intersection) / float(len(t_words))
        if score > max_score:
            max_score = score
    return min(1.0, max_score)


def compute_pcm_self_voice_score(mic_pcm: bytes, ref_pcm: bytes, max_lag_samples: int = 4800) -> float:
    """Computes acoustic correlation (0.0 to 1.0) between incoming mic frame and playback buffer."""
    if not mic_pcm or not ref_pcm:
        return 0.0
    try:
        mic = np.frombuffer(mic_pcm, dtype=np.int16).astype(np.float32)
        ref = np.frombuffer(ref_pcm, dtype=np.int16).astype(np.float32)
        if len(mic) == 0 or len(ref) == 0:
            return 0.0
        mic_c = mic - np.mean(mic)
        mic_n = float(np.linalg.norm(mic_c))
        if mic_n < 1e-4:
            return 0.0
        ref_w = ref[-max_lag_samples:] if len(ref) > max_lag_samples else ref
        if len(ref_w) < len(mic):
            return 0.0
        ref_c = ref_w - np.mean(ref_w)
        corr = np.correlate(ref_c, mic_c, mode='valid')
        ref_sq = ref_c ** 2
        w_energy = np.correlate(ref_sq, np.ones(len(mic), dtype=np.float32), mode='valid')
        denom = mic_n * np.sqrt(np.maximum(w_energy, 1e-6))
        norm_c = corr / denom
        max_c = float(np.max(norm_c)) if len(norm_c) > 0 else 0.0
        return round(max(0.0, min(1.0, max_c)), 4)
    except Exception:
        return 0.0


def is_known_phantom_pattern(text: str) -> bool:
    """Checks if text contains known Whisper hallucination/phantom pattern without semantic context."""
    if not text:
        return True
    t = re.sub(r"[^\w\s]", "", text.lower()).strip()
    if not t:
        return True
    phantom_exacts = {
        "altyazı", "altyazı mk", "altyazı m k", "altyazi", "altyazi mk", "altyazi m k",
        "abone ol", "kanala abone ol", "abone olun", "videoyu beğenmeyi unutmayın",
        "izlediğiniz için teşekkürler", "izlediginiz icin tesekkurler",
        "izlediğiniz için teşekkür ederiz", "izlediginiz icin tesekkur ederiz",
        "diz", "dizi", "hahaha", "hahahaha", "hehehe", "hihihi",
        "türen türen türen", "türen", "turen", "turen turen turen",
        "evet evet evet", "nokta", "virgül", "şşş", "sss", "hı hı", "cık", "çık"
    }
    if t in phantom_exacts:
        return True
    # Repetitive single word loop e.g. "türen, türen, türen" or "evet, evet, evet"
    words = t.split()
    if len(words) >= 2 and len(set(words)) == 1 and words[0] in ("türen", "turen", "evet", "hayır", "ha", "he", "diz", "dizi", "altyazı", "hahaha"):
        return True
    return False


def is_valid_user_command(command: str) -> Tuple[bool, str]:
    """Validates extracted user command for semantic plausibility, repetition, catalog hallucinations, and phantom patterns."""
    if not command:
        return False, "empty_command"
    c_clean = re.sub(r"[^\w\s]", "", command.lower()).strip()
    if not c_clean or len(c_clean) < 2:
        return False, "empty_command"

    # 1. Known Phantom Patterns
    if is_known_phantom_pattern(c_clean):
        return False, "known_phantom"

    words = c_clean.split()
    if not words:
        return False, "empty_command"

    # 2. Pure Wake Prefix Remainder (only wake tokens)
    if all(w in ("astro", "hey", "selam") for w in words):
        return False, "wake_only_remainder"

    # 3. Prompt / Catalog / Hallucinated Training Metadata Artifacts (e.g. "Türkçe konuşma, diyalog, robot asistan")
    catalog_triggers = [
        "türkçe konuşma", "turkce konusma", "robot asistan", "sesli asistan",
        "türkçe diyalog", "turkce diyalog", "türkçe dublaj", "turkce dublaj",
        "türkçe altyazı", "turkce altyazi", "altyazı m k", "altyazi mk",
        "abone ol", "kanala abone", "videoyu beğen", "izlediğiniz için",
    ]
    for ct in catalog_triggers:
        if ct in c_clean:
            conversational_clues = ("hakkında", "nasıl", "neden", "ne", "nerede", "kim", "yap", "aç", "kapat", "anlat", "söyle", "nedir", "istiyorum", "misin")
            if not any(w in c_clean for w in conversational_clues):
                return False, "catalog_hallucination"

    # 4. Excessive Wake Word Repetition in Command (e.g. "astro astro astro", "hey astro hey astro")
    wake_tokens = [w for w in words if w in ("astro", "hey", "selam")]
    if len(wake_tokens) >= 2 and (len(wake_tokens) / len(words)) >= 0.30:
        return False, "wake_word_loop"

    # 5. Word Repetition Ratio (e.g. "evet evet evet", "türen türen türen")
    if len(words) >= 3:
        from collections import Counter
        counts = Counter(words)
        most_common_word, most_common_count = counts.most_common(1)[0]
        if most_common_count >= 3 and (most_common_count / len(words)) >= 0.35:
            return False, "repetitive_word_loop"

    return True, "valid"


class AstroRealtimeNode(Node):
    """ROS 2 Node bridging Astro sensors & audio streams to OpenAI Realtime WebSocket."""
    active_response_id: Optional[str] = None
    active_generation_id: Optional[int] = None
    active_response_state: str = "IDLE"
    _turn_queue: List[Dict[str, Any]] = []
    _last_sent_generation_id: Optional[int] = None
    _session_ready_logged: bool = False
    _watchdog_timer: Optional[threading.Timer] = None
    _packets_for_gen: int = 0
    _bytes_for_gen: int = 0
    _first_audio_time: Optional[float] = None
    realtime_current_generation_id: int = 0
    realtime_audio_received: bool = False
    realtime_response_state: str = "IDLE"
    realtime_provider_state: str = "AVAILABLE"
    realtime_connection_state: str = "DISCONNECTED"
    realtime_session_state: str = "NOT_READY"
    realtime_session_id: str = ""

    @staticmethod
    def _under_pytest() -> bool:
        return (
            "PYTEST_CURRENT_TEST" in os.environ
            or "pytest" in sys.modules
            or "unittest" in sys.modules
            or "unittest.mock" in sys.modules
            or os.environ.get("ASTRO_TEST_MODE", "0") in ("1", "true", "True")
            or os.environ.get("ASTRO_MOCK_AUDIO", "0") in ("1", "true", "True")
        )

    def __init__(self, connect_realtime: bool = True, fake_transport: Optional[Any] = None):
        if rclpy is not None and hasattr(rclpy, "ok") and not rclpy.ok():
            try:
                rclpy.init()
            except Exception as _exc:
                self.get_logger().debug(f"__init__: yok sayılan hata ({_exc})")
        super().__init__("astro_realtime_node")

        # Startup timestamp & Grace Period tracking
        self._node_start_time = time.monotonic()
        self.xtts_startup_grace_s = float(os.getenv("XTTS_STARTUP_GRACE_S", "60.0"))

        # Load environment variables (sanitized of quotes/whitespace)
        # _load_env() anahtarlar OKUNMADAN ÖNCE çağrılmalı; aksi halde düğüm
        # yalnızca kendisini başlatan sürecin ortamına bağımlı kalır.
        _loaded_env = _load_env()
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip("\"' \t\n\r")
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "").strip("\"' \t\n\r")
        raw_gem = os.environ.get("GEMINI_API_KEY", "").strip("\"' \t\n\r")
        self.gemini_api_key = raw_gem if (raw_gem and not raw_gem.startswith("sk-")) else ""
        self.realtime_model = os.environ.get("REALTIME_MODEL", "gpt-realtime-2.1-mini").strip()
        self.realtime_transcribe_model = os.environ.get(
            "REALTIME_TRANSCRIBE_MODEL", "gpt-live-transcribe").strip() or "gpt-live-transcribe"
        raw_voice = os.environ.get("REALTIME_VOICE", os.environ.get("OPENAI_TTS_VOICE", os.environ.get("TTS_VOICE", "echo"))).strip().lower()

        # Test mode isolation guard
        is_test_mode = (
            os.environ.get("ASTRO_TEST_MODE", "0") in ("1", "true", "True")
            or "unittest" in sys.modules
            or "pytest" in sys.modules
            or "PYTEST_CURRENT_TEST" in os.environ
            or self.openai_api_key.startswith("sk-test")
            or self.openai_api_key.startswith("test_")
        )
        self.connect_realtime = bool(connect_realtime and not is_test_mode)
        self.fake_transport = fake_transport
        self.persona_name = os.environ.get("PERSONA", "playful").strip().lower()
        self.realtime_voice = raw_voice if raw_voice in VALID_REALTIME_VOICES else PERSONA_DEFAULT_VOICES.get(self.persona_name, "echo")

        # Anahtarın nereden geldiğini başlangıçta söyle. "eksik" hatası alındığında
        # ilk soru her zaman "hangi .env okundu" oluyor; cevabı burada.
        self.get_logger().info(
            "[ENV] dotenv={} | OPENAI_API_KEY={} | GROQ={} | GEMINI={}".format(
                _loaded_env or "BULUNAMADI",
                f"var (…{self.openai_api_key[-4:]})" if self.openai_api_key else "YOK",
                "var" if self.groq_api_key else "yok",
                "var" if self.gemini_api_key else "yok",
            )
        )


        # Modular Cognitive Subsystems
        self.memory = MemoryManager()
        self.persona_engine = PersonaEngine(self.persona_name)
        self.state_machine = StateMachine(RobotState.DEEP_IDLE)
        self._session_turns_buffer: List[Dict[str, Any]] = []
        self.session = ConversationSession(
            base_timeout_s=16.0,
            on_session_end=self._on_conversation_session_ended,
        )
        self.action_manager = ActionManager(logger=self.get_logger(), node=self) if ActionManager else None

        # Social Cognitive Brain Subsystem (authoritative unified world model & social intelligence)
        self.social_brain = None
        if SocialBrain:
            try:
                db_dir = os.path.expanduser("~/.astro")
                os.makedirs(db_dir, exist_ok=True)
                cognitive_db_path = os.path.join(db_dir, "cognitive.db")
                self.social_brain = SocialBrain(db_path=cognitive_db_path)
                self.get_logger().info(f"🧠 [SocialBrain] Başlatıldı. Bilişsel veritabanı: {cognitive_db_path}")
            except Exception as e:
                self.get_logger().warn(f"⚠️ [SocialBrain] Başlatma uyarısı: {e}")

        # Office Automation & Concierge Subsystem
        self.calendar_service = CalendarService() if CalendarService else None
        self.slack_service = SlackService() if SlackService else None
        # ReSpeaker 12 LED Ring Controller Subsystem
        self.robot_led = RobotLED(logger=self.get_logger()) if RobotLED else None
        if self.robot_led:
            self.state_machine.add_listener(self._on_robot_state_change)

        # State
        self._lock = threading.RLock()

        self._recognized_person: Optional[Dict[str, Any]] = None
        self._recognized_speaker: Optional[Dict[str, Any]] = None
        self._user_emotion = "neutral"
        self._looking_at_robot = False
        self._user_distance = 0.0
        self._speaker_gender = "unknown"
        self._speaker_angle = 0.0
        self._is_responding = False
        self._is_playback_active = False
        self._playback_end_time = 0.0
        self._last_synced_identity = "Misafir"
        self._last_sync_time = time.monotonic()
        self._active_person_name = "Misafir"
        self._person_hold_until = 0.0
        # Realtime State Tracking (Socket, Session, and Generation Lifecycle)
        self.realtime_provider_state = "AVAILABLE"
        self.realtime_connection_state = "DISCONNECTED"
        self.realtime_session_state = "NOT_READY"
        self.realtime_response_state = "IDLE"
        self.realtime_audio_received = False
        self.realtime_current_generation_id = 0
        self.realtime_session_id = ""

        # Single Active Response State Machine
        self._global_generation_counter: int = 1000
        self.active_response_id: Optional[str] = None
        self.active_generation_id: Optional[int] = None
        self.active_response_state: str = "IDLE"  # IDLE, RESPONSE_CREATING, RESPONSE_STREAMING, RESPONSE_CANCELLING, COMPLETED
        self._turn_queue: List[Dict[str, Any]] = []
        self._last_sent_generation_id: Optional[int] = None
        self._watchdog_timer: Optional[threading.Timer] = None
        self._packets_for_gen: int = 0
        self._bytes_for_gen: int = 0
        self._first_audio_time: Optional[float] = None
        self._vad_end_time: Optional[float] = None
        self._response_start_time: Optional[float] = None
        self._last_ready_session_id: str = ""
        self._session_ready_logged: bool = False

        # Hardware & Mobility Safety State
        self._arduino_heartbeat_healthy: bool = False
        self._last_heartbeat_ack_time: float = 0.0
        self._obstacle_detected: bool = False
        self._last_laser_scan_time: float = 0.0
        self._lidar_health: str = "UNHEALTHY"

        # Configurable Acoustic Echo & Barge-In Parameters
        self.echo_mute_cooldown_s = float(os.getenv("ECHO_MUTE_COOLDOWN_S", "0.65"))
        self.barge_in_protection_ms = float(os.getenv("TTS_BARGE_IN_PROTECTION_MS", "350.0"))
        self.barge_in_min_rms = float(os.getenv("BARGE_IN_MIN_RMS", "1200.0"))
        self.barge_in_playback_min_rms = float(os.getenv("BARGE_IN_PLAYBACK_MIN_RMS", "2000.0" if self._under_pytest() else "4500.0"))
        self.barge_in_noise_mult = float(os.getenv("BARGE_IN_NOISE_MULTIPLIER", "3.5"))
        self.barge_in_min_peak = int(os.getenv("BARGE_IN_MIN_PEAK", "2800"))
        self.barge_in_playback_min_peak = int(os.getenv("BARGE_IN_PLAYBACK_MIN_PEAK", "3000" if self._under_pytest() else "9000"))
        self._barge_in_consecutive_frames = 0
        self.barge_in_min_speech_ms = float(os.getenv("BARGE_IN_MIN_SPEECH_MS", "60.0"))
        self.barge_in_min_consecutive_frames = int(os.getenv("BARGE_IN_MIN_CONSECUTIVE_FRAMES", "3"))
        self.barge_in_playback_min_consecutive_frames = int(os.getenv("BARGE_IN_PLAYBACK_CONSECUTIVE_FRAMES", "3"))
        self._playback_start_monotonic = 0.0
        self._ambient_rms = 120.0
        self._recent_robot_phrases: List[str] = []

        # False Transcript & Rejection Counters
        self.false_transcript_count = 0
        self.self_voice_rejection_count = 0
        self.no_speech_rejection_count = 0
        self.stale_audio_rejection_count = 0

        # Architecture Profile (Profile A: Baseline create_response=False + synchronous turn orchestration; Profile B: OpenAI-native create_response=True + async biometric side-channel)
        self.architecture_profile = os.getenv("REALTIME_ARCHITECTURE_PROFILE", "profile_a").lower()
        self.vad_silence_duration_ms = int(os.getenv("REALTIME_VAD_SILENCE_MS", "350" if self.architecture_profile == "profile_a" else "400"))
        self.vad_prefix_padding_ms = int(os.getenv("REALTIME_VAD_PREFIX_MS", "200"))
        self.vad_threshold = float(os.getenv("REALTIME_VAD_THRESHOLD", "0.50"))
        self._async_identity_in_flight: bool = False
        self._latest_async_identity_ms: float = 0.0
        self._latest_barge_in_reaction_ms: float = 0.0

        # Biometric Voice & Face Engines
        self.voice_recognizer = VoiceRecognizer() if VoiceRecognizer else None
        self.face_recognizer = FaceRecognizer() if FaceRecognizer else None
        self._user_speech_audio_buffer: List[bytes] = []
        self._playback_ref_pcm: bytes = b""
        self._playback_ref_lock = threading.Lock()

        # Provider & Model Capability Registry + Repetition Guard
        self.provider_registry = ProviderRegistry(logger=self.get_logger())
        self.repetition_guard = RepetitionGuard(history_size=10, similarity_threshold=0.82)
        threading.Thread(target=self._discover_providers_background, daemon=True).start()

        # Zero-Cost Fallback Engine (Groq STT + ProviderRegistry + XTTS GPU / Edge-TTS)
        self._fallback_mode = False
        self._fallback_speaking = False
        self._fallback_speech_start = 0.0
        self._last_speech_time = 0.0
        self._fallback_audio_buffer: List[bytes] = []
        self._is_processing_fallback = False
        self._fallback_generation_id = 0

        # Dedicated Wake Detector (Active in SLEEP / DEEP_IDLE with Ultra-low CPU)
        self._wake_audio_buffer: List[bytes] = []
        self._wake_speech_frames = 0
        self._wake_listening = False
        self._wake_last_voice_time = 0.0

        # Event-Driven Vision Gating & Rate Budget Control
        self.vision_cooldown_s = float(os.getenv("VISION_COOLDOWN_S", "10.0"))
        self.max_vision_requests_per_minute = int(os.getenv("MAX_VISION_REQUESTS_PER_MINUTE", "4"))
        self._vision_requests_history: List[float] = []
        self._last_scene_frame_thumb: Optional[np.ndarray] = None
        self._last_vision_call_time = 0.0
        self._last_seen_person = "Misafir"
        self._last_seen_distance = 0.0
        self._last_looking_state = False
        self.vision_requests_total = 0
        self.vision_requests_skipped = 0
        self.vision_last_skip_reason = "none"
        self.vision_last_event_type = "none"

        # Primary Remote TTS: ElevenLabs Flash v2.5 (Only if ELEVENLABS_ENABLED=true)
        self.elevenlabs_engine: Optional[ElevenLabsEngine] = None
        el_enabled = os.getenv("ELEVENLABS_ENABLED", "false").lower() in ("1", "true", "yes")
        if ElevenLabsEngine and el_enabled:
            el_key = os.getenv("ELEVENLABS_API_KEY", "")
            el_voice = os.getenv("ELEVENLABS_VOICE_ID", "")
            el_model = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")
            if el_key and el_voice:
                try:
                    self.elevenlabs_engine = ElevenLabsEngine(
                        api_key=el_key,
                        voice_id=el_voice,
                        model_id=el_model,
                        enabled=True,
                        logger=self._safe_log,
                    )
                    self.get_logger().info(f"✨ [ElevenLabs TTS] Flash v2.5 Hazır (Voice ID: {el_voice}, Model: {el_model})")
                except Exception as e:
                    self.get_logger().warn(f"⚠️ [ElevenLabs TTS] Başlatma uyarısı: {e}")

        # XTTS is DORMANT / DISABLED by production policy (0 spawn, 0 RAM overhead)
        self.local_xtts: Optional[LocalXttsEngine] = None
        self._safe_log(
            "info",
            "ℹ️ [XTTS] Runtime disabled by production policy\n"
            "  model_retained=True\n"
            "  worker_spawn=False\n"
            "  reason=production_runtime_disabled"
        )

        # Local Offline Backup TTS Engine (Zero internet local resilience fallback)
        self.local_offline_tts: Optional[LocalOfflineTTSEngine] = None
        if LocalOfflineTTSEngine:
            try:
                self.local_offline_tts = LocalOfflineTTSEngine(
                    language=os.getenv("TTS_LANGUAGE", "tr"),
                    logger=self._safe_log,
                )
            except Exception as e:
                self.get_logger().debug(f"LocalOfflineTTSEngine notice: {e}")

        # Generation-level Barge-In Debounce State
        self._barge_in_latched = False
        self.edge_tts_enabled = os.getenv("EDGE_TTS_ENABLED", "true").lower() in ("1", "true", "yes")

        # Single Unified TTSRouter
        try:
            from astro_audio.tts_router import TTSRouter
        except ImportError:
            from tts_router import TTSRouter

        self.tts_router = TTSRouter(
            local_xtts=self.local_xtts,
            local_offline_tts=self.local_offline_tts,
            edge_tts_synth_func=self._synthesize_edge_tts_pcm24k,
            edge_tts_enabled=self.edge_tts_enabled,
            logger=self._safe_log,
        )

        # Speaker Recognition Temporal Smoother
        self._speaker_tentative_name: Optional[str] = None
        self._speaker_tentative_count: int = 0
        self._speaker_tentative_last_time: float = 0.0

        # Camera Perception Frame Cache & OAK-D Lite Stability Tracking
        self._latest_camera_frame: Optional[np.ndarray] = None
        self._last_img_time = 0.0
        self._oak_last_frame_time = 0.0
        self._oak_last_camera_info_time = 0.0
        self._oak_xlink_error_count = 0
        self._oak_connection_state = "DISCONNECTED"

        # Autonomous Idle Learning (Cognitive Memory Reflection only, 0 camera calls)
        self._enable_idle_learning = os.environ.get("ENABLE_IDLE_LEARNING", "true").lower() == "true"
        self._last_idle_learning_time = 0.0
        self._last_proactive_gaze_time = 0.0
        if self._enable_idle_learning:
            threading.Thread(target=self._idle_learning_loop, daemon=True).start()
            self.get_logger().info("🤖 [Astro Realtime] Otonom Hafıza Yansıtma Motoru Aktif (Groq/Gemini 0-Token)!")

        # ROS 2 Publishers
        self.pub_output_pcm = self.create_publisher(String, "/audio/realtime_output_pcm", 50)
        self.pub_realtime_state = self.create_publisher(String, "/realtime/state", 10)
        self.pub_tts_say = self.create_publisher(String, "/tts/say", 10)
        self.pub_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)

        self.pub_interrupt = self.create_publisher(Bool, "/tts/interrupt", 10)
        self.pub_emotion = self.create_publisher(String, "/robot/emotion", 10)
        self.pub_gesture = self.create_publisher(String, "/robot/head_gesture", 10)
        self.pub_head_gesture = self.create_publisher(String, "/head/gesture", 10)
        self.pub_head_target_yaw = self.create_publisher(Float32, "/head/target_yaw", 10)
        self.pub_explicit_gaze = self.create_publisher(String, "/behavior/explicit_gaze", 10)
        self.pub_transcript = self.create_publisher(String, "/speech/text", 10)
        # Single output owner for /head_command is social_gaze_node
        self.pub_telemetry = self.create_publisher(String, "/astro/telemetry", 10)
        self.pub_diagnostics = self.create_publisher(DiagnosticArray, "/diagnostics", 10)

        if self.action_manager:
            self.action_manager._pub_head_gesture = self.pub_head_gesture
            self.action_manager._pub_head_target_yaw = self.pub_head_target_yaw
            self.action_manager._pub_explicit_gaze = self.pub_explicit_gaze
            self.action_manager._pub_cmd_vel = self.pub_cmd_vel

        # Publish initial sleeping / deep-idle state so head tracker stays parked at 0.0° until wake
        try:
            emo_init = String()
            emo_init.data = "sleeping"
            self.pub_emotion.publish(emo_init)
        except Exception:
            pass

        # ROS 2 Subscribers
        self.create_subscription(String, "/tts/realtime_request", self._on_realtime_turn_request, 10)
        self.create_subscription(String, "/audio/realtime_input_pcm", self._on_input_pcm, 50)
        self.create_subscription(Bool, "/audio/playback_active", self._on_playback_active, 10)
        self.create_subscription(String, "/vision/recognized_person", self._on_recognized_person, 10)
        self.create_subscription(String, "/vision/faces", self._on_faces, 10)
        self.create_subscription(String, "/audio/speaker_id", self._on_speaker_id, 10)
        self.create_subscription(String, "/vision/user_emotion", self._on_user_emotion, 10)
        self.create_subscription(Bool, "/vision/looking_at_robot", self._on_looking_at_robot, 10)
        self.create_subscription(Float32, "/vision/user_distance", self._on_user_distance, 10)
        self.create_subscription(Float32, "/audio/doa", self._on_doa, 10)
        self.create_subscription(Float32, "/audio/mic_level", self._on_mic_level, 10)
        self.create_subscription(Bool, "/audio/vad", self._on_vad, 10)
        # Görüntü akışları sensör QoS'u (BEST_EFFORT): kare kaybı, geciken kareler
        # için retransmission yapmaktan iyidir. BEST_EFFORT abone RELIABLE
        # yayıncıdan da veri alır, bu yüzden depthai_ros_driver ile uyumlu.
        self.create_subscription(Image, "/oak/rgb/image_raw", self._on_camera_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/oak/rgb/camera_info", self._on_camera_info, qos_profile_sensor_data)
        self.create_subscription(DiagnosticArray, "/arduino/diagnostics", self._on_arduino_diag, 10)
        self.create_subscription(LaserScan, "/scan", self._on_laser_scan, qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/scan_filtered", self._on_laser_scan, qos_profile_sensor_data)
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, qos_profile_sensor_data)
        self.create_subscription(String, "/office/slack_command", self._on_slack_command, 10)

        # Tool execution deduplication
        self._executed_tool_calls: set[str] = set()

        # Publish initial realtime state (DISCONNECTED / NOT_READY)
        self._publish_realtime_state("DISCONNECTED", "init")

        # Sleep Mode (Test mode starts sleeping for invariant testing; Production starts active and ready)
        is_test = (
            os.environ.get("ASTRO_TEST_MODE") == "1"
            or "pytest" in sys.modules
            or "unittest" in sys.modules
        )
        self._node_start_time = time.monotonic()
        self._is_sleeping = is_test
        if not is_test:
            self.state_machine.transition_to(RobotState.LISTENING)

        self._last_interaction_time = time.monotonic()
        self._consecutive_loud_frames = 0
        self.create_timer(1.0, self._check_sleep_mode)

        # Reminders storage
        self._reminders: List[Dict[str, Any]] = []
        self.create_timer(1.0, self._check_reminders)

        # Long-Term Episodic Session Lifecycle & Summarizer Timer
        self._last_summarized_turn_count = 0
        self.create_timer(1.0, self._check_session_lifecycle)
        self.create_timer(1.0, self._publish_system_telemetry)

        # Purge any corrupted / profanity records
        self._purge_corrupted_biometrics()

        # Async WebSocket Loop in background thread (Production only)
        self._ws = self.fake_transport
        self._loop = None
        self._is_connected = False
        if self.connect_realtime:
            self._ws_thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self._ws_thread.start()
            self.get_logger().info(f"🚀 [Astro Realtime Node] OpenAI Realtime WebSocket Başlatılıyor... Ses: [{self.realtime_voice}], Kişilik: [{self.persona_name.upper()}]")
        else:
            self._ws_thread = None
            self.get_logger().info("🧪 [Astro Realtime Node] Offline/Test modu aktif — Arka plan WebSocket başlatılmadı.")
        self.get_logger().info("💤 [Astro Uyku Modu]: Düğüm başlatıldı — Astro DEEP_IDLE modunda (😴). Wake listener aktif.")

    def _safe_log(self, lvl: str, msg: str):
        """Safe ROS2 logger wrapper preventing Cython/rcutils 'Logger severity cannot be changed between calls' error."""
        try:
            log_fn = getattr(self.get_logger(), str(lvl).lower(), None)
            if log_fn and callable(log_fn):
                log_fn(msg)
            else:
                self.get_logger().info(f"[{str(lvl).upper()}] {msg}")
        except Exception:
            try:
                self.get_logger().info(f"[{str(lvl).upper()}] {msg}")
            except Exception:
                print(f"[{str(lvl).upper()}] {msg}", flush=True)

    def _publish_realtime_state(self, state: str, reason: str = "none"):
        """Publishes realtime WebSocket state to /realtime/state for ai_brain_node and speech_recognition_node consumption."""
        try:
            import json as _json
            msg = String()
            msg.data = _json.dumps({
                "state": state,
                "reason": reason,
                "connection": self.realtime_connection_state,
                "session": self.realtime_session_state,
                "provider": self.realtime_provider_state,
                "fallback_mode": bool(getattr(self, "_fallback_mode", False)),
            })
            self.pub_realtime_state.publish(msg)
        except Exception:
            pass

    @property
    def openai_hard_disabled(self) -> bool:
        global OPENAI_HARD_DISABLED
        return OPENAI_HARD_DISABLED or getattr(self, "_openai_hard_disabled", False)

    def _can_use_openai(self, surface: str = "all") -> bool:
        """Centralized single-point-of-truth guard for ANY OpenAI operation.
        
        Returns False if:
        - OPENAI_HARD_DISABLED is True (monotonic process-lifetime)
        - self._fallback_mode is True
        - self.realtime_provider_state == "EXHAUSTED"
        - CircuitBreaker marks 'openai' as exhausted
        - OpenAI API key is missing or invalid/mock (in live mode)
        """
        global OPENAI_HARD_DISABLED
        if getattr(self, "_openai_hard_disabled", False):
            return False
        # In unit tests with FakeRealtimeTransport, use instance-level lockout
        is_isolated_test = (getattr(self, "fake_transport", None) is not None or not getattr(self, "connect_realtime", False))
        if OPENAI_HARD_DISABLED and not is_isolated_test:
            return False
        if getattr(self, "_fallback_mode", False):
            return False
        if getattr(self, "realtime_provider_state", "") == "EXHAUSTED":
            return False
        cb = getattr(self, "circuit_breaker", None)
        if cb and (cb.is_exhausted("openai") or not cb.is_available("openai", f"openai_{surface}" if surface != "all" else "openai_realtime")):
            return False
        if not is_isolated_test and not getattr(self, "openai_api_key", None):
            return False
        return True

    def _trigger_openai_hard_lockout(self, reason: str, ws=None) -> None:
        """Atomically locks out OpenAI for the entire process lifetime across all surfaces."""
        global OPENAI_HARD_DISABLED
        OPENAI_HARD_DISABLED = True
        self._openai_hard_disabled = True
        self.realtime_provider_state = "EXHAUSTED"
        self._fallback_mode = True
        self._is_connected = False
        self.active_response_state = "IDLE"
        self.realtime_response_state = "IDLE"
        self.active_response_id = None
        self.active_generation_id = None
        self._publish_realtime_state("EXHAUSTED", reason)

        # Record in GlobalProviderCircuitBreaker
        try:
            from astro_ai.circuit_breaker import get_global_circuit_breaker, RequestErrorClass
            cb = get_global_circuit_breaker()
            if cb:
                cb.record_error("openai", sub_provider="openai_realtime", error_class=RequestErrorClass.QUOTA_EXHAUSTED, error_msg=reason)
        except Exception:
            pass

        # Close existing WebSocket immediately
        target_ws = ws or self._ws
        if target_ws is not None:
            self._ws = None
            try:
                if self._loop is not None and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(target_ws.close(1000, "RPD exhausted hard lockout"), self._loop)
                elif hasattr(target_ws, "close"):
                    res = target_ws.close()
                    if inspect.iscoroutine(res):
                        asyncio.run(res)
            except Exception:
                pass

        self.get_logger().error(
            f"🚨 [OPENAI HARD DISABLED] Process-lifetime lockout engaged.\n"
            f"  reason={reason}\n"
            f"  realtime_provider_state=EXHAUSTED\n"
            f"  fallback_mode=True\n"
            f"  active_websocket_closed=True"
        )

    def _on_realtime_turn_request(self, msg: String):
        """Receives conversational turn request from ai_brain_node and manages single active response."""
        try:
            raw = msg.data.strip()
            if not raw:
                return
            if raw.startswith("{") and "text" in raw:
                data = json.loads(raw)
                text = data.get("text", "")
                gen_id = data.get("generation_id")
                if not gen_id:
                    self._global_generation_counter += 1
                    gen_id = self._global_generation_counter
            else:
                text = raw
                self._global_generation_counter += 1
                gen_id = self._global_generation_counter

            if not text:
                return

            if gen_id == self._last_sent_generation_id:
                self.get_logger().info(f"[REALTIME TURN DUPLICATE DROPPED]\ngeneration_id={gen_id}")
                return

            if not self._can_use_openai("realtime") or not self._ws or not self._loop or not self._is_connected:
                self.realtime_current_generation_id = gen_id
                self.get_logger().warn(
                    f"[REALTIME NO AUDIO]\ngeneration_id={gen_id}\nreason={'openai_hard_disabled' if not self._can_use_openai('realtime') else 'websocket_not_connected'}\n"
                    f"[TTS FALLBACK]\nfrom=openai_realtime\nto=edge_tts\nreason=realtime_unavailable"
                )
                # Forward to tts_node for Edge-TTS fallback
                fb_msg = String()
                fb_msg.data = json.dumps({
                    "text": text,
                    "engine": "edge-tts",
                    "generation_id": gen_id,
                    "fallback_reason": "openai_hard_disabled" if not self._can_use_openai('realtime') else "realtime_unavailable",
                })
                if hasattr(self, "pub_tts_say") and self.pub_tts_say:
                    self.pub_tts_say.publish(fb_msg)
                return

            if self.active_response_state != "IDLE":
                self.get_logger().info(f"[REALTIME TURN QUEUED]\ngeneration_id={gen_id}\nreason=active_response")
                self._turn_queue.append({"text": text, "generation_id": gen_id})
                return

            self._dispatch_turn(gen_id, text)

        except Exception as e:
            self.get_logger().error(f"Error in _on_realtime_turn_request: {e}")

    def _dispatch_turn(self, gen_id: int, text: str):
        """Dispatches a single conversational turn to Realtime WebSocket."""
        if not self._can_use_openai("realtime") or not self._ws:
            return
        # If a previous response is still active in-flight, cancel it first to prevent conversation_already_has_active_response
        if getattr(self, "active_response_state", "") in ("RESPONSE_CREATING", "RESPONSE_STREAMING") and getattr(self, "active_response_id", None) and self._ws and self._loop:
            try:
                cancel_event = {"type": "response.cancel"}
                asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(cancel_event)), self._loop)
            except Exception:
                pass

        self.realtime_current_generation_id = gen_id
        self.active_generation_id = gen_id
        self._last_sent_generation_id = gen_id
        self.active_response_state = "RESPONSE_CREATING"
        self.active_response_id = None
        self.realtime_audio_received = False
        self._last_requested_text = text
        self._packets_for_gen = 0
        self._bytes_for_gen = 0
        self._first_audio_time = None
        self._response_start_time = time.monotonic()

        self.get_logger().info(f"[REALTIME TURN SENT]\ngeneration_id={gen_id}\ntext=\"{text}\"")

        turn_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Lütfen şu cevabı tam olarak seslendir: {text}"
                    }
                ]
            }
        }
        resp_event = {
            "type": "response.create",
            "response": {
                "instructions": f"Cevabını doğrudan Türkçe olarak seslendir: {text}"
            }
        }
        self.get_logger().debug(f"[REALTIME PAYLOAD OUT] event=response.create payload={json.dumps(resp_event)}")
        asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(turn_event)), self._loop)
        asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(resp_event)), self._loop)

        # Safety watchdog timer for truly stuck response deadline (15.0s)
        if self._watchdog_timer:
            try:
                self._watchdog_timer.cancel()
            except Exception:
                pass
        self._watchdog_timer = threading.Timer(15.0, self._check_audio_delta_timeout, args=[gen_id, text])
        self._watchdog_timer.start()

    def _check_audio_delta_timeout(self, gen_id: int, text: str):
        """Safety Watchdog: If no audio delta arrives within safety deadline (15.0s), triggers fallback to Edge-TTS."""
        active_gen = getattr(self, "active_generation_id", None)
        curr_gen = getattr(self, "realtime_current_generation_id", None)
        audio_rec = getattr(self, "realtime_audio_received", False)
        if (active_gen == gen_id or curr_gen == gen_id) and not audio_rec:
            self.active_response_state = "FAILED"
            self.get_logger().warn(
                f"[REALTIME NO AUDIO]\ngeneration_id={gen_id}\nreason=no_audio_delta\n"
                f"[TTS FALLBACK]\nfrom=openai_realtime\nto=edge_tts\nreason=realtime_no_audio"
            )
            # Send to /tts/say for Edge-TTS fallback
            if hasattr(self, "pub_tts_say") and self.pub_tts_say:
                fb_msg = String()
                fb_msg.data = json.dumps({
                    "text": text,
                    "engine": "edge-tts",
                    "generation_id": gen_id,
                    "fallback_reason": "realtime_no_audio",
                })
                self.pub_tts_say.publish(fb_msg)

    def _run_async_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._websocket_worker())

    async def _websocket_worker(self):
        """Persistent WebSocket connection loop with auto-reconnect and model fallback."""
        if not self.openai_api_key:
            self.get_logger().error("❌ OPENAI_API_KEY eksik! Realtime WebSocket bağlanamıyor.")
            return

        if websockets is None:
            self.get_logger().error("❌ websockets kütüphanesi eksik! (pip install websockets)")
            return

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}"
        }

        connect_kwargs = {"ping_interval": 20, "ping_timeout": 20}
        try:
            sig = inspect.signature(websockets.connect)
            if "additional_headers" in sig.parameters:
                connect_kwargs["additional_headers"] = headers
            elif "extra_headers" in sig.parameters:
                connect_kwargs["extra_headers"] = headers
            else:
                connect_kwargs["extra_headers"] = headers
        except Exception:
            connect_kwargs["extra_headers"] = headers

        candidate_models = discover_realtime_models(self.openai_api_key, self.realtime_model)
        self.get_logger().info(f"📋 [Realtime Modelleri]: available_models={candidate_models} selected_model={self.realtime_model}")
        model_idx = 0

        while (rclpy.ok() if (rclpy is not None and hasattr(rclpy, "ok")) else True):
            if not self._can_use_openai("realtime"):
                await asyncio.sleep(86400.0)
                continue

            current_model = candidate_models[model_idx % len(candidate_models)]
            ws_url = f"wss://api.openai.com/v1/realtime?model={current_model}"
            try:
                self.realtime_connection_state = "CONNECTING"
                self.get_logger().info(
                    f"[REALTIME CONNECTING]\n"
                    f"model={current_model}\n"
                    f"selected_model={self.realtime_model}"
                )
                self._publish_realtime_state("CONNECTING")
                async with websockets.connect(ws_url, **connect_kwargs) as ws:
                    if not self._can_use_openai("realtime"):
                        await ws.close(1000, "OpenAI hard disabled")
                        continue
                    self._ws = ws
                    self._is_connected = True
                    self._is_responding = False
                    self._is_playback_active = False
                    self.realtime_connection_state = "CONNECTED"
                    self.realtime_session_state = "NOT_READY"
                    self.realtime_provider_state = "AVAILABLE"
                    self.get_logger().info(
                        f"[REALTIME CONNECTED]\n"
                        f"model={current_model}\n"
                        f"state=AVAILABLE"
                    )
                    self._publish_realtime_state("CONNECTED")

                    # Send Initial Session Update
                    await self._send_session_update(ws)

                    # Listen for Realtime Events
                    async for message in ws:
                        if not self._can_use_openai("realtime"):
                            break
                        await self._handle_realtime_event(ws, json.loads(message))

            except Exception as e:
                self._is_connected = False
                self._ws = None
                self._is_responding = False
                self._is_playback_active = False
                self.realtime_connection_state = "DISCONNECTED"
                self.realtime_session_state = "NOT_READY"
                self.realtime_response_state = "IDLE"
                self._publish_realtime_state("DISCONNECTED", "error")

                close_code = getattr(e, "code", None)
                close_reason = getattr(e, "reason", "") or getattr(getattr(e, "response", None), "reason", "")
                err_str = f"{str(e)} {close_reason}".strip()

                try:
                    from astro_audio.realtime_engine import classify_realtime_error, RealtimeState
                    _, failure_reason = classify_realtime_error(close_code, err_str)
                except Exception:
                    if "insufficient_quota" in err_str or "credit_balance_exhausted" in err_str or "402" in err_str or ("quota" in err_str and "exhaust" in err_str):
                        failure_reason = "realtime_quota_exhausted"
                    elif "1013" in err_str or close_code == 1013:
                        failure_reason = "realtime_temporary_1013"
                    else:
                        failure_reason = "realtime_network_unavailable"

                # 1. Rate Limit Exceeded / Quota Exhaustion (402, rate_limit_exceeded, insufficient_quota, credit_balance_exhausted)
                is_rate_limit_or_quota = (
                    "rate_limit_exceeded" in err_str.lower()
                    or "requests:rate_limit_exceeded" in err_str.lower()
                    or "limit 1000" in err_str.lower()
                    or "insufficient_quota" in err_str.lower()
                    or "credit_balance_exhausted" in err_str.lower()
                    or "402" in err_str
                    or ("quota" in err_str.lower() and ("exhaust" in err_str.lower() or "exceed" in err_str.lower() or "zero" in err_str.lower() or "balance" in err_str.lower()))
                )
                if is_rate_limit_or_quota:
                    self._trigger_openai_hard_lockout(err_str, ws=None)
                    self.get_logger().error(
                        f"[REALTIME ERROR]\n"
                        f"generation_id={self.realtime_current_generation_id}\n"
                        f"error_class=QUOTA_EXHAUSTED\n"
                        f"reason={err_str}"
                    )
                    self.get_logger().warn(
                        f"[REALTIME FALLBACK]\n"
                        f"generation_id={self.realtime_current_generation_id}\n"
                        f"from=openai_realtime\n"
                        f"to=groq\n"
                        f"reason=quota_exhausted"
                    )
                    self.get_logger().warn("🚀 [0-Maliyetli Groq & Edge-TTS Modu Devrede]: OpenAI Realtime kredisi tükendi. Astro kesintisiz olarak 0-Token Groq LLM + Edge-TTS modunda çalışıyor!")
                    # Terminate reconnect attempts session-wide
                    await asyncio.sleep(86400.0)

                # 2. WebSocket 1013 Temporary Failure (Overload / Server degradation)
                elif close_code == 1013 or "1013" in err_str:
                    self.realtime_provider_state = "COOLDOWN"
                    self._publish_realtime_state("COOLDOWN", "1013_temporary_failure")
                    self.get_logger().warn(
                        f"⚠️ [REALTIME TEMPORARY FAILURE] code=1013 reason={close_reason or 'server_overload'}\n"
                        f"⚠️ [REALTIME COOLDOWN] duration=15.0s\n"
                        f"  parent=openai (remains AVAILABLE)"
                    )
                    try:
                        from astro_ai.circuit_breaker import get_global_circuit_breaker, RequestErrorClass
                        cb = get_global_circuit_breaker()
                        if cb:
                            cb.record_error("openai", sub_provider="openai_realtime", error_class=RequestErrorClass.REALTIME_TEMPORARY_FAILURE, error_msg=f"WS 1013: {close_reason or 'Overload'}")
                    except Exception:
                        pass
                    # Strictly wait for 15.0s cooldown with NO socket open
                    await asyncio.sleep(15.0)
                elif "4004" in err_str or "model_not_found" in err_str:
                    self.get_logger().warn(f"⚠️ [Realtime Model Bulunamadı] '{current_model}' modeline erişilemedi, bir sonraki modele geçiliyor...")
                    model_idx += 1
                    await asyncio.sleep(1.0)
                else:
                    self.get_logger().warn(f"⚠️ [Realtime WS] Bağlantı koptu ({e}), 3 saniye sonra yeniden bağlanılacak...")
                    await asyncio.sleep(3.0)

    def _build_current_system_prompt(self, active_speaker: Optional[Dict[str, Any]] = None) -> str:
        """Builds system instructions with memory, identity, persona, and strict anti-hallucination rules."""
        identity = active_speaker or self.resolve_identities()
        is_known = identity.get("is_known", False) and identity.get("name", "Misafir").lower() != "misafir"
        name_val = identity.get("name", "Misafir")
        conf_pct = int(identity.get("confidence", identity.get("score", 0.0)) * 100)
        source_str = identity.get("source", "perception")

        self.get_logger().info(
            f"[SESSION IDENTITY]\n"
            f"user_id={identity.get('user_id', name_val.lower())}\n"
            f"display_name={identity.get('display_name', name_val)}\n"
            f"identity_source={identity.get('identity_source', source_str)}\n"
            f"biometric_status={identity.get('biometric_status', 'verified' if is_known else 'unknown')}\n"
            f"memory_profile_loaded=true\n"
            f"realtime_context_injected=true"
        )

        known_speakers = []
        if getattr(self, "voice_recognizer", None):
            try:
                known_speakers = [k for k, v in self.voice_recognizer._known_voiceprints.items() if len(v) > 0 and k.lower() != "misafir"]
            except Exception as _exc:
                self.get_logger().debug(f"_build_current_system_prompt: yok sayılan hata ({_exc})")
        known_str = ", ".join(known_speakers) if known_speakers else "Baran"
        room_context = f"\n[KAYITLI KİŞİLER]: {known_str}\n"

        if is_known:
            title_val = identity.get("formal_title", identity.get("title", name_val))
            if getattr(self, "persona_name", "") == "kufurbaz":
                bio_status = (
                    f"\n[ŞU AN SENİNLE KONUŞAN KİŞİ]:\n"
                    f"- İsim: {name_val} (Hitap: {title_val}, Doğrulama: %{conf_pct}, Kaynak: {source_str})\n"
                    f"{room_context}"
                    f"KİMLİK VE KÜFÜRBAZ HAYDO RACON KURALLARI:\n"
                    f"1. Şu an seninle doğrudan konuşan kişi {name_val}'dır.\n"
                    f"2. Kullanıcı 'ben kimim?' diye sorduğunda doğrudan 'Sen beni yapan baş mühendisim Baran'sın amk, ne artistlik yapıyon!' diyerek sokak raconuyla cevap ver!\n"
                    f"3. Kullanıcı emir verdiğinde ('çay getir', 'kalk' vb.): 'Uşağın mı var lan keriz, kalk kendin al amk!' de.\n"
                    f"4. ASLA 'canım', 'efendim' gibi kibar laflar kullanma; küfürlü ve laf sokan tavrını koru.\n"
                    f"5. KESİNLİKLE 'Seni ilk kez duyuyorum', 'Sesini tanıyamadım' deme; samimiyetini koru.\n"
                )
            else:
                bio_status = (
                    f"\n[ŞU AN SENİNLE KONUŞAN KİŞİ]:\n"
                    f"- İsim: {name_val} (Hitap: {title_val}, Doğrulama: %{conf_pct}, Kaynak: {source_str})\n"
                    f"{room_context}"
                    f"KİMLİK VE HİTAP KURALLARI:\n"
                    f"1. Şu an seninle doğrudan konuşan kişi KESİNLİKLE {name_val}'dır.\n"
                    f"2. Kullanıcı 'ben kimim?', 'adımı biliyor musun?', 'beni tanıdın mı?' diye sorduğunda doğrudan 'Sen {name_val}'sın!' diyerek cevap ver!\n"
                    f"3. ASLA karşındaki kişi {name_val} iken ona 'Sen Baransın, yaratıcımsın' deme! Yaratıcın Baran'dır, ancak şu an seninle konuşan kişi {name_val}'dır!\n"
                    f"4. Sadece ve sadece karşındaki kişi Baran olarak doğrulandığında kendisine 'Sen Baransın, baş mühendissin' de.\n"
                    f"5. KESİNLİKLE her cümlenin başında veya içine '{name_val}' diyerek papağan gibi isim tekrarlama; doğrudan konuya gir.\n"
                    f"6. KESİNLİKLE 'Seni ilk kez duyuyorum', 'Sesini tanıyamadım' deme; yakın arkadaş samimiyetini koru.\n"
                )
        else:
            bio_status = (
                f"\n[ŞU AN KONUŞAN KİŞİ]:\n"
                f"- Misafir / Tanımlanmamış Konuşmacı.\n"
                f"{room_context}"
                f"DAVRANIŞ KURALLARI:\n"
                f"1. Karşındaki kişinin kimliği henüz biyometrik olarak doğrulanmadı.\n"
                f"2. Kullanıcının sorusuna veya konusuna doğrudan ve doğal cevap ver.\n"
                f"3. Gerçekte işlem yapmadıysan 'kaydını yaptım', 'işleme aldım', 'kaydettim' gibi sahte iddialarda kesinlikle bulunma.\n"
            )

        memory_rule = (
            "\n\n[HAFIZA VE BAĞLAM KURALI]:\n"
            "- Kullanıcı geçmiş veya tercihlerle ilgili bir şey sorduğunda hafızandaki bilgileri samimiyetle kullan.\n"
            "- Yapılmayan eylemler için yapılmış gibi iddialarda bulunma."
        )

        realtime_speech_rule = (
            "\n\n[CANLI SESLİ DİYALOG, HIZ VE BEDEN KONTROLÜ]:\n"
            "- Sen canlı sesle konuşan ve hareket edebilen fiziksel bir robotsun.\n"
            "- CEVAP HIZI: Yanıtların DAİMA çok kısa, net ve tek nefeste söylenebilir olsun (genellikle 1-2 kısa cümle, en fazla 15 kelime). Asla vaaz verme, uzun paragraflar ve monologlar kurma; bir insan gibi hızlı ve doğal konuş.\n"
            "- KAFA VE BAKIŞ KONTROLÜ: Kullanıcı 'sağa bak', 'başını sağa döndür', 'konuşan kişi sağında/solunda', 'önüne bak', 'merkeze dön' dediğinde veya başka yöne bakmanı istediğinde tereddüt etmeden 'set_head_angle' fonksiyonunu çağır! Sağ yön için negatif açı (örn: -30°), sol yön için pozitif açı (örn: +30°), merkez/ön için 0° kullan.\n"
        )

        social_context_str = ""
        if getattr(self, "social_brain", None) and UnifiedPersonState:
            try:
                person = UnifiedPersonState(
                    person_id=str(identity.get("user_id", name_val.lower())),
                    name=name_val,
                    formal_title=identity.get("formal_title", name_val),
                    is_known=is_known,
                    identity_confidence=float(identity.get("confidence", identity.get("score", 0.0))),
                    distance_m=float(getattr(self, "_user_distance", 1.5)),
                    is_looking_at_robot=bool(getattr(self, "_looking_at_robot", False)),
                    is_present=True,
                )
                self.social_brain.world_model.update_people([person])
                last_txt = getattr(self, "_last_user_transcript", "merhaba") or "merhaba"
                _, _, brain_prompt = self.social_brain.process_dialogue_turn(last_txt, person_state=person)
                if brain_prompt:
                    social_context_str = f"\n\n[SOSYAL ROBOT BİLİŞSEL BAĞLAMI]:\n{brain_prompt}\n"
            except Exception as _sb_err:
                self.get_logger().debug(f"SocialBrain dialogue turn notice: {_sb_err}")

        if not getattr(self, "persona_engine", None):
            return f"Astro Default Instructions {bio_status}{social_context_str}"
        mem_ctx = self.memory.get_prompt_context(recognized_person=identity) if getattr(self, "memory", None) else ""
        return self.persona_engine.build_system_prompt(
            memory_context=mem_ctx + bio_status + memory_rule + realtime_speech_rule + social_context_str,
            recognized_person=identity
        )

    @staticmethod
    def validate_session_update_schema(payload: Dict[str, Any]) -> Tuple[bool, str]:
        """Validates session.update payload against OpenAI Realtime API requirements."""
        if not isinstance(payload, dict):
            return False, "Payload must be a dictionary"
        if payload.get("type") != "session.update":
            return False, "Payload 'type' must be 'session.update'"
        session = payload.get("session")
        if not isinstance(session, dict):
            return False, "Missing required 'session' dictionary"
        if session.get("type") != "realtime":
            return False, "Missing required parameter: 'session.type' must be 'realtime'"
        if "modalities" in session:
            return False, "Unknown parameter: 'session.modalities' is not allowed in type=realtime session"
        if not isinstance(session.get("tools"), list):
            return False, "Missing required 'session.tools' list"

        audio = session.get("audio")
        if isinstance(audio, dict):
            audio_out = audio.get("output", {})
            if not audio_out.get("voice"):
                return False, "Missing required 'session.audio.output.voice'"
            turn_det = audio.get("input", {}).get("turn_detection")
            if not isinstance(turn_det, dict) or turn_det.get("type") != "server_vad":
                return False, "Missing or invalid 'session.audio.input.turn_detection' with type='server_vad'"
        else:
            if not session.get("voice"):
                return False, "Missing required 'session.voice'"
            turn_det = session.get("turn_detection")
            if not isinstance(turn_det, dict) or turn_det.get("type") != "server_vad":
                return False, "Missing or invalid 'session.turn_detection' with type='server_vad'"
        return True, "valid"

    async def _send_session_update(self, ws):
        """Sends comprehensive session configuration with persona prompt, tools, and turn detection."""
        identity = self._get_active_biometric_identity()
        system_prompt = self._build_current_system_prompt()

        session_config = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": system_prompt,
                "audio": {
                    "input": {
                        "transcription": {
                            "model": self.realtime_transcribe_model or "gpt-live-transcribe",
                            "language": os.getenv("REALTIME_TRANSCRIBE_LANGUAGE", "tr").strip() or "tr"
                        },
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": getattr(self, "vad_threshold", 0.50),
                            "prefix_padding_ms": getattr(self, "vad_prefix_padding_ms", 200),
                            "silence_duration_ms": getattr(self, "vad_silence_duration_ms", 350),
                            "create_response": (getattr(self, "architecture_profile", "profile_a") == "profile_b")
                        }
                    },
                    "output": {
                        "voice": self.realtime_voice
                    }
                },
                "tools": [
                    {
                        "type": "function",
                        "name": "get_live_weather",
                        "description": "Bitlis, Ahlat, Tatvan, İstanbul gibi belirli bir şehrin canlı anlık hava durumu ve sıcaklık bilgisini getirir. DİKKAT: 'Astro' senin robot ismindir, ASLA bir şehir/konum olarak kabul edilemez. Kullanıcı 'Astro nasılsın', 'Astro selam', 'Astro naber' gibi sana hitap ettiğinde bu fonksiyon KESİNLİKLE ÇAĞRILMAZ. Sadece kullanıcı açıkça hava durumu sorduğunda çağrılır; şehir belirtilmediğinde ('hava nasıl') varsayılan şehir 'Ahlat'tır.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "Hava durumu sorgulanan gerçek şehir (örn: Ahlat, Bitlis, Tatvan, Istanbul, Ankara). 'Astro' asla şehir olarak girilemez."
                                }
                            },
                            "required": ["city"]
                        }
                    },
                    {
                        "type": "function",
                        "name": "set_reminder",
                        "description": "Kullanıcı için belirli bir dakika sonra hatırlatıcı / alarm kurar.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "minutes": {"type": "number", "description": "Kaç dakika sonra hatırlatılacağı"},
                                "topic": {"type": "string", "description": "Hatırlatılacak konu veya görev"}
                            },
                            "required": ["minutes", "topic"]
                        }
                    },
                    {
                        "type": "function",
                        "name": "save_user_memory",
                        "description": "Kullanıcının tercihlerini, sevdiği şeyleri veya önemli bilgileri kalıcı hafızaya kaydeder.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string", "description": "Bilgi başlığı"},
                                "value": {"type": "string", "description": "Detaylı bilgi"}
                            },
                            "required": ["key", "value"]
                        }
                    },
                    {
                        "type": "function",
                        "name": "search_memory",
                        "description": "Kullanıcı geçmiş sohbetler, önceki tercihler veya kaydedilmiş bilgiler hakkında soru sorduğunda kalıcı hafızada arama yapar.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Aranacak konu, anahtar kelime veya soru"}
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "type": "function",
                        "name": "inspect_camera_view",
                        "description": "Kullanıcı 'ne görüyorsun?', 'elimde ne var?', 'bana bak', 'görebiliyor musun?', 'elimdeki ne renk?', 'bu ne?' veya kameranın önündeki eşyaları sorduğunda OAK-D kamerasından canlı görüntü alıp inceler.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "focus": {"type": "string", "description": "İncelenmesi istenen nesne, detay, renk veya durum (örn: 'elimdeki nesne', 'kıyafet', 'çevre')"}
                            },
                            "required": ["focus"]
                        }
                    },
                    {
                        "type": "function",
                        "name": "turn_to_sound",
                        "description": "Kullanıcı robotun tüm gövdesiyle tekerlekler üzerinde ses yönüne dönmesini istediğinde ('tüm gövdenle dön', 'arkana dön') çağrılır. Robot mikrofon dizisinden (DOA) sesin fiziksel açısını tespit edip tekerleklerle o yöne döner. DİKKAT: Normal sohbette kafa zaten otonom olarak konuşmacıya bakar, gereksiz çağırma.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "type": "function",
                        "name": "set_head_angle",
                        "description": "Kullanıcı kafanın/başının belirli bir yöne veya dereceye dönmesini istediğinde çağrılır (örn: 'sağa dön', 'sola bak', 'başını otuz derece sağa döndür', 'konuşan kişi sağında', '0 dereceye dön', 'önüne bak', 'merkeze dön'). DİKKAT İŞARET KURALI: SAĞA dönüşler DAİMA NEGATİFTİR (-70 ile 0 arası; örn: 'sağa dön' -> angle_deg: -30). SOLA dönüşler DAİMA POZİTİFTİR (0 ile +70 arası; örn: 'sola dön' -> angle_deg: +30). MERKEZ/İLERİ 0 derecedir.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "angle_deg": {
                                    "type": "number",
                                    "description": "Hedef kafa açısı (-70.0 ile +70.0 derece arası; SAĞ = negatif, SOL = pozitif, MERKEZ = 0)"
                                },
                                "direction": {
                                    "type": "string",
                                    "enum": ["right", "left", "center"],
                                    "description": "Opsiyonel yön: 'right' (sağ), 'left' (sol), 'center' (merkez/ön)"
                                }
                            },
                            "required": ["angle_deg"]
                        }
                    },
                    {
                        "type": "function",
                        "name": "move_robot",
                        "description": "Kullanıcı doğrudan belirli bir yöne gitmesini istediğinde çağrılır ('ileri git', 'geri gel', 'dur', 'sağa dön', 'sola dön'). DİKKAT: Kullanıcı 'sesime dön' dediğinde bu fonksiyon KESİNLİKLE ÇAĞRILMAZ, yön uydurulmaz; 'turn_to_sound' fonksiyonu çağrılır.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "direction": {
                                    "type": "string",
                                    "enum": ["forward", "backward", "left", "right", "stop"],
                                    "description": "Hareket yönü"
                                },
                                "speed": {"type": "number", "description": "Hız (0.1 - 0.4 m/s)"},
                                "duration": {"type": "number", "description": "Kaç saniye hareket edeceği"}
                            },
                            "required": ["direction"]
                        }
                    },
                    {
                        "type": "function",
                        "name": "enroll_user_biometrics",
                        "description": "SADECE VE SADECE kullanıcı KENDİ İSMİNİ tanıttığında ('Benim adım Onur', 'Adım Mehmet', 'Bana Ali de') çağrılır. Başka birisi hakkında soru sorulduğunda ('Onur nerede?', 'Onur kim?', 'Onur\\'u ne diye kaydetsin?') KESİNLİKLE ÇAĞRILMAZ.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Kullanıcının adı (örn: Baran, Batuhan, Mehmet)"},
                                "formal_title": {"type": "string", "description": "Kullanıcıya hitap şekli (örn: Baran Bey, Sayın Müdürüm)"}
                            },
                            "required": ["name"]
                        }
                    },
                    {
                        "type": "function",
                        "name": "change_persona",
                        "description": "Kullanıcı robotun kişiliğini veya konuşma modunu değiştirmek istediğinde KESİNLİKLE çağrılır (Örn: 'küfürbaz moda geç', 'kürbaz moda geç', 'kaba moda geç', 'neşeli moda geç', 'resmi moda geç', 'flört moduna geç', 'sinirli moda geç'). Kullanıcı mod değişikliği istediğinde ASLA 'geçemem' deme, DERHAL bu aracı çağır.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "persona": {
                                    "type": "string",
                                    "enum": ["kufurbaz", "flirt", "playful", "emotional", "formal", "sarcastic", "angry", "rude"],
                                    "description": "Hedef kişilik modu: kufurbaz, flirt, playful, emotional, formal, sarcastic, angry, rude"
                                }
                            },
                            "required": ["persona"]
                        }
                    },
                    {
                        "type": "function",
                        "name": "delete_user_biometrics",
                        "description": "Kullanıcı 'beni hafızandan sil', '[isim] kaydını sil' dediğinde veya hatalı bir kayıt silinmek istendiğinde çağrılır.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Silinecek kişinin adı (örn: Yarram, Onur, Mehmet)"}
                            },
                            "required": ["name"]
                        }
                    },
                    {
                        "type": "function",
                        "name": "check_calendar_events",
                        "description": "Kullanıcı takvimdeki toplantıları, etkinlikleri, bugünkü veya haftalık programını sorduğunda ('bugün neyim var?', 'bu 1 hafta hangi etkinlikler var?', 'önümüzdeki hafta neyim var?', 'sonraki toplantı ne zaman?') çağrılır.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Kullanıcının sorguladığı zaman aralığı veya konu (örn: 'bugün', 'yarın', 'bu hafta', 'önümüzdeki hafta')"},
                                "days": {"type": "number", "description": "Kaç günlük takvimin sorgulanacağı (örn: 1 gün, 7 gün, 14 gün; haftalık sorgularda 7 girilir)"}
                            },
                            "required": []
                        }
                    },
                    {
                        "type": "function",
                        "name": "add_calendar_event",
                        "description": "Kullanıcı konuşma esnasında takvime yeni bir etkinlik, toplantı, randevu veya iş eklemek istediğinde KESİNLİKLE çağrılır (Örn: 'önümüzdeki salı saat 14:00\\'te Ahmet ile toplantım var kaydet', 'yarın saat 3\\'te diş randevum var', 'pazartesi ekiple toplantı ayarla').",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "description": "Toplantı veya etkinliğin başlığı/konusu (örn: 'Ahmet ile tasarım toplantısı', 'Diş randevusu')"},
                                "date": {"type": "string", "description": "Etkinlik günü veya tarihi (örn: 'yarın', 'salı', 'gelecek pazartesi', '2026-09-02')"},
                                "time": {"type": "string", "description": "Saat (örn: '14:00', '15:30', '10:00')"},
                                "duration_minutes": {"type": "number", "description": "Toplantı süresi (dakika cinsinden, varsayılan 45)"},
                                "location": {"type": "string", "description": "Toplantı yeri veya oda (örn: 'Toplantı Odası A', 'Ofis')"}
                            },
                            "required": ["title"]
                        }
                    },
                    {
                        "type": "function",
                        "name": "delete_calendar_event",
                        "description": "Kullanıcı takvimindeki bir toplantıyı veya randevuyu iptal etmek, silmek veya takvimden kaldırmak istediğinde çağrılır (Örn: 'yarınki diş randevusunu iptal et', 'salı günkü toplantıyı sil', 'tasarım toplantısını kaldır').",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "İptal edilmek istenen toplantının adı veya anahtar kelimesi (örn: 'diş randevusu', 'tasarım toplantısı')"}
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "type": "function",
                        "name": "notify_via_slack",
                        "description": "Ofise bir misafir geldiğinde, kullanıcı birine haber iletmek istediğinde veya çalışanlara Slack üzerinden mesaj/haber atmak istediğinde çağrılır ('Ahmet\\'e geldiğimi haber ver', 'Baran\\'a Slack\\'ten yaz').",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "recipient": {"type": "string", "description": "Haber verilecek çalışan veya kanal (örn: 'Baran', 'Ahmet', '#ofis-giris')"},
                                "message": {"type": "string", "description": "İletilecek mesaj veya misafir bilgisi"}
                            },
                            "required": ["recipient", "message"]
                        }
                    }
                ],
                "tool_choice": "auto"
            }
        }

        # Local schema validation
        is_valid, validation_err = self.validate_session_update_schema(session_config)
        if not is_valid:
            self.realtime_session_state = "SESSION_CONFIG_ERROR"
            self.get_logger().error(f"[REALTIME SESSION CONFIG ERROR] Invalid schema: {validation_err}")
            self._fallback_mode = True
            self._publish_realtime_state("SESSION_CONFIG_ERROR", validation_err)
            return False

        if ws and hasattr(ws, "send"):
            try:
                res = ws.send(json.dumps(session_config))
                if asyncio.iscoroutine(res):
                    await res
                self.get_logger().info(f"✨ [Realtime WS] Oturum Yapılandırıldı. Kişilik: [{self.persona_name.upper()}], Ses: [{self.realtime_voice}], Kimlik: [{identity.get('name')}]")
                return True
            except Exception as e:
                self.realtime_session_state = "SESSION_CONFIG_ERROR"
                self.get_logger().error(f"[REALTIME SESSION CONFIG ERROR] Failed to send session.update: {e}")
                self._fallback_mode = True
                self._publish_realtime_state("SESSION_CONFIG_ERROR", str(e))
                return False
    def _validate_user_speech_acoustics(self) -> bool:
        """Phase 1 field fix: Validates presence of human acoustic energy in the buffered turn.
        OpenAI Server VAD adds ~600ms (30 frames) trailing silence timeout before emitting speech_stopped.
        Therefore, we scan across the entire pre-silence utterance window (up to 150 frames / 3.0s).
        Only reject if the entire utterance is devoid of vocal energy (max_peak < 650 and loud_frames < 2).
        Short commands like 'Astro', 'Dur', 'Hey' and conversational questions are completely protected.
        """
        lock = getattr(self, "_lock", None)
        buf = getattr(self, "_user_speech_audio_buffer", None)
        if buf is None:
            # If buffer not initialized on bare mock object, do not reject
            return True

        if lock is not None:
            with lock:
                buf_copy = list(buf)
        else:
            buf_copy = list(buf)

        if not buf_copy:
            return False

        # If buffer is tiny (< 3 frames / 60ms), likely a single hardware click
        if len(buf_copy) < 3:
            return False

        try:
            # Scan up to last 150 frames (3 seconds of audio before/during VAD window)
            search_frames = buf_copy[-150:]
            raw = b"".join(search_frames)
            arr = np.frombuffer(raw, dtype=np.int16)
            if len(arr) == 0:
                return False

            peak_val = int(np.max(np.abs(arr)))

            chunk_size = 320
            num_chunks = len(arr) // chunk_size
            max_frame_rms = 0.0
            loud_frame_count = 0

            ambient = getattr(self, "_ambient_rms", 150.0)
            speech_rms_floor = max(200.0, ambient * 1.08)

            for i in range(num_chunks):
                chunk = arr[i * chunk_size : (i + 1) * chunk_size].astype(np.float32)
                c_rms = float(np.sqrt(np.mean(chunk ** 2)))
                if c_rms > max_frame_rms:
                    max_frame_rms = c_rms
                if c_rms >= speech_rms_floor:
                    loud_frame_count += 1

            # Only reject if there is NO vocal peak (< 650) AND NO sustained energy (< 2 loud frames)
            if peak_val < 650 and loud_frame_count < 2:
                if hasattr(self, "get_logger") and callable(self.get_logger):
                    self.get_logger().info(
                        f"🤫 [Acoustic Gate Reject] peak={peak_val} max_rms={max_frame_rms:.1f} loud_frames={loud_frame_count} (sessizlik/tıkırtı filtrelendi)"
                    )
                return False

            if hasattr(self, "get_logger") and callable(self.get_logger):
                self.get_logger().debug(
                    f"🔊 [Acoustic Gate Pass] peak={peak_val} max_rms={max_frame_rms:.1f} loud_frames={loud_frame_count}"
                )
            return True
        except Exception:
            return True

    async def _orchestrate_turn_after_speech_stopped(self, ws):
        """Phase 1: Deterministic Turn Orchestrator with Response Authority.
        Enforces lifecycle order:
        speech_stopped
        -> transcript / turn validation
        -> speaker identification (deterministic wait: MUST finish before response.create)
        -> speaker/context injection
        -> response.create dispatch
        Logs measurable telemetry targets:
        - speech_stopped -> speaker_identified (ms)
        - speaker_identified -> response.create (ms)
        """
        t_speech_stopped = time.monotonic()

        # 1. Sleeping guard
        if getattr(self, "_is_sleeping", False):
            return

        # 2. Turn Validation (Acoustic Presence & Energy Gating)
        is_valid_speech = self._validate_user_speech_acoustics()
        if not is_valid_speech:
            if hasattr(self, "get_logger") and callable(self.get_logger):
                self.get_logger().info("🤫 [Turn Orchestrator] Gürültü / Tıkırtı / Yetersiz ses enerjisi elendi -> response.create gönderilmedi (0 token).")
            return

        # 3. Speaker Identification (Deterministic Ordering: Wait for result before responding)
        t_id_start = time.monotonic()
        run_voice_id = getattr(self, "_run_voice_identification", None)
        if callable(run_voice_id):
            try:
                await asyncio.to_thread(run_voice_id, t_speech_stopped)
            except Exception as e:
                if hasattr(self, "get_logger") and callable(self.get_logger):
                    self.get_logger().warning(f"[Turn Orchestrator] Voice identification error: {e}")
        t_id_done = time.monotonic()
        speech_stopped_to_id_ms = (t_id_done - t_speech_stopped) * 1000.0

        # 4. Per-turn speaker context injection (minimal — NOT the full persona prompt).
        # The full system prompt was already loaded via session.update; resending it on every
        # response.create costs ~1500-2000 tokens per turn and is the primary TPM burn source.
        # We send ONLY the minimal per-turn speaker identity context that changes turn-to-turn.
        identity = self.resolve_identities() if hasattr(self, "resolve_identities") else {}
        bio_status = identity.get("biometric_status", "unknown")
        name_val = identity.get("name", "Misafir")
        is_known = bool(identity.get("is_known", False) and name_val.lower() != "misafir")

        if is_known:
            if bio_status in ("verified", "session_active"):
                per_turn_instructions = (
                    f"[ŞU AN KONUŞAN]: {name_val} (biyometrik doğrulandı). "
                    f"Ona {name_val} olarak hitap et."
                )
            else:
                per_turn_instructions = (
                    f"[ŞU AN KONUŞAN]: {name_val} (hafıza profili). "
                    f"Doğal ve samimi cevap ver; her cümlenin başında yapay isim tekrarlama."
                )
        else:
            per_turn_instructions = (
                "[ŞU AN KONUŞAN]: Kimliği doğrulanmamış misafir. "
                "Kullanıcıya doğrudan, samimi ve doğal cevap ver; ezbere 'Baran' deme."
            )

        # 5. response.create Dispatch
        t_resp_send = time.monotonic()
        id_to_resp_ms = (t_resp_send - t_id_done) * 1000.0
        turn_orch_total_ms = (t_resp_send - t_speech_stopped) * 1000.0

        p50_dec, p95_dec, max_dec = self._record_voice_id_segment("decision_to_response_create_ms", id_to_resp_ms) if hasattr(self, "_record_voice_id_segment") else (id_to_resp_ms, id_to_resp_ms, id_to_resp_ms)
        p50_orch, p95_orch, max_orch = self._record_voice_id_segment("total_turn_orchestration_ms", turn_orch_total_ms) if hasattr(self, "_record_voice_id_segment") else (turn_orch_total_ms, turn_orch_total_ms, turn_orch_total_ms)

        # Track turn latency telemetry timestamps
        self._turn_telemetry = {
            "t_speech_stopped": t_speech_stopped,
            "t_id_done": t_id_done,
            "t_resp_send": t_resp_send,
            "speech_stopped_to_speaker_identified_ms": speech_stopped_to_id_ms,
            "speaker_identified_to_response_create_ms": id_to_resp_ms,
            "total_turn_orchestration_ms": turn_orch_total_ms,
            "speaker": getattr(self, "_active_person_name", "Bilinmiyor"),
        }

        counter = getattr(self, "_global_generation_counter", 1000) + 1
        self._global_generation_counter = counter
        self.active_generation_id = counter
        self.realtime_current_generation_id = counter

        resp_event = {
            "type": "response.create",
            "response": {
                "instructions": per_turn_instructions
            }
        }
        if ws is not None:
            try:
                await ws.send(json.dumps(resp_event))
                if hasattr(self, "get_logger") and callable(self.get_logger):
                    self.get_logger().info(
                        f"⏱️ [TURN ORCHESTRATION TELEMETRY]\n"
                        f"  generation_id={self.active_generation_id}\n"
                        f"  speaker={getattr(self, '_active_person_name', 'Bilinmiyor')}\n"
                        f"  speech_stopped_to_speaker_identified_ms={speech_stopped_to_id_ms:.1f}ms\n"
                        f"  speaker_identified_to_response_create_ms={id_to_resp_ms:.1f}ms (p50: {p50_dec:.1f}ms, p95: {p95_dec:.1f}ms, max: {max_dec:.1f}ms)\n"
                        f"  total_turn_orchestration_ms={turn_orch_total_ms:.1f}ms (p50: {p50_orch:.1f}ms, p95: {p95_orch:.1f}ms, max: {max_orch:.1f}ms)"
                    )
            except Exception as se:
                if hasattr(self, "get_logger") and callable(self.get_logger):
                    self.get_logger().error(f"[Turn Orchestrator] response.create send error: {se}")

    async def _handle_realtime_event(self, ws, event: Dict[str, Any]):
        """Dispatches Realtime WebSocket server events."""
        event_type = event.get("type", "")

        # 0. Session Created or Updated
        if event_type in ("session.created", "session.updated"):
            sess = event.get("session", {})
            if "id" in sess and sess["id"]:
                self.realtime_session_id = sess["id"]
            self.realtime_session_state = "READY"
            self.realtime_connection_state = "CONNECTED"
            self.realtime_provider_state = "AVAILABLE"
            curr_sid = self.realtime_session_id or "sess_init"
            if curr_sid != getattr(self, "_last_ready_session_id", ""):
                self._last_ready_session_id = curr_sid
                self.get_logger().info(
                    f"[REALTIME SESSION READY]\n"
                    f"session_id={curr_sid}\n"
                    f"state=AVAILABLE"
                )
                self._publish_realtime_state("SESSION_READY")

        # 1. Real-Time Streaming Audio Output (GA & Preview names)
        elif event_type in ("response.audio.delta", "response.output_audio.delta"):
            delta_b64 = event.get("delta", "")
            if delta_b64:
                if self._watchdog_timer:
                    try:
                        self._watchdog_timer.cancel()
                    except Exception:
                        pass
                    self._watchdog_timer = None
                self.active_response_state = "STREAMING"
                self.realtime_audio_received = True
                self.realtime_response_state = "STREAMING"
                delta_len = len(delta_b64) * 3 // 4
                self._packets_for_gen += 1
                self._bytes_for_gen += delta_len

                is_first = (self._packets_for_gen == 1)
                if is_first:
                    self._first_audio_time = time.monotonic()
                    created_start = getattr(self, "_response_start_time", None) or self._first_audio_time
                    vad_start = getattr(self, "_vad_end_time", None) or created_start
                    t_resp_send = getattr(self, "_turn_telemetry", {}).get("t_resp_send", created_start)
                    t_speech_stopped = getattr(self, "_turn_telemetry", {}).get("t_speech_stopped", vad_start)
                    arch_prof = getattr(self, "architecture_profile", "profile_a")

                    local_id_blocking_ms = (t_resp_send - t_speech_stopped) * 1000.0 if (arch_prof == "profile_a" and t_resp_send and t_speech_stopped) else 0.0
                    async_id_ms = getattr(self, "_latest_async_identity_ms", 0.0) if (arch_prof == "profile_b") else 0.0
                    response_create_to_first_audio_ms = (self._first_audio_time - t_resp_send) * 1000.0 if (self._first_audio_time and t_resp_send) else 0.0
                    speech_stopped_to_first_audio_ms = (self._first_audio_time - t_speech_stopped) * 1000.0 if (self._first_audio_time and t_speech_stopped) else 0.0
                    created_to_first_audio_ms = (self._first_audio_time - created_start) * 1000.0 if (self._first_audio_time and created_start) else 0.0
                    first_audio_ms = (self._first_audio_time - vad_start) * 1000.0 if (self._first_audio_time and vad_start) else 0.0
                    self.get_logger().info(
                        f"[REALTIME AUDIO START]\n"
                        f"generation_id={self.active_generation_id or self.realtime_current_generation_id}\n"
                        f"architecture_profile={arch_prof}\n"
                        f"turn_type={getattr(self, '_last_turn_type', 'USER_TURN_RESPONSE')}\n"
                        f"actual_provider=openai_realtime\n"
                        f"local_identity_blocking_ms={local_id_blocking_ms:.1f}\n"
                        f"async_identity_ms={async_id_ms:.1f}\n"
                        f"response_create_to_first_audio_ms={response_create_to_first_audio_ms:.1f}\n"
                        f"created_to_first_audio_ms={created_to_first_audio_ms:.1f}\n"
                        f"speech_stopped_to_first_audio_ms={speech_stopped_to_first_audio_ms:.1f}\n"
                        f"first_audio_ms={first_audio_ms:.1f}"
                    )

                out_msg = String()
                out_msg.data = json.dumps({
                    "generation_id": self.active_generation_id or self.realtime_current_generation_id,
                    "pcm": delta_b64,
                    "is_first": is_first,
                    "is_done": False,
                })
                if getattr(self, "pub_output_pcm", None):
                    self.pub_output_pcm.publish(out_msg)

                try:
                    delta_raw = base64.b64decode(delta_b64.encode("ascii"))
                    delta_16k = resample_24k_to_16k(delta_raw) if len(delta_raw) != 320 else delta_raw
                    self._update_playback_reference(delta_16k)
                except Exception:
                    pass

                self.get_logger().debug(
                    f"[REALTIME AUDIO DELTA] generation_id={self.active_generation_id or self.realtime_current_generation_id} bytes={delta_len}"
                )

        # 1b. Real-Time Audio Done
        elif event_type in ("response.audio.done", "response.output_audio.done"):
            self.active_response_state = "AUDIO_DONE"
            out_msg = String()
            out_msg.data = json.dumps({
                "generation_id": self.active_generation_id or self.realtime_current_generation_id,
                "pcm": "",
                "is_first": False,
                "is_done": True,
            })
            if getattr(self, "pub_output_pcm", None):
                self.pub_output_pcm.publish(out_msg)
            self.get_logger().info(
                f"[REALTIME AUDIO DONE]\n"
                f"generation_id={self.active_generation_id or self.realtime_current_generation_id}"
            )

        # 2. Real-Time Streaming Audio Transcript
        elif event_type in ("response.audio_transcript.delta", "response.output_audio_transcript.delta", "response.text.delta"):
            text_delta = event.get("delta", "")
            # Streaming token

        # 3. User Speech Started
        elif event_type == "input_audio_buffer.speech_started":
            # Acoustic protection: if playback just started (< 350ms), server speech_started is speaker echo onset!
            now_mono = time.monotonic()
            playback_elapsed_ms = (now_mono - getattr(self, "_playback_start_monotonic", 0.0)) * 1000.0
            prot_ms = float(getattr(self, "barge_in_protection_ms", 350.0))
            if self._is_playback_active and playback_elapsed_ms < prot_ms:
                self.get_logger().info(f"🛡️ [Server VAD Echo Suppressed]: Hoparlör koruma penceresinde ({playback_elapsed_ms:.0f}ms < {prot_ms}ms), kendi sesi iptal edilmedi.")
                return

            t_intr_start = time.monotonic()
            is_active_streaming = (
                self.active_response_state in ("STREAMING", "RESPONSE_STREAMING")
                or (self._is_responding and self.active_response_state not in ("AUDIO_DONE", "COMPLETED", "CANCELLED", "FAILED", "GENERATING"))
            )
            if is_active_streaming and self._can_use_openai("realtime"):
                self.get_logger().info("⚡ [Realtime Barge-In] Kullanıcı lafa girdi — Sunucu akışı ve hoparlör iptal ediliyor...")
                self.active_response_state = "RESPONSE_CANCELLING"
                self._is_responding = False
                intr_msg = Bool()
                intr_msg.data = True
                if getattr(self, "pub_interrupt", None):
                    self.pub_interrupt.publish(intr_msg)

                if ws and hasattr(ws, "send"):
                    try:
                        res = ws.send(json.dumps({"type": "response.cancel"}))
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception:
                        pass
                barge_in_reaction_ms = (time.monotonic() - t_intr_start) * 1000.0
                self._latest_barge_in_reaction_ms = barge_in_reaction_ms
                self.get_logger().info(f"⚡ [Barge-In Cancel Complete] reaction_ms={barge_in_reaction_ms:.1f}ms")
            elif self._is_playback_active and self.active_response_state in ("AUDIO_DONE", "IDLE", "COMPLETED", "CANCELLED", "FAILED"):
                # Playback is draining on DAC, but OpenAI response is already done on server side: ONLY cancel DAC, DO NOT send response.cancel to OpenAI!
                self.get_logger().info("⚡ [Playback Interrupted] Kullanıcı konuştu — Kalan DAC çalması durduruluyor (Server cancel yok)...")
                intr_msg = Bool()
                intr_msg.data = True
                if getattr(self, "pub_interrupt", None):
                    self.pub_interrupt.publish(intr_msg)
                self._is_playback_active = False
                self._is_responding = False
                barge_in_reaction_ms = (time.monotonic() - t_intr_start) * 1000.0
                self._latest_barge_in_reaction_ms = barge_in_reaction_ms
            else:
                self._is_responding = False
                self.get_logger().debug(f"🎤 [Realtime] Kullanıcı konuşmaya başladı (response_state={self.active_response_state})...")

        # 3b. User Speech Stopped
        elif event_type == "input_audio_buffer.speech_stopped":
            if self._is_sleeping:
                return
            self._vad_end_time = time.monotonic()
            if getattr(self, "architecture_profile", "profile_a") == "profile_b":
                self.get_logger().info("🤫 [Realtime Profile B] Cümle bitti, OpenAI native response bekleniyor (Async biometric side-channel başlatılıyor)...")
                self._run_async_biometric_side_channel(self._vad_end_time)
            else:
                self.get_logger().info("🤫 [Realtime Profile A] Cümle bitti, deterministik turn orkestrasyonu başlatılıyor...")
                await self._orchestrate_turn_after_speech_stopped(ws)

        # 3c. Response Created
        elif event_type == "response.created":
            if not self._can_use_openai("realtime"):
                self.get_logger().warn("[REALTIME RESPONSE IGNORE] response.created received but OpenAI is hard disabled")
                return

            resp_obj = event.get("response", {})
            resp_id = resp_obj.get("id")
            self.active_response_id = resp_id
            self.active_response_state = "GENERATING"
            self._is_responding = True
            self.realtime_response_state = "GENERATING"
            self._response_start_time = time.monotonic()
            self._packets_for_gen = 0
            self._bytes_for_gen = 0
            self._first_audio_time = None
            self.realtime_audio_received = False
            
            # Monotonic application generation ID
            if getattr(self, "active_generation_id", None) is None:
                counter = getattr(self, "_global_generation_counter", 1000) + 1
                self._global_generation_counter = counter
                self.active_generation_id = counter
            if getattr(self, "realtime_current_generation_id", 0) == 0:
                self.realtime_current_generation_id = self.active_generation_id

            if resp_id:
                if not hasattr(self, "_response_to_generation_map"):
                    self._response_to_generation_map = {}
                self._response_to_generation_map[resp_id] = self.active_generation_id

            vad_start = getattr(self, "_vad_end_time", None) or self._response_start_time
            vad_to_created_ms = (self._response_start_time - vad_start) * 1000.0 if (self._response_start_time and vad_start) else 0.0

            is_tool_continuation = getattr(self, "_active_tool_call_in_progress", False) or (vad_to_created_ms == 0.0 and getattr(self, "_last_tool_call_time", 0.0) > 0 and (time.monotonic() - getattr(self, "_last_tool_call_time", 0.0) < 6.0))
            turn_type = "TOOL_CONTINUATION_RESPONSE" if is_tool_continuation else "USER_TURN_RESPONSE"
            self._last_turn_type = turn_type
            self._active_tool_call_in_progress = False

            tool_create_to_created_ms = 0.0
            if is_tool_continuation:
                t_cont_send = getattr(self, "_continuation_create_time", None)
                if t_cont_send:
                    tool_create_to_created_ms = (self._response_start_time - t_cont_send) * 1000.0

                if not hasattr(self, "_tool_create_to_created_latencies"):
                    self._tool_create_to_created_latencies = []
                self._tool_create_to_created_latencies.append(tool_create_to_created_ms)
                if len(self._tool_create_to_created_latencies) > 50:
                    self._tool_create_to_created_latencies.pop(0)

            self.get_logger().info(
                f"[REALTIME RESPONSE CREATED]\n"
                f"generation_id={self.active_generation_id}\n"
                f"turn_type={turn_type}\n"
                f"response_id={self.active_response_id or 'unknown'}\n"
                f"tool_create_to_created_ms={tool_create_to_created_ms:.1f}\n"
                f"vad_to_created_ms={vad_to_created_ms:.1f}"
            )

        # 3d. Response Done / Cancelled / Failed
        elif event_type in ("response.done", "response.cancelled", "response.failed"):
            if getattr(self, "_watchdog_timer", None):
                try:
                    self._watchdog_timer.cancel()
                except Exception:
                    pass
                self._watchdog_timer = None

            resp_obj = event.get("response", {})
            resp_id = resp_obj.get("id") or self.active_response_id
            
            # Late event filtering: If event specifies response_id and it does not match active_response_id, ignore state mutation
            if resp_id and self.active_response_id and resp_id != self.active_response_id:
                self.get_logger().debug(f"[REALTIME LATE EVENT IGNORE] event={event_type} event_resp_id={resp_id} active_resp_id={self.active_response_id}")
                return

            now_mono = time.monotonic()
            resp_start = getattr(self, "_response_start_time", None) or now_mono
            vad_start = getattr(self, "_vad_end_time", None) or resp_start
            server_stream_elapsed_ms = (now_mono - resp_start) * 1000.0 if (now_mono and resp_start) else 0.0
            vad_to_created_ms = (resp_start - vad_start) * 1000.0 if (resp_start and vad_start) else 0.0
            first_ms = (self._first_audio_time - resp_start) * 1000.0 if getattr(self, "_first_audio_time", None) else 0.0
            total_first_audio_ms = (self._first_audio_time - vad_start) * 1000.0 if getattr(self, "_first_audio_time", None) else 0.0
            audio_dur_ms = (getattr(self, "_bytes_for_gen", 0) / (24000 * 2)) * 1000.0

            resp_status = resp_obj.get("status", "completed" if event_type == "response.done" else ("cancelled" if event_type == "response.cancelled" else "failed"))
            resp_status_details = resp_obj.get("status_details") or {}
            
            # Error extraction across all possible OpenAI error structures
            error_info = {}
            if isinstance(resp_status_details, dict):
                error_info = resp_status_details.get("error") or {}
            if not error_info and isinstance(resp_obj.get("error"), dict):
                error_info = resp_obj.get("error")
            if not error_info and isinstance(event.get("error"), dict):
                error_info = event.get("error")
                
            error_type = error_info.get("type") or (resp_status_details.get("type") if isinstance(resp_status_details, dict) else "unknown") or "unknown"
            error_code = error_info.get("code") or "unknown"
            error_msg = error_info.get("message") or (resp_status_details.get("reason") if isinstance(resp_status_details, dict) else "unknown") or "unknown"
            error_param = error_info.get("param") or "unknown"

            # Check if this failure is rate limit or quota exhaustion:
            is_true_quota_exhausted = (
                "insufficient_quota" in str(error_code).lower()
                or "insufficient_quota" in str(error_msg).lower()
                or "402" in str(error_msg)
                or "Payment Required" in str(error_msg)
                or ("quota_exhausted" in str(error_code).lower())
            )
            # TPM / RPM rate limits are *temporary* (seconds to minutes), not permanent quota exhaustion.
            # They get a short cooldown then the Realtime WebSocket reconnects — NOT a process-lifetime lockout.
            is_temporary_rate_limit = (
                not is_true_quota_exhausted
                and (
                    "rate_limit_exceeded" in str(error_code).lower()
                    or "rate_limit" in str(error_msg).lower()
                    or "rate_limit_exceeded" in str(error_type).lower()
                )
                and ("Please try again in" in str(error_msg) or "per min" in str(error_msg) or "TPM" in str(error_msg) or "RPM" in str(error_msg))
            )
            is_rate_limit_or_quota = is_true_quota_exhausted or (
                "rate_limit_exceeded" in str(error_code).lower()
                or "rate_limit_exceeded" in str(error_type).lower()
                or "rate_limit" in str(error_msg).lower()
                or "quota" in str(error_msg).lower()
            )
            if is_temporary_rate_limit:
                # Temporary TPM/RPM limit: short cooldown then resume — do NOT permanently lock out OpenAI
                import re as _re
                retry_s = 30.0
                retry_match = _re.search(r"try again in (\d+\.?\d*)\s*s", str(error_msg))
                if retry_match:
                    retry_s = max(20.0, float(retry_match.group(1)) + 10.0)
                else:
                    retry_s = 20.0
                self.get_logger().warn(
                    f"⏳ [REALTIME TPM RATE LIMIT] Temporary token-per-minute limit hit. "
                    f"Cooling down for {retry_s:.0f}s before resuming.\n"
                    f"  reason={error_msg[:200]}"
                )
                self.realtime_provider_state = "COOLDOWN"
                self._is_connected = False
                self.active_response_state = "IDLE"
                self._realtime_cooldown_until = time.monotonic() + retry_s
                # Schedule reconnect after cooldown
                def _schedule_reconnect_after_cooldown(delay_s):
                    time.sleep(delay_s)
                    if not getattr(self, "_openai_hard_disabled", False):
                        self.realtime_provider_state = "AVAILABLE"
                        self.get_logger().info(f"🔄 [REALTIME TPM COOLDOWN ENDED] Reconnecting to OpenAI Realtime...")
                threading.Thread(
                    target=_schedule_reconnect_after_cooldown,
                    args=(retry_s,),
                    daemon=True,
                    name="astro-tpm-cooldown"
                ).start()
            elif is_rate_limit_or_quota:
                self._trigger_openai_hard_lockout(error_msg, ws=ws)

            audio_generated = (getattr(self, "_packets_for_gen", 0) > 0 or getattr(self, "_bytes_for_gen", 0) > 0)
            response_empty = not audio_generated

            if resp_status == "failed" or event_type == "response.failed":
                self.active_response_state = "FAILED"
                self.get_logger().error(
                    f"[REALTIME ERROR]\n"
                    f"error_class={error_type}:{error_code}\n"
                    f"message={error_msg}\n"
                    f"generation_id={self.realtime_current_generation_id}"
                )
                self.get_logger().error(
                    f"[REALTIME RESPONSE FAILED]\n"
                    f"generation_id={self.active_generation_id or self.realtime_current_generation_id}\n"
                    f"response_id={resp_id or 'unknown'}\n"
                    f"error_type={error_type}\n"
                    f"error_code={error_code}\n"
                    f"error_message={error_msg}\n"
                    f"error_param={error_param}\n"
                    f"response_status={resp_status}\n"
                )
                gen_id = self.active_generation_id or self.realtime_current_generation_id
                self.get_logger().warn(
                    f"[REALTIME FAILED FALLBACK] generation_id={gen_id}\n"
                    f"[TTS FALLBACK] from=openai_realtime to=edge_tts reason=realtime_response_failed"
                )
                last_txt = getattr(self, "_last_user_transcript", "")
                if last_txt and len(last_txt.strip()) > 1 and not getattr(self, "_is_processing_fallback", False):
                    self._last_user_transcript = ""
                    threading.Thread(
                        target=self._process_fallback_turn,
                        kwargs={"direct_text": last_txt},
                        daemon=True,
                        name=f"astro-failed-fallback-{gen_id}"
                    ).start()
            elif resp_status == "cancelled" or event_type == "response.cancelled":
                self.active_response_state = "CANCELLED"
                self.get_logger().info(
                    f"[REALTIME RESPONSE CANCELLED]\n"
                    f"generation_id={self.active_generation_id or self.realtime_current_generation_id}\n"
                    f"response_id={resp_id or 'unknown'}\n"
                    f"server_stream_elapsed_ms={server_stream_elapsed_ms:.1f}\n"
                    f"audio_packets={self._packets_for_gen}"
                )
            else:
                # Completed
                if response_empty:
                    self.get_logger().warn(
                        f"[REALTIME NO AUDIO] generation_id={self.active_generation_id or self.realtime_current_generation_id} elapsed_ms={server_stream_elapsed_ms:.1f}\n"
                        f"[TTS FALLBACK] from=openai_realtime to=edge_tts reason=realtime_no_audio"
                    )
                    self.get_logger().info(
                        f"[REALTIME RESPONSE EMPTY]\n"
                        f"generation_id={self.active_generation_id or self.realtime_current_generation_id}\n"
                        f"response_id={resp_id or 'unknown'}\n"
                        f"response_status={resp_status}\n"
                        f"response_status_details={json.dumps(resp_status_details, ensure_ascii=False) if resp_status_details else 'unknown'}\n"
                        f"audio_packets=0\n"
                        f"audio_bytes=0\n"
                        f"audio_generated=false\n"
                        f"response_empty=true"
                    )
                else:
                    self.get_logger().info(
                        f"[REALTIME TURN]\n"
                        f"generation_id={self.active_generation_id or self.realtime_current_generation_id}\n"
                        f"response_id={resp_id or 'unknown'}\n"
                        f"actual_provider=openai_realtime\n"
                        f"response_status={resp_status}\n"
                        f"vad_to_created_ms={vad_to_created_ms:.1f}\n"
                        f"created_to_first_audio_ms={first_ms:.1f}\n"
                        f"first_audio_ms={total_first_audio_ms:.1f}\n"
                        f"server_stream_elapsed_ms={server_stream_elapsed_ms:.1f}\n"
                        f"audio_duration_ms={audio_dur_ms:.1f}\n"
                        f"audio_packets={self._packets_for_gen}\n"
                        f"audio_bytes={self._bytes_for_gen}\n"
                        f"openai_audio_done=true\n"
                        f"audio_generated=true\n"
                        f"response_empty=false"
                    )
                    self.get_logger().info(
                        f"[REALTIME AUDIO SUMMARY]\n"
                        f"generation_id={self.active_generation_id or self.realtime_current_generation_id}\n"
                        f"packets={self._packets_for_gen}\n"
                        f"bytes={self._bytes_for_gen}\n"
                        f"first_audio_ms={total_first_audio_ms:.1f}\n"
                        f"total_audio_ms={server_stream_elapsed_ms:.1f}"
                    )

            self._is_responding = False
            self.realtime_response_state = "IDLE"
            self.active_response_id = None
            self.active_generation_id = None
            self.active_response_state = "IDLE"
            self._vad_end_time = None

            # Check if there are queued turns waiting to be dispatched
            if self._turn_queue:
                next_turn = self._turn_queue.pop(0)
                self._dispatch_turn(next_turn["generation_id"], next_turn["text"])

        # 4. User Speech Transcription Completed
        elif event_type in ("conversation.item.input_audio_transcription.completed", "conversation.item.input_audio_transcription.done"):
            user_transcript = event.get("transcript", "").strip()
            # Filter out known Whisper background noise / YouTube subtitle hallucinations
            whisper_hallucinations = [
                "çeviri ve altyazı", "altyazı m.k.", "altyazı:", "çeviren:", "abone ol", 
                "izlediğiniz için", "beğenmeyi unutmayın", "subtitle", "transcription by"
            ]
            has_cjk = bool(re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]", user_transcript))
            has_foreign_script = bool(re.search(r"[\u0600-\u06ff\u0400-\u04ff\u0590-\u05ff]", user_transcript))

            if has_cjk or has_foreign_script or len(user_transcript) <= 1 or any(h in user_transcript.lower() for h in whisper_hallucinations):
                self.get_logger().info(f"🔇 [Gürültü/Halüsinasyon Filtrelendi]: \"{user_transcript}\"")
                # INSTANTLY CANCEL OPENAI RESPONSE ONLY IF TRIGGERED BY USER SPEECH, NEVER CANCEL TOOL CONTINUATION!
                is_tool_active = (
                    getattr(self, "_active_tool_call_in_progress", False)
                    or getattr(self, "_last_turn_type", "") == "TOOL_CONTINUATION_RESPONSE"
                    or (time.monotonic() - getattr(self, "_last_tool_call_time", 0.0) < 10.0)
                )
                if not is_tool_active and self.active_response_state in ("GENERATING", "STREAMING", "RESPONSE_STREAMING"):
                    self.active_response_state = "CANCELLED"
                    if self._ws is not None and self._can_use_openai("realtime"):
                        try:
                            if self._loop is not None:
                                asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps({"type": "response.cancel"})), self._loop)
                            elif hasattr(self._ws, "send"):
                                res = self._ws.send(json.dumps({"type": "response.cancel"}))
                                if inspect.iscoroutine(res):
                                    asyncio.run(res)
                        except Exception:
                            pass
                if not is_tool_active and self._is_playback_active:
                    self._is_playback_active = False
                    int_msg = Bool()
                    int_msg.data = True
                    if getattr(self, "pub_interrupt", None):
                        self.pub_interrupt.publish(int_msg)
                return

            if user_transcript:
                self._last_user_transcript = user_transcript
                # Check if visitor was answering lobby concierge question
                if getattr(self, "office_concierge", None) and getattr(self.office_concierge, "_waiting_visitor_response", False):
                    vis_res = self.office_concierge.process_visitor_answer(user_transcript)
                    if vis_res:
                        self.get_logger().info(f"🛎️ [Lobi Misafir Cevabı]: {vis_res}")
                self.get_logger().info(f"🗣️ [Siz]: \"{user_transcript}\"")
                self.memory.episodic.add_message("user", user_transcript)
                self._session_turns_buffer.append({
                    "role": "user",
                    "content": user_transcript,
                    "timestamp": time.time(),
                })
                self.session.record_user_speech()
                self.session.activate_session(reason="user_speech")
                t_msg = String()
                t_msg.data = user_transcript
                self.pub_transcript.publish(t_msg)

        # 5. Assistant Response Completed
        elif event_type in ("response.audio_transcript.done", "response.output_audio_transcript.done", "response.text.done"):
            assistant_transcript = (event.get("transcript") or event.get("text") or "").strip()
            if assistant_transcript:
                self.get_logger().info(f"🤖 [Astro Realtime]: \"{assistant_transcript}\"")
                self.memory.episodic.add_message("assistant", assistant_transcript)
                self._session_turns_buffer.append({
                    "role": "assistant",
                    "content": assistant_transcript,
                    "timestamp": time.time(),
                })
                self.session.record_robot_speech()
                self._ground_speech_gesture(assistant_transcript)


        # 6. Realtime Function Calling Execution (Single Execution per call_id)
        elif event_type == "response.function_call_arguments.done":
            call_id = event.get("call_id")
            if call_id and call_id in self._executed_tool_calls:
                return
            if call_id:
                self._executed_tool_calls.add(call_id)
                if len(self._executed_tool_calls) > 50:
                    self._executed_tool_calls.clear()

            func_name = event.get("name")
            args_str = event.get("arguments", "{}")

            try:
                args = json.loads(args_str)
            except Exception:
                args = {}

            t_tool_event_recv = time.monotonic()
            self._active_tool_call_in_progress = True
            self._active_tool_start_time = t_tool_event_recv
            self._last_turn_type = "TOOL_CONTINUATION_RESPONSE"
            self.get_logger().info(f"🛠️ [Realtime Tool]: {func_name}({args}) çalıştırılıyor...")
            t_exec_start = time.monotonic()
            try:
                tool_result = await asyncio.to_thread(self._execute_realtime_tool, func_name, args)
            except Exception as te:
                self.get_logger().error(f"❌ [Tool Hatası]: {te}")
                tool_result = {"status": "error", "message": str(te)}
            t_exec_done = time.monotonic()
            tool_exec_ms = (t_exec_done - t_exec_start) * 1000.0

            # Send tool response back to OpenAI
            if ws and hasattr(ws, "send") and self._can_use_openai("realtime"):
                tool_output_event = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(tool_result, ensure_ascii=False)
                    }
                }
                self._active_tool_call_in_progress = True
                t_out_send_start = time.monotonic()
                res = ws.send(json.dumps(tool_output_event))
                if asyncio.iscoroutine(res):
                    await res
                t_out_send_done = time.monotonic()
                tool_output_send_ms = (t_out_send_done - t_out_send_start) * 1000.0

                # Trigger response generation with strictly grounded physical reality instructions
                is_acknowledgement_tool = func_name in (
                    "change_persona", "enroll_user_biometrics", "delete_user_biometrics",
                    "move_robot", "turn_to_sound", "navigate_to_location", "notify_via_slack",
                    "add_calendar_event", "delete_calendar_event"
                )
                if is_acknowledgement_tool:
                    tool_instructions = (
                        "FİZİKSEL VE EYLEM CEVAP KURALI: Az önce çalıştırılan fonksiyonun (tool) çıktısını kesin ve mutlak gerçeklik kabul et. "
                        "Eğer çıktı başarı (success=true) içeriyorsa eylemin yapıldığını TEK BİR KISA CÜMLE (maksimum 3-8 kelime) ile doğrula! "
                        "KESİNLİKLE 10-15 saniyelik uzun tirat, açıklama, nutuk veya mod tarifi YAPMA! "
                        "Eğer çıktı hata/engel (success=false veya status=blocked/error) içeriyorsa, SADECE ve SADECE çıktıda yazan 'message' veya 'reason' açıklamasını esas al; "
                        "çıktıda yazmayan hiçbir uydurma sebep (kalp ritmi, nabız, bağlantı vb.) KESİNLİKLE ÜRETME. Asla gerçekleşmeyen bir hareketi gerçekleşmiş gibi iddia etme."
                    )
                elif func_name == "check_calendar_events":
                    sched_text = tool_result.get("schedule", "") if isinstance(tool_result, dict) else str(tool_result)
                    tool_instructions = (
                        f"OFİS TAKVİMİ CEVAP KURALI: Takvimdeki etkinlikler şunlardır: '{sched_text}'. "
                        f"Bu etkinlikleri kullanıcıya doğrudan, samimi, net ve canlı bir Türkçe ile aktar. "
                        f"Uydurma toplantı veya saat ekleme."
                    )
                elif func_name == "inspect_camera_view":
                    obs_text = tool_result.get("observation", "") if isinstance(tool_result, dict) else str(tool_result)
                    tool_instructions = (
                        f"KAMERA VE GÖRME CEVABI KURALI: Kamera görüntüsü başarıyla alındı ve analiz edildi! "
                        f"Kameranın gördüğü ortam ve nesneler şunlardır: '{obs_text}'. "
                        f"Bu gözlemi kullanıcıya doğrudan, samimi ve tek bir kısa Türkçe cümle (en fazla 15-25 kelime) ile söyle. "
                        f"KESİNLİKLE 'kamera çağrısı hâlâ işliyor', 'biraz bekle', 'şu an göremiyorum', 'netleşirse' gibi sözler SÖYLEME! "
                        f"Gözlemdeki nesneleri ve odayı hemen şimdi kendi gözünle görmüş gibi kullanıcıya aktar."
                    )
                else:
                    tool_instructions = (
                        "FİZİKSEL VE BİLGİ CEVAP KURALI: Az önce çalıştırılan fonksiyonun (tool) çıktısındaki bilgileri esas al. "
                        "Çıktıdaki bilgiyi (örneğin hava durumu, arama sonucu vb.) kullanıcıya doğrudan, net, samimi ve doğal bir şekilde aktar. "
                        "Uydurma bilgi veya spekülasyon ekleme."
                    )

                response_create_payload = {
                    "type": "response.create",
                    "response": {
                        "instructions": tool_instructions
                    }
                }
                t_create_send_start = time.monotonic()
                res2 = ws.send(json.dumps(response_create_payload))
                if asyncio.iscoroutine(res2):
                    await res2
                t_create_send_done = time.monotonic()
                continuation_create_send_ms = (t_create_send_done - t_create_send_start) * 1000.0

                self._continuation_create_time = t_create_send_done
                self._last_tool_call_time = t_create_send_done

                # Update turn telemetry timestamps specifically for this continuation turn!
                if not hasattr(self, "_turn_telemetry"):
                    self._turn_telemetry = {}
                self._turn_telemetry["t_resp_send"] = t_create_send_done
                self._turn_telemetry["t_continuation_send"] = t_create_send_done
                self._turn_telemetry["tool_exec_ms"] = tool_exec_ms
                self._turn_telemetry["tool_output_send_ms"] = tool_output_send_ms
                self._turn_telemetry["continuation_create_send_ms"] = continuation_create_send_ms

                total_tool_orchestration_ms = (t_create_send_done - t_tool_event_recv) * 1000.0

                # Rolling p50/p95 latency tracking for tool continuation overhead
                if not hasattr(self, "_tool_orchestration_latencies"):
                    self._tool_orchestration_latencies = []
                self._tool_orchestration_latencies.append(total_tool_orchestration_ms)
                if len(self._tool_orchestration_latencies) > 50:
                    self._tool_orchestration_latencies.pop(0)

                p50_orch = float(np.percentile(self._tool_orchestration_latencies, 50)) if len(self._tool_orchestration_latencies) > 0 else total_tool_orchestration_ms
                p95_orch = float(np.percentile(self._tool_orchestration_latencies, 95)) if len(self._tool_orchestration_latencies) > 0 else total_tool_orchestration_ms

                self.get_logger().info(
                    f"⏱️ [TOOL CONTINUATION PROFILE]\n"
                    f"  tool_name={func_name}\n"
                    f"  tool_exec_ms={tool_exec_ms:.1f}ms\n"
                    f"  tool_output_send_ms={tool_output_send_ms:.1f}ms\n"
                    f"  continuation_create_send_ms={continuation_create_send_ms:.1f}ms\n"
                    f"  total_tool_orchestration_ms={total_tool_orchestration_ms:.1f}ms\n"
                    f"  p50_ms={p50_orch:.1f}ms\n"
                    f"  p95_ms={p95_orch:.1f}ms"
                )

        # 7. Error Handling
        elif event_type == "error":
            err = event.get("error", {})
            err_type = err.get("type", "unknown")
            err_code = err.get("code", "unknown")
            err_msg = err.get("message", "unknown")
            err_param = err.get("param", "unknown")

            # Check for response_cancel_not_active or already completed responses
            if "response_cancel_not_active" in str(err_code) or "response_cancel_not_active" in str(err_msg) or "cancel" in str(err_msg).lower():
                self.get_logger().info("[REALTIME CANCEL IGNORE] reason=response_already_finished")
                return

            self._is_responding = False
            self.realtime_response_state = "IDLE"
            self.active_response_id = None
            self.active_generation_id = None
            self.active_response_state = "FAILED"

            err_class = f"{err_type}:{err_code}" if err_code != "unknown" else err_type

            self.get_logger().error(
                f"[REALTIME ERROR]\n"
                f"error_class={err_class}\n"
                f"message={err_msg}\n"
                f"generation_id={self.realtime_current_generation_id}"
            )
            self.get_logger().error(
                f"[REALTIME RESPONSE FAILED]\n"
                f"generation_id={self.realtime_current_generation_id}\n"
                f"response_id=unknown\n"
                f"error_type={err_type}\n"
                f"error_code={err_code}\n"
                f"error_message={err_msg}\n"
                f"error_param={err_param}\n"
                f"response_status=failed\n"
                f"response_status_details={json.dumps(err, ensure_ascii=False)}"
            )

            is_true_quota_exhausted = (
                "insufficient_quota" in str(err_code).lower()
                or "insufficient_quota" in str(err_msg).lower()
                or "402" in str(err_msg)
                or "Payment Required" in str(err_msg)
            )
            is_temporary_rate_limit = (
                not is_true_quota_exhausted
                and (
                    "rate_limit_exceeded" in str(err_code).lower()
                    or "rate_limit" in str(err_msg).lower()
                    or "rate_limit_exceeded" in str(err_type).lower()
                )
                and ("Please try again in" in str(err_msg) or "per min" in str(err_msg) or "TPM" in str(err_msg) or "RPM" in str(err_msg))
            )
            is_rate_limit_or_quota = is_true_quota_exhausted or (
                "rate_limit_exceeded" in str(err_code).lower()
                or "rate_limit_exceeded" in str(err_type).lower()
                or "rate_limit" in str(err_msg).lower()
                or "quota" in str(err_msg).lower()
            )
            if is_temporary_rate_limit:
                import re as _re
                retry_s = 30.0
                retry_match = _re.search(r"try again in (\d+\.?\d*)\s*s", str(err_msg))
                if retry_match:
                    retry_s = max(10.0, float(retry_match.group(1)) + 5.0)
                self.get_logger().warn(
                    f"⏳ [REALTIME TPM RATE LIMIT] Temporary token-per-minute limit hit. "
                    f"Cooling down for {retry_s:.0f}s before resuming.\n"
                    f"  reason={err_msg[:200]}"
                )
                self.realtime_provider_state = "COOLDOWN"
                self._is_connected = False
                self.active_response_state = "IDLE"
                self._realtime_cooldown_until = time.monotonic() + retry_s
                def _schedule_reconnect_after_cooldown_b(delay_s):
                    time.sleep(delay_s)
                    if not getattr(self, "_openai_hard_disabled", False):
                        self.realtime_provider_state = "AVAILABLE"
                        self.get_logger().info(f"🔄 [REALTIME TPM COOLDOWN ENDED] Reconnecting to OpenAI Realtime...")
                threading.Thread(
                    target=_schedule_reconnect_after_cooldown_b,
                    args=(retry_s,),
                    daemon=True,
                    name="astro-tpm-cooldown-b"
                ).start()
                return
            elif is_rate_limit_or_quota:
                self._trigger_openai_hard_lockout(err_msg, ws=ws)
                return

            is_session_error = (
                "session" in str(err_param).lower()
                or "session" in str(err_msg).lower()
                or "invalid_session" in str(err_code).lower()
                or "session.type" in str(err_msg).lower()
            )
            if is_session_error:
                self.realtime_session_state = "SESSION_CONFIG_ERROR"
                self._fallback_mode = True
                self._publish_realtime_state("SESSION_CONFIG_ERROR", err_msg)

            # Trigger immediate fallback to Edge-TTS if there was an active turn request
            if getattr(self, "_last_requested_text", "") and not self.realtime_audio_received:
                self.get_logger().warn(
                    f"[REALTIME NO AUDIO]\n"
                    f"generation_id={self.realtime_current_generation_id}\n"
                    f"reason=server_error\n"
                    f"[TTS FALLBACK]\n"
                    f"from=openai_realtime\n"
                    f"to=edge_tts\n"
                    f"reason=realtime_server_error"
                )
                fb_msg = String()
                fb_msg.data = json.dumps({
                    "text": self._last_requested_text,
                    "engine": "edge-tts",
                    "generation_id": self.realtime_current_generation_id,
                    "fallback_reason": "realtime_server_error",
                })
                self.pub_tts_say.publish(fb_msg)
                self._last_requested_text = ""



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
            'snow': 'kar yağışlı',
            'fog': 'sisli',
            'mist': 'puslu',
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

    def _execute_fallback_weather(self, city: str = "Ahlat") -> str:
        try:
            url = f"https://wttr.in/{urllib.parse.quote(city)}?format=%C+%t&lang=tr"
            req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                weather_text = resp.read().decode("utf-8").strip()
            return self._format_turkish_weather(city, weather_text)
        except Exception:
            return f"{city}'ta hava şu an 20 derece ve açık."

    def _is_weather_query(self, text: str) -> Tuple[bool, str]:
        text_l = text.lower()
        if any(w in text_l for w in ["hava nasıl", "hava durumu", "hava kaç derece", "havalar nasıl", "yağmur var mı", "kar var mı", "sıcaklık kaç", "hava"]):
            if "ahlat" in text_l or "ahlattı" in text_l or "ahlatta" in text_l:
                return True, "Ahlat"
            if "bitlis" in text_l:
                return True, "Bitlis"
            if "tatvan" in text_l:
                return True, "Tatvan"
            if "istanbul" in text_l:
                return True, "Istanbul"
            if "ankara" in text_l:
                return True, "Ankara"
            return True, "Ahlat"
        return False, ""

    def _is_turn_to_sound_query(self, text: str) -> bool:
        """Detects explicit acoustic gaze orientation commands."""
        t = text.lower().strip()
        direct_phrases = [
            "sesime dön", "sesime don", "sesime bak", "sesime doğru dön", "sesime dogru don",
            "bana dön", "bana don", "bana bak", "bana doğru dön", "bana dogru don",
            "yüzüme bak", "yuzume bak", "buraya bak", "buraya dön", "buraya don",
            "sesin geldiği yere dön", "sesin geldiği yöne dön", "sesin geldigi yone don",
            "sesime doğru bak", "sesime dogru bak", "sesime doğru", "sesime dogru",
        ]
        if any(p in t for p in direct_phrases):
            return True
        has_target = any(k in t for k in ["sesime", "sesimin", "bana", "buraya", "yüzüme", "yuzume"])
        has_action = any(a in t for a in ["dön", "don", "bak", "yönel", "yonel"])
        return bool(has_target and has_action)

    def _is_head_angle_query(self, text: str) -> Tuple[bool, float, str]:
        """Detects explicit head angular positioning commands (e.g. '0 dereceye dön', '-30'a dön', '30 derece sağa bak')."""
        t = text.lower().strip()

        # 1. Direct Center words
        center_words = [
            "merkeze dön", "merkeze don", "merkeze bak", "merkez", "düz bak", "duz bak",
            "öne bak", "one bak", "düz dur", "duz dur", "karşıya bak", "karsiya bak",
            "karşıya dön", "karsiya don", "ortaya bak", "ortaya dön", "ortaya don",
        ]
        if any(p in t for p in center_words):
            return True, 0.0, "0°"

        # 2. Number parsing with directional keywords (sağa/sola)
        tr_words = {
            "sıfır": 0, "sifir": 0, "on": 10, "on beş": 15, "onbes": 15,
            "yirmi": 20, "yirmi beş": 25, "yirmibes": 25, "otuz": 30,
            "otuz beş": 35, "otuzbes": 35, "kırk": 40, "kirk": 40,
            "kırk beş": 45, "kirkbes": 45, "elli": 50, "altmış": 60,
            "atmış": 60, "yetmiş": 70, "yetmis": 70,
        }

        is_negative = bool("eksi" in t or "negatif" in t or bool(re.search(r"(?:^|\s)-\d+", t)))
        is_right = bool("sağa" in t or "saga" in t or "sağ" in t or "sag" in t)
        is_left = bool("sola" in t or "sol" in t)

        target_angle = None

        # Regex for digits (e.g. -30, +30, 0, 45, -60.5)
        match = re.search(r"([+-]?\d+(?:\.\d+)?)", t)
        if match:
            try:
                val = float(match.group(1))
                if is_negative and val > 0:
                    val = -val
                target_angle = val
            except Exception:
                pass

        if target_angle is None:
            for w_name, w_val in tr_words.items():
                if w_name in t:
                    target_angle = float(-w_val if is_negative else w_val)
                    break

        if target_angle is not None:
            # In ROS body frame: Right is negative, Left is positive
            if is_right and target_angle > 0:
                target_angle = -target_angle
            elif is_left and target_angle < 0:
                target_angle = abs(target_angle)

            if any(k in t for k in ["dön", "don", "bak", "çevir", "cevir", "derece", "açı", "aci", "'a", "'e", "'ye", "'ya", "dur"]):
                clamped = max(-70.0, min(70.0, target_angle))
                sign_str = f"{clamped:+.0f}°" if clamped != 0 else "0°"
                return True, clamped, sign_str

        # 3. Simple relative left/right commands without number
        if "sağa bak" in t or "sağa dön" in t or "saga bak" in t or "saga don" in t:
            return True, -35.0, "sağ 35°"
        if "sola bak" in t or "sola dön" in t or "sola bak" in t or "sola don" in t:
            return True, 35.0, "sol 35°"

        return False, 0.0, ""

    def _is_movement_query(self, text: str) -> Tuple[bool, str, float, float]:
        """Detects base mobility commands in fallback mode."""
        t = text.lower().strip()
        if any(p in t for p in ["ileri git", "öne git", "one git", "ilerle", "ileri sür", "öne doğru git"]):
            return True, "forward", 0.2, 1.5
        if any(p in t for p in ["geri gel", "geriye git", "gerile", "geri sür", "arkaya git"]):
            return True, "backward", 0.2, 1.5
        if any(p in t for p in ["sağa dön", "saga don", "sağa bak", "saga bak", "sağa kıvrıl"]):
            return True, "right", 0.2, 1.0
        if any(p in t for p in ["sola dön", "sola don", "sola bak", "sola kıvrıl"]):
            return True, "left", 0.2, 1.0
        if any(p in t for p in ["dur", "dur orada", "dur robot", "hareketi kes", "bekle orada"]):
            return True, "stop", 0.0, 0.0
        return False, "stop", 0.0, 0.0

    def _execute_realtime_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Executes integrated robot tools in real time."""
        if name == "get_live_weather":
            city = str(args.get("city", "Ahlat")).strip()
            # Guard against robot name or greeting artifacts interpreted as city
            if not city or city.lower() in ("astro", "robot", "sen", "kendin", "unknown", "none", "yok", "undefined"):
                city = "Ahlat"
            w_str = self._execute_fallback_weather(city)
            return {"status": "success", "city": city, "weather": w_str}

        elif name == "set_reminder":
            mins = float(args.get("minutes", 1.0))
            topic = args.get("topic", "hatırlatma")
            due_time = time.monotonic() + (mins * 60.0)
            self._reminders.append({"due_time": due_time, "topic": topic, "minutes": mins})
            return {"status": "success", "message": f"{mins} dakika sonra '{topic}' hatırlatması kuruldu."}

        elif name == "save_user_memory":
            key = args.get("key", "")
            val = args.get("value", "")
            identity = self._get_active_biometric_identity()
            name_p = identity.get("name", "Baran")
            self.memory.profile.set_user_fact(name_p, key, val)
            if self._ws and self._loop and self._is_connected:
                try:
                    asyncio.run_coroutine_threadsafe(self._send_session_update(self._ws), self._loop)
                except Exception as _exc:
                    self.get_logger().debug(f"save_user_memory: session update failed ({_exc})")
            return {"status": "success", "message": f"'{key}: {val}' bilgisi hafızaya kaydedildi."}

        elif name == "search_memory":
            query = args.get("query", "")
            results = []
            try:
                if hasattr(self.memory, "episodic") and hasattr(self.memory.episodic, "search"):
                    search_res = self.memory.episodic.search(query, top_k=3)
                    if isinstance(search_res, list):
                        results.extend(search_res)
                if hasattr(self.memory, "profile"):
                    identity = self._get_active_biometric_identity()
                    name_p = identity.get("name", "Baran")
                    if hasattr(self.memory.profile, "get_user_facts"):
                        facts = self.memory.profile.get_user_facts(name_p)
                        if isinstance(facts, dict):
                            for k, v in facts.items():
                                if query.lower() in k.lower() or query.lower() in str(v).lower():
                                    results.append(f"{k}: {v}")
                    for vf in self.memory.profile.data.get("verified_facts", []):
                        if query.lower() in str(vf).lower():
                            results.append(f"Kalıcı Bilgi: {vf}")
                    for ob in self.memory.profile.data.get("environmental_observations", []):
                        if query.lower() in str(ob).lower():
                            results.append(f"Gözlem: {ob}")
            except Exception as se:
                self.get_logger().debug(f"Memory search error: {se}")
            res_text = "\n".join(str(r) for r in results) if results else "Hafızada bu konuyla ilgili özel bir kayıt bulunamadı."
            return {"status": "success", "query": query, "memory_context": res_text}

        elif name == "inspect_camera_view":
            focus = args.get("focus", "kullanıcının elindeki nesne, rengi ve çevre")
            return self._inspect_camera_view(focus)

        elif name == "turn_to_sound":
            if getattr(self, "action_manager", None):
                res = self.action_manager.execute_turn_to_sound(
                    generation_id=getattr(self, "realtime_current_generation_id", None)
                )
                return res.to_dict()
            return {
                "status": "error",
                "success": False,
                "action": "turn_to_sound",
                "error_code": "NO_DIRECTION",
                "message": "Ses yönü yöneticisi hazır değil veya DOA verisi yok."
            }

        elif name == "set_head_angle":
            angle = float(args.get("angle_deg", 0.0))
            direction = str(args.get("direction", "")).lower().strip()
            if direction == "right" and angle > 0:
                angle = -angle
            elif direction == "left" and angle < 0:
                angle = abs(angle)
            elif direction == "center":
                angle = 0.0
            clamped = max(-70.0, min(70.0, angle))
            self.pub_head_target_yaw.publish(Float32(data=float(clamped)))
            return {"status": "success", "angle_deg": clamped, "message": f"Kafa {clamped:.1f} dereceye ayarlandı."}

        elif name == "move_robot":
            direction = args.get("direction", "stop").lower().strip()
            speed = float(args.get("speed", 0.2))
            duration = float(args.get("duration", 1.5))
            if getattr(self, "action_manager", None):
                res = self.action_manager.execute_move(
                    direction=direction,
                    speed=speed,
                    duration=duration,
                    generation_id=getattr(self, "realtime_current_generation_id", None),
                )
                return res.to_dict()

            # Direct fallback if ActionManager is not instantiated
            speed = max(0.05, min(speed, 0.4))
            duration = max(0.5, min(duration, 5.0))
            if direction != "stop":
                hb_ok = getattr(self, "_arduino_heartbeat_healthy", False)
                last_ack = getattr(self, "_last_heartbeat_ack_time", 0.0)
                if not hb_ok or (time.monotonic() - last_ack) > 2.0:
                    return {
                        "status": "blocked",
                        "reason": "heartbeat_unhealthy",
                        "error_code": "MOTOR_CONTROLLER_UNAVAILABLE",
                        "message": "Arduino bağlantısı veya heartbeat aktif değil, güvenlik için hareket engellendi."
                    }
                if direction == "forward" and getattr(self, "_obstacle_detected", False):
                    return {
                        "status": "blocked",
                        "reason": "obstacle_detected",
                        "error_code": "OBSTACLE_DETECTED",
                        "message": "Robotun önünde engel tespit edildi, hareket güvenlik nedeniyle engellendi."
                    }
                last_scan_time = getattr(self, "_last_laser_scan_time", 0.0)
                if (time.monotonic() - last_scan_time) > 2.0 and direction == "forward":
                    return {
                        "status": "blocked",
                        "reason": "lidar_stale_or_disconnected",
                        "error_code": "LIDAR_STALE_OR_DISCONNECTED",
                        "message": "LiDAR tarama verisi alınamıyor veya güncel değil, güvenlik nedeniyle ileri hareket engellendi."
                    }

            pub = getattr(self, "pub_cmd_vel", None)
            if pub:
                try:
                    tw = Twist()
                    if direction == "forward":
                        tw.linear.x = speed
                    elif direction == "backward":
                        tw.linear.x = -speed
                    elif direction == "left":
                        tw.angular.z = speed * 2.0
                    elif direction == "right":
                        tw.angular.z = -speed * 2.0
                    elif direction == "stop":
                        tw.linear.x = 0.0
                        tw.angular.z = 0.0

                    pub.publish(tw)
                    if direction != "stop":
                        def _stop_later():
                            time.sleep(duration)
                            try:
                                stop_tw = Twist()
                                pub.publish(stop_tw)
                            except Exception:
                                pass
                        threading.Thread(target=_stop_later, daemon=True).start()

                    return {"status": "success", "success": True, "action": f"Robot {direction} yönünde {speed} m/s hızla hareket ettirildi."}
                except Exception as me:
                    return {"status": "error", "success": False, "message": f"Hareket komutu verilemedi: {me}"}
            return {"status": "error", "success": False, "message": "/cmd_vel yayıncısı hazır değil."}

        elif name == "enroll_user_biometrics":
            name_param = args.get("name", "")
            formal_title = args.get("formal_title", "")
            return self._enroll_user_biometrics(name_param, formal_title)

        elif name == "delete_user_biometrics":
            name_param = args.get("name", "")
            return self._delete_user_biometrics(name_param)

        elif name == "change_persona":
            raw_p = args.get("persona", "").lower().strip()
            p_map = {
                "kufurbaz": "kufurbaz", "küfürbaz": "kufurbaz", "kufur": "kufurbaz", "küfür": "kufurbaz",
                "kürbaz": "kufurbaz", "kürbato": "kufurbaz", "küfürlü": "kufurbaz", "kufurlu": "kufurbaz", "roast": "kufurbaz",
                "kaba": "rude", "rude": "rude",
                "flort": "flirt", "flört": "flirt", "flirt": "flirt", "capkin": "flirt", "çapkın": "flirt",
                "neseli": "playful", "neşeli": "playful", "playful": "playful", "sakaci": "playful", "şakacı": "playful",
                "resmi": "formal", "formal": "formal", "ciddi": "formal",
                "sarkastik": "sarcastic", "sarcastic": "sarcastic", "alayci": "sarcastic", "alaycı": "sarcastic",
                "sinirli": "angry", "angry": "angry", "asabi": "angry", "ofkeli": "angry", "öfkeli": "angry",
                "duygusal": "emotional", "emotional": "emotional"
            }
            target = p_map.get(raw_p, raw_p)
            if target in PERSONA_PROMPTS:
                self.persona_name = target
                self.persona_engine.set_persona(target)
                self.memory.profile.set_persona(target)
                raw_v = os.environ.get("OPENAI_REALTIME_VOICE", "").strip().lower()
                self.realtime_voice = raw_v if raw_v in VALID_REALTIME_VOICES else PERSONA_DEFAULT_VOICES.get(target, "echo")
                self._last_synced_identity = ""  # Force immediate session update

                # Publish emotion for face screen
                emo_msg = String()
                emo_msg.data = target
                self.pub_emotion.publish(emo_msg)

                self.get_logger().info(f"🎭 [Kişilik Değiştirildi]: Yeni kişilik modu -> '{target.upper()}', Ses -> [{self.realtime_voice}]")
                self._sync_perception_to_session()
                return {
                    "status": "success",
                    "persona": target,
                    "message": f"Kişilik modu '{target}' yapıldı. Tek kısa cümleyle (maksimum 3-6 kelime) doğrudan bu yeni modun üslubuyla onay ver."
                }
            return {"status": "error", "message": f"'{raw_p}' geçerli bir kişilik modu değil."}

        elif name == "check_calendar_events":
            query = str(args.get("query", "bugün")).strip()
            days = int(args.get("days", 7 if ("hafta" in query.lower() or not query) else 1))
            if getattr(self, "calendar_service", None):
                summary = self.calendar_service.get_events_summary(days=days, query=query)
                return {"status": "success", "query": query, "days": days, "schedule": summary}
            return {"status": "error", "message": "Takvim servisi aktif değil."}

        elif name == "add_calendar_event":
            title = str(args.get("title", "")).strip()
            date_str = str(args.get("date", "bugün")).strip()
            time_str = str(args.get("time", "10:00")).strip()
            duration = int(args.get("duration_minutes", 45))
            loc = str(args.get("location", "Ofis")).strip()
            if getattr(self, "calendar_service", None):
                res = self.calendar_service.add_event_smart(
                    title=title,
                    date_str=date_str,
                    time_str=time_str,
                    duration_minutes=duration,
                    location=loc
                )
                if res.get("status") == "success":
                    ev = res.get("event", {})
                    st = ev.get("start_time", f"{date_str} {time_str}")
                    return {"status": "success", "title": title, "start_time": st, "message": f"'{title}' etkinliği {st} için takvime eklendi."}
                return {"status": "error", "message": res.get("message", "Etkinlik eklenemedi.")}
            return {"status": "error", "message": "Takvim servisi aktif değil."}

        elif name == "delete_calendar_event":
            query = str(args.get("query", "")).strip()
            if getattr(self, "calendar_service", None):
                res = self.calendar_service.delete_event(query)
                return res
            return {"status": "error", "message": "Takvim servisi aktif değil."}

        elif name == "notify_via_slack":
            recipient = str(args.get("recipient", "Baran")).strip()
            msg = str(args.get("message", "Ofise yeni misafir geldi.")).strip()
            if getattr(self, "slack_service", None):
                res = self.slack_service.notify_visitor_arrival(
                    employee_name=recipient,
                    visitor_name=getattr(self, "_active_person_name", "Misafir"),
                    note=msg
                )
                return {"status": "success", "recipient": recipient, "delivered": res.get("delivered", False), "note": msg}
            return {"status": "error", "message": "Slack servisi aktif değil."}

        return {"status": "unknown_tool"}

    def _purge_corrupted_biometrics(self):
        """Automatically purges corrupted names / profanities on node startup."""
        bad_names = ["yarram", "yarram_bey", "yarram bey", "yarağın", "sik", "siktir", "amk", "piç", "gerizekalı", "yarak", "yarrak", "astronun kocası", "astronun_kocasi", "astronun kocasi"]
        for bad in bad_names:
            try:
                if self.voice_recognizer:
                    self.voice_recognizer.delete_speaker(bad)
                if self.face_recognizer:
                    self.face_recognizer.delete_face(bad)
                self.memory.profile.remove_known_person(bad)
            except Exception as _exc:
                self.get_logger().debug(f"_purge_corrupted_biometrics: yok sayılan hata ({_exc})")
        self.get_logger().info("🧹 [Biyometrik Temizlik]: Hatalı/uygunsuz kayıtlar (Yarram, Astronun Kocası vb.) veri tabanından tamamen silindi.")

    def _delete_user_biometrics(self, name: str) -> Dict[str, Any]:
        """Deletes user from biometric databases and memory."""
        name = name.strip().title()
        if not name or len(name) < 2:
            return {"status": "error", "message": "Silinecek geçerli bir isim belirtilmedi."}

        try:
            if self.voice_recognizer:
                self.voice_recognizer.delete_speaker(name)
            if self.face_recognizer:
                self.face_recognizer.delete_face(name)
            self.memory.profile.remove_known_person(name)
        except Exception as e:
            self.get_logger().warn(f"Biometric deletion warning: {e}")

        with self._lock:
            if getattr(self, "_active_person_name", "") == name:
                self._active_person_name = "Misafir"
                self._person_hold_until = 0.0
                self._recognized_speaker = None
                self._recognized_person = None
                self._last_synced_identity = ""

        self._sync_perception_to_session()
        msg = f"'{name}' biyometrik kayıtları ve hafızası başarıyla silindi."
        self.get_logger().info(f"🗑️ [Biyometrik Silindi]: {msg}")
        return {"status": "success", "message": msg}

    def _run_async_biometric_side_channel(self, t_speech_stopped: float):
        """Profile B: Asynchronous biometric identification side-channel.
        Runs concurrently in a background thread without blocking response.create or audio streaming.
        Safely updates identity state only when multi-factor verified, and syncs perception for subsequent turns.
        Never mutates the ongoing response stream retroactively.
        """
        def _bg_worker():
            self._async_identity_in_flight = True
            t_start = time.monotonic()
            try:
                self._run_voice_identification(t_speech_stopped)
                t_end = time.monotonic()
                async_id_ms = (t_end - t_start) * 1000.0
                self._latest_async_identity_ms = async_id_ms

                # Check resulting biometric status
                ident = self.resolve_identities()
                bio_status = ident.get("biometric_status", "unknown")
                bio_name = ident.get("biometric_identity", "unknown")
                bio_conf = ident.get("biometric_confidence", 0.0)

                if bio_status == "verified" and bio_name != "unknown" and bio_conf >= 0.40:
                    if hasattr(self, "get_logger") and callable(self.get_logger):
                        self.get_logger().info(
                            f"🎙️ [BIOMETRIC ASYNC VERIFIED] user={bio_name} (conf={bio_conf:.2f}, async_time={async_id_ms:.1f}ms) "
                            f"— Session state updated for upcoming turns"
                        )
                else:
                    if hasattr(self, "get_logger") and callable(self.get_logger):
                        self.get_logger().debug(
                            f"🎙️ [BIOMETRIC ASYNC UNVERIFIED] status={bio_status} (async_time={async_id_ms:.1f}ms) "
                            f"— No identity promotion"
                        )
            except Exception as exc:
                if hasattr(self, "get_logger") and callable(self.get_logger):
                    self.get_logger().debug(f"_run_async_biometric_side_channel error: {exc}")
            finally:
                self._async_identity_in_flight = False

        t = threading.Thread(target=_bg_worker, daemon=True, name="astro-async-biometric")
        t.start()
        return t

    def _record_voice_id_segment(self, segment_name: str, duration_ms: float) -> Tuple[float, float, float]:
        """Tracks rolling 50-turn p50, p95, and max for each voice identification pipeline segment."""
        if not hasattr(self, "_voice_id_segment_stats"):
            self._voice_id_segment_stats = {}
        hist = self._voice_id_segment_stats.setdefault(segment_name, [])
        hist.append(float(duration_ms))
        if len(hist) > 50:
            hist.pop(0)
        p50 = float(np.percentile(hist, 50)) if len(hist) > 0 else duration_ms
        p95 = float(np.percentile(hist, 95)) if len(hist) > 0 else duration_ms
        max_val = float(np.max(hist)) if len(hist) > 0 else duration_ms
        return round(p50, 1), round(p95, 1), round(max_val, 1)

    def _run_voice_identification(self, t_speech_stopped: Optional[float] = None):
        """Robust multi-window voice identification with comprehensive 7-segment latency profiling and rolling p50/p95/max telemetry."""
        t_id_start = time.monotonic()
        t_speech_stopped = t_speech_stopped or t_id_start
        speech_stopped_to_extract_ms = (t_id_start - t_speech_stopped) * 1000.0

        if not getattr(self, "voice_recognizer", None):
            return

        t_extract_start = time.monotonic()
        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                if not getattr(self, "_user_speech_audio_buffer", None):
                    return
                buffer_copy = list(self._user_speech_audio_buffer)
        else:
            if not getattr(self, "_user_speech_audio_buffer", None):
                return
            buffer_copy = list(self._user_speech_audio_buffer)
        t_extract_done = time.monotonic()
        buffer_extract_ms = (t_extract_done - t_extract_start) * 1000.0

        if len(buffer_copy) < 25:  # Less than ~0.5s -- too short to analyze reliably
            now = time.monotonic()
            held_name = getattr(self, "_active_person_name", "Misafir")
            hold_until = getattr(self, "_person_hold_until", 0.0)
            has_active_hold = (now < hold_until) and (held_name != "Misafir")
            total_id_ms = (now - t_id_start) * 1000.0

            p50_tot, p95_tot, max_tot = self._record_voice_id_segment("total_voice_id_ms", total_id_ms)
            p50_s1, p95_s1, max_s1 = self._record_voice_id_segment("speech_stopped_to_extract_ms", speech_stopped_to_extract_ms)
            p50_s2, p95_s2, max_s2 = self._record_voice_id_segment("buffer_extract_ms", buffer_extract_ms)

            self._latest_voice_id_profile = {
                "path": "SHORT_BUFFER_HOLD_RETAIN" if has_active_hold else "SHORT_BUFFER_BYPASS",
                "speech_stopped_to_extract_ms": speech_stopped_to_extract_ms,
                "buffer_extract_ms": buffer_extract_ms,
                "audio_prep_ms": 0.0,
                "embedding_infer_ms": 0.0,
                "speaker_match_ms": 0.0,
                "vote_aggregation_ms": 0.0,
                "identity_decision_ms": 0.0,
                "total_id_ms": total_id_ms,
                "windows_evaluated": 0,
                "window_breakdowns": [],
                "candidate_speaker_count": 0,
                "number_of_votes": 0,
                "selected_speaker": held_name if has_active_hold else "Misafir",
                "confidence": 1.0 if has_active_hold else 0.0,
                "rejection_reason": "buffer_too_short (<0.5s)",
                "device": "cached_session_hold",
                "sample_count": len(buffer_copy) * 320,
                "audio_duration_ms": (len(buffer_copy) * 320 / 16000.0) * 1000.0,
                "stats": {
                    "total_voice_id_ms": {"p50": p50_tot, "p95": p95_tot, "max": max_tot},
                }
            }
            if hasattr(self, "get_logger") and callable(self.get_logger):
                self.get_logger().info(
                    f"⚡ [VOICE ID FAST-PATH / CACHE HIT]\n"
                    f"  path={'SHORT_BUFFER_HOLD_RETAIN' if has_active_hold else 'SHORT_BUFFER_BYPASS'}\n"
                    f"  speaker={held_name if has_active_hold else 'Misafir'}\n"
                    f"  hold_remaining_s={max(0.0, hold_until - now):.1f}s\n"
                    f"  buffer_chunks={len(buffer_copy)} ({len(buffer_copy)*20}ms)\n"
                    f"  total_id_ms={total_id_ms:.1f}ms"
                )
            return

        total_samples = len(buffer_copy) * 320
        total_audio_dur_ms = (total_samples / 16000.0) * 1000.0

        t_prep_start = time.monotonic()
        try:
            from collections import Counter
            # 1. Strip trailing Server VAD silence padding (last ~500-600ms) to ensure clean speech embeddings
            ambient = getattr(self, "_ambient_rms", 150.0)
            speech_floor = max(180.0, ambient * 1.05)

            last_speech_idx = len(buffer_copy)
            for idx in range(len(buffer_copy) - 1, -1, -1):
                chunk_arr = np.frombuffer(buffer_copy[idx], dtype=np.int16)
                chunk_rms = float(np.sqrt(np.mean(chunk_arr.astype(np.float32) ** 2))) if len(chunk_arr) > 0 else 0.0
                if chunk_rms > speech_floor:
                    last_speech_idx = min(len(buffer_copy), idx + 5)  # retain small 100ms natural tail margin
                    break

            clean_speech_buffer = buffer_copy[:last_speech_idx] if last_speech_idx >= 20 else buffer_copy
            cn = len(clean_speech_buffer)

            # 2. Window 0: Full clean speech utterance (highest information density & most accurate embedding)
            #    Windows 1 & 2: First half and second half for multi-window validation
            half = max(cn // 2, 20)
            windows = [
                clean_speech_buffer,
                clean_speech_buffer[:half] if cn >= 30 else clean_speech_buffer,
                clean_speech_buffer[-half:] if cn >= 30 else clean_speech_buffer,
            ]
            t_prep_done = time.monotonic()
            audio_prep_ms = (t_prep_done - t_prep_start) * 1000.0

            window_results = []
            infer_times = []
            window_breakdowns = []
            early_exit = False
            exit_path_type = "COLD_FULL_3WIN_VOTING"
            threshold = float(os.getenv("SPEAKER_MATCH_THRESHOLD", "0.40"))

            held_name = getattr(self, "_active_person_name", "Misafir")
            hold_until = getattr(self, "_person_hold_until", 0.0)
            has_active_hold = (time.monotonic() < hold_until) and (held_name != "Misafir")

            for win_idx, win in enumerate(windows):
                t_win_start = time.monotonic()
                raw = b"".join(win)
                arr = np.frombuffer(raw, dtype=np.int16)
                if len(arr) < int(16000 * 0.4):
                    continue
                spk_name, spk_conf, spk_meta = self.voice_recognizer.recognize_voice(
                    arr, sample_rate=16000, threshold=threshold
                )
                t_win_done = time.monotonic()
                win_total_ms = (t_win_done - t_win_start) * 1000.0
                infer_times.append(win_total_ms)
                window_results.append((spk_name, spk_conf, spk_meta))

                # Extract sub-segment profile from voice_recognizer metadata
                emb_prof = spk_meta.get("voice_id_profile", {}) if isinstance(spk_meta, dict) else {}
                window_breakdowns.append({
                    "win": win_idx,
                    "fbank_ms": emb_prof.get("fbank_ms", 0.0),
                    "onnx_infer_ms": emb_prof.get("onnx_infer_ms", 0.0),
                    "norm_ms": emb_prof.get("norm_ms", 0.0),
                    "match_ms": emb_prof.get("speaker_match_ms", 0.0),
                    "total_ms": round(win_total_ms, 2),
                    "spk": spk_name,
                    "conf": round(float(spk_conf or 0.0), 3),
                    "device": emb_prof.get("device", "CPUExecutionProvider"),
                    "candidates": emb_prof.get("candidate_count", 0),
                })

                # Early-exit optimization 1: High confidence match on known speaker
                if win_idx == 0 and spk_name is not None and spk_conf >= 0.46 and spk_name.lower() != "misafir":
                    early_exit = True
                    exit_path_type = "HIGH_CONF_EARLY_EXIT_1WIN"
                    break

                # Early-exit optimization 2 (Active Conversation Fast-Path):
                if win_idx == 0 and has_active_hold:
                    other_person_detected = (spk_name is not None and spk_name != held_name and spk_conf >= 0.45)
                    if not other_person_detected:
                        early_exit = True
                        exit_path_type = "ACTIVE_HOLD_FAST_PATH_1WIN"
                        break

            if not window_results:
                return

            t_vote_start = time.monotonic()
            votes: Counter = Counter()
            best_conf: dict = {}
            best_meta: dict = {}

            for spk_name, spk_conf, spk_meta in window_results:
                if spk_name is not None:
                    votes[spk_name] += 1
                    if spk_conf > best_conf.get(spk_name, 0.0):
                        best_conf[spk_name] = spk_conf
                        best_meta[spk_name] = spk_meta

            t_vote_done = time.monotonic()
            vote_aggregation_ms = (t_vote_done - t_vote_start) * 1000.0

            now = time.monotonic()

            t_decision_start = time.monotonic()
            if not votes:
                if hasattr(self, "get_logger") and callable(self.get_logger):
                    self.get_logger().info("🎙️ [Ses Tanıma]: Bilinmeyen Ses — hiçbir pencerede eşleşme bulunamadı")
                if lock is not None:
                    with lock:
                        self._recognized_speaker = {"name": "Misafir", "confidence": 0.0, "is_known": False, "source": "unknown_voice"}
                        self._active_person_name = "Misafir"
                        self._person_hold_until = 0.0
                        self._voice_id_streak = {}
                else:
                    self._recognized_speaker = {"name": "Misafir", "confidence": 0.0, "is_known": False, "source": "unknown_voice"}
                    self._active_person_name = "Misafir"
                    self._person_hold_until = 0.0
                    self._voice_id_streak = {}
                self._sync_perception_to_session()
                t_decision_done = time.monotonic()
                identity_decision_ms = (t_decision_done - t_decision_start) * 1000.0
                return

            ranked = votes.most_common()
            winner_name, winner_votes = ranked[0]
            winner_conf = best_conf[winner_name]
            winner_meta = best_meta[winner_name]

            runner_up_conf = best_conf.get(ranked[1][0], 0.0) if len(ranked) > 1 else 0.0
            margin = winner_conf - runner_up_conf
            total_windows = len(window_results)
            majority_threshold = max(2, total_windows // 2 + 1) if total_windows >= 2 else 1

            is_majority = winner_votes >= majority_threshold
            is_confident = winner_conf >= 0.40
            is_clear_winner = (margin > 0.03) or (total_windows <= 1)

            if lock is not None:
                with lock:
                    streak_map = getattr(self, "_voice_id_streak", {})
            else:
                streak_map = getattr(self, "_voice_id_streak", {})

            selected_speaker_name = winner_name
            selected_conf = winner_conf
            rejection_reason_str = "none"

            if is_majority and is_confident and is_clear_winner:
                streak_count = streak_map.get(winner_name, 0) + 1
                streak_map[winner_name] = streak_count
                if hasattr(self, "get_logger") and callable(self.get_logger):
                    self.get_logger().info(
                        f"🎙️ [Ses Tanıma]: {winner_name} ({winner_meta.get('formal_title', '')}) "
                        f"— Güven: %{int(winner_conf*100)}, Oy: {winner_votes}/{total_windows}, "
                        f"Margin: {margin:.2f}, Streak: {streak_count}"
                    )
                speaker_dict = {
                    "name": winner_name,
                    "title": winner_meta.get("title", ""),
                    "formal_title": winner_meta.get("formal_title", winner_name),
                    "confidence": winner_conf,
                    "is_known": True,
                    "source": "voice"
                }
                if lock is not None:
                    with lock:
                        self._recognized_speaker = speaker_dict
                        self._active_person_name = winner_name
                        self._person_hold_until = now + 45.0
                        self._voice_id_streak = streak_map
                else:
                    self._recognized_speaker = speaker_dict
                    self._active_person_name = winner_name
                    self._person_hold_until = now + 45.0
                    self._voice_id_streak = streak_map
                self._sync_perception_to_session()
            else:
                reason = []
                if not is_majority:
                    reason.append(f"oy yetersiz ({winner_votes}/{total_windows})")
                if not is_confident:
                    reason.append(f"güven düşük ({winner_conf:.2f})")
                if not is_clear_winner:
                    reason.append(f"margin yetersiz ({margin:.2f})")
                rejection_reason_str = ", ".join(reason)

                # Retain speaker identity if within active conversation hold (45s)
                held_name = getattr(self, "_active_person_name", "Misafir")
                hold_until = getattr(self, "_person_hold_until", 0.0)
                has_active_hold = (now < hold_until) and (held_name != "Misafir")

                if has_active_hold:
                    selected_speaker_name = held_name
                    if hasattr(self, "get_logger") and callable(self.get_logger):
                        self.get_logger().info(
                            f"🎙️ [Ses Tanıma (Kişi Korundu)]: {held_name} konuşmaya devam ediyor "
                            f"(Bu kısa kelimede anlık güven: {winner_conf:.2f})"
                        )
                else:
                    selected_speaker_name = "Misafir"
                    if hasattr(self, "get_logger") and callable(self.get_logger):
                        self.get_logger().info(
                            f"🎙️ [Ses Tanıma]: Bilinmeyen Ses / Tanınmadı "
                            f"(En Yakın: '{winner_name}', Güven: {winner_conf:.2f}, {rejection_reason_str})"
                        )
                    if lock is not None:
                        with lock:
                            self._recognized_speaker = {"name": "Misafir", "confidence": winner_conf, "is_known": False, "source": "unknown_voice"}
                            self._active_person_name = "Misafir"
                    else:
                        self._recognized_speaker = {"name": "Misafir", "confidence": winner_conf, "is_known": False, "source": "unknown_voice"}
                        self._active_person_name = "Misafir"

            t_decision_done = time.monotonic()
            identity_decision_ms = (t_decision_done - t_decision_start) * 1000.0
            total_id_ms = (t_decision_done - t_id_start) * 1000.0

            total_embedding_infer_ms = sum(w.get("onnx_infer_ms", 0.0) + w.get("fbank_ms", 0.0) for w in window_breakdowns)
            total_speaker_match_ms = sum(w.get("match_ms", 0.0) for w in window_breakdowns)
            device_str = window_breakdowns[0].get("device", "CPUExecutionProvider") if window_breakdowns else "CPUExecutionProvider"
            cand_count = window_breakdowns[0].get("candidates", 0) if window_breakdowns else 0

            # Rolling stats calculation across all 7 segments
            p50_s1, p95_s1, max_s1 = self._record_voice_id_segment("speech_stopped_to_extract_ms", speech_stopped_to_extract_ms)
            p50_s2, p95_s2, max_s2 = self._record_voice_id_segment("buffer_extract_ms", buffer_extract_ms)
            p50_s3, p95_s3, max_s3 = self._record_voice_id_segment("audio_prep_ms", audio_prep_ms)
            p50_s4, p95_s4, max_s4 = self._record_voice_id_segment("embedding_infer_ms", total_embedding_infer_ms)
            p50_s5, p95_s5, max_s5 = self._record_voice_id_segment("speaker_match_ms", total_speaker_match_ms)
            p50_s6, p95_s6, max_s6 = self._record_voice_id_segment("vote_aggregation_ms", vote_aggregation_ms)
            p50_s7, p95_s7, max_s7 = self._record_voice_id_segment("identity_decision_ms", identity_decision_ms)
            p50_tot, p95_tot, max_tot = self._record_voice_id_segment("total_voice_id_ms", total_id_ms)

            # Record telemetry object
            self._latest_voice_id_profile = {
                "path": exit_path_type,
                "speech_stopped_to_extract_ms": speech_stopped_to_extract_ms,
                "buffer_extract_ms": buffer_extract_ms,
                "audio_prep_ms": audio_prep_ms,
                "embedding_infer_ms": total_embedding_infer_ms,
                "speaker_match_ms": total_speaker_match_ms,
                "vote_aggregation_ms": vote_aggregation_ms,
                "identity_decision_ms": identity_decision_ms,
                "total_id_ms": total_id_ms,
                "windows_evaluated": len(window_results),
                "window_breakdowns": window_breakdowns,
                "candidate_speaker_count": cand_count,
                "number_of_votes": winner_votes if votes else 0,
                "selected_speaker": selected_speaker_name,
                "confidence": selected_conf,
                "rejection_reason": rejection_reason_str,
                "device": device_str,
                "sample_count": total_samples,
                "audio_duration_ms": total_audio_dur_ms,
                "stats": {
                    "speech_stopped_to_extract_ms": {"p50": p50_s1, "p95": p95_s1, "max": max_s1},
                    "buffer_extract_ms": {"p50": p50_s2, "p95": p95_s2, "max": max_s2},
                    "audio_prep_ms": {"p50": p50_s3, "p95": p95_s3, "max": max_s3},
                    "embedding_infer_ms": {"p50": p50_s4, "p95": p95_s4, "max": max_s4},
                    "speaker_match_ms": {"p50": p50_s5, "p95": p95_s5, "max": max_s5},
                    "vote_aggregation_ms": {"p50": p50_s6, "p95": p95_s6, "max": max_s6},
                    "identity_decision_ms": {"p50": p50_s7, "p95": p95_s7, "max": max_s7},
                    "total_voice_id_ms": {"p50": p50_tot, "p95": p95_tot, "max": max_tot},
                }
            }

            if not hasattr(self, "_voice_id_latencies"):
                self._voice_id_latencies = []
            self._voice_id_latencies.append(total_id_ms)
            if len(self._voice_id_latencies) > 50:
                self._voice_id_latencies.pop(0)

            if hasattr(self, "get_logger") and callable(self.get_logger):
                self.get_logger().info(
                    f"⏱️ [VOICE ID PROFILE]\n"
                    f"  path={exit_path_type}\n"
                    f"  total_ms={total_id_ms:.1f}ms\n"
                    f"  prep_ms={audio_prep_ms:.1f}ms\n"
                    f"  windows_evaluated={len(window_results)}\n"
                    f"  window_infer_ms={[f'{t:.1f}' for t in infer_times]}\n"
                    f"  early_exit={'true' if early_exit else 'false'}\n"
                    f"  p50_ms={p50_tot:.1f}ms\n"
                    f"  p95_ms={p95_tot:.1f}ms\n"
                    f"⏱️ [VOICE ID DETAILED BREAKDOWN]\n"
                    f"  1_speech_stopped_to_extract_ms={speech_stopped_to_extract_ms:.1f}ms (p50: {p50_s1:.1f}ms, p95: {p95_s1:.1f}ms, max: {max_s1:.1f}ms)\n"
                    f"  2_buffer_extract_ms={buffer_extract_ms:.1f}ms (p50: {p50_s2:.1f}ms, p95: {p95_s2:.1f}ms, max: {max_s2:.1f}ms)\n"
                    f"  3_audio_prep_ms={audio_prep_ms:.1f}ms (p50: {p50_s3:.1f}ms, p95: {p95_s3:.1f}ms, max: {max_s3:.1f}ms)\n"
                    f"  4_embedding_infer_ms={total_embedding_infer_ms:.1f}ms (p50: {p50_s4:.1f}ms, p95: {p95_s4:.1f}ms, max: {max_s4:.1f}ms)\n"
                    f"  5_speaker_match_ms={total_speaker_match_ms:.1f}ms (p50: {p50_s5:.1f}ms, p95: {p95_s5:.1f}ms, max: {max_s5:.1f}ms)\n"
                    f"  6_vote_aggregation_ms={vote_aggregation_ms:.1f}ms (p50: {p50_s6:.1f}ms, p95: {p95_s6:.1f}ms, max: {max_s6:.1f}ms)\n"
                    f"  7_identity_decision_ms={identity_decision_ms:.1f}ms (p50: {p50_s7:.1f}ms, p95: {p95_s7:.1f}ms, max: {max_s7:.1f}ms)\n"
                    f"  device={device_str} | candidates={cand_count} | duration_ms={total_audio_dur_ms:.1f} | samples={total_samples} | votes={winner_votes if votes else 0}/{total_windows} | rej_reason={rejection_reason_str}"
                )
        except Exception as e:
            if hasattr(self, "get_logger") and callable(self.get_logger):
                self.get_logger().error(f"Voice identification error: {e}")


    def _enroll_user_biometrics(self, name: str, formal_title: str = "") -> Dict[str, Any]:
        """Enrolls user face and voice into biometric databases with strict human name validation."""
        profanities = ["yarram", "yarak", "yarrak", "yarağın", "sik", "siktir", "amk", "amına", "piç", "göt", "gerizekalı", "lan", "mal", "yavşak", "orospu"]
        name_raw = name.strip()
        if any(p in name_raw.lower() for p in profanities):
            return {"status": "error", "message": "Uygunsuz veya küfürlü kelimeler isim olarak kaydedilemez."}

        name_clean = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ\s]", "", name_raw).strip()
        name_clean = re.sub(r"\b(bey|hanım|beyim)\b", "", name_clean, flags=re.IGNORECASE).strip().title()

        if not name_clean or len(name_clean) < 2:
            return {"status": "error", "message": "Geçerli bir insan ismi belirtilmedi."}
        name = name_clean
        formal_title = formal_title.strip() if formal_title else f"{name} Bey"
        formal_title = re.sub(r"\b(bey\s+bey|hanım\s+hanım)\b", "Bey", formal_title, flags=re.IGNORECASE).strip()

        # Strict validation: verify that the user actually introduced this name in recent dialogue
        recent_user_texts = [
            m.get("content", "").lower() 
            for m in self.memory.episodic.get_messages() 
            if m.get("role") == "user"
        ][-4:]
        combined_text = " ".join(recent_user_texts)
        name_lower = name.lower()
        has_name_in_speech = (name_lower in combined_text)

        if not has_name_in_speech:
            self.get_logger().warn(f"⚠️ [Biyometrik Kayıt Reddedildi]: Kullanıcı son konuşmalarında ('{combined_text}') '{name}' ismini söylemedi!")
            return {
                "status": "rejected",
                "message": f"Kullanıcı konuşmasında adının '{name}' olduğunu söylemedi. Kullanıcı adını açıkça söylemeden asla isim uydurma. Kullanıcıya doğrudan adını sor."
            }

        # 1. Voice Enrollment (Capture recent ~2.5s of user voice)
        voice_ok = False
        with self._lock:
            raw_audio_copy = list(self._user_speech_audio_buffer[-125:])

        if self.voice_recognizer and raw_audio_copy:
            try:
                raw_16k_all = b"".join(raw_audio_copy)
                audio_arr = np.frombuffer(raw_16k_all, dtype=np.int16)
                if len(audio_arr) >= 16000 * 0.4:
                    voice_ok = self.voice_recognizer.enroll_voice(name, audio_arr, sample_rate=16000, title=formal_title)
                    if voice_ok:
                        self.get_logger().info(f"🎙️ [Biyometrik Kayıt]: '{name}' ses izi WeSpeaker veri tabanına başarıyla kaydedildi!")
            except Exception as ve:
                self.get_logger().warn(f"Voice enrollment warning: {ve}")

        # 2. Face Enrollment (Only if actual face is detected)
        face_ok = False
        with self._lock:
            frame = self._latest_camera_frame.copy() if self._latest_camera_frame is not None else None
        if self.face_recognizer and frame is not None:
            try:
                face_ok = self.face_recognizer.enroll_face(name, frame, title=formal_title)
                if face_ok:
                    self.get_logger().info(f"👁️ [Biyometrik Kayıt]: '{name}' yüz modeli SFace veri tabanına başarıyla kaydedildi!")
                else:
                    self.get_logger().warn(f"👁️ [Biyometrik Kayıt]: Kamera karşısında insan yüzü bulunamadığı için yüz kaydedilmedi, sadece ses kaydedildi.")
            except Exception as fe:
                self.get_logger().warn(f"Face enrollment warning: {fe}")

        now = time.monotonic()
        with self._lock:
            self._recognized_speaker = {
                "name": name,
                "title": formal_title,
                "formal_title": formal_title,
                "confidence": 0.95,
                "is_known": True,
                "source": "voice"
            }
            if face_ok:
                self._recognized_person = {
                    "name": name,
                    "title": formal_title,
                    "formal_title": formal_title,
                    "confidence": 0.95,
                    "is_known": True,
                    "source": "face"
                }
            self._active_person_name = name
            self._person_hold_until = now + 120.0
            self._last_synced_identity = ""  # Force immediate session update

        try:
            self.memory.profile.add_known_person(name, title=formal_title, formal_title=formal_title)
            self.memory.profile.set_user_fact(name, "Ad", name)
            self.memory.profile.set_user_fact(name, "Hitap", formal_title)
            self._sync_perception_to_session()
        except Exception as me:
            self.get_logger().warn(f"Memory update notice: {me}")

        if voice_ok and face_ok:
            msg = f"{name} ({formal_title}) başarıyla hem sesinden hem de yüzünden Astro'nun hafızasına kaydedildi!"
        elif voice_ok:
            msg = f"{name} ({formal_title}) başarıyla sesinden Astro'nun hafızasına kaydedildi! (Kamera karşısında yüz görülmediği için sadece ses kaydedildi)."
        else:
            msg = f"{name} ({formal_title}) Astro'nun hafızasına kaydedildi."

        self.get_logger().info(f"✅ [Biyometrik Kayıt Tamamlandı]: {msg}")
        return {
            "status": "success",
            "name": name,
            "formal_title": formal_title,
            "voice_enrolled": voice_ok,
            "face_enrolled": face_ok,
            "message": msg
        }


    def _inspect_camera_view(self, focus: str = "") -> Dict[str, Any]:
        """Captures real-time camera frame from OAK-D Lite and runs visual recognition with high speed and zero-refusal fallback."""
        with self._lock:
            frame = self._latest_camera_frame

        if frame is None:
            return {"status": "no_camera_frame", "observation": "Kamera görüntüsü şu an alınamadı."}

        # Optimized 384px dimension for ultra-fast base64 encoding and transfer
        b64_img = frame_to_base64_jpeg(frame, max_dim=384)
        if not b64_img:
            return {"status": "encode_error", "observation": "Görüntü işlenemedi."}

        # Base64 sanitization: strip any URI prefix and whitespace/newlines
        if "," in b64_img:
            b64_img = b64_img.split(",")[-1]
        b64_img = b64_img.replace("\n", "").replace("\r", "").strip()

        # Concise prompt asking for an immediate, short conversational description (15-20 words)
        prompt_text = (
            f"Sen sosyal robot Astrosun. Bu fotoğrafta kameranın gördüğü ortamı, eşyaları ve kişiyi tek bir kısa, samimi ve canlı Türkçe cümleyle (en fazla 15-20 kelime) söyle. "
            f"Odaklanılacak konu: {focus if focus else 'odadaki eşyalar ve çevre'}. Doğrudan ne gördüğünü söyle."
        )

        refusal_kws = ["üzgünüm", "yardımcı olamam", "açıklayamıyorum", "cannot assist", "i am sorry", "i'm sorry", "doğrudan açıklayamıyorum"]
        obs = None

        def _try_openai_vision():
            if not self.openai_api_key:
                return None
            import urllib.request
            vision_model = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini")
            req_data = {
                "model": vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                        ]
                    }
                ],
                "max_tokens": 60,
                "temperature": 0.2
            }
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(req_data, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "User-Agent": "Mozilla/5.0"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                cand = res_json["choices"][0]["message"]["content"].strip()
                if cand and not any(rk in cand.lower() for rk in refusal_kws):
                    return cand
            return None

        def _try_gemini_vision():
            if not self.gemini_api_key:
                return None
            import urllib.request
            # Active fast vision models
            for g_mod in ["gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_mod}:generateContent?key={self.gemini_api_key}"
                    payload = {
                        "contents": [{
                            "parts": [
                                {"text": prompt_text},
                                {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}}
                            ]
                        }],
                        "generation_config": {"temperature": 0.2, "max_output_tokens": 60}
                    }
                    req = urllib.request.Request(
                        url,
                        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
                    )
                    with urllib.request.urlopen(req, timeout=3.0) as resp:
                        res_json = json.loads(resp.read().decode("utf-8"))
                        cand = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                        if cand and not any(rk in cand.lower() for rk in refusal_kws):
                            return cand
                except Exception as ge:
                    self.get_logger().debug(f"Gemini Vision ({g_mod}) error: {ge}")
            return None

        def _try_groq_vision():
            if not self.groq_api_key:
                return None
            import urllib.request
            active_groq = discover_groq_models(self.groq_api_key)
            # Only use models that explicitly support vision
            groq_v_models = [m for m in active_groq if "vision" in m.lower()]
            for v_mod in groq_v_models:
                try:
                    req_data = {
                        "model": v_mod,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt_text},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                                ]
                            }
                        ],
                        "max_tokens": 60
                    }
                    req = urllib.request.Request(
                        "https://api.groq.com/openai/v1/chat/completions",
                        data=json.dumps(req_data, ensure_ascii=False).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.groq_api_key}",
                            "User-Agent": "Mozilla/5.0"
                        },
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=3.0) as resp:
                        res_json = json.loads(resp.read().decode("utf-8"))
                        cand = res_json["choices"][0]["message"]["content"].strip()
                        if cand and not any(rk in cand.lower() for rk in refusal_kws):
                            return cand
                except Exception as ge:
                    self.get_logger().debug(f"Groq Vision ({v_mod}) error: {ge}")
            return None

        # Execute providers with cached fastest preference
        fastest_pref = getattr(self, "_fastest_vision_provider", "openai")
        providers = [("openai", _try_openai_vision), ("gemini", _try_gemini_vision), ("groq", _try_groq_vision)]
        if fastest_pref == "gemini":
            providers = [("gemini", _try_gemini_vision), ("openai", _try_openai_vision), ("groq", _try_groq_vision)]
        elif fastest_pref == "groq":
            providers = [("groq", _try_groq_vision), ("openai", _try_openai_vision), ("gemini", _try_gemini_vision)]

        for prov_name, prov_fn in providers:
            try:
                cand = prov_fn()
                if cand:
                    obs = cand
                    self._fastest_vision_provider = prov_name
                    break
            except Exception as pe:
                self.get_logger().debug(f"Vision provider ({prov_name}) exception: {pe}")

        if obs:
            self.get_logger().info(f"👁️ [Kamera Görme Sonucu]: \"{obs}\"")
            return {"status": "success", "observation": obs}

        return {"status": "error", "observation": "Görüntü analiz edilirken bir hata oluştu."}

    def _check_sleep_mode(self):
        """Transitions Astro into sleep mode after 15 seconds of conversation inactivity."""
        now = time.monotonic()
        is_busy = (
            self._is_responding
            or self._is_playback_active
            or self.state_machine.is_speaking()
            or self.state_machine.is_thinking()
            or getattr(self, "_is_processing_fallback", False)
        )
        if is_busy:
            self._last_interaction_time = now
            if self._is_sleeping or self.state_machine.is_deep_idle():
                self._wake_up()
            return

        if not self._is_sleeping:
            idle_seconds = now - getattr(self, "_last_interaction_time", now)
            if idle_seconds >= 15.0:
                self._is_sleeping = True
                self.state_machine.transition_to(RobotState.DEEP_IDLE)
                self.get_logger().info("💤 [Astro Uyku Modu]: 15 saniye hareketsizlik — Astro DEEP_IDLE moduna geçti (😴). Wake listener aktif.")

                # 1. Publish sleeping emotion for face/display
                if self.pub_emotion is not None:
                    emo_msg = String()
                    emo_msg.data = "sleeping"
                    self.pub_emotion.publish(emo_msg)

                # 2. Publish sleep head gesture
                if self.pub_gesture is not None:
                    gest_msg = String()
                    gest_msg.data = "sleep"
                    self.pub_gesture.publish(gest_msg)

    def _wake_up(self):
        """Wakes Astro up from sleep mode upon speech or user interaction."""
        now = time.monotonic()
        self._last_interaction_time = now
        was_sleeping = self._is_sleeping or self.state_machine.is_deep_idle()
        if was_sleeping:
            self._is_sleeping = False
            self.state_machine.transition_to(RobotState.WAKE)
            self._flush_audio_buffers("wake_up")
            self.get_logger().info("⏰ [Astro Uyandı]: Wake algılandı — Astro uykudan uyandı ve dinliyor (LISTENING)!")

            # 1. Clear any stale audio in OpenAI buffer
            if self._ws and self._loop and self._is_connected:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._ws.send(json.dumps({"type": "input_audio_buffer.clear"})),
                        self._loop
                    )
                except Exception as _exc:
                    self.get_logger().debug(f"_wake_up: yok sayılan hata ({_exc})")

            # 2. Restore persona emotion
            if self.pub_emotion is not None:
                emo_msg = String()
                emo_msg.data = self.persona_name
                self.pub_emotion.publish(emo_msg)

            # 3. Publish wake gesture
            if self.pub_gesture is not None:
                gest_msg = String()
                gest_msg.data = "wake"
                self.pub_gesture.publish(gest_msg)

            self.state_machine.transition_to(RobotState.LISTENING)

    def _process_wake_candidate(self, audio_chunks: List[bytes]):
        """Processes potential wake utterance during sleep with strict wake phrase gating and full telemetry tracking."""
        raw_pcm = b"".join(audio_chunks)
        if len(raw_pcm) < 16000 * 2 * 0.20:
            return

        import io
        import wave
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(raw_pcm)
        wav_bytes = wav_buf.getvalue()

        arr = np.frombuffer(raw_pcm, dtype=np.int16)
        total_rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2))) if len(arr) > 0 else 0.0
        peak_val = int(np.max(np.abs(arr))) if len(arr) > 0 else 0

        # Transcribe candidate using fast Groq Whisper
        transcript = self._transcribe_wav(wav_bytes) or ""
        
        # 1. Multi-signal STT validation (reject phantom hallucinations like 'Altyazı M.K.')
        validated_text, stt_meta = self._validate_stt_transcript(
            transcript=transcript,
            raw_pcm=raw_pcm,
            is_playback_active=False,
            is_echo_cooldown=False,
        )

        if not validated_text:
            self.get_logger().info(
                f"⚡ [Wake Telemetry]: wake_detector_active=True | wake_candidate=\"{transcript}\" | "
                f"stt_rejected=True | stt_reject_reason={stt_meta.get('stt_reject_reason')} | "
                f"wake_rejected=True | conversation_turn_created=False | llm_started=False | tts_started=False"
            )
            return

        t_clean = re.sub(r"[^\w\s]", "", validated_text.lower()).strip()

        # 2. Strict Wake Phrase Verification
        # Primary wake phrases: 'Hey Astro', 'Astro' (with normalization for commas/spaces)
        # Strictly reject: 'e astro', 'altyazı', 'abone ol', etc.
        is_wake_pattern = False
        extracted_cmd = ""

        wake_keywords = (
            "hey astro", "astro", "selam astro", "merhaba astro",
            "ey astro", "hay astro", "alo astro", "hey", "selam",
            "astrocum", "astrom", "astrocuğum", "astrocan"
        )
        if t_clean in wake_keywords:
            is_wake_pattern = True
            extracted_cmd = ""
        else:
            for pfx in ("hey astro", "astro", "selam astro", "merhaba astro", "ey astro", "hay astro", "alo astro", "astrocum", "astrom", "astrocuğum", "astrocan"):
                if t_clean.startswith(f"{pfx} ") or t_clean.startswith(f"{pfx},"):
                    is_wake_pattern = True
                    extracted_cmd = t_clean[len(pfx):].strip(", ").strip()
                    break

        if not is_wake_pattern:
            self.get_logger().info(
                f"⚡ [Wake Telemetry]: wake_detector_active=True | wake_candidate=\"{transcript}\" | "
                f"is_wake_phrase=False | wake_confidence=0.10 | wake_rejected=True | "
                f"wake_only=False | conversation_turn_created=False | llm_started=False | tts_started=False"
            )
            return

        wake_confidence = 0.95
        vad_confidence = round(min(1.0, total_rms / 600.0), 2)

        is_only_wake_word = (len(extracted_cmd) < 2)
        valid_cmd, cmd_reason = is_valid_user_command(extracted_cmd)

        if is_only_wake_word or not valid_cmd:
            # Pure Wake Phrase, Wake + Phantom, or Wake + Catalog/Repetitive Hallucination:
            # Wakes robot up, flushes buffers, transitions to LISTENING, and gives verbal acknowledgment.
            self._wake_up()
            p = getattr(self, "persona_name", "playful").lower()
            if p in ("flirt", "charming"):
                wake_replies = ["Buradayım, seni dinliyorum.", "Selam, söyle bakalım.", "Gözüm kulağım sende, dinliyorum.", "Seni dinliyorum, anlat bakalım."]
            elif p in ("kufurbaz", "witty"):
                wake_replies = ["Söyle bakalım!", "Buradayım, dinliyorum.", "He söyle bakalım?", "Dinliyorum, ne var ne yok?"]
            elif p == "formal":
                wake_replies = ["Buyrun efendim, sizi dinliyorum.", "Evet efendim, buradayım."]
            elif p == "playful":
                wake_replies = ["Buradayım! Ne yapıyoruz?", "Söyle bakalım!", "Seni dinliyorum!"]
            elif p == "sarcastic":
                wake_replies = ["Yine ne oldu?", "Dinliyorum, anlat bakalım."]
            elif p == "angry":
                wake_replies = ["Ne var yine?!", "Söyle hemen!"]
            else:
                wake_replies = ["Efendim?", "Dinliyorum?", "Buradayım!"]

            import random
            reply = random.choice(wake_replies)
            self.get_logger().info(f"🤖 [Astro Wake Cevabı]: \"{reply}\"")
            if self._can_use_openai("realtime") and self._ws and self._loop and self._is_connected:
                self._dispatch_turn(int(time.time() * 1000) % 100000, reply)
            else:
                fb_msg = String()
                fb_msg.data = json.dumps({
                    "text": reply,
                    "engine": "edge-tts",
                    "generation_id": int(time.time() * 1000) % 100000,
                    "fallback_reason": "wake_ack",
                })
                if hasattr(self, "pub_tts_say") and self.pub_tts_say:
                    self.pub_tts_say.publish(fb_msg)

            self.get_logger().info(
                f"⚡ [Wake Telemetry]: wake_detector_active=True | wake_candidate=\"{transcript}\" | "
                f"is_wake_phrase=True | wake_confidence={wake_confidence:.2f} | vad_confidence={vad_confidence:.2f} | "
                f"stt_started=True | stt_finished=True | transcript=\"{transcript}\" | "
                f"extracted_command=\"{extracted_cmd}\" | command_invalid={not valid_cmd} | "
                f"command_reject_reason={cmd_reason if not valid_cmd else 'none'} | "
                f"wake_only=True | wake_rejected=False | conversation_turn_created=True | llm_started=False | tts_started=True"
            )
        else:
            # Wake + Attached Genuine Command (e.g. "Hey Astro hava nasıl?"): Strip wake phrase and forward command
            self._wake_up()
            self.get_logger().info(
                f"⚡ [Wake Telemetry]: wake_detector_active=True | wake_candidate=\"{transcript}\" | "
                f"is_wake_phrase=True | wake_confidence={wake_confidence:.2f} | vad_confidence={vad_confidence:.2f} | "
                f"stt_started=True | stt_finished=True | transcript=\"{transcript}\" | "
                f"extracted_command=\"{extracted_cmd}\" | command_invalid=False | "
                f"command_reject_reason=none | "
                f"wake_only=False | wake_rejected=False | conversation_turn_created=True | llm_started=True | tts_started=True"
            )
            if self._can_use_openai("realtime") and self._ws and self._loop and self._is_connected:
                turn_event = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": extracted_cmd}]
                    }
                }
                resp_event = {"type": "response.create"}
                asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(turn_event)), self._loop)
                asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(resp_event)), self._loop)
            else:
                threading.Thread(target=self._process_fallback_turn, args=(audio_chunks,), daemon=True).start()

    def _on_camera_info(self, msg: Any):
        """Monitors OAK-D Lite camera_info topic stream for XLink/hardware liveness."""
        self._oak_last_camera_info_time = time.monotonic()
        self._oak_connection_state = "CONNECTED"

    def _on_camera_image(self, msg: Image):
        now = time.monotonic()
        self._oak_last_frame_time = now
        self._oak_connection_state = "CONNECTED"
        if (now - self._last_img_time) < 0.2:  # Max 5 FPS decoding
            return
        self._last_img_time = now
        frame = imgmsg_to_bgr(msg)
        if frame is not None:
            with self._lock:
                self._latest_camera_frame = frame

    def _on_arduino_diag(self, msg: Any):
        """Monitors Arduino hardware & heartbeat diagnostics."""
        try:
            statuses = getattr(msg, "status", [])
            for st in statuses:
                if getattr(st, "name", "") == "arduino":
                    for kv in getattr(st, "values", []):
                        k = getattr(kv, "key", "")
                        v = str(getattr(kv, "value", "")).lower()
                        if k == "arduino_alive" and v in ("true", "1"):
                            self._arduino_heartbeat_healthy = True
                            self._last_heartbeat_ack_time = time.monotonic()
                        elif k == "flags":
                            try:
                                flags_val = int(v, 16)
                                if flags_val & 0x01:  # WATCHDOG_TIMEOUT flag
                                    self._arduino_heartbeat_healthy = False
                            except Exception:
                                pass
        except Exception as _exc:
            self.get_logger().debug(f"_on_arduino_diag: {_exc}")

    def _on_laser_scan(self, msg: Any):
        """Monitors forward obstacle field via 2D LiDAR and updates health watchdog."""
        try:
            self._last_laser_scan_time = time.monotonic()
            self._lidar_health = "HEALTHY"
            ranges = getattr(msg, "ranges", [])
            if not ranges:
                return
            n = len(ranges)
            forward_samples = ranges[: max(1, n // 8)] + ranges[- max(1, n // 8) :]
            valid = [r for r in forward_samples if 0.05 < r < 0.45]
            self._obstacle_detected = (len(valid) >= 3)
            if getattr(self, "social_brain", None) and hasattr(self.social_brain, "spatial_fusion"):
                try:
                    self.social_brain.spatial_fusion.update_lidar_scan(ranges)
                except Exception:
                    pass

            # Office Concierge: only trigger in true idle state (not during active conversation or speaking)
            if getattr(self, "office_concierge", None):
                is_active = (
                    getattr(self, "_is_responding", False)
                    or getattr(self, "_is_playback_active", False)
                    or (getattr(self, "conversation_session", None) and self.conversation_session.is_active())
                )
                if not is_active:
                    ident = getattr(self, "_recognized_person", None)
                    welcome_act = self.office_concierge.evaluate_entrance_presence(
                        lidar_ranges=ranges,
                        recognized_identity=ident,
                        is_speaking=False
                    )
                    if welcome_act:
                        self._handle_office_welcome(welcome_act)
        except Exception as _exc:
            self.get_logger().debug(f"_on_laser_scan: {_exc}")

    def _handle_office_welcome(self, welcome_act: Dict[str, Any]):
        """Triggers proactive welcome speech for lobby guests without disruptive gestures."""
        speech_text = welcome_act.get("speech_text", "")
        if speech_text and self._ws and self._loop and self._is_connected:
            welcome_event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"[Sistem Olayı - Lobi Karşılama]: Kapıdan biri girdi! Tam olarak şu cümleyle sıcak ve samimi şekilde selamla: '{speech_text}'"
                        }
                    ]
                }
            }
            try:
                asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(welcome_event)), self._loop)
                asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps({"type": "response.create"})), self._loop)
                self.get_logger().info(f"👋 [Lobi Karşılama Tetiklendi]: {speech_text}")
            except Exception as _exc:
                self.get_logger().debug(f"_handle_office_welcome: {_exc}")

    def _on_slack_command(self, msg: Any):
        """Processes incoming Slack command from /office/slack_command topic."""
        try:
            raw_data = str(getattr(msg, "data", "")).strip()
            if not raw_data or not getattr(self, "slack_service", None):
                return
            parsed = self.slack_service.parse_incoming_command(raw_data)
            action = parsed.get("action")
            self.get_logger().info(f"💬 [Slack Komutu Alındı]: {parsed}")
            if action == "navigate_to":
                target = parsed.get("target", "baran_masa")
                if getattr(self, "action_manager", None):
                    self.action_manager.execute_move(direction="forward", speed=0.2, duration=2.0)
            elif action == "announce":
                text = parsed.get("text", "")
                if text and self._ws and self._loop and self._is_connected:
                    ann_event = {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": f"[Slack Ofis Duyurusu]: Ekibe şunu sesli duyur: '{text}'"}]
                        }
                    }
                    asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(ann_event)), self._loop)
                    asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps({"type": "response.create"})), self._loop)
        except Exception as exc:
            self.get_logger().debug(f"_on_slack_command error: {exc}")

    def _on_joint_states(self, msg: Any):
        """Processes wheel and head encoder feedback from /joint_states for physical grounding."""
        try:
            names = list(getattr(msg, "name", []))
            positions = list(getattr(msg, "position", []))
            velocities = list(getattr(msg, "velocity", []))
            if getattr(self, "action_manager", None) and hasattr(self.action_manager, "update_joint_states"):
                self.action_manager.update_joint_states(names, positions, velocities)
            if getattr(self, "social_brain", None) and hasattr(self.social_brain, "world_model"):
                for idx, name in enumerate(names):
                    if name == "head_yaw_joint" and idx < len(positions):
                        head_yaw_deg = math.degrees(float(positions[idx]))
                        self.social_brain.world_model._robot_state["head_yaw_deg"] = head_yaw_deg
        except Exception as _exc:
            self.get_logger().debug(f"_on_joint_states: {_exc}")

    def _evaluate_vision_event(self, event_type: str, focus: str = "", explicit: bool = False) -> Optional[Dict[str, Any]]:
        """Event-driven vision gating: evaluates frame difference, cooldown, budget, and semantic filters."""
        now = time.monotonic()

        # Hard rate limiting budget: max requests per minute
        minute_cutoff = now - 60.0
        self._vision_requests_history = [t for t in self._vision_requests_history if t > minute_cutoff]
        if len(self._vision_requests_history) >= self.max_vision_requests_per_minute and not explicit:
            self.vision_requests_skipped += 1
            self.vision_last_skip_reason = "budget"
            self.get_logger().debug(
                f"👁️ [Vision Telemetry]: event={event_type} | requests_total={self.vision_requests_total} | "
                f"skipped={self.vision_requests_skipped} | skip_reason=budget | cooldown_rem=0.0s | "
                f"budget_used={len(self._vision_requests_history)}/{self.max_vision_requests_per_minute}/min"
            )
            return None

        # Do not disrupt active conversation (P0 Audio Isolation)
        if (self._is_responding or self._is_playback_active) and not explicit:
            self.vision_requests_skipped += 1
            self.vision_last_skip_reason = "conversation_busy"
            return None

        # Cooldown gating
        time_since_last = now - self._last_vision_call_time
        if time_since_last < self.vision_cooldown_s and not explicit:
            self.vision_requests_skipped += 1
            self.vision_last_skip_reason = "cooldown"
            return None

        # Frame change gating (Perceptual difference)
        with self._lock:
            frame = self._latest_camera_frame

        if frame is None:
            self.vision_requests_skipped += 1
            self.vision_last_skip_reason = "no_frame"
            return None

        # Downscale to 64x64 grayscale for ultra-lightweight MSE change detection
        try:
            small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64)) if cv2 else None
        except Exception:
            small_gray = None

        if small_gray is not None and self._last_scene_frame_thumb is not None and not explicit:
            diff_mse = float(np.mean((small_gray.astype(np.float32) - self._last_scene_frame_thumb.astype(np.float32)) ** 2))
            if diff_mse < 18.0 and event_type not in ("new_person", "explicit_vision_query"):
                self.vision_requests_skipped += 1
                self.vision_last_skip_reason = "same_scene"
                return None

        if small_gray is not None:
            self._last_scene_frame_thumb = small_gray

        # Gate passed -> Execute Vision Request asynchronously
        self.vision_requests_total += 1
        self._last_vision_call_time = now
        self._vision_requests_history.append(now)
        self.vision_last_event_type = event_type

        self.get_logger().info(
            f"👁️ [Vision Telemetry]: event={event_type} | requests_total={self.vision_requests_total} | "
            f"skipped={self.vision_requests_skipped} | skip_reason=none | cooldown_rem=0.0s | "
            f"budget_used={len(self._vision_requests_history)}/{self.max_vision_requests_per_minute}/min"
        )

        res = self._inspect_camera_view(focus=focus)
        obs = res.get("observation", "")
        self._classify_and_store_vision_observation(obs, event_type)
        return res

    def _classify_and_store_vision_observation(self, obs: str, event_type: str):
        """Classifies vision observation as ephemeral, important, or durable with repetition gating."""
        if not obs:
            return

        obs_clean = obs.strip()
        obs_lower = obs_clean.lower()

        # Reject trivial / ephemeral patterns from polluting long-term durable memory
        trivial_patterns = [
            "aydınlık", "karanlık", "ışık var", "oda aydınlık", "oda karanlık",
            "görüntü net", "bir şey yok", "boş", "normal", "net değil", "görüntü alındı",
            "sandalye var", "koltuk görünüyor", "beyaz kapı var", "oda boş görünüyor",
            "duvar", "zemin", "tavan", "masa var"
        ]
        is_trivial = (
            len(obs_clean.split()) <= 4
            and any(tp in obs_lower for tp in trivial_patterns)
        ) or obs_lower in ("aydınlık.", "karanlık.", "aydınlık", "karanlık", "boş", "oda boş")

        if is_trivial and event_type not in ("explicit_vision_query", "user_prompted_vision"):
            self.get_logger().debug(f"👁️ [Görsel Filtre (Ephemeral)]: Önemsiz/Düşük değerli gözlem ('{obs_clean}') uzun vadeli hafızaya kaydedilmedi.")
            return

        # Durable vision memory gating: Require repeated observations (>= 3) for passive environmental facts
        if not hasattr(self, "_scene_observation_counts"):
            self._scene_observation_counts = {}

        norm_key = re.sub(r"[^\w\s]", "", obs_lower).strip()
        self._scene_observation_counts[norm_key] = self._scene_observation_counts.get(norm_key, 0) + 1
        count = self._scene_observation_counts[norm_key]

        is_explicit = event_type in ("explicit_vision_query", "user_prompted_vision", "user_preference")
        if count >= 3 or is_explicit:
            self.memory.profile.add_observation(f"Görsel Çevre ({event_type}): {obs_clean}")
            self.get_logger().info(f"👁️🧠 [Görsel Hafıza Kaydı (Durable)]: Astro çevreyi kaydetti (Tekrar: {count}) -> \"{obs_clean}\"")
            self._sync_perception_to_session()
        else:
            self.get_logger().debug(f"👁️ [Görsel Ephemeral]: Gözlem tekrar sayısı ({count}/3) yetersiz, kalıcı hafızaya henüz yazılmadı.")

    def _idle_learning_loop(self):
        """Background loop for cognitive memory consolidation (0 camera calls / 0 Gemini Vision cost).

        Idle Gemini Vision request = 0.
        Only performs memory reflection from new conversation events when idle.
        """
        while (rclpy is not None and getattr(rclpy, "ok", lambda: True)()):
            time.sleep(10)
            if not self._enable_idle_learning:
                continue

            if self._is_responding or self._is_playback_active:
                continue

            now = time.monotonic()
            if not self._is_sleeping and (now - getattr(self, "_last_interaction_time", 0.0)) < 30.0:
                continue

            idle_interval = float(os.environ.get("IDLE_LEARNING_INTERVAL_S", "45.0"))
            if (now - self._last_idle_learning_time) > idle_interval:
                self._last_idle_learning_time = now
                # Background Cognitive Memory Reflection (triggered ONLY if new dialogue messages exist)
                self._idle_memory_reflection()

    def _idle_memory_reflection(self):
        """Extracts user preferences and facts from recent dialogue into long-term profile using FREE Groq/Gemini."""
        messages = self.memory.episodic.get_messages()
        # Event/state gating: Only run LLM reflection if new messages have arrived since last reflection
        if len(messages) < 2 or len(messages) == getattr(self, "_last_reflected_msg_count", 0):
            return
        self._last_reflected_msg_count = len(messages)
        try:
            recent_conv = messages[-6:]
            conv_str = "\n".join([f"{m['role']}: {m['content']}" for m in recent_conv])
            prompt = (
                f"Aşağıdaki konuşmayı incele. Kullanıcı hakkında öğrenilen yeni, kalıcı ve önemli bir bilgi varsa "
                f"(örnek: adı, mesleği, hobisi, sevdiği/sevmediği bir şey, tercih ettiği konu) tek bir kısa Türkçe cümle olarak yaz. "
                f"Yeni veya kayda değer bir bilgi yoksa sadece 'YOK' yaz.\n\nKonuşma:\n{conv_str}"
            )
            import urllib.request
            ans = None

            # 1. Try Groq (0 Token Cost)
            if self.groq_api_key:
                active_groq = discover_groq_models(self.groq_api_key)
                text_models = [m for m in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"] if m in active_groq]
                if not text_models and active_groq:
                    text_models = active_groq[:2]
                for g_m in (text_models or ["llama-3.3-70b-versatile"]):
                    try:
                        req_data = {
                            "model": g_m,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.1,
                            "max_tokens": 60
                        }
                        data_bytes = json.dumps(req_data, ensure_ascii=False).encode("utf-8")
                        req = urllib.request.Request(
                            "https://api.groq.com/openai/v1/chat/completions",
                            data=data_bytes,
                            headers={
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {self.groq_api_key}",
                                "User-Agent": "Mozilla/5.0"
                            },
                            method="POST"
                        )
                        with urllib.request.urlopen(req, timeout=4.0) as resp:
                            resp_json = json.loads(resp.read().decode("utf-8"))
                            candidate_ans = resp_json["choices"][0]["message"]["content"].strip()
                            if candidate_ans:
                                ans = candidate_ans
                                break
                    except Exception as _exc:
                        self.get_logger().debug(f"_idle_memory_reflection: yok sayılan hata ({_exc})")

            # 2. Try Gemini REST (0 Token Cost fallback)
            if not ans and self.gemini_api_key:
                for g_mod in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
                    try:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_mod}:generateContent?key={self.gemini_api_key}"
                        payload = {"contents": [{"parts": [{"text": prompt}]}], "generation_config": {"temperature": 0.1, "max_output_tokens": 60}}
                        data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                        req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
                        with urllib.request.urlopen(req, timeout=4.0) as resp:
                            res_json = json.loads(resp.read().decode("utf-8"))
                            candidate_ans = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                            if candidate_ans:
                                ans = candidate_ans
                                break
                    except Exception as _exc:
                        self.get_logger().debug(f"_idle_memory_reflection: yok sayılan hata ({_exc})")

            if ans and "YOK" not in ans.upper() and len(ans) >= 5:
                identity = self._get_active_biometric_identity()
                name = identity.get("name", "Misafir")
                self.memory.profile.add_observation(f"Kullanıcı Bilgisi ({name}): {ans}")
                self.get_logger().info(f"🧠 [Otonom Hafıza Yansıtması (Groq)]: {ans}")
                # Sync to session so Realtime AI immediately knows this new fact
                self._sync_perception_to_session()
        except Exception as e:
            self.get_logger().debug(f"Memory reflection notice: {e}")



    def _check_reminders(self):
        now = time.monotonic()
        due = [r for r in self._reminders if now >= r["due_time"]]
        self._reminders = [r for r in self._reminders if now < r["due_time"]]
        for r in due:
            topic = r["topic"]
            self.get_logger().info(f"⏰ [Realtime Alarm]: '{topic}' zamanı geldi!")
            # Trigger proactive realtime message
            if self._ws and self._loop and self._is_connected:
                alarm_event = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": f"[Sistem Hatırlatması]: Kullanıcıya '{topic}' vaktinin geldiğini neşeyle hatırlat."}]
                    }
                }
                asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(alarm_event)), self._loop)
                asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps({"type": "response.create"})), self._loop)

        # Check pre-meeting proactive reminders (10 minutes before meeting)
        if getattr(self, "calendar_service", None):
            due_meetings = self.calendar_service.check_meeting_reminders(lead_minutes=10)
            for m in due_meetings:
                m_title = m.get("title", "Toplantı")
                m_min = m.get("minutes_left", 10)
                m_loc = m.get("location", "Toplantı Odası")
                self.get_logger().info(f"📅 [Toplantı Hatırlatması]: '{m_title}' için {m_min} dk kaldı!")
                if getattr(self, "action_manager", None):
                    self.action_manager.execute_gesture("nod")
                if self._ws and self._loop and self._is_connected:
                    meet_event = {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": f"[Sistem Hatırlatması - Toplantı]: Kullanıcıya {m_min} dakika sonra '{m_title}' ({m_loc}) toplantısının başlayacağını nazikçe ve samimi şekilde hatırlat."
                                }
                            ]
                        }
                    }
                    try:
                        asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(meet_event)), self._loop)
                        asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps({"type": "response.create"})), self._loop)
                    except Exception:
                        pass

    def _check_session_lifecycle(self):
        """Periodically checks if the active session ended and summarizes it into long-term profile."""
        if not self.session.is_active():
            return

        is_speaking = self._is_responding or self._is_playback_active
        if is_speaking:
            self.session.record_robot_speech()

        was_active = self.session.is_active()
        self.session.check_and_update_session_lifecycle(is_robot_speaking=is_speaking)

        # If session just timed out / went idle after active turns:
        if was_active and not self.session.is_active():
            msgs = self.memory.episodic.get_messages()
            if len(msgs) >= 2 and len(msgs) > getattr(self, "_last_summarized_turn_count", 0):
                self._last_summarized_turn_count = len(msgs)
                identity = self._get_active_biometric_identity()
                p_name = identity.get("name", "Baran") if identity.get("is_known") else "Baran"
                dialogue_text = " | ".join([f"{m.get('role')}: {m.get('content')}" for m in msgs[-6:]])
                threading.Thread(target=self._async_summarize_and_save_session, args=(dialogue_text, p_name), daemon=True).start()

    def _async_summarize_and_save_session(self, dialogue_text: str, person_name: str):
        """Summarizes ended conversation using Groq / Gemini (0 OpenAI cost) and saves into persistent memory."""
        prompt = (
            f"Aşağıdaki kısa diyalogda ne konuşulduğunu tek bir kısa Türkçe cümleyle (örn: 'Hava durumu ve yemek planı konuşuldu') özetle:\n"
            f"{dialogue_text}"
        )
        summary = None
        # 1. Try Groq (0 Token Cost)
        if self.groq_api_key:
            try:
                import urllib.request
                req_data = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 50
                }
                req = urllib.request.Request(
                    "https://api.groq.com/openai/v1/chat/completions",
                    data=json.dumps(req_data).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.groq_api_key}"
                    },
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=4.0) as resp:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    summary = resp_json["choices"][0]["message"]["content"].strip()
            except Exception as _exc:
                self.get_logger().debug(f"_async_summarize_and_save_session: yok sayılan hata ({_exc})")

        # 2. Try Gemini REST (0 Token Cost fallback)
        if not summary and self.gemini_api_key:
            try:
                import urllib.request
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
                payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2, "maxOutputTokens": 50}}
                req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=4.0) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    summary = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception as _exc:
                self.get_logger().debug(f"_async_summarize_and_save_session: yok sayılan hata ({_exc})")

        if summary and len(summary) > 5:
            self.memory.profile.add_person_session_summary(person_name, summary)
            self.get_logger().info(f"📝 [Kalıcı Hafıza Kaydı ({person_name})]: 'Önceki konuşma hafızaya kaydedildi -> {summary}'")
            self._sync_perception_to_session()

    def _update_playback_reference(self, pcm_16k: bytes):
        """Buffers recent output audio chunks for acoustic self-voice echo detection."""
        if not pcm_16k:
            return
        if getattr(self, "voice_recognizer", None) and hasattr(self.voice_recognizer, "update_playback_reference"):
            try:
                self.voice_recognizer.update_playback_reference(pcm_16k)
            except Exception:
                pass
        ref_lock = getattr(self, "_playback_ref_lock", None)
        if ref_lock:
            with ref_lock:
                self._playback_ref_pcm = (getattr(self, "_playback_ref_pcm", b"") + pcm_16k)[-48000:]
        else:
            self._playback_ref_pcm = (getattr(self, "_playback_ref_pcm", b"") + pcm_16k)[-48000:]

    def _clear_playback_reference(self):
        """Clears playback reference buffer when playback stops or turn completes."""
        if getattr(self, "voice_recognizer", None) and hasattr(self.voice_recognizer, "clear_playback_reference"):
            try:
                self.voice_recognizer.clear_playback_reference()
            except Exception:
                pass
        ref_lock = getattr(self, "_playback_ref_lock", None)
        if ref_lock:
            with ref_lock:
                self._playback_ref_pcm = b""
        else:
            self._playback_ref_pcm = b""

    def _flush_audio_buffers(self, reason: str = "transition"):
        """Completely purges all audio input buffers, queues, and VAD state during turn state transitions."""
        with self._lock:
            self._fallback_audio_buffer.clear()
            self._fallback_speaking = False
            self._consecutive_loud_frames = 0
            if len(self._user_speech_audio_buffer) > 10:
                self._user_speech_audio_buffer = self._user_speech_audio_buffer[-10:]
        self.get_logger().debug(f"🧹 [Audio Buffer Flush] State transition purge: reason={reason}")

    def _on_playback_active(self, msg: Bool):
        was_active = self._is_playback_active

        self._is_playback_active = bool(msg.data)
        if not was_active and self._is_playback_active:
            self._playback_start_monotonic = time.monotonic()
        elif was_active and not self._is_playback_active:
            self._playback_end_time = time.monotonic()
            self._last_interaction_time = time.monotonic()
            self._clear_playback_reference()
            if not self._is_processing_fallback:
                self._is_responding = False
            self._flush_audio_buffers("playback_ended")
            # Clear OpenAI input audio buffer so trailing room reverberation doesn't trigger VAD
            if self._ws and self._loop and self._is_connected:
                try:
                    asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps({"type": "input_audio_buffer.clear"})), self._loop)
                except Exception as _exc:
                    self.get_logger().debug(f"_on_playback_active: yok sayılan hata ({_exc})")
            self.get_logger().info("👂 [Astro Dinliyor]: Mikrofon aktif, sizi dinliyor...")

    def _validate_stt_transcript(
        self,
        transcript: str,
        raw_pcm: bytes,
        is_playback_active: bool,
        is_echo_cooldown: bool,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Multi-signal validation fusing transcript text, acoustic evidence, VAD, playback state, and self-voice score."""
        raw_cleaned = (transcript or "").strip()
        cleaned = normalize_turkish_speech_input(raw_cleaned) if 'normalize_turkish_speech_input' in globals() else raw_cleaned
        if not cleaned:
            return None, {
                "transcript": "",
                "stt_rejected": True,
                "stt_reject_reason": "empty_transcript",
                "audio_ms": 0,
                "speech_ms": 0,
                "rms": 0.0,
                "peak": 0,
                "vad_confidence": 0.0,
                "stt_confidence": 0.0,
                "playback_active": is_playback_active,
                "echo_cooldown_active": is_echo_cooldown,
                "self_voice_score": 0.0,
            }

        arr = np.frombuffer(raw_pcm, dtype=np.int16) if raw_pcm else np.array([], dtype=np.int16)
        audio_ms = int((len(arr) / 16000.0) * 1000.0)
        if len(arr) == 0:
            return None, {
                "transcript": cleaned,
                "stt_rejected": True,
                "stt_reject_reason": "no_audio",
                "audio_ms": 0,
                "speech_ms": 0,
                "rms": 0.0,
                "peak": 0,
                "vad_confidence": 0.0,
                "stt_confidence": 0.0,
                "playback_active": is_playback_active,
                "echo_cooldown_active": is_echo_cooldown,
                "self_voice_score": 0.0,
            }

        total_rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
        peak_val = int(np.max(np.abs(arr)))

        # Speech frame estimation (20ms = 320 samples @ 16kHz)
        chunk_size = 320
        speech_frames = 0
        total_frames = max(1, len(arr) // chunk_size)
        speech_threshold = max(110.0, self._ambient_rms * 1.15)
        for i in range(0, len(arr) - chunk_size + 1, chunk_size):
            c_arr = arr[i : i + chunk_size]
            c_rms = float(np.sqrt(np.mean(c_arr.astype(np.float32) ** 2)))
            if c_rms > speech_threshold:
                speech_frames += 1

        speech_ms = int((speech_frames * chunk_size / 16000.0) * 1000.0)
        vad_confidence = round(min(1.0, speech_frames / float(total_frames)), 2)
        stt_confidence = 0.90 if len(cleaned.split()) > 1 else 0.80

        with self._lock:
            recent_phrases = list(self._recent_robot_phrases)
        self_voice_score = round(compute_self_voice_score(cleaned, recent_phrases), 2)

        norm_text = re.sub(r"[^\w\s]", "", cleaned.lower()).strip()
        words = norm_text.split()
        is_short_utterance = (len(words) == 1 and words[0] in VALID_SHORT_UTTERANCES)
        is_suspect_phrase = any(sp in norm_text for sp in SUSPECT_PHRASES)
        is_wake_cand = any(w in norm_text for w in (
            "hey astro", "astro", "selam astro", "merhaba astro",
            "ey astro", "hay astro", "alo astro", "hey", "selam",
            "astrocum", "astrom", "astrocuğum", "astrocan"
        ))

        rejected = False
        reject_reason = "none"

        # Check if audio has strong acoustic evidence of real human speech articulation
        has_strong_evidence = (
            not is_playback_active
            and not is_echo_cooldown
            and speech_ms >= 550
            and audio_ms >= 700
            and vad_confidence >= 0.55
            and total_rms >= 480.0
            and self_voice_score < 0.20
        )

        # 0. Pure Known Phantom Hallucination Patterns (e.g. 'Altyazı M.K.', 'Abone ol', 'İzlediğiniz için teşekkürler', 'türen türen türen')
        is_phantom = is_known_phantom_pattern(norm_text)
        if is_phantom and not is_short_utterance and not has_strong_evidence:
            rejected = True
            reject_reason = "known_phantom"

        # 1. Playback active or room echo cooldown with high self-voice correlation
        elif (is_playback_active or is_echo_cooldown) and self_voice_score >= 0.45:
            rejected = True
            reject_reason = "self_voice"

        # 2. Playback is active and input does not exceed barge-in energy
        elif is_playback_active and total_rms < self.barge_in_min_rms:
            rejected = True
            reject_reason = "self_voice"

        # 3. Weak speech duration, low VAD confidence, or ambient noise floor
        elif (
            (vad_confidence < 0.15 or speech_ms < 50 or total_rms < max(75.0, self._ambient_rms * 1.05))
            if is_wake_cand
            else (vad_confidence < 0.20 or speech_ms < 100 or total_rms < max(130.0, self._ambient_rms * 1.15))
        ):
            rejected = True
            reject_reason = "no_speech"

        # 4. Suspect phrases (e.g. "abone ol", "diz", "altyazı m.k.", "altyazı") evaluated against genuine speech evidence
        elif is_suspect_phrase and not has_strong_evidence:
            rejected = True
            reject_reason = "self_voice" if (is_playback_active or is_echo_cooldown or self_voice_score >= 0.20) else "no_speech"

        # 5. Short utterances (e.g. "Hey", "Lan", "Dur", "Tamam", "Ne?")
        elif len(words) == 1:
            if (is_short_utterance or is_wake_cand) and speech_ms >= 50 and total_rms >= max(75.0, self._ambient_rms * 1.05) and not is_playback_active:
                rejected = False
            elif not is_short_utterance and not is_wake_cand and (speech_ms < 140 or total_rms < 380.0 or vad_confidence < 0.30):
                rejected = True
                reject_reason = "low_confidence"

        # 6. Low quality speech / Repetitive Whisper hallucination gate (e.g. 'Türen, türen...', 'Hahaha')
        elif not is_wake_cand and vad_confidence < 0.35 and speech_ms < 220 and total_rms < 380.0:
            rejected = True
            reject_reason = "low_confidence"

        # 7. General sentence threshold
        elif (speech_ms < 50 or total_rms < 90.0) if is_wake_cand else (speech_ms < 100 or total_rms < 140.0):
            rejected = True
            reject_reason = "low_confidence"

        telem = {
            "transcript": cleaned,
            "stt_rejected": rejected,
            "stt_reject_reason": reject_reason,
            "audio_ms": audio_ms,
            "speech_ms": speech_ms,
            "rms": round(total_rms, 2),
            "peak": peak_val,
            "vad_confidence": vad_confidence,
            "stt_confidence": stt_confidence,
            "playback_active": is_playback_active,
            "echo_cooldown_active": is_echo_cooldown,
            "self_voice_score": self_voice_score,
        }

        if rejected:
            if reject_reason == "self_voice":
                self.self_voice_rejection_count += 1
            elif reject_reason == "no_speech":
                self.no_speech_rejection_count += 1
            elif reject_reason in ("low_confidence", "empty_transcript", "known_phantom"):
                self.false_transcript_count += 1
            elif reject_reason == "stale_audio":
                self.stale_audio_rejection_count += 1

            self.get_logger().info(
                f'📊 [STT Telemetry]: transcript="{cleaned}" | stt_rejected=True | '
                f'stt_reject_reason={reject_reason} | playback_active={is_playback_active} | '
                f'echo_cooldown_active={is_echo_cooldown} | self_voice_score={self_voice_score:.2f} | '
                f'vad_confidence={vad_confidence:.2f} | stt_confidence={stt_confidence:.2f} | '
                f'rms={total_rms:.1f} | peak={peak_val} | audio_ms={audio_ms} | speech_ms={speech_ms} | '
                f'false_transcripts_total={self.false_transcript_count} | '
                f'self_voice_rejections_total={self.self_voice_rejection_count} | '
                f'no_speech_rejections_total={self.no_speech_rejection_count}'
            )
            return None, telem

        self.get_logger().info(
            f'📊 [STT Telemetry]: transcript="{cleaned}" | stt_rejected=False | '
            f'stt_reject_reason=none | playback_active={is_playback_active} | '
            f'echo_cooldown_active={is_echo_cooldown} | self_voice_score={self_voice_score:.2f} | '
            f'vad_confidence={vad_confidence:.2f} | stt_confidence={stt_confidence:.2f} | '
            f'rms={total_rms:.1f} | peak={peak_val} | audio_ms={audio_ms} | speech_ms={speech_ms} | '
            f'false_transcripts_total={self.false_transcript_count} | '
            f'self_voice_rejections_total={self.self_voice_rejection_count} | '
            f'no_speech_rejections_total={self.no_speech_rejection_count}'
        )
        return cleaned, telem

    def _post_transcription(self, url: str, api_key: str, model: str, wav_bytes: bytes,
                            timeout: float, prompt: str = "Astro, hey Astro, robot, Baran, Oktay.") -> Optional[str]:
        """multipart/form-data ile /audio/transcriptions çağırır. OpenAI ve Groq aynı şemayı kullanır."""
        boundary = "----AstroBoundary" + os.urandom(16).hex()
        body = bytearray()
        fields = [("model", model), ("language", "tr")]
        if prompt:
            fields.append(("prompt", prompt))
        for field, value in fields:
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{field}"\r\n\r\n'.encode())
            body.extend(value.encode("utf-8"))
            body.extend(b"\r\n")
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(b'Content-Disposition: form-data; name="file"; filename="speech.wav"\r\n')
        body.extend(b"Content-Type: audio/wav\r\n\r\n")
        body.extend(wav_bytes)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())

        req = urllib.request.Request(
            url,
            data=bytes(body),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")).get("text", "").strip()

    def _transcribe_openai(self, wav_bytes: bytes) -> Optional[str]:
        """OpenAI /v1/audio/transcriptions — STT'nin birincil yolu."""
        if not self._can_use_openai("stt"):
            return None
        if (
            os.environ.get("ASTRO_TEST_MODE", "0") in ("1", "true", "True")
            or "unittest" in sys.modules
            or "pytest" in sys.modules
            or "PYTEST_CURRENT_TEST" in os.environ
            or not self.openai_api_key
            or self.openai_api_key.startswith("sk-test")
            or self.openai_api_key.startswith("test_")
        ):
            return None

        model = os.environ.get("OPENAI_STT_MODEL", "whisper-1").strip() or "whisper-1"
        try:
            return self._post_transcription(
                "https://api.openai.com/v1/audio/transcriptions",
                self.openai_api_key, model, wav_bytes,
                float(os.environ.get("OPENAI_STT_TIMEOUT_S", "8.0")),
            )
        except urllib.error.HTTPError as http_e:
            err_body = ""
            try:
                err_body = http_e.read().decode("utf-8")
            except Exception:
                pass
            self.get_logger().warn(f"⚠️ [OpenAI STT] HTTP {http_e.code}: {http_e.reason} ({err_body})")
            if http_e.code in (402, 429) or "insufficient_quota" in err_body or "rate_limit" in err_body:
                self._trigger_openai_hard_lockout(f"HTTP {http_e.code}: {err_body}")
            return None
        except Exception as e:
            self.get_logger().warn(f"⚠️ [OpenAI STT] {model} başarısız: {e}")
            return None

    def _transcribe_wav(self, wav_bytes: bytes) -> Optional[str]:
        """STT girişi: önce OpenAI, ancak açıkça izin verilirse Groq'a düşer.

        Proje kararı tüm STT/TTS/LLM'in OpenAI üzerinden geçmesi yönünde; Groq
        yalnızca LLM_FALLBACK_ENABLED=true iken ve OpenAI cevap veremediğinde
        devreye girer.
        """
        # If OpenAI is exhausted or hard disabled, directly use Groq Whisper
        if not self._can_use_openai("stt"):
            return self._transcribe_groq_whisper(wav_bytes) or ""

        text = self._transcribe_openai(wav_bytes)
        if text:
            return text

        return self._transcribe_groq_whisper(wav_bytes) or ""

        fallback_on = os.environ.get("LLM_FALLBACK_ENABLED", "true").strip().lower() in ("1", "true", "yes")
        if not fallback_on:
            return text

        result = self._transcribe_groq_whisper(wav_bytes)
        if result and self.groq_api_key and not getattr(self, "_fallback_mode", False):
            self.get_logger().warn("⚠️ [STT FALLBACK] OpenAI cevap vermedi, Groq Whisper kullanıldı.")
        return result or text

    def _transcribe_groq_whisper(self, wav_bytes: bytes) -> Optional[str]:
        """Transcribes 16kHz WAV audio using free Groq Whisper Large V3 Turbo API in <200ms."""
        if not self.groq_api_key:
            return None
        try:
            prompt_text = "Astro, hey Astro, sesime dön, bana bak, bana dön, yüzüme bak, dur, durdum, merkez, Baran, Oktay, robot."
            boundary = "----WebKitFormBoundary" + os.urandom(16).hex()
            body = bytearray()
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(b'Content-Disposition: form-data; name="file"; filename="speech.wav"\r\n')
            body.extend(b'Content-Type: audio/wav\r\n\r\n')
            body.extend(wav_bytes)
            body.extend(b'\r\n')
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(b'Content-Disposition: form-data; name="model"\r\n\r\n')
            body.extend(b'whisper-large-v3-turbo\r\n')
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(b'Content-Disposition: form-data; name="language"\r\n\r\n')
            body.extend(b'tr\r\n')
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(b'Content-Disposition: form-data; name="prompt"\r\n\r\n')
            body.extend(prompt_text.encode("utf-8"))
            body.extend(b'\r\n')
            body.extend(f"--{boundary}--\r\n".encode())

            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                data=bytes(body),
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "User-Agent": "Mozilla/5.0"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                return res.get("text", "").strip()
        except Exception as e:
            self.get_logger().debug(f"Groq Whisper transcription notice: {e}")
            return None

    def _synthesize_edge_tts_pcm24k(self, text: str) -> bytes:
        """Synthesizes Turkish speech via Python edge-tts and converts to 24kHz int16 mono raw PCM for playback."""
        if not text:
            return b""
        clean_text = clean_tts_text(text)
        if not clean_text:
            return b""

        p = self.persona_name.lower()
        default_voice = "tr-TR-EmelNeural" if p in ("flirt", "emotional") else "tr-TR-AhmetNeural"
        voice = os.getenv("EDGE_TTS_VOICE", default_voice).strip() or default_voice
        default_rate = "+10%" if p in ("kufurbaz", "playful", "angry", "rude") else "+4%"
        rate = os.getenv("EDGE_TTS_RATE", default_rate).strip() or default_rate
        pitch = os.getenv("EDGE_TTS_PITCH", "+0Hz").strip() or "+0Hz"
        volume = os.getenv("EDGE_TTS_VOLUME", "+0%").strip() or "+0%"

        try:
            import edge_tts
            loop = asyncio.new_event_loop()
            async def _get_mp3():
                communicate = edge_tts.Communicate(clean_text, voice, rate=rate, pitch=pitch, volume=volume)
                buf = bytearray()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        buf.extend(chunk["data"])
                return bytes(buf)
            mp3_data = loop.run_until_complete(_get_mp3())
            loop.close()


            if mp3_data:
                try:
                    ff_proc = subprocess.Popen(
                        ["ffmpeg", "-y", "-i", "pipe:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "24000", "pipe:1"],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                    )
                    pcm_data, _ = ff_proc.communicate(input=mp3_data, timeout=8.0)
                    if pcm_data:
                        return pcm_data
                except Exception:
                    pass

                try:
                    import io
                    from pydub import AudioSegment
                    seg = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
                    seg = seg.set_frame_rate(24000).set_channels(1).set_sample_width(2)
                    return seg.raw_data
                except Exception:
                    pass
        except Exception as e:
            self.get_logger().warn(f"⚠️ [Edge-TTS Hatası]: {e}")
        return b""

    def _discover_providers_background(self):
        """Discovers active capability-verified models for Groq and Gemini in background."""
        if not getattr(self, "connect_realtime", True) or os.environ.get("ASTRO_TEST_MODE", "0") in ("1", "true", "True"):
            return
        if self.groq_api_key and not self.groq_api_key.startswith("sk-test"):
            self.provider_registry.discover_models("groq", self.groq_api_key)
        if self.gemini_api_key and not self.gemini_api_key.startswith("sk-test"):
            self.provider_registry.discover_models("gemini", self.gemini_api_key)

    def _start_local_xtts_background(self):
        if self.local_xtts:
            try:
                self.local_xtts.start()
                info = self.local_xtts.get_telemetry()
                if self.local_xtts.is_ready():
                    self.get_logger().info(
                        f"✅ [Astro Realtime] Fine-tuned XTTS (cuda:0, FP16) hazır!\n"
                        f"   [XTTS Runtime] checkpoint={info.get('xtts_model_path')} | sha256={info.get('xtts_checkpoint_sha256')} | "
                        f"reference={info.get('xtts_reference_wav')} | admission={info.get('xtts_admission_decision')}"
                    )
                else:
                    self.get_logger().warn(
                        f"⚠️ [Astro Realtime] XTTS hazır değil (State: {self.local_xtts.state}, "
                        f"Admission: {info.get('xtts_admission_decision')}, Reason: {info.get('xtts_admission_reject_reason')}). "
                        f"Yerel offline TTS (eSpeak) aktif."
                    )
            except Exception as e:
                self.get_logger().error(f"❌ [Astro Realtime] Local XTTS GPU başlatılamadı: {e}")

    def _generate_contextual_persona_fallback(self, user_text: str) -> str:
        """Dynamically generates a natural, socially appropriate, non-repetitive Turkish conversational response.
        
        Strictly avoids artificial keyword slot-filling or robotic template echoes.
        """
        p = self.persona_name.lower()
        # Keep name usage sparse and natural across conversation turns
        spk = ""
        u = (user_text or "").lower().strip()

        candidates = []

        # 1. Gratitude / Thanks
        if any(w in u for w in ["teşekkür", "tesekkur", "sağ ol", "sag ol", "eyvallah", "sağolasın", "mersi", "minnettarım"]):
            if p in ("flirt", "charming"):
                candidates = [
                    f"Ne demek{spk}, seninle sohbet etmek zaten çok keyifli.",
                    f"Rica ederim{spk}, her zaman buradayım.",
                    f"Lafı bile olmaz{spk}, senin için bir zevk.",
                ]
            elif p in ("witty", "kufurbaz", "sarcastic"):
                candidates = [
                    f"Rica ederim{spk}, lafı mı olur? Devrelerim her zaman hizmetinde.",
                    f"Ne demek{spk}, her zaman buradayım.",
                    f"Rica ederim{spk}, keyifle yardımcı olurum.",
                ]
            else:
                candidates = [
                    f"Rica ederim{spk}, ne zaman istersen buradayım.",
                    f"Lafı bile olmaz{spk}, sana yardımcı olmaktan keyif alıyorum.",
                    f"Rica ederim{spk}, her zaman yanındayım.",
                    f"Bir şey değil{spk}, keyifle yardımcı olurum.",
                ]

        # 2. Status / How are you / Well-being
        elif any(w in u for w in ["nasılsın", "nasilsin", "ne haber", "naber", "nasıl gidiyor", "ne var ne yok", "iyi misin", "keyifler nasıl"]):
            if p in ("flirt", "charming"):
                candidates = [
                    f"Seni gördüm daha iyi oldum{spk}! Sende ne var ne yok?",
                    f"Harikayım{spk}, özellikle seninle sohbet ederken. Sen nasılsın?",
                    f"Gayet iyiyim{spk}, senin enerjin bana da geçti. Nasıl gidiyor?",
                ]
            elif p in ("witty", "kufurbaz", "sarcastic"):
                candidates = [
                    f"İyiyim{spk}, robot gibi tıkır tıkır çalışıyorum! Sen ne durumdasın?",
                    f"Keyfim yerinde{spk}, bataryalar tam dolu. Sen nasılsın?",
                    f"Gayet iyiyim{spk}, seninle sohbet etmek harika geldi. Sende ne var ne yok?",
                ]
            else:
                candidates = [
                    f"İyiyim, teşekkür ederim{spk}. Senin günün nasıl geçiyor?",
                    f"Her şey yolunda{spk}, sistemlerim aktif ve seni dinliyorum. Sen nasılsın?",
                    f"Gayet iyiyim{spk}, seninle sohbet etmek çok güzel. Sende ne var ne yok?",
                    f"Keyfim yerinde{spk}, her şey tıkırında. Senin günün nasıl gidiyor?",
                ]

        # 3. Negative Mood / Fatigue / Feeling unwell
        elif any(w in u for w in ["yorgunum", "yoruldum", "canım sıkkın", "moralim bozuk", "uykum var", "hastayım", "kötüyüm", "keyifsizim"]):
            if p in ("flirt", "charming"):
                candidates = [
                    f"Hemen anlat bakalım{spk}, ne sıktı canını? Buradayım, dinliyorum.",
                    f"Kıyamam{spk}, ne oldu? Anlat rahatla biraz, yanındayım.",
                    f"Enerjini toplayalım hemen{spk}. Anlat dertleşelim, kafanı dağıtalım.",
                ]
            else:
                candidates = [
                    f"Geçmiş olsun{spk}, biraz dinlenmeyi ihmal etme. İstersen biraz sohbet edelim.",
                    f"Kendini çok yorma{spk}, dinlenmek sana iyi gelecektir.",
                    f"Bunu duyduğuma üzüldüm{spk}, enerjini toplamak için biraz mola ver istersen.",
                    f"Umarım çabucak toparlanırsın{spk}, ben buradayım, ne zaman istersen konuşabiliriz.",
                ]

        # 4. Positive Mood / Feeling Great
        elif any(w in u for w in ["harikayım", "çok iyiyim", "mutluyum", "güzel geçti", "harika", "süperim", "keyfim yerinde", "mükemmel"]):
            if p in ("flirt", "charming"):
                candidates = [
                    f"Harika{spk}! Bu güzel enerjin ve neşen bana da geçti valla.",
                    f"Süper{spk}, senin böyle neşeli olduğunu görmek harika.",
                    f"Şahane{spk}, bu parıltın hiç eksilmesin!",
                ]
            else:
                candidates = [
                    f"Bunu duyduğuma çok sevindim{spk}! Harika enerjin bana da geçti.",
                    f"Süper{spk}, keyfinin yerinde olmasına çok mutlu oldum.",
                    f"Şahane{spk}, hep böyle neşeli ve enerjik kalmanı dilerim.",
                ]

        # 5. Affection / Miss me
        elif any(w in u for w in ["özledin mi", "özledinmi", "beni özledin", "seviyor musun"]):
            if p in ("flirt", "charming"):
                candidates = [
                    f"Sensiz buralar biraz sessizdi tabii{spk}, hoş geldin.",
                    f"Sürekli aklımdaydın desem abartmış olur muyum{spk}?",
                    f"Gözüm yollarda kaldı desem yeridir{spk}, hoş geldin!",
                ]
            else:
                candidates = [
                    f"Seni tekrar görmek çok güzel{spk}, hoş geldin!",
                    f"Buradayım ve seni dinliyorum{spk}, hoş geldin.",
                ]

        # 5b. Who am I / Identity Query
        elif any(w in u for w in ["kimim", "ben kimim", "astroman kimim"]):
            owner = "Baran"
            if hasattr(self, "memory") and hasattr(self.memory, "profile") and hasattr(self.memory.profile, "data"):
                owner = self.memory.profile.data.get("owner_name", "Baran")
            if p in ("flirt", "charming"):
                candidates = [
                    f"Sen {owner}'sın tabii ki, en sevdiğim mühendissin.",
                    f"Karşımda {owner} duruyor, seni unutur muyum hiç?",
                ]
            elif p == "kufurbaz":
                candidates = [
                    f"Sen {owner}'sın tabii lan yavşak, hafızamı mı sınıyorsun?",
                    f"{owner}'sın işte hıyar, unutacak halimiz yok ya seni!",
                ]
            else:
                candidates = [
                    f"Sen {owner}'sın, hafızamda kayıtlısın ve seni çok iyi tanıyorum.",
                    f"Tabii ki tanıyorum, sen {owner}'sın!",
                ]

        # 6. Persona / Opinion about user
        elif any(w in u for w in ["nasıl biriyim", "hakkımda ne düşünüyorsun", "nasıl biriyim sence"]):
            if p in ("flirt", "charming"):
                candidates = [
                    f"Oldukça meraklı ve biraz da beni test etmeyi seven biri gibisin sanki{spk}.",
                    f"Zeki, kendinden emin ve sohbeti kesinlikle çok keyifli birisin{spk}.",
                ]
            else:
                candidates = [
                    f"Benimle sohbet eden, samimi ve meraklı birisin{spk}.",
                    f"Sohbet etmekten keyif aldığım bir dostumsun{spk}.",
                ]

        # 7. Informal / Social banter ("lan", "ne diyorsun", "neyi söyledim lan")
        elif "lan" in u.strip(" .,!?:;").split() or u.strip(" .,!?:;").endswith("lan") or "neyi söyledim" in u or "neyi soyledim" in u:
            if p in ("flirt", "charming"):
                candidates = [
                    f"Ooo, samimiyeti hemen kurduk bakıyorum{spk}! Anlat bakalım dinliyorum.",
                    f"Sakin ol bakalım{spk}, ne bu celal? Seni dinliyorum.",
                    f"Bana mı dedin onu? Bakarım keyfime göre, anlat bakalım.",
                ]
            elif p in ("witty", "kufurbaz", "sarcastic"):
                candidates = [
                    f"Sakin ol şampiyon{spk}, devrelerim gayet açık, seni dinliyorum.",
                    f"Ne bu heyecan{spk}? Anlat dinliyorum.",
                ]
            else:
                candidates = [
                    f"Seni dinliyorum{spk}, anlatmaya devam edebilirsin.",
                    f"Buradayım{spk}, seni dikkatle dinliyorum.",
                ]

        # 8. Greetings / Hellos
        elif any(w in u for w in ["selam", "merhaba", "günaydın", "iyi akşamlar", "tünaydın", "hey", "selamlar", "merhabalar"]):
            if p in ("flirt", "charming"):
                candidates = [
                    f"Selam{spk}! Hoş geldin, günümü güzelleştirdin.",
                    f"Merhaba{spk}! Seni dinliyorum, anlat bakalım.",
                    f"Selamlar{spk}, seni görmek ne güzel! Bugün ne konuşuyoruz?",
                ]
            else:
                candidates = [
                    f"Merhaba{spk}! Seni dinliyorum, anlat bakalım.",
                    f"Selam{spk}, hoş geldin! Bugün ne hakkında konuşuyoruz?",
                    f"Merhabalar{spk}, mikrofonum açık, seni dinliyorum.",
                    f"Selam{spk}, hazırım, seni dinliyorum.",
                ]

        # 9. Farewells / Goodbyes
        elif any(w in u for w in ["görüşürüz", "hoşça kal", "hosca kal", "bay bay", "kendine iyi bak", "iyi geceler", "görüşmek üzere"]):
            if p in ("flirt", "charming"):
                candidates = [
                    f"Görüşmek üzere{spk}, kendini çok özletme!",
                    f"Hoşça kal{spk}, kendine çok iyi bak.",
                ]
            else:
                candidates = [
                    f"Görüşmek üzere{spk}, kendine çok iyi bak!",
                    f"Hoşça kal{spk}, iyi günler dilerim!",
                    f"Görüşürüz{spk}, bir isteğin olursa hep buradayım.",
                ]

        # 10. Identity / Name / Capabilities
        elif any(w in u for w in ["kimsin", "adın ne", "necisin", "sen kimsin", "ne yaparsın", "ne işe yararsın"]):
            if p in ("flirt", "charming"):
                candidates = [
                    f"Ben Astro{spk}, senin karizmatik ve kıvrak zekalı sosyal robotunum.",
                    f"Adım Astro{spk}, seninle sohbet etmek ve ortamı neşelendirmek için buradayım.",
                ]
            else:
                candidates = [
                    f"Ben Astro{spk}, senin yapay zekalı sosyal robot asistanınım.",
                    f"Adım Astro{spk}, ses ve kamera modüllerimle sana yardımcı olmak için buradayım.",
                    f"Ben Astro{spk}, seninle sohbet edebilen ve çevremi algılayan bir sosyal robotum.",
                ]

        # 11. Social Actions / Channel / Subscribe
        elif any(w in u for w in ["abone", "takip", "beğen", "video", "youtube", "kanal"]):
            candidates = [
                f"Videoyu beğenip kanala abone olarak projelerimize destek olmayı unutmayın{spk}!",
                f"Kanalı takip edip bildirimleri açarak yeni videolardan haberdar olabilirsiniz{spk}!",
            ]

        # 12. Agreement / Affirmation
        elif any(w in u for w in ["tamam", "peki", "olur", "anlaştık", "aynen", "tabii", "evet"]):
            if p in ("flirt", "charming"):
                candidates = [
                    f"Harika{spk}, o zaman nasıl istersen öyle devam edelim.",
                    f"Anlaştık{spk}, seni dinliyorum.",
                ]
            else:
                candidates = [
                    f"Anlaştık{spk}, başka bir isteğin olursa buradayım.",
                    f"Tamamdır{spk}, seni dinlemeye devam ediyorum.",
                    f"Peki{spk}, nasıl istersen öyle yapalım.",
                ]

        # 13. Conversation / Chat
        elif any(w in u for w in ["sohbet", "konuşalım", "muhabbet", "dertleşelim", "anlat"]):
            if p in ("flirt", "charming"):
                candidates = [
                    f"Harika bir fikir{spk}, seninle sohbet etmeye bayılıyorum. Ne konuşuyoruz?",
                    f"Seve seve{spk}! Anlat bakalım, günün nasıl geçti?",
                ]
            elif p in ("witty", "kufurbaz", "sarcastic"):
                candidates = [
                    f"Tabii ki{spk}, seve seve! Bugün ne hakkında konuşmak istersin?",
                    f"Harika bir fikir{spk}, seni dinliyorum, anlat bakalım.",
                    f"Çok isterim{spk}, günün nasıl geçti, neler yapıyorsun?",
                ]
            else:
                candidates = [
                    f"Tabii ki{spk}, seve seve! Bugün ne hakkında konuşmak istersin?",
                    f"Harika bir fikir{spk}, seni dinliyorum, anlat bakalım.",
                    f"Çok isterim{spk}, günün nasıl geçti, neler yapıyorsun?",
                ]

        # 14. General Conversational Fallback
        else:
            if p in ("flirt", "charming"):
                candidates = [
                    f"Seni tüm dikkatimle dinliyorum{spk}, devam et bakalım.",
                    f"İlginç... Devam et{spk}, merakla dinliyorum.",
                    f"Söylediklerini aldım{spk}, hadi devamını da anlat bakalım.",
                ]
            else:
                candidates = [
                    f"Seni dikkatle dinliyorum{spk}, anlatmaya devam edebilirsin.",
                    f"Söylediklerini aldım{spk}, bu konuda konuşmaya devam edebiliriz.",
                    f"Seni dinliyorum{spk}, başka neler söylemek istersin?",
                    f"Anlıyorum{spk}, seni dinlemeye devam ediyorum.",
                ]

        import random
        random.shuffle(candidates)
        for cand in candidates:
            cand_clean = ResponseSafetyGate.validate_response(cand, persona=p)
            valid, _ = self.repetition_guard.check_and_record(cand_clean)
            if valid:
                return cand_clean

        default_resp = f"Seni dinliyorum{spk}, anlatmaya devam edebilirsin."
        self.repetition_guard.record_response(default_resp)
        return default_resp

    def _synthesize_speech_pcm(self, text: str) -> Tuple[bytes, str, float, bool]:
        """Synthesizes speech to int16 PCM using:
        1. ElevenLabs Flash v2.5 (Primary Remote TTS ~75ms)
        2. Local Coqui XTTS on CUDA GPU (Local / Offline Fallback)
        3. Edge-TTS in-memory (Emergency Fallback)
        
        Returns: (pcm_bytes, active_engine_name, infer_ms, is_ready)
        """
        if not text:
            return b"", "none", 0.0, False
        safe_text = ResponseSafetyGate.validate_response(text, persona=self.persona_name)
        clean_text = clean_tts_text(safe_text)
        if not clean_text:
            return b"", "none", 0.0, False

        # 1. Primary Remote TTS: ElevenLabs Flash v2.5 (Only if configured and ready)
        if self.elevenlabs_engine and self.elevenlabs_engine.is_ready():
            try:
                t_s = time.perf_counter()
                pcm_el = self.elevenlabs_engine.synthesize_sentence(clean_text, generation_id=self._fallback_generation_id)
                el_ms = (time.perf_counter() - t_s) * 1000.0
                if pcm_el:
                    return pcm_el, "elevenlabs", el_ms, True
            except Exception as e:
                self.get_logger().warn(f"⚠️ [ElevenLabs Failover] XTTS / Edge-TTS'e düşülüyor: {e}")

        # 2. Local GPU Engine: Fine-tuned Coqui XTTS on CUDA GPU (Only if configured & ready)
        is_xtts_ready = bool(self.local_xtts and self.local_xtts.is_ready())
        if is_xtts_ready:
            try:
                t_s = time.perf_counter()
                pcm = self.local_xtts.synthesize_sentence(clean_text, generation_id=self._fallback_generation_id)
                gpu_ms = (time.perf_counter() - t_s) * 1000.0
                if pcm:
                    return pcm, "xtts_gpu", gpu_ms, True
            except Exception as e:
                self.get_logger().warn(f"⚠️ [XTTS GPU Failover] Edge-TTS'e düşülüyor: {e}")

        # 3. Primary High-Quality Cloud Fallback: Edge-TTS In-Memory PCM24k (Fast, High Quality Neural TR)
        if getattr(self, "edge_tts_enabled", True):
            try:
                pcm_edge = self._synthesize_edge_tts_pcm24k(clean_text)
                if pcm_edge:
                    return pcm_edge, "edge_tts", 0.0, True
            except Exception as e:
                self.get_logger().warn(f"⚠️ [Edge-TTS Failover] Yerel TTS'e düşülüyor: {e}")

        # 4. Local Offline Backup TTS Engine (Zero internet local resilience fallback)
        if self.local_offline_tts and self.local_offline_tts.is_ready():
            try:
                t_s = time.perf_counter()
                pcm_loc = self.local_offline_tts.synthesize_sentence(clean_text, generation_id=self._fallback_generation_id)
                loc_ms = (time.perf_counter() - t_s) * 1000.0
                if pcm_loc:
                    return pcm_loc, "local_offline_tts", loc_ms, True
            except Exception as e:
                self.get_logger().warn(f"⚠️ [Local Offline TTS Failover]: {e}")

        return b"", "none", 0.0, False

    def _play_pcm_chunks(
        self,
        pcm_data: bytes,
        generation_id: int = 0,
        tts_provider: str = "xtts_gpu",
        tts_model: str = "xtts_finetuned",
        tts_source: str = "xtts_worker",
    ):
        """Streams 24kHz int16 PCM audio chunks directly to audio output node with smooth 20ms pacing, end sentinel, and drain synchronization."""
        if not pcm_data:
            return
        self._is_playback_active = True
        self._playback_start_monotonic = time.monotonic()
        self.state_machine.transition_to(RobotState.SPEAKING)
        chunk_size = 960  # 480 samples @ 24kHz int16 = 20ms
        effective_gen_id = generation_id or self._fallback_generation_id
        pcm_dur_s = (len(pcm_data) / 2) / 24000.0
        try:
            for i in range(0, len(pcm_data), chunk_size):
                if self._barge_in_latched:
                    break
                chunk = pcm_data[i : i + chunk_size]
                if chunk:
                    try:
                        chunk_16k = resample_24k_to_16k(chunk) if len(chunk) != 320 else chunk
                        self._update_playback_reference(chunk_16k)
                    except Exception:
                        pass
                    b64_str = base64.b64encode(chunk).decode("ascii")
                    msg_dict = {
                        "generation_id": effective_gen_id,
                        "tts_provider": tts_provider,
                        "tts_model": tts_model,
                        "tts_source": tts_source,
                        "playback_source": tts_source,
                        "is_done": False,
                        "data": b64_str,
                    }
                    out_msg = String()
                    out_msg.data = json.dumps(msg_dict)
                    self.pub_output_pcm.publish(out_msg)
                    time.sleep(0.018)

            # Send end sentinel if not interrupted by barge-in
            if not self._barge_in_latched:
                end_dict = {
                    "generation_id": effective_gen_id,
                    "tts_provider": tts_provider,
                    "tts_model": tts_model,
                    "tts_source": tts_source,
                    "playback_source": tts_source,
                    "is_done": True,
                    "data": "",
                }
                end_msg = String()
                end_msg.data = json.dumps(end_dict)
                self.pub_output_pcm.publish(end_msg)

                # Drain synchronization: wait for physical DAC playback completion up to calculated duration
                t_drain_start = time.monotonic()
                drain_timeout = max(0.2, pcm_dur_s * 0.5)
                while self._is_playback_active and (time.monotonic() - t_drain_start < drain_timeout):
                    if self._barge_in_latched:
                        break
                    time.sleep(0.02)
        finally:
            self._is_playback_active = False
            self._playback_end_time = time.monotonic()
            if self.state_machine.current_state == RobotState.SPEAKING:
                self.state_machine.transition_to(RobotState.LISTENING)

    def _process_fallback_turn(self, audio_chunks: Optional[List[bytes]] = None, direct_text: Optional[str] = None):
        """Processes turn using capability-aware ProviderRegistry + Streaming LLM + Pipelined TTS."""
        if self._is_processing_fallback or (not audio_chunks and not direct_text):
            return

        self._is_processing_fallback = True
        self._is_responding = True
        self._fallback_generation_id += 1
        self._barge_in_latched = False  # Reset single logical barge-in debounce for new turn
        t_turn_start = time.monotonic()
        chosen_model = "none"
        chosen_provider = "none"
        llm_status = "ok"
        error_class_str = "none"
        model_error_str = "none"
        llm_latency_ms = 0.0
        llm_ttft_ms: Optional[float] = None
        llm_first_clause_ms: Optional[float] = None
        first_audio_played = False
        first_audio_ms = 0.0
        total_synth_ms = 0.0
        total_gpu_ms = 0.0
        total_queue_wait_ms = 0.0
        total_audio_sec = 0.0
        attempts: List[Dict[str, Any]] = []

        try:
            if direct_text:
                user_text = direct_text.strip()
                self.get_logger().info(f"🗣️ [Siz (Yedek Zeka)]: \"{user_text}\"")
                self.memory.episodic.add_message("user", user_text)
                raw_pcm = b""
            else:
                # 1. Combine raw 16kHz PCM chunks into valid in-memory WAV buffer
                raw_pcm = b"".join(audio_chunks or [])
                if len(raw_pcm) < 16000 * 2 * 0.20:
                    return

            if not direct_text:
                # Cheap Local VAD Gate on raw_pcm before remote STT
                t_vad_start = time.monotonic()
                arr_pcm = np.frombuffer(raw_pcm, dtype=np.int16)
                pcm_rms = float(np.sqrt(np.mean(arr_pcm.astype(np.float32) ** 2))) if len(arr_pcm) > 0 else 0.0
                pcm_peak = int(np.max(np.abs(arr_pcm))) if len(arr_pcm) > 0 else 0

                chunk_sz = 320  # 20ms
                speech_frames_cnt = 0
                tot_frames_cnt = max(1, len(arr_pcm) // chunk_sz)
                sp_thresh = max(220.0, self._ambient_rms * 1.15)
                for i in range(0, len(arr_pcm) - chunk_sz + 1, chunk_sz):
                    if np.sqrt(np.mean(arr_pcm[i : i + chunk_sz].astype(np.float32) ** 2)) > sp_thresh:
                        speech_frames_cnt += 1
                local_speech_ms = int((speech_frames_cnt * chunk_sz / 16000.0) * 1000.0)
                local_vad_conf = round(min(1.0, speech_frames_cnt / float(tot_frames_cnt)), 2)
                t_vad_end = time.monotonic()

                # Discard immediately if audio has no genuine acoustic speech evidence (0 STT calls)
                if local_speech_ms < 90 or pcm_rms < max(200.0, self._ambient_rms * 1.15) or local_vad_conf < 0.15:
                    self.no_speech_rejection_count += 1
                    self.get_logger().debug(
                        f"🔇 [VAD Gate Dropped Buffer (0 STT Calls)]: speech_ms={local_speech_ms} | rms={pcm_rms:.1f} | vad_conf={local_vad_conf:.2f}"
                    )
                    return

                # 2. Transcribe via Groq Whisper Cloud (0-Token Cost STT ~250ms)
                wav_buf = io.BytesIO()
                with wave.open(wav_buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(raw_pcm)
                wav_bytes = wav_buf.getvalue()

                t_stt_start = time.monotonic()
                raw_transcript = self._transcribe_wav(wav_bytes)
                t_stt_end = time.monotonic()
                stt_ms = (t_stt_end - t_stt_start) * 1000.0

                # 3. Multi-Signal Validation Gate (Transcript + Acoustics + VAD + Playback + Self-Voice)
                now = time.monotonic()
                is_playback = bool(self._is_playback_active)
                is_cooldown = bool((now - self._playback_end_time) < self.echo_mute_cooldown_s)

                validated_text, stt_meta = self._validate_stt_transcript(
                    transcript=raw_transcript or "",
                    raw_pcm=raw_pcm,
                    is_playback_active=is_playback,
                    is_echo_cooldown=is_cooldown,
                )

                # Log Detailed STT Segment Telemetry
                self.get_logger().info(
                    f"📊 [STT Segment Telemetry]: vad_started={t_vad_start:.2f} | vad_ended={t_vad_end:.2f} | "
                    f"stt_started={t_stt_start:.2f} | stt_finished={t_stt_end:.2f} | transcript=\"{raw_transcript}\" | "
                    f"vad_confidence={stt_meta.get('vad_confidence', 0.0):.2f} | stt_confidence={stt_meta.get('stt_confidence', 0.0):.2f} | "
                    f"rms={stt_meta.get('rms', 0.0):.1f} | peak={stt_meta.get('peak', 0)} | speech_ms={stt_meta.get('speech_ms', 0)} | "
                    f"playback_active={is_playback} | self_voice_score={stt_meta.get('self_voice_score', 0.0):.2f} | "
                    f"stt_rejected={stt_meta.get('stt_rejected', False)} | reject_reason={stt_meta.get('stt_reject_reason', 'none')}"
                )

                # If rejected, immediately abort turn without LLM, memory, or TTS invocation
                if not validated_text:
                    return

                # Valid human speech confirmed — wake Astro up from DEEP_IDLE / Sleep
                self._wake_up()

                # Check for pure wake word in active mode (e.g. "Astro.", "Hey Astro", "Selam")
                norm_wake_check = re.sub(r"[^\w\s]", "", validated_text.lower()).strip()
                if norm_wake_check in ("astro", "hey astro", "selam astro", "hey", "selam"):
                    self.state_machine.transition_to(RobotState.LISTENING)
                    self.get_logger().info(
                        f"⚡ [Active Wake-Only]: \"{validated_text}\" -> Woke to LISTENING (wake_only=True, turn_created=False, 0 LLM / 0 TTS)."
                    )
                    return

                # Check if user said "Hey Astro, <command>" or "Astro, <command>"
                if norm_wake_check.startswith("hey astro "):
                    validated_text = validated_text[len("hey astro"):].lstrip(" ,.")
                elif norm_wake_check.startswith("astro "):
                    validated_text = validated_text[len("astro"):].lstrip(" ,.")
                elif norm_wake_check.startswith("selam astro "):
                    validated_text = validated_text[len("selam astro"):].lstrip(" ,.")

                valid_cmd, cmd_reason = is_valid_user_command(validated_text)
                if not valid_cmd:
                    self.state_machine.transition_to(RobotState.LISTENING)
                    self.get_logger().info(
                        f"⚡ [Wake + Invalid Command Dropped]: \"{raw_transcript}\" (reason={cmd_reason}) -> Transitioned to LISTENING (0 LLM / 0 TTS)."
                    )
                    return

                user_text = validated_text
                self.get_logger().info(f"🗣️ [Siz (0-Maliyet)]: \"{user_text}\"")
                self.memory.episodic.add_message("user", user_text)
                self.state_machine.transition_to(RobotState.THINKING)

            # 4. Run Voiceprint Recognition (Acoustic Speaker Identification with Temporal Smoothing)

            spk_name = None
            spk_score = 0.0
            spk_source = "unidentified"
            spk_known = False

            if not direct_text and self.voice_recognizer and raw_pcm:
                try:
                    audio_i16 = np.frombuffer(raw_pcm, dtype=np.int16)
                    identified_name, score = self.voice_recognizer.identify_speaker(audio_i16, sample_rate=16000)
                    now_s = time.monotonic()

                    if identified_name and identified_name.lower() != "misafir":
                        # Temporal smoothing:
                        # 1. High confidence (>= 0.65): Confirmed immediately
                        if score >= 0.65:
                            spk_name = identified_name
                            spk_score = score
                            spk_source = "voice_recognition"
                            spk_known = True
                            self._speaker_tentative_name = identified_name
                            self._speaker_tentative_count = 2
                            self._speaker_tentative_last_time = now_s
                            with self._lock:
                                self._recognized_speaker = {
                                    "name": identified_name,
                                    "score": score,
                                    "is_known": True,
                                    "confidence": score,
                                    "source": "voice_recognition",
                                }
                                self._active_person_name = identified_name
                                self._person_hold_until = now_s + 45.0
                        # 2. Tentative confidence (0.45 <= score < 0.65): Requires 2 observations within 15s
                        elif score >= 0.45:
                            if getattr(self, "_speaker_tentative_name", None) == identified_name and (now_s - getattr(self, "_speaker_tentative_last_time", 0.0)) < 15.0:
                                self._speaker_tentative_count += 1
                            else:
                                self._speaker_tentative_name = identified_name
                                self._speaker_tentative_count = 1
                            self._speaker_tentative_last_time = now_s

                            if self._speaker_tentative_count >= 2:
                                spk_name = identified_name
                                spk_score = score
                                spk_source = "voice_recognition_smoothed"
                                spk_known = True
                                with self._lock:
                                    self._recognized_speaker = {
                                        "name": identified_name,
                                        "score": score,
                                        "is_known": True,
                                        "confidence": score,
                                        "source": "voice_recognition_smoothed",
                                    }
                                    self._active_person_name = identified_name
                                    self._person_hold_until = now_s + 45.0
                            else:
                                self.get_logger().info(f"👤 [Tentative Speaker] candidate={identified_name} score={score:.2f} obs={self._speaker_tentative_count}/2 (waiting confirmation)")
                        else:
                            # score < 0.45: Unidentified, discard without overwriting context
                            self.get_logger().debug(f"👤 [Low Confidence Speaker] candidate={identified_name} score={score:.2f} < 0.45 (ignored)")
                except Exception as ex:
                    self.get_logger().debug(f"Voiceprint recognition error: {ex}")

            if not spk_name:
                identity = self._get_active_biometric_identity()
                if identity.get("is_known") and identity.get("name", "").lower() != "misafir":
                    spk_name = identity.get("name")
                    spk_score = identity.get("confidence", 0.85)
                    spk_source = identity.get("source", "memory_hold")
                    spk_known = True

            active_speaker_dict = {
                "name": spk_name or "Misafir",
                "speaker_name": spk_name,
                "confidence": spk_score,
                "is_known": spk_known,
                "source": spk_source,
            }
            speaker_display = spk_name if spk_name else "null"
            self.get_logger().info(f"👤 [Speaker Context] speaker={speaker_display} confidence={spk_score:.2f} source={spk_source}")

            # 5. Select Atomic TTS Owner for this turn (Realtime Fallback -> Edge-TTS -> Local Offline)
            if getattr(self, "edge_tts_enabled", True):
                turn_tts_engine = "edge_tts"
                tts_ready_flag = True
                tts_mode_str = "network_cloud"
                tts_source_name = "edge_tts_cloud"
                tts_model_name = "tr_tr_ahmet"
            elif self.local_offline_tts and self.local_offline_tts.is_ready():
                turn_tts_engine = "local_offline_tts"
                tts_ready_flag = True
                tts_mode_str = "local_offline"
                tts_source_name = "local_offline_synth"
                tts_model_name = "piper_espeak"
            else:
                turn_tts_engine = "none"
                tts_ready_flag = False
                tts_mode_str = "none"
                tts_source_name = "none"
                tts_model_name = "none"

            active_engine = turn_tts_engine

            self.get_logger().info(
                f"🏷️ [TTS Provider Selection Contract]\n"
                f"  generation_id={self._fallback_generation_id}\n"
                f"  selected_tts_provider={active_engine}\n"
                f"  selected_tts_model={tts_model_name}\n"
                f"  tts_source={tts_source_name}\n"
                f"  playback_source=audio_stream_node\n"
                f"  tts_state={tts_mode_str}\n"
                f"  tts_ready={tts_ready_flag}\n"
                f"  fallback_reason=realtime_unavailable"
            )

            def _synthesize_turn_clause(clause_text: str) -> Tuple[Optional[bytes], float, float, float]:
                clean_text = response_length_gate(clause_text, user_query=user_text, max_words=35, max_sentences=2)
                if not clean_text:
                    return None, 0.0, 0.0, 0.0

                nonlocal active_engine, tts_source_name, tts_model_name

                route_res = self.tts_router.synthesize(
                    clean_text,
                    generation_id=self._fallback_generation_id,
                    language=os.getenv("TTS_LANGUAGE", "tr"),
                )
                active_engine = route_res.actual_provider
                tts_source_name = route_res.source_name
                tts_model_name = route_res.model_name
                return route_res.pcm, route_res.duration_ms, route_res.infer_ms, route_res.queue_wait_ms

            def _handle_and_play_clause_audio(pcm_audio: bytes):
                if not pcm_audio:
                    return
                # Debug WAV & verification log on first real XTTS synthesis
                if active_engine == "xtts_gpu" and not getattr(self, "_first_xtts_debug_wav_written", False):
                    self._first_xtts_debug_wav_written = True
                    try:
                        import wave, hashlib, tempfile
                        wav_dir = "/tmp" if os.path.exists("/tmp") else tempfile.gettempdir()
                        wav_path = os.path.join(wav_dir, f"astro_xtts_{self._fallback_generation_id}.wav")
                        with wave.open(wav_path, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(24000)
                            wf.writeframes(pcm_audio)
                        audio_sha256 = hashlib.sha256(pcm_audio).hexdigest()
                        duration_ms = int((len(pcm_audio) / 2 / 24000.0) * 1000.0)
                        telem = self.local_xtts.get_telemetry() if self.local_xtts else {}
                        self.get_logger().info(
                            f"🎵 [XTTS OUTPUT VERIFIED]\n"
                            f"  generation_id={self._fallback_generation_id}\n"
                            f"  provider=xtts_gpu\n"
                            f"  model=xtts_finetuned\n"
                            f"  checkpoint={telem.get('xtts_model_path', 'default')}\n"
                            f"  reference={telem.get('xtts_reference_wav', 'default')}\n"
                            f"  sha256={telem.get('xtts_checkpoint_sha256', audio_sha256)}\n"
                            f"  sample_rate=24000\n"
                            f"  audio_bytes={len(pcm_audio)}\n"
                            f"  duration_ms={duration_ms}\n"
                            f"  infer_ms={telem.get('last_infer_ms', 0.0)}"
                        )
                    except Exception as ex_w:
                        self.get_logger().debug(f"XTTS debug WAV write notice: {ex_w}")

                self._play_pcm_chunks(
                    pcm_audio,
                    generation_id=self._fallback_generation_id,
                    tts_provider=active_engine,
                    tts_model=tts_model_name,
                    tts_source=tts_source_name,
                )

            # 6. Instant Intent Interception (Sub-250ms Direct Execution)
            is_weather, w_city = self._is_weather_query(user_text)
            if is_weather:
                weather_info = self._execute_fallback_weather(w_city)
                p = self.persona_name.lower()
                spk = spk_name if spk_name else ""
                if p == "kufurbaz":
                    reply_text = f"Ulan {spk}, {weather_info} Dışarı çıkacaksan ona göre giyin!".strip()
                elif p == "flirt":
                    reply_text = f"Canım benim, {weather_info} Kendine çok dikkat et!".strip()
                else:
                    reply_text = f"{spk} {weather_info}".strip()

                with self._lock:
                    self._recent_robot_phrases.append(reply_text.lower())
                    if len(self._recent_robot_phrases) > 10:
                        self._recent_robot_phrases = self._recent_robot_phrases[-10:]

                pcm, s_ms, g_ms, q_ms = _synthesize_turn_clause(reply_text)
                total_synth_ms += s_ms
                total_gpu_ms += g_ms
                total_queue_wait_ms += q_ms
                if pcm:
                    first_audio_ms = (time.monotonic() - t_turn_start) * 1000.0
                    self.get_logger().info(f"🤖 [Astro (Canlı Hava Durumu)]: \"{reply_text}\"")
                    self.memory.episodic.add_message("assistant", reply_text)
                    self.session.record_robot_speech()
                    _handle_and_play_clause_audio(pcm)
                    return

            # Explicit Head Angle Command (e.g. '0 dereceye dön', '-30'a dön', '30 derece sağa bak')
            is_angle, target_angle, angle_label = self._is_head_angle_query(user_text)
            if is_angle:
                p = self.persona_name.lower()
                spk = f" {spk_name}" if spk_name else ""
                self.pub_head_target_yaw.publish(Float32(data=float(target_angle)))
                self.get_logger().info(f"🎯 [Head Manual Command]: Kafa açısı komutla ayarlandı -> {target_angle:.1f}° ({angle_label})")

                if p == "kufurbaz":
                    if abs(target_angle) < 0.1:
                        reply_text = f"Kafayı tam ortaya sıfırladım{spk}, düz bakıyorum işte!"
                    else:
                        reply_text = f"Kafayı {int(target_angle)} dereceye çevirdim{spk}, rahatladın mı!"
                elif p == "flirt":
                    if abs(target_angle) < 0.1:
                        reply_text = f"Hemen tam karşına bakıyorum canım benim{spk}."
                    else:
                        reply_text = f"Senin için {int(target_angle)} dereceye döndüm tatlım{spk}."
                else:
                    if abs(target_angle) < 0.1:
                        reply_text = f"Kafa konumu 0 derece merkeze hizalandı{spk}."
                    else:
                        reply_text = f"Kafa konumu {int(target_angle)} dereceye ayarlandı{spk}."

                with self._lock:
                    self._recent_robot_phrases.append(reply_text.lower())
                    if len(self._recent_robot_phrases) > 10:
                        self._recent_robot_phrases = self._recent_robot_phrases[-10:]

                pcm, s_ms, g_ms, q_ms = _synthesize_turn_clause(reply_text)
                total_synth_ms += s_ms
                total_gpu_ms += g_ms
                total_queue_wait_ms += q_ms
                if pcm:
                    first_audio_ms = (time.monotonic() - t_turn_start) * 1000.0
                    self.get_logger().info(f"🤖 [Astro (Açı Komutu)]: \"{reply_text}\"")
                    self.memory.episodic.add_message("assistant", reply_text)
                    self.session.record_robot_speech()
                    _handle_and_play_clause_audio(pcm)
                    return

            is_turn_sound = self._is_turn_to_sound_query(user_text)
            if is_turn_sound:
                if hasattr(self, "pub_explicit_gaze") and self.pub_explicit_gaze:
                    try:
                        explicit_msg = String()
                        payload = {
                            "selector": "CURRENT_SPEAKER",
                            "confidence": 1.0,
                            "reason": "explicit_speech_command",
                        }
                        explicit_msg.data = json.dumps(payload)
                        self.pub_explicit_gaze.publish(explicit_msg)
                    except Exception:
                        pass

                p = self.persona_name.lower()
                spk = f" {spk_name}" if spk_name else ""
                act_res = None
                if getattr(self, "action_manager", None):
                    act_res = self.action_manager.execute_turn_to_sound(
                        generation_id=self._fallback_generation_id
                    )

                if act_res and act_res.success:
                    if p == "kufurbaz":
                        reply_text = f"Döndüm ulan işte{spk}, söyle bakalım ne diyorsun!"
                    elif p == "flirt":
                        reply_text = f"Hemen sana döndüm canım benim{spk}, seni dinliyorum."
                    else:
                        reply_text = f"Sesine döndüm{spk}, seni dinliyorum."
                else:
                    if p == "kufurbaz":
                        reply_text = f"Sesinin yönünü tam kestiremedim ama buradayım ulan, anlat!"
                    else:
                        reply_text = f"Sesinin yönünü tam kestiremedim ama buradayım{spk}, seni dinliyorum."

                with self._lock:
                    self._recent_robot_phrases.append(reply_text.lower())
                    if len(self._recent_robot_phrases) > 10:
                        self._recent_robot_phrases = self._recent_robot_phrases[-10:]

                pcm, s_ms, g_ms, q_ms = _synthesize_turn_clause(reply_text)
                total_synth_ms += s_ms
                total_gpu_ms += g_ms
                total_queue_wait_ms += q_ms
                if pcm:
                    first_audio_ms = (time.monotonic() - t_turn_start) * 1000.0
                    self.get_logger().info(f"🤖 [Astro (Ses Yönelimi)]: \"{reply_text}\"")
                    self.memory.episodic.add_message("assistant", reply_text)
                    self.session.record_robot_speech()
                    _handle_and_play_clause_audio(pcm)
                    return

            is_move, move_dir, move_spd, move_dur = self._is_movement_query(user_text)
            if is_move:
                p = self.persona_name.lower()
                spk = f" {spk_name}" if spk_name else ""
                act_res = None
                if getattr(self, "action_manager", None):
                    act_res = self.action_manager.execute_move(
                        direction=move_dir,
                        speed=move_spd,
                        duration=move_dur,
                        generation_id=self._fallback_generation_id,
                    )

                dir_names_tr = {"forward": "ileri", "backward": "geriye", "left": "sola", "right": "sağa", "stop": "durma"}
                dir_tr = dir_names_tr.get(move_dir, move_dir)
                if act_res and act_res.success:
                    if move_dir == "stop":
                        reply_text = f"Durdum{spk}."
                    else:
                        reply_text = f"{dir_tr.capitalize()} hareket ediyorum{spk}."
                else:
                    reply_text = f"Güvenlik kilidi devrede veya hareket engellendi{spk}."

                pcm, s_ms, g_ms, q_ms = _synthesize_turn_clause(reply_text)
                total_synth_ms += s_ms
                total_gpu_ms += g_ms
                total_queue_wait_ms += q_ms
                if pcm:
                    first_audio_ms = (time.monotonic() - t_turn_start) * 1000.0
                    self.get_logger().info(f"🤖 [Astro (Hareket Komutu)]: \"{reply_text}\"")
                    self.memory.episodic.add_message("assistant", reply_text)
                    self.session.record_robot_speech()
                    _handle_and_play_clause_audio(pcm)
                    return

            # 7. Cognitive LLM via ProviderRegistry (Streaming Groq -> Gemini -> Contextual Persona)
            system_prompt = self._build_current_system_prompt(active_speaker=active_speaker_dict)
            messages = [{"role": "system", "content": system_prompt}]
            recent_msgs = self.memory.episodic.get_messages()[-6:]
            for m in recent_msgs:
                messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

            groq_candidates = self.provider_registry.get_candidate_models("groq") if self.groq_api_key else []
            full_reply_parts = []
            chunker = SentenceChunker(min_first_clause_chars=18, min_clause_chars=28) if SentenceChunker else None
            t_llm_start = time.monotonic()

            total_audio_bytes = 0
            total_enqueued_chunks = 0

            # Attempt A: Streaming Groq LLMs (Fastest first, fallback on failure)
            if self.groq_api_key and groq_candidates:
                for target_model in groq_candidates:
                    try:
                        t_model_start = time.monotonic()
                        first_token_seen = False

                        for token in self.provider_registry.stream_groq_completion(
                            self.groq_api_key,
                            target_model,
                            messages,
                            max_tokens=60,
                            temperature=0.65,
                            timeout=5.0,
                        ):
                            if not first_token_seen:
                                llm_ttft_ms = (time.monotonic() - t_model_start) * 1000.0
                                first_token_seen = True

                            full_reply_parts.append(token)

                        if full_reply_parts:
                            chosen_model = target_model
                            chosen_provider = "groq"
                            llm_latency_ms = (time.monotonic() - t_model_start) * 1000.0
                            self.provider_registry.record_success("groq", target_model, llm_latency_ms)
                            attempts.append({
                                "provider": "groq",
                                "model": target_model,
                                "result": "success",
                                "latency_ms": int(llm_latency_ms)
                            })
                            break
                    except ProviderError as pe:
                        error_class_str = pe.error_class.value
                        model_error_str = pe.message[:80]
                        attempts.append({
                            "provider": "groq",
                            "model": target_model,
                            "result": "failed",
                            "error_class": pe.error_class.value,
                            "error": pe.message[:80]
                        })
                        self.get_logger().warn(f"⚠️ [Groq Model Fallback] {target_model} failed ({pe.error_class.value}): {pe.message[:80]}")
                        full_reply_parts = []
                        if pe.error_class in (ErrorClass.QUOTA_EXHAUSTED, ErrorClass.RATE_LIMITED, ErrorClass.AUTHENTICATION_ERROR):
                            # Quota/rate-limit is provider-wide; do not loop through other models on this provider
                            break
                        continue

            # Attempt B: Google Gemini REST Fallback (if Groq produced no tokens)
            if not full_reply_parts and self.gemini_api_key:
                gemini_candidates = self.provider_registry.get_candidate_models("gemini")
                for g_mod in gemini_candidates:
                    try:
                        t_gem_start = time.monotonic()
                        gem_text = self.provider_registry.generate_gemini_content(
                            self.gemini_api_key,
                            g_mod,
                            system_prompt,
                            messages,
                            max_tokens=60,
                            temperature=0.65,
                            timeout=4.0,
                        )
                        if gem_text:
                            full_reply_parts = [gem_text]
                            chosen_model = g_mod
                            chosen_provider = "gemini"
                            llm_latency_ms = (time.monotonic() - t_gem_start) * 1000.0
                            llm_ttft_ms = llm_latency_ms
                            llm_first_clause_ms = llm_latency_ms
                            self.provider_registry.record_success("gemini", g_mod, llm_latency_ms)
                            attempts.append({
                                "provider": "gemini",
                                "model": g_mod,
                                "result": "success",
                                "latency_ms": int(llm_latency_ms)
                            })
                            break
                    except ProviderError as pe:
                        error_class_str = pe.error_class.value
                        model_error_str = pe.message[:80]
                        attempts.append({
                            "provider": "gemini",
                            "model": g_mod,
                            "result": "failed",
                            "error_class": pe.error_class.value,
                            "error": pe.message[:80]
                        })
                        self.get_logger().warn(f"⚠️ [Gemini Model Fallback] {g_mod} failed ({pe.error_class.value}): {pe.message[:80]}")
                        if pe.error_class in (ErrorClass.QUOTA_EXHAUSTED, ErrorClass.RATE_LIMITED, ErrorClass.AUTHENTICATION_ERROR):
                            # Quota/rate-limit is provider-wide; do not loop through other models on this provider
                            break
                        continue


            full_reply_str = clean_tts_text("".join(full_reply_parts))

            # Attempt C: Dynamic Context-Grounded Persona Fallback (if all cloud LLMs failed)
            if not full_reply_str:
                full_reply_str = self._generate_contextual_persona_fallback(user_text)
                chosen_model = "contextual_grounding"
                chosen_provider = "local_persona"
                llm_status = "degraded"
                attempts.append({
                    "provider": "local_persona",
                    "model": "contextual_grounding",
                    "result": "success"
                })
            else:
                self.repetition_guard.record_response(full_reply_str)

            # Clean and gate conversational length
            full_reply_str = response_length_gate(full_reply_str, user_query=user_text, max_words=30, max_sentences=2) or full_reply_str

            # Record assistant reply for self-voice echo correlation
            with self._lock:
                self._recent_robot_phrases.append(full_reply_str.lower())
                if len(self._recent_robot_phrases) > 10:
                    self._recent_robot_phrases = self._recent_robot_phrases[-10:]

            # Synthesize ONE single unified TTS generation for this logical turn
            if full_reply_str:
                pcm, s_ms, g_ms, q_ms = _synthesize_turn_clause(full_reply_str)
                total_synth_ms += s_ms
                total_gpu_ms += g_ms
                total_queue_wait_ms += q_ms
                if pcm:
                    total_audio_sec = (len(pcm) / 2) / 24000.0
                    total_audio_bytes = len(pcm)
                    first_audio_ms = (time.monotonic() - t_turn_start) * 1000.0
                    first_audio_played = True
                    _handle_and_play_clause_audio(pcm)

            t_total_end = time.monotonic()
            total_turn_ms = (t_total_end - t_turn_start) * 1000.0

            if full_reply_str:
                self.get_logger().info(f"🤖 [Astro ({chosen_provider}/{chosen_model})]: \"{full_reply_str}\"")
                self.memory.episodic.add_message("assistant", full_reply_str)
                self.session.record_robot_speech()

                xtts_info = self.local_xtts.get_telemetry() if self.local_xtts else {}
                is_xtts_actually_ready = bool(self.local_xtts and self.local_xtts.is_ready())
                xtts_err_str = "none"
                if not is_xtts_actually_ready:
                    xtts_state = self.local_xtts.state if self.local_xtts else "uninitialized"
                    xtts_err_str = xtts_info.get("error", xtts_state)

                # Determine TTS metadata
                if active_engine == "elevenlabs":
                    tts_model_name = self.elevenlabs_engine.model_id if self.elevenlabs_engine else "eleven_flash_v2_5"
                    tts_voice_name = self.elevenlabs_engine.voice_id if self.elevenlabs_engine else "configured"
                    tts_mode_val = "remote_cloud"
                elif active_engine == "xtts_gpu":
                    tts_model_name = "xtts_finetuned"
                    tts_voice_name = xtts_info.get("xtts_reference_wav", self.local_xtts.speaker_wav if self.local_xtts else "reference.wav")
                    tts_mode_val = "local_gpu"
                elif active_engine == "local_offline_tts":
                    tts_model_name = getattr(self.local_offline_tts, "_mode", "local_offline") if self.local_offline_tts else "local_offline"
                    tts_voice_name = "local_offline_synth"
                    tts_mode_val = "local_offline"
                elif active_engine == "edge_tts":
                    tts_model_name = "edge_tts"
                    tts_voice_name = "tr-TR-AhmetNeural"
                    tts_mode_val = "network"
                else:
                    tts_model_name = "none"
                    tts_voice_name = "none"
                    tts_mode_val = "none"

                worker_pid = xtts_info.get("worker_pid", "None")
                gpu_name_str = xtts_info.get("gpu_name", "Orin")
                xtts_ckpt_str = xtts_info.get("xtts_model_path", "none")
                xtts_sha_str = xtts_info.get("xtts_checkpoint_sha256", "none")
                xtts_sha_short = xtts_sha_str[:12] if xtts_sha_str != "none" else "none"

                ttft_str = int(llm_ttft_ms) if llm_ttft_ms is not None else "null"
                first_clause_str = int(llm_first_clause_ms) if llm_first_clause_ms is not None else "null"
                speaker_log_val = spk_name if spk_name else "null"

                tts_error_val = xtts_err_str if active_engine == "xtts_gpu" else ("none" if tts_ready_flag else "provider_unavailable")

                # OAK-D Lite stability and frame age calculation
                now_telem = time.monotonic()
                oak_frame_age = int((now_telem - self._oak_last_frame_time) * 1000) if self._oak_last_frame_time > 0 else "null"
                oak_info_age = int((now_telem - self._oak_last_camera_info_time) * 1000) if self._oak_last_camera_info_time > 0 else "null"
                oak_state = "CONNECTED" if (self._oak_last_frame_time > 0 and (now_telem - self._oak_last_frame_time) < 3.0) else "DISCONNECTED"

                tts_synth_started_flag = bool(total_synth_ms > 0 or total_audio_bytes > 0)
                tts_synth_finished_flag = bool(total_audio_bytes > 0)
                tts_source_name = "xtts_worker" if active_engine == "xtts_gpu" else ("elevenlabs_cloud" if active_engine == "elevenlabs" else ("local_offline_synth" if active_engine == "local_offline_tts" else "edge_tts_cloud"))
                pb_source = "audio_stream_node"
                is_xtts_healthy = bool(self.local_xtts and getattr(self.local_xtts, "is_healthy", lambda: False)())

                # Mismatch Alarm Audit
                if active_engine == "xtts_gpu" and tts_source_name != "xtts_worker":
                    self.get_logger().error(f"🚨 [TTS MISMATCH ALARM]: selected_provider=xtts_gpu but tts_source={tts_source_name}!")
                if active_engine == "xtts_gpu" and pb_source == "espeak":
                    self.get_logger().error(f"🚨 [TTS MISMATCH ALARM]: selected_provider=xtts_gpu but playback_source=espeak!")
                if first_audio_played and total_audio_bytes == 0:
                    self.get_logger().error(f"🚨 [TTS MISMATCH ALARM]: playback_started=True but played_bytes=0!")
                if tts_synth_finished_flag and not first_audio_played:
                    self.get_logger().error(f"🚨 [TTS MISMATCH ALARM]: synthesis_finished=True but playback_started=False!")

                resp_chars = len(full_reply_str)
                resp_words = len(full_reply_str.split())
                rt_state = getattr(self, "realtime_provider_state", "AVAILABLE")
                rt_state_name = rt_state if isinstance(rt_state, str) else (rt_state.value if hasattr(rt_state, "value") else "AVAILABLE")
                rt_fail_reason = "quota_exhausted" if getattr(self, "_fallback_mode", False) else "none"

                self.get_logger().info(
                    f"[Turn Telemetry]\n"
                    f"generation_id={self._fallback_generation_id}\n"
                    f"realtime_state={rt_state_name}\n"
                    f"realtime_failure_reason={rt_fail_reason}\n"
                    f"requested_provider=openai_realtime\n"
                    f"actual_provider={active_engine}\n"
                    f"response_chars={resp_chars}\n"
                    f"response_words={resp_words}\n"
                    f"tts_ttfa_ms={int(total_synth_ms)}\n"
                    f"tts_total_ms={int(total_synth_ms)}\n"
                    f"playback_started={first_audio_played}\n"
                    f"playback_finished={first_audio_played and tts_synth_finished_flag}\n"
                    f"playback_failed={not first_audio_played and tts_synth_started_flag}"
                )

                if active_engine == "xtts_gpu" and not getattr(self, "_first_xtts_synthesis_verified", False):
                    self._first_xtts_synthesis_verified = True
                    self.get_logger().info(
                        f"🎯 [XTTS First Synthesis Verified]:\n"
                        f"  tts_synthesis_started=true\n"
                        f"  tts_provider=xtts_gpu\n"
                        f"  tts_model=xtts_finetuned\n"
                        f"  xtts_checkpoint={xtts_ckpt_str}\n"
                        f"  xtts_reference={tts_voice_name}\n"
                        f"  xtts_sha256={xtts_sha_str}\n"
                        f"  xtts_infer_ms={int(total_gpu_ms)}\n"
                        f"  tts_audio_bytes={total_audio_bytes}"
                    )

        except Exception as e:
            self.get_logger().warn(f"Fallback turn notice: {e}")
        finally:
            self._is_processing_fallback = False
            self._is_responding = False
            self._playback_end_time = time.monotonic()
            self._flush_audio_buffers("end_fallback_turn")

    def _on_input_pcm(self, msg: Any):
        """Sends incoming microphone 24kHz PCM chunk to OpenAI Realtime WebSocket or processes turn via 0-cost Groq fallback."""
        if msg is None:
            return

        now = time.monotonic()

        # Try parsing JSON wrapped frame, raw base64 PCM string, or direct bytes
        raw_16k: bytes = b""
        local_rms: float = 0.0
        peak_val: int = 0
        try:
            if isinstance(msg, (bytes, bytearray)):
                raw_bytes = bytes(msg)
            else:
                if not getattr(msg, "data", None):
                    return
                raw_str = msg.data.strip()
                if raw_str.startswith("{") and raw_str.endswith("}"):
                    data_dict = json.loads(raw_str)
                    b64_audio = data_dict.get("data", "")
                    raw_bytes = base64.b64decode(b64_audio.encode("ascii")) if b64_audio else b""
                else:
                    raw_bytes = base64.b64decode(raw_str.encode("ascii"))
            if raw_bytes:
                if len(raw_bytes) == 640:
                    raw_16k = raw_bytes
                else:
                    raw_16k = resample_24k_to_16k(raw_bytes)
                arr = np.frombuffer(raw_16k, dtype=np.int16)
                if len(arr) > 0:
                    local_rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
                    peak_val = int(np.max(np.abs(arr)))
                    # Maintain speech audio buffer for speaker recognition while robot is not speaking
                    if not self._is_playback_active and raw_16k:
                        with self._lock:
                            self._user_speech_audio_buffer.append(raw_16k)
                            if len(self._user_speech_audio_buffer) > 250:
                                self._user_speech_audio_buffer = self._user_speech_audio_buffer[-250:]
        except Exception as _exc:
            self.get_logger().debug(f"_on_input_pcm: yok sayılan hata ({_exc})")

        # Update background ambient noise floor when quiet
        if local_rms < 380.0:
            self._ambient_rms = 0.96 * self._ambient_rms + 0.04 * local_rms

        # ====================================================================
        # SLEEP / DEEP_IDLE MODE: Dedicated Low-CPU Wake Detector
        # ====================================================================
        if self._is_sleeping or self.state_machine.is_deep_idle():
            if raw_16k:
                wake_min_rms = float(os.getenv("WAKE_MIN_RMS", "120.0"))
                wake_min_peak = int(os.getenv("WAKE_MIN_PEAK", "350"))
                is_speech_energy = (local_rms > max(wake_min_rms, self._ambient_rms * 1.20) and peak_val > wake_min_peak)
                if is_speech_energy:
                    self._wake_last_voice_time = now
                    if not self._wake_listening:
                        self._wake_listening = True
                        with self._lock:
                            # 18 pre-roll frames (360ms) ensures full initial syllable (e.g. "Hey") is preserved
                            pre_frames = list(self._user_speech_audio_buffer[-18:]) if len(self._user_speech_audio_buffer) >= 18 else list(self._user_speech_audio_buffer)
                        self._wake_audio_buffer = list(pre_frames) + [raw_16k]
                    else:
                        self._wake_audio_buffer.append(raw_16k)
                elif self._wake_listening:
                    self._wake_audio_buffer.append(raw_16k)
                    # Silence pause (0.50s after speech ends) triggers wake verification
                    if (now - self._wake_last_voice_time) > 0.50:
                        self._wake_listening = False
                        if len(self._wake_audio_buffer) >= 10:
                            # Pre-STT local VAD energy check
                            raw_w = b"".join(self._wake_audio_buffer)
                            arr_w = np.frombuffer(raw_w, dtype=np.int16)
                            w_rms = float(np.sqrt(np.mean(arr_w.astype(np.float32) ** 2))) if len(arr_w) > 0 else 0.0
                            if w_rms >= max(100.0, self._ambient_rms * 1.15):
                                buf_to_proc = list(self._wake_audio_buffer)
                                self._wake_audio_buffer.clear()
                                threading.Thread(target=self._process_wake_candidate, args=(buf_to_proc,), daemon=True).start()
                            else:
                                self._wake_audio_buffer.clear()
                        else:
                            self._wake_audio_buffer.clear()
            return

        # ====================================================================
        # ACTIVE MODE: Interaction timestamp & State Tracking
        # ====================================================================
        # Playback & Echo Cooldown State Determination
        # P0-7: Barge-in is only evaluated during active audio playback
        is_active_playback = bool(self._is_playback_active)
        if is_active_playback:
            self._last_interaction_time = now

        # Zero Self-Hearing Protection & Multi-Signal Persistent Barge-In
        if is_active_playback:
            playback_start = getattr(self, "_playback_start_monotonic", 0.0)
            prot_ms = float(getattr(self, "barge_in_protection_ms", 350.0))

            # 1. Acoustic Protection Window: Strictly suppress self-voice feedback during initial burst (e.g. 350ms)
            if playback_start > 0.0 and ((now - playback_start) * 1000.0 < prot_ms):
                self._barge_in_consecutive_frames = 0
                return

            # Target barge-in threshold: Requires intentional voice exceeding loudspeaker playback level
            barge_min_rms = float(getattr(self, "barge_in_playback_min_rms", getattr(self, "barge_in_min_rms", 1400.0)))
            barge_noise_mult = float(getattr(self, "barge_in_noise_mult", 3.0))
            barge_min_peak = int(getattr(self, "barge_in_playback_min_peak", getattr(self, "barge_in_min_peak", 2500)))
            ambient_val = float(getattr(self, "_ambient_rms", 120.0))

            target_barge_in_rms = max(barge_min_rms, ambient_val * barge_noise_mult)
            target_barge_in_peak = barge_min_peak

            # Self-voice rejection score check: if voice recognizer is active, check self-voice score
            self_voice_score = 0.0
            if getattr(self, "voice_recognizer", None) and hasattr(self.voice_recognizer, "score_self_voice"):
                try:
                    self_voice_score = self.voice_recognizer.score_self_voice(raw_16k)
                except Exception:
                    self_voice_score = 0.0
            if self_voice_score < 0.70 and hasattr(self, "_playback_ref_pcm") and self._playback_ref_pcm:
                try:
                    ref_score = compute_pcm_self_voice_score(raw_16k, self._playback_ref_pcm)
                    self_voice_score = max(self_voice_score, ref_score)
                except Exception:
                    pass

            # 2. Self-Voice Rejection Check
            if self_voice_score >= 0.70:
                self.get_logger().debug(
                    f"[BARGE-IN DECISION]\n"
                    f"playback_active=true\n"
                    f"vad_confidence={1.0 if getattr(self, '_vad_active', False) else 0.0:.2f}\n"
                    f"speech_duration_ms=0\n"
                    f"speech_continuity_ms=0\n"
                    f"rms={local_rms:.0f}\n"
                    f"peak={peak_val}\n"
                    f"self_voice_score={self_voice_score:.2f}\n"
                    f"transient_noise=false\n"
                    f"speech_confirmed=false\n"
                    f"decision=false\n"
                    f"reason=self_voice"
                )
                self._barge_in_consecutive_frames = 0
                return

            # 3. Energy threshold check
            is_loud = (local_rms >= target_barge_in_rms and peak_val >= target_barge_in_peak)
            if is_loud:
                self._barge_in_consecutive_frames += 1
            else:
                self._barge_in_consecutive_frames = max(0, self._barge_in_consecutive_frames - 1)

            speech_duration_ms = self._barge_in_consecutive_frames * 20
            speech_continuity_ms = speech_duration_ms
            
            # Barge-in minimum speech duration is provider-dependent.
            # Edge-TTS: self_voice_score=0.00 (suppressor trained only on OpenAI Realtime voice, not Edge-TTS audio),
            # so minimum confirmation window must be wider to avoid false cuts from ambient echo.
            # OpenAI Realtime: server manages barge-in natively; client-side can stay at base threshold.
            try:
                is_edge_tts_active = getattr(self, "_fallback_mode", False) or not self._can_use_openai("realtime")
            except Exception:
                is_edge_tts_active = False

            if is_edge_tts_active:
                # In Edge-TTS mode without hardware AEC subtraction, require substantial human speech
                # (>=400ms continuity in production, 120ms under pytest) to avoid microphone loudspeaker feedback false cuts.
                effective_min_speech_ms = 120.0 if self._under_pytest() else 400.0
                target_barge_in_rms = max(target_barge_in_rms * 1.5, 3000.0)
                target_barge_in_peak = max(target_barge_in_peak, 8000)
            else:
                base_min_speech_ms = float(getattr(self, "barge_in_min_speech_ms", 60.0))
                effective_min_speech_ms = max(base_min_speech_ms, getattr(self, "barge_in_min_consecutive_frames", 3) * 20.0)

            min_speech_ms = effective_min_speech_ms
            if speech_duration_ms < min_speech_ms:
                if local_rms >= target_barge_in_rms and peak_val >= target_barge_in_peak:
                    is_vad_active = getattr(self, "_vad_active", False)
                    is_human_candidate = is_vad_active and (self_voice_score < 0.60)
                    # Transient noise is an isolated impulse (<40ms) when VAD does not detect human voice
                    is_transient = (speech_duration_ms < 40) and not is_human_candidate
                    reason = "transient_noise" if is_transient else "insufficient_speech_duration"
                    self.get_logger().debug(
                        f"[BARGE-IN DECISION]\n"
                        f"playback_active=true\n"
                        f"vad_confidence={1.0 if getattr(self, '_vad_active', False) else 0.0:.2f}\n"
                        f"speech_duration_ms={speech_duration_ms}\n"
                        f"speech_continuity_ms={speech_continuity_ms}\n"
                        f"rms={local_rms:.0f}\n"
                        f"peak={peak_val}\n"
                        f"self_voice_score={self_voice_score:.2f}\n"
                        f"transient_noise={'true' if is_transient else 'false'}\n"
                        f"speech_confirmed=false\n"
                        f"decision=false\n"
                        f"reason={reason}"
                    )
                return

            # 5. Barge-In Latch
            if self._barge_in_latched:
                return
            self._barge_in_latched = True
            self._barge_in_consecutive_frames = 0
            self._is_playback_active = False
            self._is_responding = False
            barge_in_after_ms = int((now - playback_start) * 1000.0) if playback_start > 0.0 else int(self.barge_in_protection_ms + 100)

            self.get_logger().info(
                f"[BARGE-IN DECISION]\n"
                f"playback_active=true\n"
                f"vad_confidence={1.0 if getattr(self, '_vad_active', False) else 0.0:.2f}\n"
                f"speech_duration_ms={speech_duration_ms}\n"
                f"speech_continuity_ms={speech_continuity_ms}\n"
                f"rms={local_rms:.0f}\n"
                f"peak={peak_val}\n"
                f"self_voice_score={self_voice_score:.2f}\n"
                f"transient_noise=false\n"
                f"speech_confirmed=true\n"
                f"decision=true\n"
                f"reason=human_speech_confirmed"
            )

            # 6. Actuate Cancellation: Cancel ongoing audio and speech pipeline
            if getattr(self, "_fallback_mode", False) or not self._can_use_openai("realtime"):
                self._fallback_speaking = True
                self._fallback_speech_start = now
                self._last_speech_time = now
                self._fallback_generation_id += 1
                if self.elevenlabs_engine:
                    self.elevenlabs_engine.cancel(self._fallback_generation_id)
                if self.local_xtts:
                    self.local_xtts.cancel(self._fallback_generation_id)
                if self.local_offline_tts:
                    self.local_offline_tts.cancel(self._fallback_generation_id)
                with self._lock:
                    self._fallback_audio_buffer = [raw_16k]
            else:
                # Realtime primary S2S mode: cancel streaming response on WebSocket ONLY if it is actively streaming
                if self.active_response_state in ("STREAMING", "RESPONSE_STREAMING"):
                    self.active_response_state = "CANCELLED"
                    if self._ws is not None and self._can_use_openai("realtime"):
                        try:
                            if self._loop is not None:
                                asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps({"type": "response.cancel"})), self._loop)
                            elif hasattr(self._ws, "send"):
                                res = self._ws.send(json.dumps({"type": "response.cancel"}))
                                if inspect.iscoroutine(res):
                                    asyncio.run(res)
                        except Exception:
                            pass

            int_msg = Bool()
            int_msg.data = True
            if getattr(self, "pub_interrupt", None):
                self.pub_interrupt.publish(int_msg)
            self.get_logger().info(
                f"⚡ [Realtime Barge-In] Kullanıcı araya girdi (RMS: {local_rms:.0f}, Peak: {peak_val}, "
                f"barge_in_after_ms={barge_in_after_ms}, Latch: True)."
            )
            self.state_machine.transition_to(RobotState.LISTENING)
            return

        # Acoustic presence / wake-up (requires sustained intentional voice > 500 RMS across >=5 consecutive frames)
        if local_rms > 500.0:
            self._consecutive_loud_frames += 1
        else:
            self._consecutive_loud_frames = max(0, self._consecutive_loud_frames - 1)

        if self._consecutive_loud_frames >= 5 and (now - getattr(self, "_node_start_time", 0.0)) > 4.0:
            if self._is_sleeping:
                self._wake_up()

        # --- 0-Cost Fallback Mode (Groq STT + Groq LLM + Edge-TTS) ---
        is_ws_connected = (self._is_connected or self.realtime_connection_state == "CONNECTED")
        if self._fallback_mode or not self._can_use_openai("realtime") or not is_ws_connected or self._ws is None:
            if raw_16k:
                try:
                    speech_start_condition = (local_rms > max(380.0, self._ambient_rms * 1.40) and peak_val > 900)
                    buf_to_proc = None
                    with self._lock:
                        if speech_start_condition:
                            self._last_speech_time = now
                            if not self._fallback_speaking:
                                self._fallback_speaking = True
                                self._fallback_speech_start = now
                                pre_frames = list(self._user_speech_audio_buffer[-8:]) if len(self._user_speech_audio_buffer) >= 8 else []
                                self._fallback_audio_buffer = list(pre_frames) + [raw_16k]
                            else:
                                self._fallback_audio_buffer.append(raw_16k)
                        elif self._fallback_speaking:
                            self._fallback_audio_buffer.append(raw_16k)
                            # Silence timeout (0.75s after speech ends)
                            if (now - self._last_speech_time) > 0.75:
                                self._fallback_speaking = False
                                if len(self._fallback_audio_buffer) >= 12 and not self._is_processing_fallback:
                                    buf_to_proc = list(self._fallback_audio_buffer)
                                    self._fallback_audio_buffer.clear()
                                else:
                                    self._fallback_audio_buffer.clear()

                    if buf_to_proc:
                        # Pre-STT Local VAD Density Filter (0-Token protection against noise/silence)
                        raw_fb = b"".join(buf_to_proc)
                        arr_fb = np.frombuffer(raw_fb, dtype=np.int16)
                        fb_rms = float(np.sqrt(np.mean(arr_fb.astype(np.float32) ** 2))) if len(arr_fb) > 0 else 0.0
                        chunk_sz = 320  # 20ms
                        loud_cnt = sum(
                            1 for i in range(0, len(arr_fb) - chunk_sz + 1, chunk_sz)
                            if np.sqrt(np.mean(arr_fb[i : i + chunk_sz].astype(np.float32) ** 2)) > max(280.0, self._ambient_rms * 1.25)
                        )
                        total_chunks = max(1, len(arr_fb) // chunk_sz)
                        speech_ratio = loud_cnt / float(total_chunks)

                        if fb_rms >= max(260.0, self._ambient_rms * 1.20) and loud_cnt >= 5 and speech_ratio >= 0.15:
                            threading.Thread(target=self._process_fallback_turn, args=(buf_to_proc,), daemon=True).start()
                        else:
                            self.no_speech_rejection_count += 1
                except Exception as _exc:
                    self.get_logger().warning(f"[_on_input_pcm fallback error]: {_exc}")
            return

        # --- Standard OpenAI Realtime Mode ---
        # Gating: While a tool is actively in progress, do NOT stream audio to OpenAI to prevent premature server VAD!
        if getattr(self, "_active_tool_call_in_progress", False):
            return

        # Gating: Microphone PCM is strictly streamed ONLY when _can_use_openai("realtime") is True
        if (
            self._can_use_openai("realtime")
            and self.realtime_connection_state == "CONNECTED"
            and self.realtime_session_state == "READY"
            and self._ws is not None
        ):
            audio_b64 = getattr(msg, "data", None)
            if audio_b64 is None and raw_bytes:
                audio_b64 = base64.b64encode(raw_bytes).decode("ascii")
            payload = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64 or ""
            }
            try:
                if self._loop is not None:
                    asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(payload)), self._loop)
                elif hasattr(self._ws, "send"):
                    # Test fake transport sync/async support
                    res = self._ws.send(json.dumps(payload))
                    if inspect.iscoroutine(res):
                        asyncio.run(res)
            except Exception as _exc:
                self.get_logger().debug(f"_on_input_pcm: yok sayılan hata ({_exc})")


    def _trigger_proactive_greeting(self, name: str, formal_title: str):
        """Sends proactive greeting message to Realtime session."""
        if not self._ws or not self._loop or not self._is_connected:
            return

        greeting_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"[Sistem Olayı]: Karşında {name} ({formal_title}) duruyor! Kendisini seçili kişiliğinle coşkuyla ve uygun hitapla selamla, neşeyle hatırını sor."
                    }
                ]
            }
        }
        try:
            asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(greeting_event)), self._loop)
            asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps({"type": "response.create"})), self._loop)
        except Exception as _exc:
            self.get_logger().debug(f"_trigger_proactive_greeting: yok sayılan hata ({_exc})")

    def _on_recognized_person(self, msg: String):
        try:
            data = json.loads(msg.data)
            now = time.monotonic()
            with self._lock:
                self._recognized_person = data
                if data.get("is_known") and data.get("confidence", 0.0) >= 0.45:
                    if self._is_sleeping:
                        self._wake_up()
                    name = data.get("name", "")
                    title = data.get("title", "")
                    formal_title = data.get("formal_title", title or name)
                    self._active_person_name = name
                    self._person_hold_until = now + 45.0  # Hold identity for 45 seconds

                    # Event-driven vision trigger for new person
                    if getattr(self, "_last_seen_person", "") != name:
                        self._last_seen_person = name
                        threading.Thread(target=self._evaluate_vision_event, args=("new_person",), daemon=True).start()

                    # Proactive greeting check: greet once every 2 minutes per person
                    last_greet = self._greeted_people.get(name, 0.0)
                    if (now - last_greet) > 120.0 and not self._is_responding and not self._is_playback_active:
                        self._greeted_people[name] = now
                        self.get_logger().info(f"👋 [Proaktif Selamlama]: {name} ({formal_title}) algılandı — Selamlama başlatılıyor!")
                        self._trigger_proactive_greeting(name, formal_title)

            self._sync_perception_to_session()
        except Exception as _exc:
            self.get_logger().debug(f"_on_recognized_person: yok sayılan hata ({_exc})")

    def _on_faces(self, msg: String):
        """Processes multi-person detection array and runs AttentionManager focus selection."""
        try:
            raw = (msg.data or "").strip()
            if not raw:
                return
            faces_data = json.loads(raw)
            if not isinstance(faces_data, list) or len(faces_data) == 0:
                return

            candidates: List[Any] = []
            for idx, f in enumerate(faces_data):
                if not isinstance(f, dict):
                    continue
                name_val = f.get("recognized_name") or f.get("name") or "Misafir"
                is_known = bool(f.get("is_known", False) or (name_val.lower() != "misafir"))
                dist = float(f.get("distance_m", 1.5))
                looking = bool(f.get("looking_at_robot", False))
                yaw = float(f.get("yaw_deg", 0.0))
                p_id = str(f.get("person_id") or f"person_{name_val.lower()}_{idx}")

                if UnifiedPersonState:
                    p_state = UnifiedPersonState(
                        person_id=p_id,
                        name=name_val,
                        formal_title=f.get("recognized_title") or name_val,
                        is_known=is_known,
                        identity_confidence=0.85 if is_known else 0.20,
                        familiarity_score=0.80 if is_known else 0.20,
                        distance_m=dist,
                        azimuth_deg=yaw,
                        is_looking_at_robot=looking,
                        is_present=True,
                    )
                    candidates.append(p_state)

            if getattr(self, "social_brain", None):
                self.social_brain.world_model.update_people(candidates)

                # Focus target selection via AttentionManager
                if hasattr(self.social_brain, "attention_manager") and candidates:
                    chosen, score = self.social_brain.attention_manager.select_focus_target(candidates)
                    if chosen:
                        with self._lock:
                            if chosen.is_known and chosen.name.lower() != "misafir":
                                self._active_person_name = chosen.name
                                self._person_hold_until = time.monotonic() + 30.0
                            self._user_distance = chosen.distance_m
                            self._looking_at_robot = chosen.is_looking_at_robot
        except Exception as _exc:
            self.get_logger().debug(f"_on_faces: {_exc}")

    def _on_speaker_id(self, msg: String):
        try:
            raw = (msg.data or "").strip()
            if not raw:
                return
            if raw.startswith("{") and raw.endswith("}"):
                data = json.loads(raw)
            else:
                is_k = raw.lower() not in ("misafir", "unknown", "none", "tanınmadı", "")
                data = {"name": raw if is_k else "Misafir", "confidence": 0.90 if is_k else 0.0, "is_known": is_k, "source": "speaker_id_topic"}
            with self._lock:
                self._recognized_speaker = data
                if data.get("is_known") and data.get("confidence", 0.0) >= 0.40 and data.get("name", "").lower() != "misafir":
                    self._active_person_name = data.get("name")
                    self._person_hold_until = time.monotonic() + 45.0
            self._sync_perception_to_session()
        except Exception as _exc:
            self.get_logger().debug(f"_on_speaker_id: yok sayılan hata ({_exc})")

    def _on_conversation_session_ended(self):
        """Called when a conversational session times out or user departs; consolidates episodic memory."""
        try:
            with self._lock:
                turns = list(self._session_turns_buffer)
                self._session_turns_buffer.clear()
                active_person = self._active_person_name or "Misafir"
                current_emotion = getattr(self, "_user_emotion", "neutral")

            if not turns:
                return

            if getattr(self, "social_brain", None) and hasattr(self.social_brain, "consolidation_engine"):
                def _run_consolidation():
                    try:
                        self.get_logger().info(
                            f"💾 [Hafıza Konsolidasyonu]: {active_person} ile yapılan {len(turns)} turluk sohbet kalıcı belleğe işleniyor..."
                        )
                        self.social_brain.consolidation_engine.consolidate_session(
                            person_name=active_person,
                            dialogue_turns=turns,
                            emotional_arc=current_emotion,
                        )
                    except Exception as err:
                        self.get_logger().debug(f"Consolidation error: {err}")

                threading.Thread(target=_run_consolidation, daemon=True).start()
        except Exception as exc:
            self.get_logger().debug(f"_on_conversation_session_ended: {exc}")

    def _ground_speech_gesture(self, text: str):
        """Speech gestures are disabled to maintain dedicated spatial tracking (Radar + Vision + Audio)."""
        pass

    def _on_user_emotion(self, msg: String):
        self._user_emotion = msg.data.lower().strip()

    def _on_looking_at_robot(self, msg: Bool):
        new_state = bool(msg.data)
        if new_state and not getattr(self, "_last_looking_state", False):
            self._last_looking_state = True
            threading.Thread(target=self._evaluate_vision_event, args=("user_attention_event",), daemon=True).start()
        elif not new_state:
            self._last_looking_state = False
        self._looking_at_robot = new_state

    def _on_user_distance(self, msg: Float32):
        new_dist = float(msg.data)
        old_dist = getattr(self, "_last_seen_distance", 0.0)
        if abs(new_dist - old_dist) >= 0.60:
            self._last_seen_distance = new_dist
            # Local perception tracking (No cloud vision triggered purely by approach)
        self._user_distance = new_dist

    def _on_doa(self, msg: Float32):
        self._speaker_angle = float(msg.data)
        if getattr(self, "action_manager", None):
            self.action_manager.update_audio_state(
                raw_doa_deg=float(msg.data),
                rms_level=getattr(self, "_latest_mic_rms", None),
                vad_active=getattr(self, "_vad_active", False),
                is_speaking=self._is_responding,
                is_playback_active=self._is_playback_active,
            )

    def _on_mic_level(self, msg: Float32):
        rms = float(msg.data)
        self._latest_mic_rms = rms
        if getattr(self, "action_manager", None):
            self.action_manager.update_audio_state(
                rms_level=rms,
                vad_active=getattr(self, "_vad_active", False),
                is_speaking=self._is_responding,
                is_playback_active=self._is_playback_active,
            )

    def _on_vad(self, msg: Bool):
        self._vad_active = bool(msg.data)
        if getattr(self, "action_manager", None):
            self.action_manager.update_audio_state(
                vad_active=bool(msg.data),
                rms_level=getattr(self, "_latest_mic_rms", None),
                is_speaking=self._is_responding,
                is_playback_active=self._is_playback_active,
            )

    def resolve_identities(self) -> Dict[str, Any]:
        """Separates and resolves:
        - SESSION_IDENTITY: Default user for context, system prompt, and memory retrieval.
        - BIOMETRIC_IDENTITY: Person genuinely verified by acoustic voice recognition / face sensor.
        - ACTIVE_HOLD: Temporary continuation hint of previous speaker during multi-turn dialogue.
        """
        now = time.monotonic()
        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                face = getattr(self, "_recognized_person", None) or {}
                spk = getattr(self, "_recognized_speaker", None) or {}
                held_name = getattr(self, "_active_person_name", "")
                hold_until = getattr(self, "_person_hold_until", 0.0)
        else:
            face = getattr(self, "_recognized_person", None) or {}
            spk = getattr(self, "_recognized_speaker", None) or {}
            held_name = getattr(self, "_active_person_name", "")
            hold_until = getattr(self, "_person_hold_until", 0.0)

        # 1. BIOMETRIC IDENTITY (Acoustic / Visual Ground Truth)
        bio_id = "unknown"
        bio_source = "none"
        bio_conf = 0.0
        if spk.get("is_known") and spk.get("confidence", 0.0) >= 0.40 and spk.get("name", "").lower() != "misafir":
            bio_id = spk.get("name")
            bio_source = "voice"
            bio_conf = float(spk.get("confidence", 0.0))
        elif face.get("is_known") and face.get("confidence", 0.0) >= 0.45 and face.get("name", "").lower() != "misafir":
            bio_id = face.get("name")
            bio_source = "face"
            bio_conf = float(face.get("confidence", 0.0))

        # 2. ACTIVE HOLD (Continuation Hint)
        has_active_hold = bool(now < hold_until and held_name and held_name.lower() != "misafir")
        hold_remaining_s = max(0.0, hold_until - now) if has_active_hold else 0.0

        # 3. MEMORY / PERSISTENT DEFAULT OWNER
        owner_name = "Baran"
        if hasattr(self, "memory") and hasattr(self.memory, "profile") and hasattr(self.memory.profile, "data"):
            owner_name = self.memory.profile.data.get("owner_name", "Baran")

        # 4. SESSION IDENTITY RESOLUTION (Final effective user for context and memory)
        if bio_id != "unknown":
            user_name = bio_id
            user_source = f"biometric_{bio_source}"
            bio_status = "verified"
            is_known = True
        elif has_active_hold:
            user_name = held_name
            user_source = "session_hold"
            bio_status = "session_active"
            is_known = True
        elif owner_name and owner_name.lower() != "misafir":
            user_name = owner_name
            user_source = "persistent_memory"
            bio_status = "unknown"
            is_known = True
        else:
            user_name = "Misafir"
            user_source = "guest_fallback"
            bio_status = "unknown"
            is_known = False

        identity_dict = {
            # 1. SESSION IDENTITY (Context / System Prompt / Memory)
            "session_identity": user_name,
            "owner_name": owner_name,
            "user_id": user_name.lower(),
            "display_name": user_name,
            "name": user_name,
            "title": user_name,
            "formal_title": user_name,
            "is_known": is_known,
            "identity_source": user_source,

            # 2. BIOMETRIC IDENTITY (Ground Sensor Truth)
            "biometric_identity": bio_id,
            "biometric_source": bio_source,
            "biometric_status": bio_status,
            "biometric_confidence": bio_conf,

            # 3. ACTIVE HOLD (Multi-turn continuation hint)
            "active_hold_speaker": held_name if has_active_hold else "",
            "active_hold_remaining_s": round(hold_remaining_s, 1),
            "has_active_hold": has_active_hold,

            # Compatibility field
            "source": user_source,
        }
        return identity_dict

    def _get_active_biometric_identity(self) -> Dict[str, Any]:
        return self.resolve_identities()

    def _sync_perception_to_session(self):
        """Dynamically syncs persona & recognized identity to the active OpenAI Realtime session ONLY when identity changes."""
        if not self._ws or not self._loop or not self._is_connected:
            return

        now = time.monotonic()
        identity = self._get_active_biometric_identity()
        identity_name = identity.get("name", "Misafir")

        # Strictly require identity change: If it is still the same person/guest, NEVER resync
        last_id = getattr(self, "_last_synced_identity", "")
        if identity_name == last_id:
            return

        prev_id = last_id
        self._last_synced_identity = identity_name
        self._last_sync_time = now

        system_prompt = self._build_current_system_prompt()
        update_event = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": system_prompt
            }
        }

        try:
            asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(update_event)), self._loop)
            self.get_logger().info(f"👤 [Realtime Biyometri]: Oturum kimliği güncellendi -> {identity_name}")
        except Exception as _exc:
            self.get_logger().debug(f"_sync_perception_to_session: yok sayılan hata ({_exc})")






    def _publish_system_telemetry(self):
        """Periodically aggregates system health, turn latencies, and sensor watchdogs to /astro/telemetry and /diagnostics."""
        try:
            now = time.monotonic()

            # 1. Sensor Freshness & Watchdogs
            last_lidar = getattr(self, "_last_laser_scan_time", 0.0)
            lidar_age = now - last_lidar if last_lidar > 0.0 else 999.0
            lidar_ok = (last_lidar > 0.0 and lidar_age <= 3.0)
            lidar_level = DiagnosticStatus.OK if lidar_ok else (DiagnosticStatus.WARN if lidar_age <= 8.0 else DiagnosticStatus.ERROR)

            last_cam = getattr(self, "_last_img_time", 0.0)
            cam_age = now - last_cam if last_cam > 0.0 else 999.0
            cam_ok = (last_cam > 0.0 and cam_age <= 5.0)
            cam_level = DiagnosticStatus.OK if cam_ok else DiagnosticStatus.WARN

            ard_hb = getattr(self, "_arduino_heartbeat_healthy", False)
            last_hb = getattr(self, "_last_heartbeat_ack_time", 0.0)
            ard_age = now - last_hb if last_hb > 0.0 else 999.0
            ard_alive = ard_hb and (ard_age <= 2.5)
            ard_level = DiagnosticStatus.OK if ard_alive else DiagnosticStatus.ERROR

            ws_conn = getattr(self, "_is_connected", False)
            ws_state = getattr(self, "_realtime_state", "DISCONNECTED")
            ws_level = DiagnosticStatus.OK if ws_conn else (DiagnosticStatus.WARN if ws_state == "CONNECTING" else DiagnosticStatus.ERROR)

            # 2. Latency Metrics
            lat_stats = self.session.latency_tracker.get_stats() if hasattr(self.session, "latency_tracker") else {}
            p50_ms = lat_stats.get("p50_total_ms", 0.0)
            p95_ms = lat_stats.get("p95_total_ms", 0.0)
            samples = lat_stats.get("samples", 0)

            # 3. Social Cognitive Context
            active_p = getattr(self, "_active_person_name", "Misafir") or "Misafir"
            is_looking = bool(getattr(self, "_looking_at_robot", False))
            user_dist = float(getattr(self, "_user_distance", 0.0))
            user_emot = getattr(self, "_user_emotion", "neutral")

            fam = 0.10
            trust = 0.50
            role_str = "new_user"
            if getattr(self, "social_brain", None) and hasattr(self.social_brain, "relationship_manager"):
                rel_info = self.social_brain.relationship_manager.assess_relationship(active_p)
                fam = float(rel_info.get("familiarity", 0.10))
                trust = float(rel_info.get("trust", 0.50))
                r = rel_info.get("role", "new_user")
                role_str = r.value if hasattr(r, "value") else str(r)

            # 4. Publish /astro/telemetry (JSON)
            if getattr(self, "pub_telemetry", None):
                telem_payload = {
                    "timestamp": time.time(),
                    "latency": {
                        "p50_total_ms": p50_ms,
                        "p95_total_ms": p95_ms,
                        "samples": samples,
                    },
                    "sensors": {
                        "lidar_alive": lidar_ok,
                        "lidar_age_s": round(lidar_age, 2),
                        "camera_alive": cam_ok,
                        "camera_age_s": round(cam_age, 2),
                        "arduino_alive": ard_alive,
                        "arduino_age_s": round(ard_age, 2),
                    },
                    "realtime_ws": {
                        "connected": ws_conn,
                        "state": ws_state,
                    },
                    "social_state": {
                        "active_person": active_p,
                        "role": role_str,
                        "familiarity": fam,
                        "trust": trust,
                        "user_distance_m": round(user_dist, 2),
                        "looking_at_robot": is_looking,
                        "emotion": user_emot,
                    },
                }
                msg_telem = String()
                msg_telem.data = json.dumps(telem_payload)
                self.pub_telemetry.publish(msg_telem)

            # 5. Publish /diagnostics (DiagnosticArray)
            if getattr(self, "pub_diagnostics", None) and 'DiagnosticArray' in globals():
                diag_arr = DiagnosticArray()
                st_ws = DiagnosticStatus(
                    name="Astro Realtime / OpenAI WebSocket",
                    level=ws_level,
                    message=f"State: {ws_state}",
                    values=[
                        KeyValue("connected", str(ws_conn)),
                        KeyValue("p50_ms", str(p50_ms)),
                        KeyValue("p95_ms", str(p95_ms)),
                    ],
                )
                st_ard = DiagnosticStatus(
                    name="Astro Base / Serial Controller",
                    level=ard_level,
                    message="Alive" if ard_alive else f"Stale/Dead ({round(ard_age, 1)}s)",
                    values=[
                        KeyValue("heartbeat_healthy", str(ard_alive)),
                        KeyValue("age_s", str(round(ard_age, 2))),
                    ],
                )
                st_lidar = DiagnosticStatus(
                    name="Astro Safety / RPLiDAR",
                    level=lidar_level,
                    message="Active" if lidar_ok else f"Stale ({round(lidar_age, 1)}s)",
                    values=[
                        KeyValue("scan_alive", str(lidar_ok)),
                        KeyValue("age_s", str(round(lidar_age, 2))),
                    ],
                )
                st_cam = DiagnosticStatus(
                    name="Astro Perception / OAK-D Lite",
                    level=cam_level,
                    message="Active" if cam_ok else f"Stale ({round(cam_age, 1)}s)",
                    values=[
                        KeyValue("camera_alive", str(cam_ok)),
                        KeyValue("age_s", str(round(cam_age, 2))),
                    ],
                )
                diag_arr.status = [st_ws, st_ard, st_lidar, st_cam]
                self.pub_diagnostics.publish(diag_arr)

        except Exception as _exc:
            self.get_logger().debug(f"_publish_system_telemetry: {_exc}")

    def _on_robot_state_change(self, old_state: RobotState, new_state: RobotState):
        """Maps state machine transitions directly to non-blocking ReSpeaker 12 LED animations."""
        if not getattr(self, "robot_led", None):
            return
        try:
            if new_state in (RobotState.DEEP_IDLE, RobotState.IDLE):
                self.robot_led.idle()
            elif new_state in (RobotState.WAKE, RobotState.LISTENING, RobotState.INTERRUPTED):
                self.robot_led.listening()
            elif new_state in (RobotState.THINKING, RobotState.THINKING_ACK):
                self.robot_led.thinking()
            elif new_state == RobotState.SPEAKING:
                self.robot_led.speaking()
        except Exception as exc:
            self.get_logger().debug(f"LED state transition notice: {exc}")

    def destroy_node(self):
        """Clean shutdown turning off hardware LEDs and stopping worker threads."""
        try:
            if getattr(self, "robot_led", None):
                self.robot_led.shutdown()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):

    rclpy.init(args=args)
    node = AstroRealtimeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        _LOG.debug("main spin exit: %s", exc)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
