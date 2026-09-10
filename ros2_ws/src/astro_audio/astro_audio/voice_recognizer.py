#!/usr/bin/env python3
"""ASTRO V1 — Acoustic Speaker Recognition & Voiceprint Identity Engine.

Features:
  - 16kHz PCM Acoustic Feature Extraction (MFCCs, Pitch F0, Spectral Moments)
  - Normalized Voiceprint Embeddings & Cosine Metric Matching
  - Pre-seeded voiceprint profiles for Creators & Officials
  - Dynamic On-The-Fly Enrollment (learn new voices in real time)
"""

import json
import logging

_LOG = logging.getLogger(__name__)

import os
import re
import time
import threading
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


try:
    from astro_audio.speaker_db import SpeakerEngine, SpeakerEngineUnavailable
except ImportError:  # paket kaynaktan çalıştırılıyorsa
    try:
        from speaker_db import SpeakerEngine, SpeakerEngineUnavailable
    except ImportError:
        SpeakerEngine = None

        class SpeakerEngineUnavailable(RuntimeError):
            pass

# Ölçüm: aynı kişi 0.46-0.81, farklı kişi 0.16-0.34 -> 0.42 ikisinin arasında.
VOICE_MATCH_THRESHOLD = float(os.getenv("SPEAKER_MATCH_THRESHOLD", "0.42"))

_ENGINE = None
_ENGINE_TRIED = False


def _get_engine():
    """WeSpeaker ONNX motorunu bir kez yükler; model yoksa None döner."""
    global _ENGINE, _ENGINE_TRIED
    if not _ENGINE_TRIED:
        _ENGINE_TRIED = True
        if SpeakerEngine is not None:
            try:
                _ENGINE = SpeakerEngine()
            except SpeakerEngineUnavailable as exc:
                print(f"[VoiceRecognizer] Konuşmacı modeli yüklenemedi: {exc}")
    return _ENGINE


class VoiceRecognizer:
    """Manages acoustic voiceprints, speaker profiles, and real-time voice identification."""

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            candidates = [
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "known_voices")),
                os.path.abspath("./data/known_voices")
            ]
            self.data_dir = candidates[0]
            for c in candidates:
                if os.path.exists(c):
                    self.data_dir = c
                    break
        else:
            self.data_dir = data_dir

        os.makedirs(self.data_dir, exist_ok=True)
        self._lock = threading.RLock()
        self._known_voiceprints: Dict[str, List[np.ndarray]] = {}
        self._speaker_metadata: Dict[str, Dict[str, Any]] = {}
        self._playback_ref_bytes: bytes = b""
        self._playback_ref_lock = threading.Lock()

        self._init_default_speakers()
        self.reload_voiceprints()

    def _init_default_speakers(self):
        defaults = {
            "Baran": {"title": "Baş Mühendis & Yaratıcı", "formal_title": "Baran Bey", "gender": "male"},
            "Erol Karaömeroğlu": {"title": "Bitlis Valisi", "formal_title": "Sayın Valim", "gender": "male"},
            "Nesrullah Tanğlay": {"title": "Bitlis Belediye Başkanı", "formal_title": "Sayın Başkanım", "gender": "male"},
            "Batuhan Bingöl": {"title": "Ahlat Kaymakamı", "formal_title": "Sayın Kaymakamım", "gender": "male"},
            "Yavuz Gülmez": {"title": "Ahlat Belediye Başkanı", "formal_title": "Sayın Başkanım", "gender": "male"},
            "Recep Tayyip Erdoğan": {"title": "Cumhurbaşkanı", "formal_title": "Sayın Cumhurbaşkanım", "gender": "male"}
        }
        for name, meta in defaults.items():
            norm = self._normalize_name(name)
            self._speaker_metadata[norm] = {"name": name, **meta}

    def _normalize_name(self, name: str) -> str:
        tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
        clean = name.translate(tr_map).lower()
        clean = re.sub(r"[^a-z0-9_]+", "_", clean).strip("_")
        return clean or "unknown"

    def extract_voiceprint_with_profile(self, audio_arr: np.ndarray, sample_rate: int = 16000) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Sesten L2-normalize edilmiş WeSpeaker vektörü ve mikro zamanlama profili çıkarır."""
        prof = {
            "sample_count": len(audio_arr) if audio_arr is not None else 0,
            "audio_duration_ms": (len(audio_arr) / sample_rate * 1000.0) if (audio_arr is not None and sample_rate > 0) else 0.0,
            "fbank_ms": 0.0,
            "onnx_infer_ms": 0.0,
            "norm_ms": 0.0,
            "device": "CPUExecutionProvider",
        }
        if audio_arr is None or len(audio_arr) == 0:
            return None, prof

        engine = _get_engine()
        if engine is None:
            return None, prof
        try:
            if hasattr(engine, "embed_with_profile"):
                emb, eng_prof = engine.embed_with_profile(np.asarray(audio_arr), sample_rate)
                prof.update(eng_prof)
                return emb, prof
            emb = engine.embed(np.asarray(audio_arr), sample_rate)
            return emb, prof
        except Exception:
            return None, prof

    def extract_voiceprint(self, audio_arr: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """Sesten L2-normalize edilmiş WeSpeaker (VoxCeleb) vektörü çıkarır."""
        emb, _ = self.extract_voiceprint_with_profile(audio_arr, sample_rate)
        return emb

    def reload_voiceprints(self):
        """Scans data_dir for saved speaker .npy and loads SpeakerEngine database (~/.astro/voices/speakers.json)."""
        engine = _get_engine()
        if engine is not None:
            try:
                engine.load()
            except Exception as _exc:
                _LOG.debug("reload_voiceprints: yok sayılan hata (%s)", _exc)

        with self._lock:
            self._known_voiceprints.clear()
            # 1. Load from SpeakerEngine (~/.astro/voices/speakers.json)
            if engine is not None and getattr(engine, "people", None):
                for person_name, vectors in engine.people.items():
                    norm = self._normalize_name(person_name)
                    self._known_voiceprints.setdefault(norm, []).extend(vectors)
                    if norm not in self._speaker_metadata:
                        self._speaker_metadata[norm] = {
                            "name": person_name,
                            "title": "Tanınan Konuşmacı",
                            "formal_title": person_name,
                            "gender": "unknown"
                        }

            # 2. Load from .npy files in data_dir
            if os.path.exists(self.data_dir):
                for f in os.listdir(self.data_dir):
                    if f.endswith(".npy"):
                        spk_name = os.path.splitext(f)[0]
                        norm = self._normalize_name(spk_name)
                        try:
                            emb = np.load(os.path.join(self.data_dir, f))
                            self._known_voiceprints.setdefault(norm, []).append(emb)
                        except Exception as _exc:
                            _LOG.debug("reload_voiceprints: yok sayılan hata (%s)", _exc)

    def enroll_voice(self, name: str, audio_arr: np.ndarray, sample_rate: int = 16000, title: Optional[str] = None) -> bool:
        """Dynamically learns and saves a speaker voiceprint."""
        if audio_arr is None or not name:
            return False

        emb = self.extract_voiceprint(audio_arr, sample_rate)
        if emb is None:
            return False

        norm_name = self._normalize_name(name)
        with self._lock:
            self._known_voiceprints.setdefault(norm_name, []).append(emb)
            if norm_name not in self._speaker_metadata:
                self._speaker_metadata[norm_name] = {
                    "name": name,
                    "title": title or "Misafir",
                    "formal_title": title or name,
                    "gender": "unknown"
                }

            # Save to disk as .npy
            try:
                save_path = os.path.join(self.data_dir, f"{norm_name}.npy")
                np.save(save_path, emb)
            except Exception as _exc:
                _LOG.debug("enroll_voice: yok sayılan hata (%s)", _exc)

            # Save to SpeakerEngine database
            engine = _get_engine()
            if engine is not None:
                try:
                    engine.add_person(name, [emb])
                    engine.save()
                except Exception as _exc:
                    _LOG.debug("enroll_voice: yok sayılan hata (%s)", _exc)
            return True

    def recognize_voice(self, audio_arr: np.ndarray, sample_rate: int = 16000, threshold: float = VOICE_MATCH_THRESHOLD) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """Matches audio array against known voiceprints. Returns (name, best_score, metadata).

        IMPORTANT: This method always returns the BEST candidate and their score, even below threshold.
        Threshold enforcement and margin-based accept/reject logic is handled by the caller
        (_run_voice_identification multi-window voter). This enables proper margin calculation.
        """
        # Check if database was updated on disk by enroll_speaker.py and reload dynamically
        engine = _get_engine()
        if engine is not None and hasattr(engine, "db_path") and hasattr(engine.db_path, "exists") and engine.db_path.exists():
            try:
                mtime = engine.db_path.stat().st_mtime
                if mtime > getattr(self, "_last_db_mtime", 0.0):
                    self._last_db_mtime = mtime
                    self.reload_voiceprints()
            except Exception:
                pass

        t0 = time.monotonic()
        emb, emb_prof = self.extract_voiceprint_with_profile(audio_arr, sample_rate)
        t_emb_done = time.monotonic()
        extract_ms = (t_emb_done - t0) * 1000.0
        if emb is None:
            return None, 0.0, {"voice_id_profile": emb_prof}

        # Collect all speaker scores in one pass for margin calculation
        all_scores: list = []  # list of (norm_name, sim, meta)

        # 1. In-memory embeddings (primary — fastest path)
        with self._lock:
            known_vp = dict(self._known_voiceprints)
            known_meta = dict(self._speaker_metadata)

        t_match_start = time.monotonic()
        for spk_norm, emb_list in known_vp.items():
            best_sim = max(float(np.dot(emb, kn_emb)) for kn_emb in emb_list) if emb_list else -1.0
            meta = dict(known_meta.get(spk_norm, {
                "name": spk_norm.replace("_", " ").title(),
                "title": "Tanınan Konuşmacı",
                "formal_title": spk_norm.replace("_", " ").title()
            }))
            all_scores.append((spk_norm, best_sim, meta))

        # 2. Also query SpeakerEngine (may have additional speakers from speakers.json)
        engine = _get_engine()
        if engine is not None:
            try:
                engine.load()
                matched_name, sim = engine.identify(emb)
                if matched_name is not None:
                    norm = self._normalize_name(matched_name)
                    if not any(s[0] == norm for s in all_scores):
                        eng_meta = dict(known_meta.get(norm, {
                            "name": matched_name,
                            "title": "Tanınan Konuşmacı",
                            "formal_title": matched_name
                        }))
                        all_scores.append((norm, float(sim), eng_meta))
            except Exception as _exc:
                _LOG.debug("recognize_voice: yok sayılan hata (%s)", _exc)
        t_match_done = time.monotonic()
        match_ms = (t_match_done - t_match_start) * 1000.0

        if not all_scores:
            return None, 0.0, {"voice_id_profile": emb_prof}

        # Sort by similarity descending
        all_scores.sort(key=lambda x: x[1], reverse=True)
        best_norm, best_sim, best_meta = all_scores[0]

        best_meta["voice_id_profile"] = {
            "fbank_ms": emb_prof.get("fbank_ms", 0.0),
            "onnx_infer_ms": emb_prof.get("onnx_infer_ms", 0.0),
            "norm_ms": emb_prof.get("norm_ms", 0.0),
            "extract_ms": round(extract_ms, 2),
            "speaker_match_ms": round(match_ms, 2),
            "device": emb_prof.get("device", "CPUExecutionProvider"),
            "candidate_count": len(all_scores),
            "sample_count": len(audio_arr) if audio_arr is not None else 0,
            "audio_duration_ms": round((len(audio_arr) / sample_rate) * 1000.0, 1) if (audio_arr is not None and sample_rate > 0) else 0.0,
        }

        # Always return best candidate — let caller decide accept/reject via margin
        return best_meta.get("name", best_norm.replace("_", " ").title()), round(max(0.0, best_sim), 4), best_meta

    def identify_speaker(self, audio_arr: np.ndarray, sample_rate: int = 16000) -> Tuple[Optional[str], float]:
        """Convenience method returning (name, score) for direct pipeline consumption."""
        name, score, _ = self.recognize_voice(audio_arr, sample_rate)
        return name, score

    def update_playback_reference(self, pcm_bytes: bytes, max_history_bytes: int = 48000):
        """Updates reference PCM ring buffer from recent robot TTS/playback audio (16kHz int16)."""
        if not pcm_bytes:
            return
        with self._playback_ref_lock:
            self._playback_ref_bytes = (self._playback_ref_bytes + pcm_bytes)[-max_history_bytes:]

    def clear_playback_reference(self):
        """Clears playback reference buffer when playback stops or turn finishes."""
        with self._playback_ref_lock:
            self._playback_ref_bytes = b""

    def score_self_voice(self, audio_data: Any, reference_pcm: Optional[bytes] = None, max_lag_samples: int = 4800) -> float:
        """Calculates acoustic self-voice / playback echo score (0.0 to 1.0) using normalized cross-correlation.

        Returns >= 0.70 when audio matches robot playback echo, and < 0.40 for human speech.
        """
        if audio_data is None:
            return 0.0
        if isinstance(audio_data, (bytes, bytearray)):
            mic_bytes = bytes(audio_data)
        elif isinstance(audio_data, np.ndarray):
            mic_bytes = audio_data.astype(np.int16).tobytes()
        else:
            return 0.0

        if reference_pcm is not None:
            ref_bytes = reference_pcm
        else:
            with self._playback_ref_lock:
                ref_bytes = self._playback_ref_bytes

        if not mic_bytes or not ref_bytes:
            return 0.0

        mic = np.frombuffer(mic_bytes, dtype=np.int16).astype(np.float32)
        ref = np.frombuffer(ref_bytes, dtype=np.int16).astype(np.float32)

        if len(mic) == 0 or len(ref) == 0:
            return 0.0

        mic_centered = mic - np.mean(mic)
        mic_norm = float(np.linalg.norm(mic_centered))
        if mic_norm < 1e-4:
            return 0.0

        ref_window = ref[-max_lag_samples:] if len(ref) > max_lag_samples else ref
        if len(ref_window) < len(mic):
            return 0.0

        ref_centered = ref_window - np.mean(ref_window)
        corr = np.correlate(ref_centered, mic_centered, mode='valid')
        ref_sq = ref_centered ** 2
        window_energy = np.correlate(ref_sq, np.ones(len(mic), dtype=np.float32), mode='valid')
        denom = mic_norm * np.sqrt(np.maximum(window_energy, 1e-6))
        norm_corr = corr / denom
        max_corr = float(np.max(norm_corr)) if len(norm_corr) > 0 else 0.0
        return round(max(0.0, min(1.0, max_corr)), 4)

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns runtime speaker recognition telemetry."""
        engine = _get_engine()
        model_p = str(getattr(engine, "model_path", "none")) if engine else "none"
        model_exists = bool(engine and getattr(engine, "_session", None) is not None)
        db_p = str(getattr(engine, "db_path", self.data_dir)) if engine else self.data_dir

        with self._lock:
            known_list = [v.get("name", k) for k, v in self._speaker_metadata.items()]

        return {
            "speaker_model_path": model_p,
            "speaker_model_exists": model_exists,
            "known_voices_path": db_p,
            "known_speakers": known_list,
        }

    def delete_speaker(self, name: str) -> bool:
        """Deletes speaker from memory, .npy files, and speakers.json."""
        norm_name = self._normalize_name(name)
        with self._lock:
            self._known_voiceprints.pop(norm_name, None)
            self._speaker_metadata.pop(norm_name, None)

        try:
            npy_path = os.path.join(self.data_dir, f"{norm_name}.npy")
            if os.path.exists(npy_path):
                os.remove(npy_path)
        except Exception as _exc:
            _LOG.debug("delete_speaker: yok sayılan hata (%s)", _exc)

        engine = _get_engine()
        if engine is not None:
            try:
                engine.remove_person(name)
                engine.save()
            except Exception as _exc:
                _LOG.debug("delete_speaker: yok sayılan hata (%s)", _exc)
        return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ASTRO Speaker Recognition Telemetry & Test CLI")
    parser.add_argument("--test-speaker", help="Test if speaker is enrolled and ready")
    parser.add_argument("--list", action="store_true", help="List enrolled speakers and paths")
    args = parser.parse_args()

    rec = VoiceRecognizer()
    telemetry = rec.get_telemetry()
    print("=" * 60)
    print("[ASTRO Speaker Recognition Runtime Telemetry]")
    print(f"   Model Path:          {telemetry['speaker_model_path']}")
    print(f"   Model Exists/Ready:  {telemetry['speaker_model_exists']}")
    print(f"   Known Voices Path:   {telemetry['known_voices_path']}")
    print(f"   Enrolled Speakers:   {', '.join(telemetry['known_speakers']) if telemetry['known_speakers'] else 'None'}")
    print("=" * 60)

    if args.test_speaker:
        name_query = args.test_speaker.strip().lower()
        matched = [s for s in telemetry['known_speakers'] if s.lower() == name_query]
        if matched:
            print(f"[OK] Speaker '{args.test_speaker}' is verified and registered in database!")
        else:
            print(f"[ERROR] Speaker '{args.test_speaker}' NOT found in enrolled speakers: {telemetry['known_speakers']}")

