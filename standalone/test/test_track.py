"""Sabit dizüstü kamerası, dönmeyen kafayı dönmüş gibi göstermemeli."""

import contextlib
import io
import re
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import track
from tracker import Detection


class _SabitKamera:
    available = True
    backend = "webcam"
    detector_name = "test"

    def __init__(self, **kwargs):
        self.kalan = 200
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def read(self):
        self.kalan -= 1
        return self.kalan >= 0, self.frame

    def detect(self, frame):
        # Merkez x=380: 640 piksellik görüntüde hafif sağda sabit bir yüz.
        return [Detection(x=330, y=160, w=100, h=100, confidence=0.95)]

    def close(self):
        pass


class _SessizKaynak:
    available = False
    error = "testte ses yok"

    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def close(self):
        pass


def test_sabit_kamera_yuzu_kafa_limitine_suruklemez():
    """Gerçek main ve ortak beyin; yalnız aygıtlar ve saat denetim altında."""
    saat = iter(100 + i / 30 for i in range(1000))
    cikti = io.StringIO()
    with patch.object(track, "CameraSource", _SabitKamera), \
            patch.object(track, "AudioSource", _SessizKaynak), \
            patch.object(track.time, "monotonic", side_effect=lambda: next(saat)), \
            contextlib.redirect_stdout(cikti):
        assert track.main(["--no-window", "--no-voice", "--fixed-head"]) == 0

    satirlar = [line for line in cikti.getvalue().splitlines() if "istenen" in line]
    hedefler = [float(re.search(r"istenen\s*([+-][\d.]+)", line)[1]) for line in satirlar]
    assert len(hedefler) > 3
    assert all(-10 < hedef < -3 for hedef in hedefler)
    assert max(hedefler) - min(hedefler) < 0.1
    assert all("sabit" in line and "kafa:X" in line for line in satirlar)
    assert "gercek" not in cikti.getvalue()
    assert "Encoder hiç konuşmadı" not in cikti.getvalue()


def test_sabit_referans_motor_baglantisiyla_birlikte_acilamaz():
    with pytest.raises(SystemExit) as exc:
        track.main(["--fixed-head", "--serial", "/dev/test"])
    assert exc.value.code == 2


# ── Legacy calibrated_yaw tests (backward compat) ──────────────────

# 1. DOA 32 -> yaw -45
def test_doa_32_maps_to_yaw_minus_45():
    from track import ReSpeakerAudioLocalizer
    assert ReSpeakerAudioLocalizer.calibrated_yaw(32.0) == pytest.approx(-45.0, abs=1e-3)


# 2. DOA 78 -> yaw 0
def test_doa_78_maps_to_yaw_0():
    from track import ReSpeakerAudioLocalizer
    assert ReSpeakerAudioLocalizer.calibrated_yaw(78.0) == pytest.approx(0.0, abs=1e-3)


# 3. DOA 142 -> yaw +45
def test_doa_142_maps_to_yaw_plus_45():
    from track import ReSpeakerAudioLocalizer
    assert ReSpeakerAudioLocalizer.calibrated_yaw(142.0) == pytest.approx(45.0, abs=1e-3)


# 4. DOA 149 -> yaw +90
def test_doa_149_maps_to_yaw_plus_90():
    from track import ReSpeakerAudioLocalizer
    assert ReSpeakerAudioLocalizer.calibrated_yaw(149.0) == pytest.approx(90.0, abs=1e-3)


# 5. DOA 90 gibi ölçülmemiş/ambiguous değer -> güvenli davranış
def test_unmeasured_doa_90_safe_behavior():
    from track import ReSpeakerAudioLocalizer
    yaw = ReSpeakerAudioLocalizer.calibrated_yaw(90.0)
    assert yaw is not None
    assert 0.0 < yaw < 45.0
    assert yaw == pytest.approx(8.4375, abs=0.1)
    assert abs(yaw) <= 90.0
    assert yaw not in (180.0, -180.0)


# ── Sector mapping tests ──────────────────────────────────────────

def test_doa_to_sector_left():
    from track import ReSpeakerAudioLocalizer
    assert ReSpeakerAudioLocalizer.doa_to_sector(25.0) == "LEFT"
    assert ReSpeakerAudioLocalizer.doa_to_sector(32.0) == "LEFT"
    assert ReSpeakerAudioLocalizer.doa_to_sector(54.9) == "LEFT"


def test_doa_to_sector_center():
    from track import ReSpeakerAudioLocalizer
    assert ReSpeakerAudioLocalizer.doa_to_sector(55.0) == "CENTER"
    assert ReSpeakerAudioLocalizer.doa_to_sector(78.0) == "CENTER"
    assert ReSpeakerAudioLocalizer.doa_to_sector(99.9) == "CENTER"


def test_doa_to_sector_right():
    from track import ReSpeakerAudioLocalizer
    assert ReSpeakerAudioLocalizer.doa_to_sector(100.0) == "RIGHT"
    assert ReSpeakerAudioLocalizer.doa_to_sector(142.0) == "RIGHT"
    assert ReSpeakerAudioLocalizer.doa_to_sector(149.0) == "RIGHT"
    assert ReSpeakerAudioLocalizer.doa_to_sector(155.0) == "RIGHT"


def test_doa_to_sector_invalid():
    from track import ReSpeakerAudioLocalizer
    assert ReSpeakerAudioLocalizer.doa_to_sector(180.0) is None
    assert ReSpeakerAudioLocalizer.doa_to_sector(270.0) is None
    assert ReSpeakerAudioLocalizer.doa_to_sector(350.0) is None
    assert ReSpeakerAudioLocalizer.doa_to_sector(10.0) is None
    assert ReSpeakerAudioLocalizer.doa_to_sector(0.0) is None


# ── Sector-based update tests ─────────────────────────────────────

def test_sector_first_detection_immediate():
    """First valid DOA reading is accepted immediately (no confirmation wait)."""
    from track import ReSpeakerAudioLocalizer
    loc = ReSpeakerAudioLocalizer()

    # DOA 32 → LEFT sector → target +60 (robot's physical right)
    loc.update(doa_raw=32.0, voice_activity=True, timestamp=1.0)
    assert loc.is_tracking()
    assert loc.confirmed_sector == "LEFT"
    assert loc.target_yaw_deg == 60.0


def test_sector_center_first_detection():
    from track import ReSpeakerAudioLocalizer
    loc = ReSpeakerAudioLocalizer()
    loc.update(doa_raw=78.0, voice_activity=True, timestamp=1.0)
    assert loc.confirmed_sector == "CENTER"
    assert loc.target_yaw_deg == 0.0


def test_sector_right_first_detection():
    from track import ReSpeakerAudioLocalizer
    loc = ReSpeakerAudioLocalizer()
    loc.update(doa_raw=142.0, voice_activity=True, timestamp=1.0)
    assert loc.confirmed_sector == "RIGHT"
    assert loc.target_yaw_deg == -60.0


def test_sector_same_sector_no_jitter():
    """Multiple DOA readings in the same sector keep the same target."""
    from track import ReSpeakerAudioLocalizer
    loc = ReSpeakerAudioLocalizer()
    loc.update(doa_raw=32.0, voice_activity=True, timestamp=1.0)
    assert loc.target_yaw_deg == 60.0

    # Different DOA values but same LEFT sector
    for i, doa in enumerate([25.0, 40.0, 35.0, 50.0, 30.0]):
        loc.update(doa_raw=doa, voice_activity=True, timestamp=1.1 + i * 0.05)
        assert loc.target_yaw_deg == 60.0
        assert loc.confirmed_sector == "LEFT"


def test_sector_switch_requires_confirmation():
    """Sector changes while actively tracking require SECTOR_CONFIRM_COUNT consecutive readings."""
    from track import ReSpeakerAudioLocalizer
    loc = ReSpeakerAudioLocalizer()

    # Establish LEFT
    loc.update(doa_raw=32.0, voice_activity=True, timestamp=1.0)
    assert loc.confirmed_sector == "LEFT"

    # Single reading in RIGHT while tracking LEFT → does NOT switch (needs 2)
    loc.update(doa_raw=142.0, voice_activity=True, timestamp=1.1)
    assert loc.confirmed_sector == "LEFT"
    assert loc.target_yaw_deg == 60.0

    # Second reading in RIGHT → NOW it switches (count = 2)
    loc.update(doa_raw=142.0, voice_activity=True, timestamp=1.2)
    assert loc.confirmed_sector == "RIGHT"
    assert loc.target_yaw_deg == -60.0


def test_sector_switch_interrupted_resets_count():
    """A reading back to the confirmed sector resets the pending count."""
    from track import ReSpeakerAudioLocalizer
    loc = ReSpeakerAudioLocalizer()

    loc.update(doa_raw=32.0, voice_activity=True, timestamp=1.0)
    assert loc.confirmed_sector == "LEFT"

    # One reading in RIGHT
    loc.update(doa_raw=142.0, voice_activity=True, timestamp=1.1)
    assert loc.confirmed_sector == "LEFT"

    # Back to LEFT → resets pending count
    loc.update(doa_raw=32.0, voice_activity=True, timestamp=1.2)
    assert loc.confirmed_sector == "LEFT"

    # Reading 1 in RIGHT → still not enough
    loc.update(doa_raw=142.0, voice_activity=True, timestamp=1.3)
    assert loc.confirmed_sector == "LEFT"

    # Reading 2 in RIGHT → switches
    loc.update(doa_raw=142.0, voice_activity=True, timestamp=1.4)
    assert loc.confirmed_sector == "RIGHT"


def test_sector_motor_yaw_matches_target():
    """motor_yaw_deg always equals target_yaw_deg."""
    from track import ReSpeakerAudioLocalizer
    for doa, expected_sector, expected_target in [
        (32.0, "LEFT", 60.0),
        (78.0, "CENTER", 0.0),
        (142.0, "RIGHT", -60.0),
    ]:
        loc = ReSpeakerAudioLocalizer()
        loc.update(doa_raw=doa, voice_activity=True, timestamp=1.0)
        assert loc.target_yaw_deg == expected_target
        assert loc.motor_yaw_deg == expected_target
        assert loc.confirmed_sector == expected_sector


# ── Sector front jitter suppression ───────────────────────────────

def test_front_jitter_stays_in_center():
    """DOA values around 78° all map to CENTER sector → target stays at 0.0."""
    from track import ReSpeakerAudioLocalizer
    loc = ReSpeakerAudioLocalizer()
    jitter_sequence = [78.0, 75.0, 80.0, 77.0, 79.0, 76.0, 81.0, 78.0]
    for i, doa in enumerate(jitter_sequence):
        target = loc.update(doa_raw=doa, voice_activity=True, timestamp=1.0 + i * 0.05)
        assert target == 0.0
        assert loc.confirmed_sector == "CENTER"


# ── Hold timeout and VAD tests ────────────────────────────────────

# 8. VOICEACTIVITY false iken yeni hedef üretilmiyor
def test_voice_activity_false_produces_no_new_target():
    from track import ReSpeakerAudioLocalizer
    loc = ReSpeakerAudioLocalizer(hold_timeout_s=1.2)
    # Speech active at DOA 32 -> LEFT sector -> target +60.0 (robot's physical right)
    loc.update(doa_raw=32.0, voice_activity=True, timestamp=1.0)
    assert loc.is_tracking() is True
    assert loc.target_yaw_deg == 60.0

    # Speech inactive, new DOA 149 arrives -> target MUST NOT change
    loc.update(doa_raw=149.0, voice_activity=False, timestamp=1.5)
    assert loc.is_tracking() is True  # still holding within 1.2s timeout
    assert loc.target_yaw_deg == 60.0


# 9. VOICEACTIVITY tekrar true olduğunda DOA tracking devam ediyor
def test_tracking_resumes_when_voice_activity_returns():
    from track import ReSpeakerAudioLocalizer
    loc = ReSpeakerAudioLocalizer(hold_timeout_s=1.2)
    loc.update(doa_raw=32.0, voice_activity=True, timestamp=1.0)
    assert loc.target_yaw_deg == 60.0

    # Hold timeout expires (1.5s > 1.2s)
    loc.update(doa_raw=None, voice_activity=False, timestamp=2.5)
    assert loc.is_tracking() is False
    assert loc.target_yaw_deg == 0.0

    # Speech resumes at same sector (LEFT, DOA 40) → immediate resume
    loc.update(doa_raw=40.0, voice_activity=True, timestamp=3.0)
    assert loc.is_tracking() is True
    assert loc.target_yaw_deg == 60.0
    assert loc.confirmed_sector == "LEFT"

    # Speech resumes at DIFFERENT sector (RIGHT, DOA 142) while idle → immediate resume
    loc.update(doa_raw=None, voice_activity=False, timestamp=4.5)  # timeout
    loc.is_tracking(now=5.0)  # trigger timeout
    assert loc.target_yaw_deg == 0.0

    loc.update(doa_raw=142.0, voice_activity=True, timestamp=5.1)
    assert loc.target_yaw_deg == -60.0  # idle -> accepts immediately!
    assert loc.confirmed_sector == "RIGHT"


# 10. Arka ve invalid DOA -> hiçbir zaman ±180 veya arka hedef üretme
@pytest.mark.parametrize("invalid_doa", [180.0, 200.0, 250.0, 270.0, 285.0, 300.0, 330.0, 350.0, 0.0, 10.0])
def test_rear_and_invalid_doa_never_produce_rear_target(invalid_doa):
    from track import ReSpeakerAudioLocalizer

    # 1. calibrated_yaw arka/invalid değerler için None döner
    assert ReSpeakerAudioLocalizer.calibrated_yaw(invalid_doa) is None

    # 2. doa_to_sector arka/invalid değerler için None döner
    assert ReSpeakerAudioLocalizer.doa_to_sector(invalid_doa) is None

    # 3. Localizer boşta iken geçersiz DOA geldiğinde yeni hedef üretmez
    loc = ReSpeakerAudioLocalizer()
    target = loc.update(doa_raw=invalid_doa, voice_activity=True, timestamp=1.0)
    assert target == 0.0
    assert loc.target_yaw_deg == 0.0
    assert loc.is_tracking() is False


# 10b. Invalid DOA geldiğinde mevcut geçerli audio target kısa süre korunmalı
def test_invalid_doa_retains_valid_target_then_times_out():
    from track import ReSpeakerAudioLocalizer
    loc = ReSpeakerAudioLocalizer(hold_timeout_s=1.2)

    # 1. Geçerli konuşma DOA 142 -> RIGHT sector -> hedef -60°
    loc.update(doa_raw=142.0, voice_activity=True, timestamp=1.0)
    assert loc.is_tracking() is True
    assert loc.target_yaw_deg == -60.0

    # 2. Arka/invalid DOA geldiğinde mevcut hedef -60° korunur
    loc.update(doa_raw=300.0, voice_activity=True, timestamp=1.5)
    assert loc.is_tracking() is True
    assert loc.target_yaw_deg == -60.0

    # 3. Timeout dolduğunda hedef 0°'ye döner
    loc.update(doa_raw=300.0, voice_activity=True, timestamp=2.5)
    assert loc.is_tracking() is False
    assert loc.target_yaw_deg == 0.0


# 10c. Geçerli çalışma aralığı içinde sol/sağ saturation [-90°, +90°] (legacy calibrated_yaw)
def test_valid_workspace_saturation():
    from track import ReSpeakerAudioLocalizer
    assert ReSpeakerAudioLocalizer.calibrated_yaw(22.0) == -90.0
    assert ReSpeakerAudioLocalizer.calibrated_yaw(25.0) == -90.0
    assert ReSpeakerAudioLocalizer.calibrated_yaw(149.0) == 90.0
    assert ReSpeakerAudioLocalizer.calibrated_yaw(152.0) == 90.0


# 11. eski continuous tracker target_yaw değerleri asla dışarı sızmıyor
def test_continuous_tracker_angles_never_leak():
    from track import ReSpeakerAudioLocalizer
    from tracker import GazeTracker, PrioritySource

    tracker = GazeTracker()
    loc = ReSpeakerAudioLocalizer()

    result = tracker.step(
        faces=[],
        frame_size=(640, 480),
        doa_deg=None,
        measured_head_deg=0.0,
        timestamp=10.0,
        speech=None,
    )
    assert result.owner == PrioritySource.IDLE
    assert result.target_yaw_deg == 0.0

    # Audio target comes exclusively from ReSpeakerAudioLocalizer (sector-based)
    loc.update(doa_raw=32.0, voice_activity=True, timestamp=10.0)
    assert loc.target_yaw_deg == 60.0


# 12. audio target tek authoritative kaynak
def test_audio_target_sole_authoritative_source():
    from track import ReSpeakerAudioLocalizer
    from tracker import GazeResult, PrioritySource, GazeStateEnum, Detection

    loc = ReSpeakerAudioLocalizer(hold_timeout_s=1.2)

    # Step 1: Speech active -> LEFT sector
    loc.update(doa_raw=32.0, voice_activity=True, timestamp=1.0)
    assert loc.is_tracking() is True
    target_yaw = loc.target_yaw_deg
    assert target_yaw == 60.0

    # Step 2: Vision detected -> localizer drops audio tracking
    det = Detection(x=300, y=200, w=80, h=80, confidence=0.95)
    vision_result = GazeResult(
        target_yaw_deg=18.5,
        gaze_state=GazeStateEnum.TRACKING,
        owner=PrioritySource.VISUAL_TRACKING,
        target_id="person_face_1",
        confidence=0.95,
        head_angle_deg=0.0,
    )
    if vision_result.owner == PrioritySource.VISUAL_TRACKING:
        loc.on_vision_active()
        head_target = vision_result.target_yaw_deg

    assert head_target == 18.5
    assert loc.is_tracking() is False
    assert loc.target_yaw_deg == 0.0


# ── Legacy static method tests ────────────────────────────────────

# 5. circular wrap: 358, 359, 0, 1 -> ortalama ~0
def test_circular_wrap_mean_near_zero():
    from track import ReSpeakerAudioLocalizer
    angles = [358.0, 359.0, 0.0, 1.0]
    mean = ReSpeakerAudioLocalizer.circular_mean(angles)
    dist = ReSpeakerAudioLocalizer.circular_dist(mean, 0.0)
    assert dist < 1.0


# 6. tek outlier reddi (legacy)
def test_single_outlier_rejection():
    from track import ReSpeakerAudioLocalizer
    loc = ReSpeakerAudioLocalizer(outlier_threshold_deg=30.0)
    assert loc.reject_outlier(69.0) is False
    assert loc.reject_outlier(70.0) is False
    assert loc.reject_outlier(71.0) is False
    assert loc.reject_outlier(140.0) is True
    assert loc.reject_outlier(70.0) is False
    assert loc.filtered_doa == pytest.approx(70.0, abs=2.0)
    assert abs(loc.filtered_doa - 140.0) > 50.0


# 7. deadband (legacy)
def test_deadband_suppresses_small_changes():
    from track import ReSpeakerAudioLocalizer
    loc = ReSpeakerAudioLocalizer(deadband_deg=5.0)
    loc.active_target_yaw = -45.0
    assert loc.apply_deadband(-43.0) == -45.0
    assert loc.apply_deadband(-47.0) == -45.0
    assert loc.apply_deadband(-41.0) == -45.0
    assert loc.apply_deadband(-39.0) == -39.0
    assert loc.active_target_yaw == -39.0


# ── HID integration tests (sector-based) ──────────────────────────

def test_mock_hid_voice_activity_and_doa_mapping():
    """Mock HID sector-based mapping:
    - voice_activity() -> True, doa_angle() -> 142 => RIGHT sector, target -60
    - voice_activity() -> True, doa_angle() -> 32  => LEFT sector, target +60
    - voice_activity() -> False, doa_angle() -> 149 => target NOT changed
    """
    from track import ReSpeakerAudioLocalizer

    mock_hid = MagicMock(spec=["voice_activity", "doa_angle"])
    localizer = ReSpeakerAudioLocalizer(hid=mock_hid, hold_timeout_s=1.2)

    # 1. voice_activity() -> True, doa_angle() -> 142 => RIGHT sector, -60
    mock_hid.voice_activity.return_value = True
    mock_hid.doa_angle.return_value = 142.0
    target_1 = localizer.read_and_update(now=1.0)
    assert localizer.is_tracking() is True
    assert target_1 == pytest.approx(-60.0, abs=1.0)
    assert localizer.confirmed_sector == "RIGHT"

    # 2. voice_activity() -> True, doa_angle() -> 32 => LEFT sector
    # Need SECTOR_CONFIRM_COUNT (2) consecutive readings to switch while tracking
    mock_hid.voice_activity.return_value = True
    mock_hid.doa_angle.return_value = 32.0
    localizer.read_and_update(now=1.1)
    target_2 = localizer.read_and_update(now=1.2)
    assert localizer.is_tracking() is True
    assert target_2 == pytest.approx(60.0, abs=1.0)
    assert localizer.confirmed_sector == "LEFT"

    # 3. voice_activity() -> False => target stays at +60
    mock_hid.voice_activity.return_value = False
    mock_hid.doa_angle.return_value = 149.0
    target_3 = localizer.read_and_update(now=1.5)
    assert target_3 == pytest.approx(60.0, abs=1.0)


def test_mock_hid_speech_detected_compatibility():
    from track import ReSpeakerAudioLocalizer

    mock_hid = MagicMock(spec=["speech_detected", "doa_angle"])
    mock_hid.speech_detected.return_value = True
    mock_hid.doa_angle.return_value = 142.0
    localizer = ReSpeakerAudioLocalizer(hid=mock_hid)

    target = localizer.read_and_update(now=1.0)
    assert localizer.is_tracking() is True
    assert target == pytest.approx(-60.0, abs=1.0)


def test_hardware_vad_unreadable_disables_tracking():
    from track import ReSpeakerAudioLocalizer

    mock_hid = MagicMock(spec=["voice_activity", "doa_angle"])
    mock_hid.voice_activity.return_value = None
    mock_hid.doa_angle.return_value = 69.0
    localizer = ReSpeakerAudioLocalizer(hid=mock_hid)

    target = localizer.read_and_update(now=1.0)
    assert localizer.is_tracking() is False
    assert target == 0.0


def test_audio_source_not_used_for_target_calculation():
    """Audio target hesabında AudioSource kullanılmadığını kanıtlar."""
    from track import ReSpeakerAudioLocalizer

    mock_audio_source = MagicMock()
    mock_audio_source.latest_doa_deg.return_value = 180.0
    mock_audio_source.latest_speech.return_value = MagicMock(is_speech=True)

    mock_hid = MagicMock(spec=["voice_activity", "doa_angle"])
    mock_hid.voice_activity.return_value = True
    mock_hid.doa_angle.return_value = 78.0

    localizer = ReSpeakerAudioLocalizer(hid=mock_hid)
    target = localizer.read_and_update(now=1.0)
    assert target == pytest.approx(0.0, abs=1.0)

    assert not mock_audio_source.latest_doa_deg.called
    assert not mock_audio_source.latest_speech.called


def test_track_main_uses_hid_voice_activity():
    mock_hid = MagicMock(spec=["voice_activity", "doa_angle"])
    mock_hid.voice_activity.return_value = True
    mock_hid.doa_angle.return_value = 78.0
    with patch.object(track, "CameraSource", _SabitKamera), \
            patch.object(track, "AudioSource", _SessizKaynak):
        code = track.main(["--fixed-head", "--no-window", "--no-voice", "--seconds", "0.05"], hid=mock_hid)
        assert code == 0
    assert mock_hid.voice_activity.called


# ── main() integration: sector-based audio log ────────────────────

def test_track_main_sends_sector_target_and_logs():
    """main() döngüsünde audio aktifken sector-based target gönderildiğini
    ve 'AUDIO sector=... DOA=... target=...' logunun basıldığını doğrular.
    """
    import io
    import contextlib
    import track

    class _BosKamera(_SabitKamera):
        def detect(self, frame):
            return []

    mock_hid = MagicMock(spec=["voice_activity", "doa_angle"])
    mock_hid.voice_activity.return_value = True
    # DOA 32 -> LEFT sector -> target +60 (robot's physical right)
    mock_hid.doa_angle.return_value = 32.0

    mock_head = MagicMock()
    mock_head.has_feedback = False

    cikti = io.StringIO()
    with patch.object(track, "CameraSource", _BosKamera), \
         patch.object(track, "AudioSource", _SessizKaynak), \
         patch.object(track, "HeadLink", return_value=mock_head), \
         contextlib.redirect_stdout(cikti):
        code = track.main(
            ["--fixed-head", "--no-window", "--no-voice", "--seconds", "0.05"],
            hid=mock_hid
        )
        assert code == 0

    assert mock_head.send_angle.called
    sent_angles = [call[0][0] for call in mock_head.send_angle.call_args_list]
    assert any(pytest.approx(60.0, abs=1.0) == a for a in sent_angles)
    assert "AUDIO sector=LEFT DOA=32 target=+60.0" in cikti.getvalue()


# ── DOA noise resilience ──────────────────────────────────────────

def test_doa_noise_does_not_cause_opposite_turn():
    """The original bug: DOA ~148 with user in front caused +83° RIGHT turn.
    With sectors, DOA 148 maps to RIGHT (-60° = robot's physical left), and noise can't cause
    the old 80°+ wrong-direction turns.
    """
    from track import ReSpeakerAudioLocalizer
    loc = ReSpeakerAudioLocalizer()

    # DOA 148 → RIGHT sector → target -60 (robot's physical left)
    loc.update(doa_raw=148.0, voice_activity=True, timestamp=1.0)
    assert loc.target_yaw_deg == -60.0
    assert loc.confirmed_sector == "RIGHT"

    # DOA 32 → would be LEFT, but needs 3 consecutive confirmations
    loc.update(doa_raw=32.0, voice_activity=True, timestamp=1.1)
    assert loc.target_yaw_deg == -60.0  # still RIGHT

    # DOA bounces back to RIGHT → resets pending
    loc.update(doa_raw=140.0, voice_activity=True, timestamp=1.2)
    assert loc.target_yaw_deg == -60.0  # still RIGHT


# ── Visual Tracking Coasting & Resilience ─────────────────────────

def test_visual_tracking_coasts_through_face_dropout_in_main():
    """Görsel takip sırasında yüz birkaç kare algılanamadığında (hareket bulanıklığı vb.),
    sistem hemen motor_yaw=0.0 göndermemeli veya audio'ya geçmemeli;
    hedef açısını korumalıdır (HOLDING_ATTENTION / TARGET_LOST coasting).
    """
    class _CoastingKamera:
        available = True
        backend = "webcam"
        detector_name = "test"
        def __init__(self, **kwargs):
            self.frame_idx = 0
            self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        def read(self):
            self.frame_idx += 1
            return self.frame_idx <= 12, self.frame
        def detect(self, frame):
            # İlk 5 kare: yüz sağda (x=400)
            if self.frame_idx <= 5:
                return [Detection(x=400, y=240, w=100, h=100, confidence=0.88)]
            # Sonraki 7 kare: yüz geçici olarak algılanamadı
            return []
        def close(self):
            pass

    mock_hid = MagicMock(spec=["voice_activity", "doa_angle"])
    mock_hid.voice_activity.return_value = True
    mock_hid.doa_angle.return_value = 149.0  # RIGHT sector -> -60.0

    mock_head = MagicMock()
    mock_head.has_feedback = False

    cikti = io.StringIO()
    with patch.object(track, "CameraSource", _CoastingKamera), \
         patch.object(track, "AudioSource", _SessizKaynak), \
         patch.object(track, "HeadLink", return_value=mock_head), \
         contextlib.redirect_stdout(cikti):
        code = track.main(
            ["--fixed-head", "--no-window", "--no-voice", "--seconds", "1.0"],
            hid=mock_hid
        )
        assert code == 0

    sent_angles = [call[0][0] for call in mock_head.send_angle.call_args_list]
    assert len(sent_angles) == 12
    # Yüz kaybolduğunda (kare 6..12) motor 0.0'a fırlamamalı ve audio (-60°) araya girmemeli
    for angle in sent_angles[5:]:
        assert angle != 0.0, f"Açı sıfıra fırladı: {angle}"
        assert angle != -60.0, f"Ses görsel takibi böldü: {angle}"
        assert pytest.approx(-13.3, abs=1.0) == angle

