"""8 Eylül saha özetinin temsilî örnekleri; 49 ham kaydın tekrarı değildir."""

from types import SimpleNamespace

import pytest

from astro_base.gaze.coordinate_frames import CoordinateTransformer
from astro_base.gaze.respeaker_sectors import ReSpeakerEyeSectors
from astro_base.gaze.types import PrioritySource
from astro_base.standalone_gaze_ros_node import StandaloneGazeRosNode
from astro_base.gaze.gaze_tracker import Detection


@pytest.mark.parametrize("raw,expected", [(0, 0), (40, -40), (320, 40)])
def test_geometrik_donusum_montaj_sektorlerinden_bagimsiz(raw, expected):
    assert CoordinateTransformer().raw_audio_doa_to_head_bearing(raw) == expected


@pytest.mark.parametrize("raw,expected", [(35, 55), (52, 55), (68, 0), (141, -55), (147, -55)])
def test_olculen_medyanlar_uc_yeni_ornekten_sonra_dogru_yone(raw, expected):
    mapper = ReSpeakerEyeSectors()
    assert mapper.update(raw, 10.0) is None
    assert mapper.update(raw, 10.1) is None
    doa = mapper.update(raw, 10.2)
    assert CoordinateTransformer().raw_audio_doa_to_head_bearing(doa) == expected


@pytest.mark.parametrize("raw", [0, 40, 145, 290, 300, 322, 330, 180, float("nan"), float("inf")])
def test_olculen_aykiri_ve_belirsiz_arka_yonler_hedef_uretmez(raw):
    mapper = ReSpeakerEyeSectors()
    for i in range(5):
        assert mapper.update(raw, 20.0 + i * 0.1) is None


def test_ayni_ornek_tekrari_ve_sessizlik_israr_sayilmaz():
    mapper = ReSpeakerEyeSectors()
    for _ in range(10):
        assert mapper.update(141, 10.0) is None
    assert mapper.update(141, 12.0) is None
    assert mapper.update(141, 12.1) is None
    assert mapper.update(141, 12.2) == 55.0
    assert mapper.update(322, 12.3) is None
    assert mapper.update(141, 12.4) is None


@pytest.mark.parametrize("raw,sign", [(35, 1), (52, 1), (141, -1), (147, -1)])
def test_ros_topicten_idle_donus_ve_yuz_gorununce_goruntu_onceligi(monkeypatch, raw, sign):
    node = StandaloneGazeRosNode(
        use_camera_source=False, enable_audio=True, enable_voice=False,
        audio_source_mode="topics", audio_doa_profile="respeaker_eye_20260908",
    )
    now = [100.0]
    monkeypatch.setattr("astro_base.standalone_gaze_ros_node.time.monotonic", lambda: now[0])
    try:
        node._on_audio_doa_conf(SimpleNamespace(data=0.60))
        for i in range(6):
            now[0] += 0.1
            # Sabit sensör sahnesi: encoder da sabit 0 bildirir. Açık çevrim
            # motor tahmini kullanmak, bu sabit DOA/yüz sahnesini tutarsız yapar.
            node._on_head_state(SimpleNamespace(position_deg=0.0, velocity_deg_s=0.0))
            node._on_audio_vad(SimpleNamespace(data=True))
            node._on_audio_doa(SimpleNamespace(data=raw))
            doa, speech, speaking = node._sample_acoustic_state(now[0])
            res = node.step_frame([], (640, 480), timestamp=now[0], doa_deg=doa, speech=speech)
            if i < 2:
                assert res.owner != PrioritySource.ACTIVE_SPEAKER
        assert res.owner == PrioritySource.ACTIVE_SPEAKER
        assert sign * res.target_yaw_deg > 20.0
        # Görünen yüz sesin tersinde: ses kamera hedefini çekmemeli.
        face = Detection(x=280, y=200, w=80, h=80, confidence=0.95)
        for i in range(4):
            now[0] += 0.1
            node._on_head_state(SimpleNamespace(position_deg=0.0, velocity_deg_s=0.0))
            res = node.step_frame([face], (640, 480), timestamp=now[0], doa_deg=doa, speech=speech)
        assert res.owner == PrioritySource.VISUAL_TRACKING
        assert abs(res.target_yaw_deg) < 5.0
        node._on_audio_vad(SimpleNamespace(data=False))
        node._on_audio_vad(SimpleNamespace(data=True))
        node._on_audio_doa(SimpleNamespace(data=raw))
        assert node._sample_acoustic_state(now[0])[0] is None
    finally:
        node.destroy_node()
