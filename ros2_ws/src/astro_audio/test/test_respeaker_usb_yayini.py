"""Gerçek USB istek sözleşmesi ve topic yayın kapıları; donanım açılmaz."""

import struct
import sys
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from astro_audio.audio_stream_node import AudioStreamNode, ReSpeakerHID
from astro_audio.audio_capture_node import ReSpeakerHID as CaptureReader


class SahteUsb:
    """Seeed tuning.py'nin adresleri: AGC ile konuşma durumu bilerek farklı."""

    def ctrl_transfer(self, request_type, request, value, index, length, timeout):
        assert (request_type, request, length) == (0xC0, 0, 8)
        values = {(0xC0, 19): 0, (0xD6, 19): 1, (0xC0, 21): 141}
        return struct.pack("<ii", values[(value, index)], 0)


@pytest.mark.parametrize("reader_class", [ReSpeakerHID, CaptureReader])
def test_usb_konusma_agc_yerine_offset_22_okur(monkeypatch, reader_class):
    module = sys.modules[reader_class.__module__]
    monkeypatch.setattr(module.usb.core, "find", lambda **kw: SahteUsb())
    reader = reader_class()
    assert reader.speech_detected() is True
    assert reader.doa_angle() == 141.0


def test_gercek_okuyucu_gec_takilan_ve_kopup_gelen_usb_icin_toparlanir(monkeypatch):
    module = sys.modules[ReSpeakerHID.__module__]
    now = [0.0]
    device = SahteUsb()
    find = MagicMock(side_effect=[None, device, device])
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.usb.core, "find", find)
    reader = ReSpeakerHID()
    assert reader.speech_detected() is None
    assert find.call_count == 1
    now[0] = 5.1
    assert reader.speech_detected() is True
    # Eksik cevap açı sıfır diye kabul edilmez; okuma kesilir ve geri denenir.
    monkeypatch.setattr(device, "ctrl_transfer", lambda *args: b"\x00\x00")
    assert reader.doa_angle() is None
    assert reader.dev is None
    assert "2/8" in reader.last_error
    assert reader.doa_angle() is None
    assert find.call_count == 2
    monkeypatch.setattr(device, "ctrl_transfer", SahteUsb().ctrl_transfer)
    now[0] = 10.2
    assert reader.doa_angle() == 141.0
    assert reader.last_error is None


def _kaydeden_yayinci():
    """Sahte std_msgs mesaj sınıfı tekil olabilir; değer yayın anında saklanır."""
    pub = MagicMock()
    pub.gonderilen = []
    pub.publish.side_effect = lambda msg: pub.gonderilen.append(msg.data)
    return pub


@pytest.fixture
def yayin():
    node = object.__new__(AudioStreamNode)
    node._respeaker = MagicMock()
    node._respeaker.speech_detected.return_value = True
    node._respeaker.doa_angle.return_value = 141.0
    node._respeaker.last_error = None
    node._is_playing = False
    node._last_playback_time = 0.0
    node._last_output_chunk_time = 0.0
    node.echo_mute_cooldown_s = 0.65
    node._play_queue = queue.Queue()
    node._output_stream = SimpleNamespace(active=True)
    node.pub_vad = _kaydeden_yayinci()
    node.pub_doa = _kaydeden_yayinci()
    node.pub_doa_confidence = _kaydeden_yayinci()
    node._hid_status = None
    node.get_logger = MagicMock()
    return node


def test_acik_ama_sessiz_dac_yon_yayinini_kesmez(yayin):
    yayin._poll_respeaker_hid()
    assert yayin.pub_vad.gonderilen[-1] is True
    assert yayin.pub_doa.gonderilen == [141.0]


def test_usb_sonradan_takilinca_okuma_yeniden_denenir(yayin):
    yayin._respeaker.dev = None
    yayin._poll_respeaker_hid()
    yayin._respeaker.speech_detected.assert_called_once()
    assert yayin.pub_doa.gonderilen == [141.0]


def test_usb_yokken_vad_false_ama_sahte_aci_yok(yayin):
    yayin._respeaker.speech_detected.return_value = None
    yayin._respeaker.doa_angle.return_value = None
    yayin._poll_respeaker_hid()
    assert yayin.pub_vad.gonderilen[-1] is False
    yayin.pub_doa.publish.assert_not_called()


@pytest.mark.parametrize("neden", ["oynuyor", "kuyruk", "yanki"])
def test_robot_sesi_ve_yanki_suresince_yon_yayinlanmaz(yayin, monkeypatch, neden):
    monkeypatch.setattr("astro_audio.audio_stream_node.time.monotonic", lambda: 100.0)
    yayin._output_stream = None
    if neden == "oynuyor":
        yayin._is_playing = True
    elif neden == "kuyruk":
        yayin._play_queue.put(b"ses")
    else:
        yayin._last_playback_time = 99.7
    yayin._poll_respeaker_hid()
    assert yayin.pub_vad.gonderilen[-1] is False
    yayin.pub_doa.publish.assert_not_called()


@pytest.mark.parametrize("write_error", [False, True])
def test_ilk_uzun_dac_yaziminda_yanki_engeli_acilmaz(yayin, monkeypatch, write_error):
    now = [100.0]
    monkeypatch.setattr("astro_audio.audio_stream_node.time.monotonic", lambda: now[0])
    yayin._last_output_chunk_time = now[0]
    yayin._out_dev_idx = None
    yayin._out_device_name = "sahte DAC"
    yayin._playback_lock = threading.Lock()
    yayin._stop_event = threading.Event()
    yayin._total_played_bytes = 0
    yayin._play_queue.put({"pcm": b"ses", "is_done": True})
    output = MagicMock()

    def write(chunk):
        assert chunk == b"ses"
        assert yayin._play_queue.empty()
        # İlk yazım hâlâ sürüyor, paketin gelişinden sonraki 0.65 s dolmuş.
        now[0] = 100.8
        yayin._poll_respeaker_hid()
        yayin._stop_event.set()
        if write_error:
            raise OSError("DAC bağlantısı koptu")

    output.write.side_effect = write
    monkeypatch.setattr("astro_audio.audio_stream_node.sd.RawOutputStream", lambda **kw: output)
    yayin._playback_worker()
    assert yayin.pub_vad.gonderilen[-1] is False
    yayin.pub_doa.publish.assert_not_called()
    assert yayin._is_playing is False
