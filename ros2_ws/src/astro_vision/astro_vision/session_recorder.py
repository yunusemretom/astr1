#!/usr/bin/env python3
"""ASTRO — hata ayıklama oturumu kaydedici.

Takibin kesildiği an robotun üstünde oluyor ve masaüstünde tekrar üretilemedi, bu
yüzden koşuyu geri getirmek gerekiyor. Tek bir dosya üç soruyu birden cevaplamalı:
kamera ne gördü, dedektör neyi yüz saydı, gaze yığını neyi takip ediyordu.

"Kafa dönmedi" üç ayrı arızaya karşılık gelebilir ve videoda yan yana görünmedikçe
ayırt edilemezler:

  - kutu yok            -> algılama sorunu
  - kutu var, owner IDLE -> arbitrasyon sorunu
  - istenen açı değişiyor ama gerçek açı takip etmiyor -> aktüatör sorunu

Kullanım (sistem çalışırken ayrı bir terminalde):

    ros2 run astro_vision session_recorder
    ros2 run astro_vision session_recorder --seconds 60
"""

import argparse
import json
import os
import shutil
import time
from datetime import datetime

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
except ImportError:
    try:
        from astro_vision.ros_compat import MockMsg, MockNode as Node, MockRclpy
    except ImportError:
        from ros_compat import MockMsg, MockNode as Node, MockRclpy
    rclpy = MockRclpy()
    qos_profile_sensor_data = 10
    Image = String = MockMsg

try:
    from astro_vision.image_utils import imgmsg_to_bgr
except ImportError:
    from image_utils import imgmsg_to_bgr


DEFAULT_OUTPUT_DIR = os.path.expanduser("~/astro_recordings")
# MJPG at quality 95 runs about 120-150 MB per minute at 640x480/30 Hz, and the
# robot's disk sits near full, so refuse to start rather than fill it mid-run.
MIN_FREE_BYTES = 2 * 1024 ** 3
RECORD_FPS = 30.0


def default_dds_profile() -> str | None:
    """astro_vision'ın kurduğu paylaşımlı bellek profilinin yolu (yoksa None).

    /vision/face_image 900 KB'lık bir topic ve Linux'un varsayılan 208 KB'lık UDP
    alım tamponu onun çoğunu düşürüyor: dedektör 30 Hz yayınlarken ölçülen 2.1 Hz.
    camera.launch.py yayıncıyı zaten paylaşımlı belleğe alıyor; eşleşmeyen bir
    kaydedici koşunun onda birini kaydedip bunu hiç söylemez.
    """
    try:
        from ament_index_python.packages import get_package_share_directory

        path = os.path.join(get_package_share_directory("astro_vision"), "config", "fastdds_shm.xml")
    except Exception:
        return None
    return path if os.path.exists(path) else None


class NotEnoughDiskSpace(RuntimeError):
    """Kayda başlamadan önce yeterli boş alan yok."""


def require_free_space(path: str, min_free_bytes: int = MIN_FREE_BYTES) -> int:
    """Yeterli boş alan yoksa hiçbir kare yazmadan hata verir."""
    free = shutil.disk_usage(path).free
    if free < min_free_bytes:
        raise NotEnoughDiskSpace(
            f"{path} üzerinde {free / 1024 ** 3:.1f} GB boş var, "
            f"en az {min_free_bytes / 1024 ** 3:.1f} GB gerekli"
        )
    return free


def format_status_lines(gaze: dict | None) -> tuple[str, str]:
    """Kareye basılacak iki satırlık durum şeridi.

    İkinci satır istenen ve gerçek açıyı yan yana koyar: arbitrasyonun mu yoksa
    aktüatörün mü sustuğunu ayıran tek bilgi bu.
    """
    if not gaze:
        return "gaze: veri yok (social_gaze_node calismiyor mu?)", ""

    target = gaze.get("active_target_id") or "-"
    first = (
        f"{gaze.get('gaze_state', '?')}  owner={gaze.get('attention_owner', '?')}  "
        f"hedef={target}  conf={float(gaze.get('target_confidence', 0.0)):.2f}"
    )
    second = (
        f"istenen {float(gaze.get('desired_yaw_deg', 0.0)):+.1f}  ->  "
        f"gercek {float(gaze.get('actual_yaw_deg', 0.0)):+.1f}"
    )
    return first, second


MIN_FONT_SCALE = 0.25


def fit_text(text: str, max_width: int, font=cv2.FONT_HERSHEY_SIMPLEX) -> tuple[str, float]:
    """Metni verilen genişliğe sığdırır: önce küçültür, gerekirse kısaltır.

    Taşan bir satır sessizce kesilir ve hangi bilginin kaybolduğu belli olmaz; ilk
    kayıtta conf değeri böyle gitmişti. Kısaltma hiç değilse görünür.
    """
    scale = 0.55
    while scale > MIN_FONT_SCALE and cv2.getTextSize(text, font, scale, 1)[0][0] > max_width:
        scale -= 0.05

    if cv2.getTextSize(text, font, scale, 1)[0][0] <= max_width:
        return text, scale

    base = text
    while base and cv2.getTextSize(base + "..", font, scale, 1)[0][0] > max_width:
        base = base[:-1]
    return (base + "..") if base else "", scale


def draw_status(frame: np.ndarray, lines: tuple[str, str]) -> np.ndarray:
    """Şeridi karenin altına, okunur bir bant üzerine çizer.

    Yazı kareye sığacak şekilde küçültülür: ilk kayıtta şerit sağ kenardan taşıp
    conf değerini kesmişti — görünen bir kutunun kafayı almaya yetip yetmediğini
    söyleyen tek sayı oydu.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    band_h = 54
    margin = 8
    cv2.rectangle(out, (0, h - band_h), (w, h), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        if not text:
            continue
        text, scale = fit_text(text, w - 2 * margin)
        if not text:
            continue
        cv2.putText(
            out, text, (margin, h - band_h + 22 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 120), 1, cv2.LINE_AA,
        )
    return out


class SessionRecorder(Node):
    """Açıklamalı kamera akışını gaze durumuyla birlikte tek videoya yazar."""

    def __init__(self, output_dir: str = None, seconds: float = None):
        super().__init__("session_recorder")
        base = output_dir or DEFAULT_OUTPUT_DIR
        os.makedirs(base, exist_ok=True)
        require_free_space(base)

        self.session_dir = os.path.join(base, datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.session_dir, exist_ok=True)
        self.video_path = os.path.join(self.session_dir, "session.avi")

        self.seconds = seconds
        self.started_at = time.monotonic()
        self.frames = 0
        self._writer = None
        self._latest_gaze = None

        # Dedektörün kutuları ve etiketleri zaten çizdiği akış; ham kamera değil.
        self.create_subscription(Image, "/vision/face_image", self._on_image, qos_profile_sensor_data)
        self.create_subscription(String, "/gaze/debug", self._on_gaze, 10)

        self.get_logger().info(f"⏺️  Kayıt: {self.video_path}")

    def _on_gaze(self, msg) -> None:
        try:
            self._latest_gaze = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().debug(f"_on_gaze: yok sayılan hata ({exc})")

    def _on_image(self, msg) -> None:
        try:
            frame = imgmsg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warn(f"Kare çözülemedi: {exc}")
            return

        annotated = draw_status(frame, format_status_lines(self._latest_gaze))

        if self._writer is None:
            h, w = annotated.shape[:2]
            self._writer = cv2.VideoWriter(
                self.video_path, cv2.VideoWriter_fourcc(*"MJPG"), RECORD_FPS, (w, h)
            )
            self._writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)

        self._writer.write(annotated)
        self.frames += 1

        if self.frames % 150 == 0:
            size_mb = os.path.getsize(self.video_path) / 1024 ** 2
            elapsed = time.monotonic() - self.started_at
            self.get_logger().info(
                f"⏺️  {self.frames} kare | {elapsed:.0f} sn | {size_mb:.0f} MB "
                f"({size_mb / max(elapsed, 1e-6) * 60:.0f} MB/dk)"
            )

    def is_finished(self) -> bool:
        return self.seconds is not None and (time.monotonic() - self.started_at) >= self.seconds

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self.frames:
            size_mb = os.path.getsize(self.video_path) / 1024 ** 2
            self.get_logger().info(f"✅ {self.frames} kare, {size_mb:.0f} MB → {self.video_path}")
        else:
            self.get_logger().warn(
                "Hiç kare gelmedi — /vision/face_image yayınlanıyor mu? "
                "(face_detector_node çalışıyor olmalı)"
            )


def main(args=None):
    parser = argparse.ArgumentParser(description="ASTRO hata ayıklama oturumu kaydedici")
    parser.add_argument("--seconds", type=float, default=None, help="Kayıt süresi (sn)")
    parser.add_argument("--output-dir", default=None, help=f"Çıktı dizini (varsayılan {DEFAULT_OUTPUT_DIR})")
    parser.add_argument(
        "--no-dds-profile", action="store_true",
        help="Paylaşımlı bellek profilini uygulama (yayıncı da kullanmıyorsa)",
    )
    opts, ros_args = parser.parse_known_args(args)

    # Yayıncıyla aynı taşımada olmak zorunlu; aksi hâlde kayıt sessizce 2 Hz olur.
    if not opts.no_dds_profile and not os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE"):
        profile = default_dds_profile()
        if profile:
            os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = profile
            print(f"ℹ️  DDS profili: {profile}")

    rclpy.init(args=ros_args)
    try:
        node = SessionRecorder(output_dir=opts.output_dir, seconds=opts.seconds)
    except NotEnoughDiskSpace as exc:
        print(f"❌ {exc}")
        rclpy.shutdown()
        return

    try:
        while rclpy.ok() and not node.is_finished():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
