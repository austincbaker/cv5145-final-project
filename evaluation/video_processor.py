import cv2
from pathlib import Path
from PIL import Image
from dataclasses import dataclass


@dataclass
class VideoProcessorConfig:
    num_frames: int = 8
    frame_size: tuple[int, int] | None = None


class VideoProcessor:
    def __init__(self, config: VideoProcessorConfig | None = None):
        self.config = config or VideoProcessorConfig()

    def extract_frames(self, video_path: str | Path) -> list[Image.Image]:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise RuntimeError(f"Video has no frames: {video_path}")

            frame_indices = self._compute_frame_indices(total_frames)
            frames = []

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                if self.config.frame_size:
                    pil_image = pil_image.resize(self.config.frame_size)

                frames.append(pil_image)

            if not frames:
                raise RuntimeError(f"Failed to extract frames from: {video_path}")

            return frames

        finally:
            cap.release()

    def _compute_frame_indices(self, total_frames: int) -> list[int]:
        num_frames = min(self.config.num_frames, total_frames)

        if num_frames == 1:
            return [total_frames // 2]

        if num_frames >= total_frames:
            return list(range(total_frames))

        step = (total_frames - 1) / (num_frames - 1)
        indices = [int(round(i * step)) for i in range(num_frames)]

        return indices

    def get_video_info(self, video_path: str | Path) -> dict:
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return {"error": f"Failed to open: {video_path}"}

        try:
            info = {
                "path": str(video_path),
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            if info["fps"] > 0:
                info["duration_seconds"] = info["total_frames"] / info["fps"]
            return info
        finally:
            cap.release()