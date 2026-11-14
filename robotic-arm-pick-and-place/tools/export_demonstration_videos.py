"""Export recorded front/wrist frames without resimulating or changing datasets.

MP4s play at the recorded policy rate, then hold the last frame for two seconds.
An episode with no executed actions produces a labeled still image instead.
Requires ffmpeg on PATH; uses the repository's h5py/numpy/Pillow dependencies.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.demonstrations import validate_episode

WIDTH, HEIGHT = 1280, 688


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def export_episode(path, destination):
    validation = validate_episode(path)
    with h5py.File(path, "r") as episode:
        meta = validation["metadata"]
        rate = json.loads(episode.attrs["action_contract"])["policy_hz"]
        image_sizes = {name: list(episode[f"observations/rgb/{name}"].shape[1:3])
                       for name in ("front", "wrist")}
        count = validation["transitions"]
        outcome = "SUCCESS" if validation["success"] else "FAILED"
        scenario = meta["recovery"] if meta["recovery"] != "none" else "nominal"
        if meta.get("asset_set", "nominal") != "nominal":
            scenario = f"{meta['asset_set']}_{scenario}"
        if meta["environment"] == "bookshelf-hard":
            scenario = f"{meta['layout']}_{scenario}"
        stem = f"{meta['environment']}_{meta['task_id']}_{scenario}_seed{meta['seed']}"
        output = destination / (stem + (".mp4" if count else "_no_motion.png"))
        if output.exists():
            raise FileExistsError(output)
        heading, regular, small = font(23), font(21), font(18)
        stages = episode["transitions/stage"].asstr()[:] if count else []
        sources = episode["transitions/source"].asstr()[:] if count else []

        def frame(index, *, held=False):
            canvas = Image.new("RGB", (WIDTH, HEIGHT), "#131b27")
            draw = ImageDraw.Draw(canvas)
            title = f"Forte | {meta['task_id'].replace('_', ' ')} | {scenario.replace('_', ' ')}"
            draw.text((16, 5), title, font=heading, fill="#f4f6fa")
            for camera, x in (("front", 0), ("wrist", 640)):
                image = Image.fromarray(episode[f"observations/rgb/{camera}"][index])
                h, w = image_sizes[camera]
                draw.text((x+16, 36), f"{camera.upper()} CAMERA | source {w}x{h}", font=small, fill="#a9bdcf")
                # Native 640x480 frames are copied exactly. Other capture sizes
                # retain aspect ratio; the source label makes any scaling explicit.
                if image.size != (640, 480):
                    image = ImageOps.contain(image, (640, 480), Image.Resampling.LANCZOS)
                canvas.paste(image, (x+(640-image.width)//2, 64+(480-image.height)//2))
            seconds = episode["observations/time_s"][index]
            stage = "initial state" if index == 0 else stages[index-1].replace("_", " ")
            active = index > 0 and sources[index-1] == "perturbed"
            status = "FINAL FRAME HELD" if held else ("GRIPPER FAULT ACTIVE" if active else "1x playback")
            draw.text((16, 551), f"t = {seconds:5.2f} s | Last action: {stage} | {status}",
                      font=regular, fill="#ffc063" if active else "#f4f6fa")
            for line, text in enumerate(textwrap.wrap(episode.attrs["instruction"], width=117)):
                draw.text((16, 581+line*22), text, font=small, fill="#c6d3e1")
            explanation = "Recorded outcome: " + outcome
            if not count:
                explanation += " - planning failed before any action; no motion recorded"
            elif index == count and not validation["success"]:
                explanation += " - " + episode.attrs["collector_error"]
            draw.text((16, 650), explanation, font=small,
                      fill="#7ee3b4" if validation["success"] else "#ff9a9a")
            return canvas

        if count:
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
                       "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}",
                       "-framerate", str(rate), "-i", "pipe:0", "-an", "-c:v", "libx264",
                       "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart", str(output)]
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
            try:
                for index in range(count+1):
                    encoder.stdin.write(np.asarray(frame(index)).tobytes())
                final = np.asarray(frame(count, held=True)).tobytes()
                for _ in range(round(2*rate)):
                    encoder.stdin.write(final)
            finally:
                encoder.stdin.close()
                status = encoder.wait()
            if status:
                raise RuntimeError(f"ffmpeg exited with status {status}")
        else:
            frame(0).save(output)
        result = dict(source=str(path.resolve()), case_id=meta["case_id"], file=str(output.resolve()),
                      success=validation["success"], actions=count, fps=rate,
                      simulation_seconds=float(episode["observations/time_s"][-1]),
                      final_frame_hold_seconds=2 if count else 0,
                      source_image_size_hw=image_sizes,
                      fault_events=json.loads(episode.attrs["events"]))
        print(json.dumps(result), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        parser.error("ffmpeg is required on PATH")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [export_episode(path, args.output_dir) for path in args.episodes]
    (args.output_dir/"exports.json").write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
