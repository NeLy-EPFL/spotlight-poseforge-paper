from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from sppaper.common.resources import get_outputs_dir
from sppaper.common.plot import find_font_path

OUTPUT_CODEC = "libx265"
CODEC_OPTIONS = {
    "crf": "20",  # high quality
    "preset": "veryslow",  # best compression (slowest encoding speed)
    "x265-params": "log-level=error",
}
OUTPUT_FPS = 66  # 0.2x speed
PANEL_WIDTH_PX = 1472
FONT_LARGE = ImageFont.truetype(find_font_path("Arial", weight="bold"), 50)
FONT_SMALL = ImageFont.truetype(find_font_path("Arial"), 40)


def add_text_to_frame(frame_array: np.ndarray) -> np.ndarray:
    img = Image.fromarray(frame_array)
    draw = ImageDraw.Draw(img)

    draw.text(
        (60, 50),
        "Behavior recording",
        fill=(255, 255, 255),
        font=FONT_LARGE,
    )
    draw.text(
        (60, 110),
        "Recorded at 330 Hz",
        fill=(255, 255, 255),
        font=FONT_SMALL,
    )
    draw.text(
        (60, 155),
        "Played at 0.2x real-time",
        fill=(255, 255, 255),
        font=FONT_SMALL,
    )
    draw.text(
        (PANEL_WIDTH_PX + 60, 50),
        "Muscle imaging (GCaMP8m)",
        fill=(255, 255, 255),
        font=FONT_LARGE,
    )
    draw.text(
        (PANEL_WIDTH_PX + 60, 110),
        "Recorded at 33 Hz",
        fill=(255, 255, 255),
        font=FONT_SMALL,
    )
    draw.text(
        (PANEL_WIDTH_PX + 60, 155),
        "Played at 0.2x real-time",
        fill=(255, 255, 255),
        font=FONT_SMALL,
    )

    return np.array(img)


def process_video(input_path, output_path, output_nframes):
    with av.open(str(input_path)) as in_container:
        in_stream = in_container.streams.video[0]
        in_stream.thread_type = "AUTO"  # enable ffmpeg multithreading for decoding

        with av.open(str(output_path), mode="w") as out_container:
            out_stream = out_container.add_stream(OUTPUT_CODEC, rate=OUTPUT_FPS)
            out_stream.pix_fmt = "yuv420p"
            out_stream.thread_type = "AUTO"  # multithreaded encoding
            out_stream.options = CODEC_OPTIONS

            # Copy width/height from input
            out_stream.width = in_stream.width
            out_stream.height = in_stream.height

            frame_count = 0
            with tqdm(total=output_nframes) as pbar:
                for packet in in_container.demux(in_stream):
                    for frame in packet.decode():
                        if frame_count >= output_nframes:
                            break

                        # Convert to RGB numpy array, annotate, convert back
                        rgb = frame.to_ndarray(format="rgb24")
                        rgb = add_text_to_frame(rgb)

                        out_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                        for out_packet in out_stream.encode(out_frame):
                            out_container.mux(out_packet)

                        frame_count += 1
                        pbar.update(1)

                    if frame_count >= output_nframes:
                        break

            # Flush encoder
            for out_packet in out_stream.encode():
                out_container.mux(out_packet)

    print(f"Done. {frame_count} frames saved to {output_path}")


if __name__ == "__main__":
    input_path = Path(
        "~/data/spotlight/20250613-fly1b-012/processed/fullsize_summary_video.mp4"
    ).expanduser()
    output_path = get_outputs_dir() / "spotlight_data_sample/sample_fullsize.mp4"
    output_nframes = 10 * 330  # 10 sec at 330 FPS

    output_path.parent.mkdir(parents=True, exist_ok=True)

    process_video(input_path, output_path, output_nframes)
