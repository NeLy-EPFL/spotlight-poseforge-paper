import av
import numpy as np

def load_precise_sparse_frames(
    path, frames: list[int] | None = None
) -> list[np.ndarray]:
    """Efficiently load a set of frames from a video file (with precise seeking).
    If frames is None, all frames are read. Frames must be sorted in ascending order.
    """
    SEEK_THRESHOLD = 50

    container = av.open(path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    time_base = float(stream.time_base)

    def pts_to_index(pts):
        return round(pts * time_base * fps)

    def seek_to(target: int):
        timestamp = int(target / fps / time_base)
        container.seek(timestamp, stream=stream, backward=True, any_frame=False)

    if frames is None:
        result = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
        container.close()
        return result

    results = {}
    frame_idx = -1
    target_iter = iter(frames)
    target = next(target_iter, None)

    while target is not None:
        if frame_idx < 0 or target - frame_idx > SEEK_THRESHOLD:
            seek_to(target)
            frame_idx = -1

        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame.pts is None:
                    continue
                frame_idx = pts_to_index(frame.pts)
                if frame_idx < target:
                    continue
                if frame_idx == target:
                    results[target] = frame.to_ndarray(format="rgb24")
                    target = next(target_iter, None)
                if target is None or target - frame_idx > SEEK_THRESHOLD:
                    break
            else:
                continue
            break

    container.close()
    return [results[f] for f in frames]