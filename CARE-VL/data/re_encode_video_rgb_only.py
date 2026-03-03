import subprocess
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Source and destination directories (update to your paths)
SRC_ROOT = Path("/path/to/source_video_data")
DST_ROOT = Path("/path/to/output_re_encoded")
# 시작 폴더 인덱스 (AI-151 이후만 처리)
START_INDEX = 150
# 병렬 처리 스레드 수
NUM_WORKERS = 4

def get_rgb_stream_index(mkv_path: Path):
    """ffprobe로 RGB 스트림 인덱스를 자동 탐색"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v",
        "-show_entries", "stream=index,codec_name,pix_fmt,width,height:stream_tags=title,handler_name",
        "-of", "json", str(mkv_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    if not streams:
        return None

    rgb_idx = None
    max_area = 0
    for s in streams:
        idx = s.get("index")
        w, h = s.get("width", 0), s.get("height", 0)
        pix = (s.get("pix_fmt") or "").lower()
        tags = " ".join([(s.get("tags", {}) or {}).get(k, "").lower() for k in ("title", "handler_name")])
        codec = (s.get("codec_name") or "").lower()

        # depth/IR 제외
        if any(x in pix for x in ["gray", "mono", "y16"]) or any(x in tags for x in ["depth", "ir"]):
            continue

        # COLOR, RGB, YUV, MJPEG/H264 등 선호
        if any(x in tags for x in ["color", "rgb"]) or "yuv" in pix or codec in ["mjpeg", "h264"]:
            area = w * h
            if area > max_area:
                max_area = area
                rgb_idx = idx

    return rgb_idx

def get_depth_stream_index(mkv_path: Path):
    """ffprobe로 depth 스트림 인덱스 탐색"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v",
        "-show_entries", "stream=index,pix_fmt:stream_tags=title,handler_name",
        "-of", "json", str(mkv_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    if not streams:
        return None

    for s in streams:
        idx = s.get("index")
        pix = (s.get("pix_fmt") or "").lower()
        tags = " ".join([(s.get("tags", {}) or {}).get(k, "").lower() for k in ("title", "handler_name")])
        if "depth" in tags or "ir" in tags or "gray" in pix or "y16" in pix:
            return idx
    return None

def reencode_rgb_only(cap_path: Path):
    """RGB 트랙만 추출해서 새 파일로 저장"""
    # AI 번호 추출
    parts = cap_path.parts
    ai_folder = next((p for p in parts if p.startswith("AI-")), None)
    if not ai_folder:
        return f"⚠️ No AI folder: {cap_path}"
    try:
        ai_idx = int(ai_folder.split("-")[1])
    except Exception:
        return f"⚠️ Invalid AI name: {ai_folder}"
    if ai_idx <= START_INDEX:
        return f"⏩ Skip {ai_folder} (≤ {START_INDEX})"

    rel = cap_path.relative_to(SRC_ROOT)
    out_dir = DST_ROOT / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (cap_path.stem + "_rgb.mp4")

    if out_file.exists():
        return f" Skip(already exists): {out_file.name}"

    rgb_idx = get_rgb_stream_index(cap_path)
    if rgb_idx is None:
        return f"❌ No RGB stream found: {cap_path}"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(cap_path),
        "-map_metadata", "0",
        "-map", f"0:v:{rgb_idx}", "-an",
        "-c:v", "h264_nvenc", "-crf", "20", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(out_file)
    ]
    subprocess.run(cmd, check=True, close_fds=True)
    return f"✅ Done: {cap_path.name}"

def reencode_rgb_depth(cap_path: Path):
    """RGB + Depth를 하나의 MKV로 재인코딩"""
    parts = cap_path.parts
    ai_folder = next((p for p in parts if p.startswith("AI-")), None)
    if not ai_folder:
        return f"⚠️ No AI folder: {cap_path}"
    try:
        ai_idx = int(ai_folder.split("-")[1])
    except Exception:
        return f"⚠️ Invalid AI name: {ai_folder}"
    if ai_idx <= START_INDEX:
        return f"⏩ Skip {ai_folder} (≤ {START_INDEX})"

    rel = cap_path.relative_to(SRC_ROOT)
    out_dir = DST_ROOT / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (cap_path.stem + "_rgbd.mkv")

    if out_file.exists():
        return f" Skip(already exists): {out_file.name}"

    rgb_idx = get_rgb_stream_index(cap_path)
    depth_idx = get_depth_stream_index(cap_path)
    if rgb_idx is None:
        return f"❌ No RGB stream found: {cap_path}"
    if depth_idx is None:
        return f"❌ No Depth stream found: {cap_path}"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(cap_path),

        # RGB: 재인코딩 (CPU)
        "-map", f"0:v:{rgb_idx}",
        "-c:v:0", "libx264",
        "-preset:v:0", "fast",
        "-crf:v:0", "20",
        "-pix_fmt:v:0", "yuv420p",

        # Depth: 무손실 ffv1 인코딩
        "-map", f"0:v:{depth_idx}",
        "-c:v:1", "ffv1",
        "-pix_fmt:v:1", "gray16le",

        "-an", str(out_file)
    ]



    subprocess.run(cmd, check=True, close_fds=True)
    return f"✅ Done (RGB+Depth): {cap_path.name}"

# def main():
#     files = list(SRC_ROOT.glob("AI*/Rec/Kinect/cap*.mkv"))
#     print(f"총 {len(files)}개 파일 발견. 병렬 처리 중...")
#
#     with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
#         futures = [ex.submit(reencode_rgb_only, f) for f in files]
#         for fut in as_completed(futures):
#             print(fut.result())
#     ex.shutdown(wait=True, cancel_futures=False)
#     print("re-encoding process done......")

def main():
    files = list(SRC_ROOT.glob("AI*/Rec/Kinect/cap*.mkv"))
    print(f"총 {len(files)}개 파일 발견. 병렬 처리 중...")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = [ex.submit(reencode_rgb_depth, f) for f in files]
        for fut in as_completed(futures):
            print(fut.result())
    print("RGB+Depth re-encoding complete ✅")


if __name__ == "__main__":
    main()
