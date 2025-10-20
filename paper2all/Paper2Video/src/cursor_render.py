import subprocess
import json
import pdb
from PIL import Image, ImageDraw


def render_cursor_on_video(
    input_video: str,
    output_video: str,
    cursor_points: list,          # list of (time, x, y)
    transition_duration: float = 0.1,
    cursor_size: int = 10,
    cursor_img_path: str = "cursor.png"):

    img = Image.open(cursor_img_path)
    img_resized = img.resize((cursor_size, cursor_size))
    img_resized.save(cursor_img_path)


    def get_video_resolution(path):
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json", path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        width = info["streams"][0]["width"]
        height = info["streams"][0]["height"]
        return width, height

    w, h = get_video_resolution(input_video)
    print(f"Video resolution: {w}x{h}")

    filters = []

    t_first, _, _ = cursor_points[0]
    if t_first > transition_duration:
        cx = w / 2 - cursor_size / 2
        cy = h / 2 - cursor_size / 2
        global_hold = (
            f"overlay=x={cx}:y={cy-20}:"
            f"enable='between(t,0,{round(t_first - transition_duration, 3)})'"
        )
        filters.append(global_hold)
        
    for i in range(1, len(cursor_points)):
        t0, x0, y0 = cursor_points[i - 1]
        t1, x1, y1 = cursor_points[i]

        hold_start = round(t0, 3)
        hold_end = round(t1 - transition_duration, 3)
        if hold_end > hold_start:
            x_hold = x0 - cursor_size / 2
            y_hold = y0 - cursor_size / 2
            hold_expr = (
                f"overlay=x={x_hold}:y={y_hold}:"
                f"enable='between(t,{hold_start},{hold_end})'"
            )
            filters.append(hold_expr)

        move_start = round(t1 - transition_duration, 3)
        move_end = t1
        dx = x1 - x0
        dy = y1 - y0
        x_expr = f"{x0 - cursor_size/2} + ({dx})*(t-{move_start})/{transition_duration}"
        y_expr = f"{y0 - cursor_size/2} + ({dy})*(t-{move_start})/{transition_duration}"
        move_expr = (
            f"overlay=x={x_expr}:y={y_expr}:"
            f"enable='between(t,{move_start},{move_end})'"
        )
        filters.append(move_expr)

    filter_lines = []
    stream_in = "[0][1]"
    for i, expr in enumerate(filters):
        stream_out = f"[tmp{i}]" if i < len(filters) - 1 else "[vout]"
        filter_lines.append(f"{stream_in} {expr} {stream_out}")
        stream_in = f"{stream_out}[1]"

    filter_complex = "; ".join(filter_lines)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-i", cursor_img_path,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-c:a", "copy",
        output_video
    ]
    subprocess.run(cmd, check=True)
    print(f"\n✅ Done! Output saved to: {output_video}")


def render_video_with_cursor_from_json(
    video_path,
    out_video_path,
    json_path,
    cursor_img_path,
    transition_duration=0.1,
    cursor_size=16
):
    with open(json_path, "r") as f:
        data = json.load(f)

    cursor_points = []
    for idx, slide in enumerate(data):
        if idx == 0: start_time = slide["start"]
        else: start_time = slide["start"] + 0.5
        x, y = slide["cursor"]
        cursor_points.append((start_time, x, y))
    
    render_cursor_on_video(
        input_video=video_path,
        output_video=out_video_path,
        cursor_points=cursor_points,
        transition_duration=transition_duration,
        cursor_size=cursor_size,
        cursor_img_path=cursor_img_path
    )
