
import whisper
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
def create_subtitle_image(text, font_size=32, font_path="arial.ttf"):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"[Warning] Failed to load font from '{font_path}': {e}")
        print("Using default font (fixed size, font_size will be ignored!)")
        font = ImageFont.load_default()

    dummy_img = Image.new("RGBA", (70, 70))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding = 20
    box_w = text_w + 2*padding
    box_h = text_h + 2*padding
    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 160))  # semi-transparent black

    draw = ImageDraw.Draw(img)
    draw.text((padding, padding), text, font=font, fill=(255, 255, 255, 255))

    return img

def generate_subtitle_clips(segments, video_w, video_h, font_size):
    clips = []
    for seg in segments:
        img = create_subtitle_image(seg["text"], font_size=font_size, font_path="./fonts/LiberationSans-Regular.ttf")
        img_array = np.array(img)
        clip = (ImageClip(img_array, ismask=False)
                .set_duration(seg["end"] - seg["start"])
                .set_start(seg["start"])
                .set_position(("center", video_h - font_size*2)))
        clips.append(clip)
    return clips

def add_subtitles(video_path, output_path, font_size):
    print("[Step 1] Transcribing with Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(video_path, language="en")
    segments = result["segments"]

    print("[Step 2] Generating subtitle clips...")
    video = VideoFileClip(video_path)
    subs = generate_subtitle_clips(segments, video.w, video.h, font_size)

    print("[Step 3] Rendering final video...")
    final = CompositeVideoClip([video] + subs)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")

