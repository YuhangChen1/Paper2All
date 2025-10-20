"""
speech_gen.py
-------------
Speech generation module for Paper2Video.

功能：
1. 自动检测 ElevenLabs 克隆权限；
2. 没权限时自动回退到默认 voice（Rachel）；
3. 逐页生成语音文件；
4. 缓存声音对象，避免重复 API 调用。
"""

import os
from os import path
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

client = None
voice_cache = {}
# 官方 Rachel 的 voice_id（作为默认回退）
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"


# 初始化 ElevenLabs 客户端
try:
    print("INFO: Initializing ElevenLabs API client...")
    from elevenlabs.client import ElevenLabs as Client
    from elevenlabs import VoiceSettings

    # 优先使用官方环境变量名，兼容你的旧命名
    api_key = os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVEN_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY environment variable not set.")

    client = Client(api_key=api_key)
    print("INFO: ElevenLabs client initialized successfully.")
except Exception as e:
    print("❌ ERROR: Failed to initialize ElevenLabs client.")
    print("Error details:", e)
    client = None


# ------------------ 解析字幕脚本 ------------------
def parse_script(script_text):
    """
    Parses script text delimited by '###' for pages
    and '|' for text/cursor prompts.
    Returns List of pages, each page is List[text, cursor].
    """
    pages = script_text.strip().split("###\n")
    result = []
    for page in pages:
        if not page.strip():
            continue
        lines = page.strip().split("\n")
        page_data = []
        for line in lines:
            if "|" not in line:
                continue
            text, cursor = line.split("|", 1)
            page_data.append([text.strip(), cursor.strip()])
        if page_data:
            result.append(page_data)
    return result


# ------------------ TTS 主函数 ------------------
def tts_per_slide(model_type, script_path, speech_save_dir, ref_audio, ref_text=None):
    """
    Generates speech for each slide using ElevenLabs API.
    Falls back to DEFAULT_VOICE_ID if cloning fails or no permission.

    Args:
        model_type: 预留参数（可用于未来切换模型/策略）
        script_path: 字幕脚本路径
        speech_save_dir: 语音输出目录
        ref_audio: 用于 IVC 声音克隆的参考音频文件路径；若为 None 则直接使用默认 voice
        ref_text: 预留参数（可用于引导韵律/风格一致性）
    """
    if client is None:
        print("❌ ERROR: ElevenLabs client unavailable. Please set ELEVENLABS_API_KEY.")
        return

    print("\n--- Starting Speech Generation Stage ---")

    if not path.exists(script_path):
        print(f"❌ ERROR: Script file not found at {script_path}")
        return

    with open(script_path, "r", encoding="utf-8") as f:
        script_with_cursor = f.read()

    parsed_speech = parse_script(script_with_cursor)
    if not parsed_speech:
        print("⚠️  No slides found in script. Aborting.")
        return

    os.makedirs(speech_save_dir, exist_ok=True)
    print(f"INFO: Found {len(parsed_speech)} slides to process.\n")

    # ---------- 尝试克隆语音（IVC） ----------
    cloned_voice_id = None
    if ref_audio and isinstance(ref_audio, str) and path.exists(ref_audio):
        if ref_audio in voice_cache:
            cloned_voice_id = voice_cache[ref_audio]
            print(f"INFO: Using cached cloned voice ID for {ref_audio}: {cloned_voice_id}")
        else:
            try:
                print(f"INFO: Attempting to clone voice from {ref_audio} ...")
                with open(ref_audio, "rb") as rf:
                    cloned_voice = client.voices.ivc.create(
                        name=f"VoiceClone-{os.path.basename(ref_audio)}",
                        files=[BytesIO(rf.read())],
                    )
                # 返回对象可能有 voice_id 或 id
                cloned_voice_id = getattr(cloned_voice, "voice_id", None) or getattr(cloned_voice, "id", None)
                if not cloned_voice_id:
                    raise ValueError("Could not get voice_id from response.")
                voice_cache[ref_audio] = cloned_voice_id
                print(f"✅ Voice cloned successfully! voice_id = {cloned_voice_id}\n")
            except Exception as e:
                print("⚠️  Voice cloning failed — using default voice.")
                print("Error details:", e)
                cloned_voice_id = DEFAULT_VOICE_ID
                print(f"INFO: Fallback to default voice '{DEFAULT_VOICE_ID}'.\n")
    else:
        # 没有 ref_audio 或路径无效，直接用默认
        cloned_voice_id = DEFAULT_VOICE_ID
        if not ref_audio:
            print("INFO: No ref_audio provided. Using default voice.")
        else:
            print(f"⚠️  ref_audio not found: {ref_audio}. Using default voice.")
        print(f"INFO: Fallback to default voice '{DEFAULT_VOICE_ID}'.\n")

    # ---------- 逐页生成语音 ----------
    for slide_idx, slide_data in enumerate(parsed_speech):
        subtitle = "\n\n".join([text for text, _ in slide_data if text.strip()])
        if not subtitle.strip():
            print(f"⚠️  No text found for slide {slide_idx + 1}. Skipping.")
            continue

        # 推荐使用 mp3 输出；若你需要无损链路，可改 pcm_44100 并自行封装 WAV
        out_path = path.join(speech_save_dir, f"{slide_idx}.mp3")
        print(f"🎙 Generating speech for slide {slide_idx + 1} ...")

        try:
            # text_to_speech.convert 返回的是分块字节流（iterable）
            response = client.text_to_speech.convert(
                voice_id=cloned_voice_id,
                text=subtitle,
                model_id="eleven_turbo_v2_5",
                # 常用：mp3_44100_128。若套餐支持无损，可改 'pcm_44100'
                output_format="mp3_44100_128",
                voice_settings=VoiceSettings(
                    stability=0.8,
                    similarity_boost=0.5,
                    style=0.2,
                    use_speaker_boost=True,
                    # speed=1.0,  # 如需控制语速可打开
                ),
            )

            # 正确写法：逐块写入文件
            with open(out_path, "wb") as f:
                for chunk in response:
                    if chunk:
                        f.write(chunk)

            print(f"✅ Slide {slide_idx + 1} speech saved -> {out_path}\n")

        except Exception as e:
            print(f"❌ ERROR: Failed to generate speech for slide {slide_idx + 1}")
            print("Error details:", e)

    print("--- Speech Generation Stage Complete ---\n")


if __name__ == "__main__":
    # 简单自测入口（按需修改）
    # 示例：
    # tts_per_slide(
    #     model_type="turbo",
    #     script_path="./script.txt",
    #     speech_save_dir="./speech_out",
    #     ref_audio="./my_voice.wav",
    # )
    pass
