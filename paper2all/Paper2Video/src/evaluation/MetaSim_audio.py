import os, re, json
import random
import argparse
import moviepy.editor as mp
from os import path
from pathlib import Path
from typing import List
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from scipy.spatial.distance import cosine


def extract_random_audio_segment(video_path: str, output_wav_path: str, duration: float = 5.0):
    print(video_path)
    video = mp.VideoFileClip(video_path)
    audio = video.audio

    total_duration = audio.duration
    if duration >= total_duration: start_time = 0
    else: start_time = random.uniform(0, total_duration - duration)

    audio_subclip = audio.subclip(start_time, start_time + duration)
    audio_subclip.write_audiofile(output_wav_path, codec='pcm_s16le', fps=16000)

def compute_speaker_similarity(audio_path_1: str, audio_path_2: str, device: str = "cuda") -> float:
    embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=device)
    audio_loader = Audio(sample_rate=16000)

    wav1, _ = audio_loader(audio_path_1)
    wav2, _ = audio_loader(audio_path_2)
    
    wav1 = wav1[0:1].unsqueeze(0)
    wav2 = wav2[0:1].unsqueeze(0)
    
    embedding1 = embedding_model(wav1)
    embedding2 = embedding_model(wav2)
    embedding1 = embedding1.reshape(embedding1.shape[1])
    embedding2 = embedding2.reshape(embedding2.shape[1])

    similarity = 1 - cosine(embedding1, embedding2)
    return similarity


def get_audio_sim_score(gen_video_path, gt_video_path):
    extract_random_audio_segment(gen_video_path, gen_video_path.replace('.mp4', '.wav'), duration=5) 
    extract_random_audio_segment(gt_video_path, gt_video_path.replace('.mp4', '.wav'), duration=5)
    similarity = compute_speaker_similarity(gen_video_path.replace('.mp4', '.wav'), 
                                            gt_video_path.replace('.mp4', '.wav'))
    return similarity

_num_at_start = re.compile(r'^\s*["\']?(\d+)')
def sort_by_leading_number(paths: List[str]) -> List[str]:
    def key(p: str):
        name = Path(p).name  
        m = _num_at_start.match(name)
        return (int(m.group(1)) if m else float('inf'), name)
    return sorted(paths, key=key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_dir", default="/path/to/result_dir")
    parser.add_argument("-g", "--gt_dir", default="/path/to/gt_dir")
    parser.add_argument("-s", "--save_dir", default="/path/to/save_dir")
    args = parser.parse_args()
    
    ## load exist result if have
    save_dir = args.save_dir
    save_dir = path.join(save_dir, path.basename(args.result_dir))
    save_path = path.join(save_dir, "audio_sim.json")
    os.makedirs(save_dir, exist_ok=True)
    if path.exists(save_path):
        with open(save_path, 'r') as f: audio_similarity_list = json.load(f)
    else: audio_similarity_list = []
    
    ## path
    gt_dir, result_dir = args.gt_dir, args.result_dir
    groundtruth_list = sort_by_leading_number([path.join(gt_dir, name) for name in os.listdir(gt_dir)])
    result_list = sort_by_leading_number([path.join(result_dir, name) for name in os.listdir(result_dir)])
    
    for index in range(len(audio_similarity_list), 40):
        if path.basename(args.result_dir) == "paper2video":
            p2v_video_path = path.join(result_list[index], "3_merage.mp4")
        elif path.basename(args.result_dir) == "veo3":
            p2v_video_path = path.join(result_list[index])
        else:
            p2v_video_path = path.join(result_list[index], "result.mp4")
        if path.exists(p2v_video_path) is False: continue
        gt_video_path = path.join(groundtruth_list[index], "gt_presentation_video.mp4")
        if path.exists(gt_video_path) is False: continue
        print(p2v_video_path, gt_video_path)
        similarity = get_audio_sim_score(p2v_video_path, gt_video_path)
        audio_similarity_list.append({
            "data_idx": index,
            "score": similarity.item()
        })
    print(audio_similarity_list)
    with open(save_path, 'w') as f: json.dump(audio_similarity_list, f, indent=4)

    # import numpy as np
    # avg = np.average(similarity_all)
    # var = np.var(similarity_all)
    # print(avg, var)