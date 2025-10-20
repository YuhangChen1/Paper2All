import os
import shutil
import subprocess
import multiprocessing
import sys, pdb
from os import path
sys.path.insert(0, "hallo2")


def run_hallo2_inference(args):
    source_image, driving_audio, save_video_dir, gpu_id, config_path, script_path, talking_head_env = args
    print(source_image, driving_audio, save_video_dir, gpu_id, config_path, script_path)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    audio_basename = os.path.splitext(os.path.basename(driving_audio))[0]
    save_path = os.path.join(save_video_dir, f"{audio_basename}")
    config_bak = config_path.replace(".yaml", "_{}.yaml".format(audio_basename))
    shutil.copy(config_path, config_bak)
    print(save_path, config_bak)
    
    updated_lines = []
    with open(config_bak, "r") as f: lines = f.readlines()
    for line in lines:
        if line.strip().startswith("save_path:"): updated_lines.append(f"save_path: {save_path}\n")
        else: updated_lines.append(line)
    with open(config_bak, "w") as f: f.writelines(updated_lines)

    cmd = [
        talking_head_env, script_path,
        "--config", config_bak,
        "--source_image", source_image,
        "--driving_audio", driving_audio,
    ]
    result = subprocess.run(cmd, env=env)

    shutil.move(config_bak, config_path)
    return result
    
def talking_gen_per_slide(model_type, input_list, save_dir, gpu_list, env_path):
    num_gpu = len(gpu_list)
    print(gpu_list)
    if model_type == "hallo2":
        config_path="./hallo2/configs/inference/long.yaml"
        script_path="./hallo2/scripts/inference_long.py"
        task_list = []
        task_num = 0
        for idx, (ref_img_path, audio_path) in enumerate(input_list):
            if path.exists(path.join(save_dir, str(idx), path.basename(ref_img_path).replace(".png", ""), "merge_video.mp4")) is False:
                task_list.append([ref_img_path, audio_path, save_dir, gpu_list[task_num%num_gpu], config_path, script_path, env_path])
                task_num += 1
        for task in task_list:
            print(task)
        with multiprocessing.Pool(processes=num_gpu) as pool: results = pool.map(run_hallo2_inference, task_list)
    return results