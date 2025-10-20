### 1. Requirements
Prepare the environment:
```bash
cd src
conda create -n p2v python=3.10
conda activate p2v
pip install -r requirements.txt
conda install -c conda-forge tectonic ffmpeg poppler
```
**[Optional] [Skip](#2-configure-llms) this part if you do not need a human presenter.**

You need to **prepare the environment separately for talking-head generation** to potential avoide package conflicts, please refer to  <a href="git clone https://github.com/fudan-generative-vision/hallo2.git">Hallo2</a>. After installing, use `which python` to get the python environment path.
```bash
cd hallo2
conda create -n hallo python=3.10
conda activate hallo
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
huggingface-cli download fudan-generative-ai/hallo2 --local-dir ../pretrained_models
```
Once you have installed hallo2, the --talking_head_env argument should point to the Python environment where hallo2 is installed. You can find the path to your hallo2 environment by running the following command:
```bash
which python
```
This will give you the path to the Python executable used by hallo2. You should use this path in the --talking_head_env argument in the pipeline.

### 2. Inference
The script `pipeline.py` provides an automated pipeline for generating academic presentation videos. It takes **LaTeX paper sources** together with **reference image/audio** as input, and goes through multiple sub-modules (Slides → Subtitles → Speech → Cursor → Talking Head) to produce a complete presentation video. ⚡ The minimum recommended GPU for running this pipeline is **NVIDIA A6000** with 48G.

#### Example Usage
Run the following command to launch a fast generation (**without talking-head generation**):
```bash
python pipeline_light.py \
    --model_name_t gpt-4.1 \
    --model_name_v gpt-4.1 \
    --result_dir /path/to/output \
    --paper_latex_root /path/to/latex_proj \
    --ref_img /path/to/ref_img.png \
    --ref_audio /path/to/ref_audio.wav \
    --gpu_list [0,1,2,3,4,5,6,7]
```

Run the following command to launch a full generation (**with talking-head generation**):

```bash
python pipeline.py \
    --model_name_t gpt-4.1 \
    --model_name_v gpt-4.1 \
    --model_name_talking hallo2 \
    --result_dir /path/to/output \
    --paper_latex_root /path/to/latex_proj \
    --ref_img /path/to/ref_img.png \
    --ref_audio /path/to/ref_audio.wav \
    --talking_head_env /path/to/hallo2_env \
    --gpu_list [0,1,2,3,4,5,6,7]
```