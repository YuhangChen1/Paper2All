import re
import pdb
import os, sys
from os import path
from PIL import Image
from pathlib import Path
from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import ModelPlatformType
from wei_utils import get_agent_config


def subtitle_cursor_gen(slide_imgs_dir, prompt_path, model_config):
    model = ModelFactory.create(
        model_platform=model_config["model_platform"],
        model_type=model_config["model_type"],
        model_config_dict=model_config.get("model_config"),
        url=model_config.get("url", None),)
    agent = ChatAgent(model=model, system_message="",)
    
    with open(prompt_path, 'r', encoding='utf-8') as f_prompt: task_prompt = f_prompt.read()
    slide_image_list = [path.join(slide_imgs_dir, name) for name in os.listdir(slide_imgs_dir)]
    slide_image_list = sorted(slide_image_list, key=lambda x: int(re.search(r'\d+', x).group()))
    
    images = []
    for idx, img_path in enumerate(slide_image_list): images.append(Image.open(img_path))    
    messages = BaseMessage.make_user_message(role_name="user", content=task_prompt, image_list=images, meta_dict={})
    response = agent.step(messages)
    subtitle = response.msg.content.strip()
    return subtitle, response.info["usage"]
    
    

