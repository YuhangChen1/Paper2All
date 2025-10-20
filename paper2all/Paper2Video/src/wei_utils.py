from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig, QwenConfig, VLLMConfig, OpenRouterConfig, GeminiConfig

def get_agent_config(model_type):
    agent_config = {}
    if model_type == 'qwen':
        agent_config = {
            "model_type": ModelType.DEEPINFRA_QWEN_2_5_72B,
            "model_config": QwenConfig().as_dict(),
            "model_platform": ModelPlatformType.DEEPINFRA,
        }
    elif model_type == 'gemini':
        agent_config = {
            "model_type": ModelType.DEEPINFRA_GEMINI_2_FLASH,
            "model_config": GeminiConfig().as_dict(),
            "model_platform": ModelPlatformType.DEEPINFRA,
            'max_images': 99
        }
    elif model_type == 'phi4':
        agent_config = {
            "model_type": ModelType.DEEPINFRA_PHI_4_MULTIMODAL,
            "model_config": QwenConfig().as_dict(),
            "model_platform": ModelPlatformType.DEEPINFRA,
        }
    elif model_type == 'llama-4-scout-17b-16e-instruct':
        agent_config = {
            'model_type': ModelType.ALIYUN_LLAMA4_SCOUT_17B_16E,
            'model_config': QwenConfig().as_dict(),
            'model_platform': ModelPlatformType.QWEN,
            'max_images': 99
        }
    elif model_type == 'qwen-2.5-vl-72b':
        agent_config = {
            'model_type': ModelType.QWEN_2_5_VL_72B,
            'model_config': QwenConfig().as_dict(),
            'model_platform': ModelPlatformType.QWEN,
            'max_images': 99
        }
    elif model_type == 'gemma':
        agent_config = {
            "model_type": "google/gemma-3-4b-it",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:5555/v1',
            'max_images': 99
        }
    elif model_type == 'llava':
        agent_config = {
            "model_type": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:8000/v1',
            'max_images': 99
        }
    elif model_type == 'molmo-o':
        agent_config = {
            "model_type": "allenai/Molmo-7B-O-0924",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:8000/v1',
            'max_images': 99
        }
    elif model_type == 'qwen-2-vl-7b':
        agent_config = {
            "model_type": "Qwen/Qwen2-VL-7B-Instruct",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:8000/v1',
            'max_images': 99
        }
    elif model_type == 'vllm_phi4':
        agent_config = {
            "model_type": "microsoft/Phi-4-multimodal-instruct",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:8000/v1',
            'max_images': 99
        }
    elif model_type == 'o3-mini':
        agent_config = {
            "model_type": ModelType.O3_MINI,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
        }
    elif model_type == 'gpt-4.1':
        agent_config = {
            "model_type": ModelType.GPT_4_1,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
        }
    elif model_type == 'gpt-4.1-mini':
        agent_config = {
            "model_type": ModelType.GPT_4_1_MINI,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
        }
    elif model_type == '4o':
        agent_config = {
            "model_type": ModelType.GPT_4O,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
            # "model_name": '4o'
        }
    elif model_type == '4o-mini':
        agent_config = {
            "model_type": ModelType.GPT_4O_MINI,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
        }
    elif model_type == 'o1':
        agent_config = {
            "model_type": ModelType.O1,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
            # "model_name": 'o1'
        }
    elif model_type == 'o3':
        agent_config = {
            "model_type": ModelType.O3,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
        }
    elif model_type == 'gpt-5':
        agent_config = {
            "model_type": ModelType.GPT_5,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
        }
    elif model_type == 'vllm_qwen_vl':
        agent_config = {
            "model_type": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:7000/v1'
        }
    elif model_type == 'vllm_qwen':
        agent_config = {
            "model_type": "Qwen/Qwen2.5-7B-Instruct",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:8000/v1',
        }
    elif model_type == 'openrouter_qwen_72b':
        agent_config = {
            'model_type': ModelType.OPENROUTER_QWEN_2_5_72B,
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    elif model_type == 'openrouter_qwen_vl_72b':
        agent_config = {
            'model_type': ModelType.OPENROUTER_QWEN_2_5_VL_72B,
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    elif model_type == 'openrouter_qwen_vl_7b':
        agent_config = {
            'model_type': ModelType.OPENROUTER_QWEN_2_5_VL_7B,
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    elif model_type == 'openrouter_qwen_7b':
        agent_config = {
            'model_type': ModelType.OPENROUTER_QWEN_2_5_7B,
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    else:
        agent_config = {
            'model_type': model_type,
            'model_platform': ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            'model_config': None
        }
    
    return agent_config