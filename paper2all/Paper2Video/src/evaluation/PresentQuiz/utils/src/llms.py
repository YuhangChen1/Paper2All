import asyncio
import base64
import os
import re
from dataclasses import asdict, dataclass
from math import ceil

import jsonlines
import requests
import tiktoken
import yaml
from FlagEmbedding import BGEM3FlagModel
from jinja2 import Environment, Template
from oaib import Auto
from openai import OpenAI
from PIL import Image
from torch import Tensor, cosine_similarity

from src.model_utils import get_text_embedding
from src.utils import get_json_from_response, pexists, pjoin, print, tenacity

ENCODING = tiktoken.encoding_for_model("gpt-4o")


def run_async(coroutine):
    """
    Run an asynchronous coroutine in a non-async environment.

    Args:
        coroutine: The coroutine to run.

    Returns:
        The result of the coroutine.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    job = loop.run_until_complete(coroutine)
    return job


def calc_image_tokens(images: list[str]):
    """
    Calculate the number of tokens for a list of images.
    """
    tokens = 0
    for image in images:
        with open(image, "rb") as f:
            width, height = Image.open(f).size
        if width > 1024 or height > 1024:
            if width > height:
                height = int(height * 1024 / width)
                width = 1024
            else:
                width = int(width * 1024 / height)
                height = 1024
        h = ceil(height / 512)
        w = ceil(width / 512)
        tokens += 85 + 170 * h * w
    return tokens


class LLM:
    """
    A wrapper class to interact with a language model.
    """

    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        api_base: str = None,
        use_openai: bool = True,
        use_batch: bool = False,
    ) -> None:
        """
        Initialize the LLM.

        Args:
            model (str): The model name.
            api_base (str): The base URL for the API.
            use_openai (bool): Whether to use OpenAI.
            use_batch (bool): Whether to use OpenAI's Batch API, which is single thread only.
        """
        if use_openai and "OPENAI_API_KEY" in os.environ:
            self.client = OpenAI(base_url=api_base)
        if use_batch and "OPENAI_API_KEY" in os.environ:
            assert use_openai, "use_batch must be used with use_openai"
            self.oai_batch = Auto(loglevel=0)
        if "OPENAI_API_KEY" not in os.environ:
            print("Warning: no API key found")
        self.model = model
        self.api_base = api_base
        self._use_openai = use_openai
        self._use_batch = use_batch

    @tenacity
    def __call__(
        self,
        content: str,
        images: list[str] = None,
        system_message: str = None,
        history: list = None,
        delay_batch: bool = False,
        return_json: bool = False,
        return_message: bool = False,
    ) -> str | dict | list:
        """
        Call the language model with a prompt and optional images.

        Args:
            content (str): The prompt content.
            images (list[str]): A list of image file paths.
            system_message (str): The system message.
            history (list): The conversation history.
            delay_batch (bool): Whether to delay return of response.
            return_json (bool): Whether to return the response as JSON.
            return_message (bool): Whether to return the message.

        Returns:
            str | dict | list: The response from the model.
        """
        if content.startswith("You are"):
            system_message, content = content.split("\n", 1)
        if history is None:
            history = []
        if isinstance(images, str):
            images = [images]
        system, message = self.format_message(content, images, system_message)
        if self._use_batch:
            result = run_async(self._run_batch(system + history + message, delay_batch))
            if delay_batch:
                return
            try:
                response = result.to_dict()["result"][0]["choices"][0]["message"][
                    "content"
                ]
            except Exception as e:
                print("Failed to get response from batch")
                raise e
        elif self._use_openai:
            completion = self.client.chat.completions.create(
                model=self.model, messages=system + history + message
            )
            response = completion.choices[0].message.content
        else:
            response = requests.post(
                self.api_base,
                json={
                    "system": system_message,
                    "prompt": content,
                    "image": [
                        i["image_url"]["url"]
                        for i in message[-1]["content"]
                        if i["type"] == "image_url"
                    ],
                },
            )
            response.raise_for_status()
            response = response.text
        message.append({"role": "assistant", "content": response})
        if return_json:
            response = get_json_from_response(response)
        if return_message:
            response = (response, message)
        return response

    def __repr__(self) -> str:
        return f"LLM(model={self.model}, api_base={self.api_base})"

    async def _run_batch(self, messages: list, delay_batch: bool = False):
        await self.oai_batch.add(
            "chat.completions.create",
            model=self.model,
            messages=messages,
        )
        if delay_batch:
            return
        return await self.oai_batch.run()

    def format_message(
        self,
        content: str,
        images: list[str] = None,
        system_message: str = None,
    ):
        """
        Message formatter for OpenAI server call.
        """
        if system_message is None:
            system_message = "You are a helpful assistant"
        system = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            }
        ]
        message = [{"role": "user", "content": [{"type": "text", "text": content}]}]
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            for image in images:
                with open(image, "rb") as f:
                    message[0]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"
                            },
                        }
                    )
        return system, message

    def get_batch_result(self):
        """
        Get responses from delayed batch calls.
        """
        results = run_async(self.oai_batch.run())
        return [
            r["choices"][0]["message"]["content"]
            for r in results.to_dict()["result"].values()
        ]

    def clear_history(self):
        self.history = []


@dataclass
class Turn:
    """
    A class to represent a turn in a conversation.
    """

    id: int
    prompt: str
    response: str
    message: list
    images: list[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    embedding: Tensor = None

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if k != "embedding"}

    def calc_token(self):
        """
        Calculate the number of tokens for the turn.
        """
        if self.images is not None:
            self.input_tokens += calc_image_tokens(self.images)
        self.input_tokens += len(ENCODING.encode(self.prompt))
        self.output_tokens = len(ENCODING.encode(self.response))

    def __eq__(self, other):
        return self is other


class Role:
    """
    An agent, defined by its instruction template and model.
    """

    def __init__(
        self,
        name: str,
        env: Environment,
        record_cost: bool,
        llm: LLM = None,
        config: dict = None,
        text_model: BGEM3FlagModel = None,
    ):
        """
        Initialize the Agent.

        Args:
            name (str): The name of the role.
            env (Environment): The Jinja2 environment.
            record_cost (bool): Whether to record the token cost.
            llm (LLM): The language model.
            config (dict): The configuration.
            text_model (BGEM3FlagModel): The text model.
        """
        self.name = name
        if config is None:
            with open(f"roles/{name}.yaml", "r") as f:
                config = yaml.safe_load(f)
        if llm is None:
            llm = globals()[config["use_model"] + "_model"]
        self.llm = llm
        self.model = llm.model
        self.record_cost = record_cost
        self.text_model = text_model
        self.return_json = config["return_json"]
        self.system_message = config["system_prompt"]
        self.prompt_args = set(config["jinja_args"])
        self.template = env.from_string(config["template"])
        self.retry_template = Template(
            """The previous output is invalid, please carefully analyze the traceback and feedback information, correct errors happened before.
            feedback:
            {{feedback}}
            traceback:
            {{traceback}}
            Give your corrected output in the same format without including the previous output:
            """
        )
        self.system_tokens = len(ENCODING.encode(self.system_message))
        self.input_tokens = 0
        self.output_tokens = 0
        self.history: list[Turn] = []

    def calc_cost(self, turns: list[Turn]):
        """
        Calculate the cost of a list of turns.
        """
        for turn in turns:
            self.input_tokens += turn.input_tokens
            self.output_tokens += turn.output_tokens
        self.input_tokens += self.system_tokens
        self.output_tokens += 3

    def get_history(self, similar: int, recent: int, prompt: str):
        """
        Get the conversation history.
        """
        history = self.history[-recent:] if recent > 0 else []
        if similar > 0:
            embedding = get_text_embedding(prompt, self.text_model)
            history.sort(key=lambda x: cosine_similarity(embedding, x.embedding))
            for turn in history:
                if len(history) > similar + recent:
                    break
                if turn not in history:
                    history.append(turn)
        history.sort(key=lambda x: x.id)
        return history

    def save_history(self, output_dir: str):
        """
        Save the conversation history to a file.
        """
        history_file = pjoin(output_dir, f"{self.name}.jsonl")
        if pexists(history_file) and len(self.history) == 0:
            return
        with jsonlines.open(history_file, "w") as writer:
            writer.write(
                {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                }
            )
            for turn in self.history:
                writer.write(turn.to_dict())

    def retry(self, feedback: str, traceback: str, error_idx: int):
        """
        Retry a failed turn with feedback and traceback.
        """
        assert error_idx > 0, "error_idx must be greater than 0"
        prompt = self.retry_template.render(feedback=feedback, traceback=traceback)
        history = []
        for turn in self.history[-error_idx:]:
            history.extend(turn.message)
        response, message = self.llm(
            prompt,
            history=history,
            return_message=True,
        )
        turn = Turn(
            id=len(self.history),
            prompt=prompt,
            response=response,
            message=message,
        )
        return self.__post_process__(response, self.history[-error_idx:], turn)

    def __repr__(self) -> str:
        return f"Role(name={self.name}, model={self.model})"

    def __call__(
        self,
        images: list[str] = None,
        recent: int = 0,
        similar: int = 0,
        **jinja_args,
    ):
        """
        Call the agent with prompt arguments.

        Args:
            images (list[str]): A list of image file paths.
            recent (int): The number of recent turns to include.
            similar (int): The number of similar turns to include.
            **jinja_args: Additional arguments for the Jinja2 template.

        Returns:
            The response from the role.
        """
        if isinstance(images, str):
            images = [images]
        assert self.prompt_args == set(jinja_args.keys()), "Invalid arguments"
        prompt = self.template.render(**jinja_args)
        history = self.get_history(similar, recent, prompt)
        history_msg = []
        for turn in history:
            history_msg.extend(turn.message)

        response, message = self.llm(
            prompt,
            system_message=self.system_message,
            history=history_msg,
            images=images,
            return_message=True,
        )
        turn = Turn(
            id=len(self.history),
            prompt=prompt,
            response=response,
            message=message,
            images=images,
        )
        return self.__post_process__(response, history, turn, similar)

    def __post_process__(
        self, response: str, history: list[Turn], turn: Turn, similar: int = 0
    ):
        """
        Post-process the response from the agent.
        """
        self.history.append(turn)
        if similar > 0:
            turn.embedding = get_text_embedding(turn.prompt, self.text_model)
        if self.record_cost:
            turn.calc_token()
            self.calc_cost(history + [turn])
        if self.return_json:
            response = get_json_from_response(response)
        return response


def get_simple_modelname(llms: list[LLM]):
    """
    Get a abbreviation from a list of LLMs.
    """
    if isinstance(llms, LLM):
        llms = [llms]
    return "+".join(re.search(r"^(.*?)-\d{2}", llm.model).group(1) for llm in llms)


gpt4o = LLM(model="gpt-4o-2024-08-06", use_batch=True)
gpt4omini = LLM(model="gpt-4o-mini-2024-07-18", use_batch=True)
qwen2_5 = LLM(
    model="Qwen2.5-72B-Instruct-GPTQ-Int4", api_base="http://124.16.138.143:7812/v1"
)

qwen_vl = LLM(model="Qwen2-VL-72B-Instruct", api_base="http://124.16.138.144:7999/v1")
qwen_coder = LLM(
    model="Qwen2.5-Coder-32B-Instruct", api_base="http://127.0.0.1:8008/v1"
)
intern_vl = LLM(model="InternVL2_5-78B", api_base="http://124.16.138.144:8009/v1")

language_model = gpt4o
vision_model = gpt4o

if __name__ == "__main__":
    gpt4o = LLM(model="gpt-4o-2024-08-06")
    print(
        gpt4o(
            "who r u",
        )
    )
