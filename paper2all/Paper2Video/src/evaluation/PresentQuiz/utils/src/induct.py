import json
import os
import shutil
from collections import defaultdict

from jinja2 import Template

import src.llms as llms
from src.model_utils import get_cluster, get_image_embedding, images_cosine_similarity
from src.presentation import Presentation
from src.utils import Config, pexists, pjoin, tenacity


class SlideInducter:
    """
    Stage I: Presentation Analysis.
    This stage is to analyze the presentation: cluster slides into different layouts, and extract content schema for each layout.
    """

    def __init__(
        self,
        prs: Presentation,
        ppt_image_folder: str,
        template_image_folder: str,
        config: Config,
        image_models: list,
    ):
        """
        Initialize the SlideInducter.

        Args:
            prs (Presentation): The presentation object.
            ppt_image_folder (str): The folder containing PPT images.
            template_image_folder (str): The folder containing normalized slide images.
            config (Config): The configuration object.
            image_models (list): A list of image models.
        """
        self.prs = prs
        self.config = config
        self.ppt_image_folder = ppt_image_folder
        self.template_image_folder = template_image_folder
        assert (
            len(os.listdir(template_image_folder))
            == len(prs)
            == len(os.listdir(ppt_image_folder))
        )
        self.image_models = image_models
        self.slide_induction = defaultdict(lambda: defaultdict(list))
        model_identifier = llms.get_simple_modelname(
            [llms.language_model, llms.vision_model]
        )
        self.output_dir = pjoin(config.RUN_DIR, "template_induct", model_identifier)
        self.split_cache = pjoin(self.output_dir, f"split_cache.json")
        self.induct_cache = pjoin(self.output_dir, f"induct_cache.json")
        os.makedirs(self.output_dir, exist_ok=True)

    def layout_induct(self):
        """
        Perform layout induction for the presentation.
        """
        if pexists(self.induct_cache):
            return json.load(open(self.induct_cache))
        content_slides_index, functional_cluster = self.category_split()
        for layout_name, cluster in functional_cluster.items():
            for slide_idx in cluster:
                content_type = self.prs.slides[slide_idx - 1].get_content_type()
                self.slide_induction[layout_name + ":" + content_type]["slides"].append(
                    slide_idx
                )
        for layout_name, cluster in self.slide_induction.items():
            cluster["template_id"] = cluster["slides"][-1]

        functional_keys = list(self.slide_induction.keys())
        function_slides_index = set()
        for layout_name, cluster in self.slide_induction.items():
            function_slides_index.update(cluster["slides"])
        used_slides_index = function_slides_index.union(content_slides_index)
        for i in range(len(self.prs.slides)):
            if i + 1 not in used_slides_index:
                content_slides_index.add(i + 1)
        self.layout_split(content_slides_index)
        if self.config.DEBUG:
            for layout_name, cluster in self.slide_induction.items():
                cluster_dir = pjoin(self.output_dir, "cluster_slides", layout_name)
                os.makedirs(cluster_dir, exist_ok=True)
                for slide_idx in cluster["slides"]:
                    shutil.copy(
                        pjoin(self.ppt_image_folder, f"slide_{slide_idx:04d}.jpg"),
                        pjoin(cluster_dir, f"slide_{slide_idx:04d}.jpg"),
                    )
        self.slide_induction["functional_keys"] = functional_keys
        json.dump(
            self.slide_induction,
            open(self.induct_cache, "w"),
            indent=4,
            ensure_ascii=False,
        )
        return self.slide_induction

    def category_split(self):
        """
        Split slides into categories based on their functional purpose.
        """
        if pexists(self.split_cache):
            split = json.load(open(self.split_cache))
            return set(split["content_slides_index"]), split["functional_cluster"]
        category_split_template = Template(open("prompts/category_split.txt").read())
        functional_cluster = llms.language_model(
            category_split_template.render(slides=self.prs.to_text()),
            return_json=True,
        )
        functional_slides = set(sum(functional_cluster.values(), []))
        content_slides_index = set(range(1, len(self.prs) + 1)) - functional_slides

        json.dump(
            {
                "content_slides_index": list(content_slides_index),
                "functional_cluster": functional_cluster,
            },
            open(self.split_cache, "w"),
            indent=4,
            ensure_ascii=False,
        )
        return content_slides_index, functional_cluster

    def layout_split(self, content_slides_index: set[int]):
        """
        Cluster slides into different layouts.
        """
        embeddings = get_image_embedding(self.template_image_folder, *self.image_models)
        assert len(embeddings) == len(self.prs)
        template = Template(open("prompts/ask_category.txt").read())
        content_split = defaultdict(list)
        for slide_idx in content_slides_index:
            slide = self.prs.slides[slide_idx - 1]
            content_type = slide.get_content_type()
            layout_name = slide.slide_layout_name
            content_split[(layout_name, content_type)].append(slide_idx)

        for (layout_name, content_type), slides in content_split.items():
            sub_embeddings = [
                embeddings[f"slide_{slide_idx:04d}.jpg"] for slide_idx in slides
            ]
            similarity = images_cosine_similarity(sub_embeddings)
            for cluster in get_cluster(similarity):
                slide_indexs = [slides[i] for i in cluster]
                template_id = max(
                    slide_indexs,
                    key=lambda x: len(self.prs.slides[x - 1].shapes),
                )
                cluster_name = (
                    llms.vision_model(
                        template.render(
                            existed_layoutnames=list(self.slide_induction.keys()),
                        ),
                        pjoin(self.ppt_image_folder, f"slide_{template_id:04d}.jpg"),
                    )
                    + ":"
                    + content_type
                )
                self.slide_induction[cluster_name]["template_id"] = template_id
                self.slide_induction[cluster_name]["slides"] = slide_indexs

    @tenacity
    def content_induct(self):
        """
        Perform content schema extraction for the presentation.
        """
        self.slide_induction = self.layout_induct()
        content_induct_prompt = Template(open("prompts/content_induct.txt").read())
        for layout_name, cluster in self.slide_induction.items():
            if "template_id" in cluster and "content_schema" not in cluster:
                schema = llms.language_model(
                    content_induct_prompt.render(
                        slide=self.prs.slides[cluster["template_id"] - 1].to_html(
                            element_id=False, paragraph_id=False
                        )
                    ),
                    return_json=True,
                )
                for k in list(schema.keys()):
                    if "data" not in schema[k]:
                        raise ValueError(f"Cannot find `data` in {k}\n{schema[k]}")
                    if len(schema[k]["data"]) == 0:
                        print(f"Empty content schema: {schema[k]}")
                        schema.pop(k)
                assert len(schema) > 0, "No content schema generated"
                self.slide_induction[layout_name]["content_schema"] = schema
        json.dump(
            self.slide_induction,
            open(self.induct_cache, "w"),
            indent=4,
            ensure_ascii=False,
        )
        return self.slide_induction
