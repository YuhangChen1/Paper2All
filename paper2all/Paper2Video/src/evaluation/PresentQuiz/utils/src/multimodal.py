import json

import PIL.Image
from rich import print

import src.llms as llms
from src.presentation import Picture, Presentation
from src.utils import Config, pbasename, pexists, pjoin


class ImageLabler:
    """
    A class to extract images information, including caption, size, and appearance times in a presentation.
    """

    def __init__(self, presentation: Presentation, config: Config):
        """
        Initialize the ImageLabler.

        Args:
            presentation (Presentation): The presentation object.
            config (Config): The configuration object.
        """
        self.presentation = presentation
        self.slide_area = presentation.slide_width.pt * presentation.slide_height.pt
        self.image_stats = {}
        self.stats_file = pjoin(config.RUN_DIR, "image_stats.json")
        self.config = config
        self.collect_images()
        if pexists(self.stats_file):
            image_stats: dict[str, dict] = json.load(open(self.stats_file, "r"))
            for name, stat in image_stats.items():
                if pbasename(name) in self.image_stats:
                    self.image_stats[pbasename(name)] = stat

    def apply_stats(self):
        """
        Apply image captions to the presentation.
        """
        for slide in self.presentation.slides:
            for shape in slide.shape_filter(Picture):
                stats = self.image_stats[pbasename(shape.img_path)]
                shape.caption = stats["caption"]

    def caption_images(self):
        """
        Generate captions for images in the presentation.
        """
        caption_prompt = open("prompts/caption.txt").read()
        for image, stats in self.image_stats.items():
            if "caption" not in stats:
                stats["caption"] = llms.vision_model(
                    caption_prompt, pjoin(self.config.IMAGE_DIR, image)
                )
                print("captioned", image, ": ", stats["caption"])
        json.dump(
            self.image_stats,
            open(self.stats_file, "w"),
            indent=4,
            ensure_ascii=False,
        )
        self.apply_stats()
        return self.image_stats

    def collect_images(self):
        """
        Collect images from the presentation and gather other information.
        """
        for slide_index, slide in enumerate(self.presentation.slides):
            for shape in slide.shape_filter(Picture):
                image_path = pbasename(shape.data[0])
                self.image_stats[image_path] = {
                    "appear_times": 0,
                    "slide_numbers": set(),
                    "relative_area": shape.area / self.slide_area * 100,
                    "size": PIL.Image.open(
                        pjoin(self.config.IMAGE_DIR, image_path)
                    ).size,
                }
                self.image_stats[image_path]["appear_times"] += 1
                self.image_stats[image_path]["slide_numbers"].add(slide_index + 1)
        for image_path, stats in self.image_stats.items():
            stats["slide_numbers"] = sorted(list(stats["slide_numbers"]))
            ranges = self._find_ranges(stats["slide_numbers"])
            top_ranges = sorted(ranges, key=lambda x: x[1] - x[0], reverse=True)[:3]
            top_ranges_str = ", ".join(
                [f"{r[0]}-{r[1]}" if r[0] != r[1] else f"{r[0]}" for r in top_ranges]
            )
            stats["top_ranges_str"] = top_ranges_str

    def _find_ranges(self, numbers):
        """
        Find consecutive ranges in a list of numbers.
        """
        ranges = []
        start = numbers[0]
        end = numbers[0]
        for num in numbers[1:]:
            if num == end + 1:
                end = num
            else:
                ranges.append((start, end))
                start = num
                end = num
        ranges.append((start, end))
        return ranges
