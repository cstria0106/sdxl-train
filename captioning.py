import random
from pathlib import Path

import numpy as np
import onnxruntime as rt
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

KAOMOJIS = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]


def load_labels(dataframe) -> tuple:
    """Load tag names and category indexes from dataframe."""
    name_series = dataframe["name"]
    name_series = name_series.map(
        lambda x: x.replace("_", " ") if x not in KAOMOJIS else x
    )
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])

    return tag_names, rating_indexes, general_indexes, character_indexes


class WDTaggerPredictor:
    def __init__(self, model_repo: str = "SmilingWolf/wd-swinv2-tagger-v3"):
        self.model_repo = model_repo
        self.model = None
        self.tag_names = None
        self.rating_indexes = None
        self.general_indexes = None
        self.character_indexes = None
        self.model_target_size = None

    def load_model(self):
        """Download and load the ONNX model and labels."""
        print(f"Loading model from {self.model_repo}...")

        csv_path = hf_hub_download(self.model_repo, LABEL_FILENAME)
        model_path = hf_hub_download(self.model_repo, MODEL_FILENAME)

        tags_df = pd.read_csv(csv_path)
        (
            self.tag_names,
            self.rating_indexes,
            self.general_indexes,
            self.character_indexes,
        ) = load_labels(tags_df)

        self.model = rt.InferenceSession(model_path)
        _, height, _, _ = self.model.get_inputs()[0].shape
        self.model_target_size = height

        print("Model loaded successfully!")

    def prepare_image(self, image_path: Path) -> np.ndarray:
        """Prepare image for model input."""
        image = Image.open(image_path).convert("RGBA")

        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        if max_dim != self.model_target_size:
            padded_image = padded_image.resize(
                (self.model_target_size, self.model_target_size),
                Image.BICUBIC,
            )

        image_array = np.asarray(padded_image, dtype=np.float32)
        image_array = image_array[:, :, ::-1]  # RGB to BGR

        return np.expand_dims(image_array, axis=0)

    def predict_batch(self, image_arrays: np.ndarray) -> np.ndarray:
        """Run batch prediction."""
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image_arrays})[0]
        return preds


def caption_images(
    directory: Path,
    front_tags: list[str] = [],
    max_tags: int = 30,
    min_probability: float = 0.4,
    exclude_tags: list[str] = [],
    include_tags: list[str] = [],
    batch_size: int = 2,
    model_repo: str = "SmilingWolf/wd-swinv2-tagger-v3",
):
    """
    Generate captions for images using WD tagger model.

    Args:
        directory: Path to directory containing images
        front_tags: Tags to always include at the front of the caption
        max_tags: Maximum number of tags to include in caption
        min_probability: Minimum probability threshold for tag inclusion
        exclude_tags: Tags to exclude from caption
        include_tags: Tags to always include regardless of probability
        batch_size: Number of images to process in parallel
        model_repo: HuggingFace model repository ID
    """
    predictor = WDTaggerPredictor(model_repo)
    predictor.load_model()

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    image_files = []
    for ext in image_extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    image_files = list(set(image_files))

    if not image_files:
        print("No images found in directory!")
        return

    # 이미 태그 파일이 있는 이미지 필터링
    images_to_process = []
    skipped_count = 0
    for img_path in image_files:
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            skipped_count += 1
        else:
            images_to_process.append(img_path)

    print(f"Found {len(image_files)} images")
    if skipped_count > 0:
        print(f"Skipping {skipped_count} images that already have captions")
    print(f"Processing {len(images_to_process)} images")

    if not images_to_process:
        print("No images to process!")
        return

    # Normalize front_tags and exclude_tags
    front_tags_set = set(tag.replace("_", " ") for tag in front_tags)
    exclude_tags_set = set(tag.replace("_", " ") for tag in exclude_tags)
    include_tags_set = set(tag.replace("_", " ") for tag in include_tags)

    for i in tqdm(
        range(0, len(images_to_process), batch_size), desc="Processing images"
    ):
        batch_files = images_to_process[i : i + batch_size]
        batch_images = []
        valid_files = []

        for img_path in batch_files:
            try:
                img_array = predictor.prepare_image(img_path)
                batch_images.append(img_array)
                valid_files.append(img_path)
            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
                continue

        if not batch_images:
            continue

        batch_array = np.vstack(batch_images)
        preds = predictor.predict_batch(batch_array)

        for img_path, pred in zip(valid_files, preds):
            labels = list(zip(predictor.tag_names, pred.astype(float)))
            general_names = [labels[i] for i in predictor.general_indexes]

            selected_tags = []
            for tag_name, prob in general_names:
                if tag_name in exclude_tags_set:
                    continue
                if tag_name in front_tags_set:
                    continue
                if prob >= min_probability or tag_name in include_tags_set:
                    selected_tags.append((tag_name, prob))

            selected_tags.sort(key=lambda x: x[1], reverse=True)
            selected_tags = [tag for tag, _ in selected_tags[:max_tags]]

            random.shuffle(selected_tags)

            final_tags = list(front_tags) + selected_tags
            caption = ", ".join(final_tags)

            txt_path = img_path.with_suffix(".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)

    print("Done! All captions saved.")
