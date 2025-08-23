import os
import shutil
from pathlib import Path

from pydantic.v1 import BaseModel
from pydantic_argparse import ArgumentParser

from captioning import caption_images


class PrepareArgs(BaseModel):
    images_dir: str
    repeats: int = 1
    name: str
    config_path: str = "config.toml"
    prompt_path: str = "prompt.txt"
    include_tags: str = ""
    exclude_tags: str = ""


def prepare(args: PrepareArgs):
    images_dir = Path(args.images_dir)

    if not images_dir.is_dir():
        raise ValueError("Invalid images directory")

    if not args.name.isidentifier():
        raise ValueError("Invalid name. Must be a valid identifier.")

    train_dir = Path("train")
    data_dir = Path(train_dir) / "data" / f"{args.repeats}_{args.name} 1girl"

    shutil.rmtree(train_dir, ignore_errors=True)
    os.makedirs(train_dir, exist_ok=True)

    try:
        os.makedirs(data_dir, exist_ok=True)

        for file in images_dir.glob("**/*"):
            suffix = file.suffix.lower()
            if file.is_file() and suffix in {
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".gif",
                ".webp",
            }:
                hashed = hash(file.relative_to(images_dir))
                # 이미지 파일 복사
                shutil.copy2(file, data_dir / f"{hashed}{suffix}")

                # 동일한 이름의 .txt 파일이 있으면 함께 복사
                txt_file = file.with_suffix(".txt")
                if txt_file.exists() and txt_file.is_file():
                    shutil.copy2(txt_file, data_dir / f"{hashed}.txt")

        include_list = [
            tag.strip() for tag in args.include_tags.split(",") if tag.strip()
        ]
        exclude_list = [
            tag.strip() for tag in args.exclude_tags.split(",") if tag.strip()
        ]

        caption_images(
            data_dir,
            front_tags=["1girl", "solo", args.name],
            include_tags=include_list,
            exclude_tags=exclude_list,
        )

        shutil.copy2(args.config_path, train_dir / "config.toml")
        shutil.copy2(args.prompt_path, train_dir / "prompt.txt")

        shutil.make_archive(args.name, "tar", root_dir=train_dir, base_dir=".")
        print(f"{args.name}.tar")

    finally:
        shutil.rmtree(train_dir, ignore_errors=True)


if __name__ == "__main__":
    args = ArgumentParser(PrepareArgs).parse_typed_args()
    prepare(args)
