import os
import shutil
import subprocess
from pathlib import Path

from pydantic.v1 import BaseModel
from pydantic_argparse import ArgumentParser


class TrainArgs(BaseModel):
    data: str
    checkpoint: str


def train(args: TrainArgs):
    data_path = Path(args.data)
    checkpoint_path = Path(args.checkpoint)

    if not data_path.is_file():
        raise ValueError("Invalid data file")
    if not checkpoint_path.is_file():
        raise ValueError("Invalid checkpoint file")

    name = data_path.stem
    if not name.isidentifier():
        raise ValueError("Invalid name. Must be a valid identifier.")

    # Create directories
    train_dir = Path("train")
    if train_dir.is_dir():
        raise ValueError("Train directory already exists. Please remove it first.")

    shutil.rmtree(train_dir, ignore_errors=True)
    os.makedirs(train_dir, exist_ok=True)
    shutil.unpack_archive(args.data, train_dir)

    # Check train directory structure
    data_dir = train_dir / "data"
    config_path = train_dir / "config.toml"
    prompt_path = train_dir / "prompt.txt"
    if not data_dir.is_dir() or not config_path.is_file() or not prompt_path.is_file():
        raise ValueError("Train directory structure is invalid.")

    output_dir = train_dir / "output"
    os.makedirs(output_dir, exist_ok=True)

    # Complete config
    with open(config_path, "a", encoding="utf-8") as f:
        f.write(
            f"""
pretrained_model_name_or_path = "{checkpoint_path.resolve().absolute().as_posix()}"
sample_prompts = "{prompt_path.resolve().absolute().as_posix()}"
output_name = "{name}"
train_data_dir = "{data_dir.resolve().absolute().as_posix()}"
output_dir = "{output_dir.resolve().absolute().as_posix()}"
"""
        )

    # Start training
    subprocess.run(
        f"uv run accelerate launch sdxl_train_network.py --config_file {config_path.resolve().absolute().as_posix()}",
        shell=True,
        cwd="sd-scripts",
    )


if __name__ == "__main__":
    args = ArgumentParser(TrainArgs).parse_typed_args()
    train(args)
