import time
from pathlib import Path

import toml
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic_argparse import ArgumentParser

from s3 import S3Client, S3Config
from ssh import SSHConfig, SSHSession
from vastai import VastAIClient, VastAIConfig


class RemoteConfig(BaseModel):
    ssh: SSHConfig | None = None
    vastai: VastAIConfig | None = None
    s3: S3Config


def _load_remote_config(path: str = "remote.toml") -> RemoteConfig:
    config_data = toml.load(path)
    return RemoteConfig(**config_data)


class RemoteArgs(BaseModelV1):
    data: str
    checkpoint: str


def _remote_ssh(ssh: SSHSession, s3: S3Client, args: RemoteArgs, conf: RemoteConfig):
    ssh.run("git clone https://github.com/cstria0106/sdxl-train || true")
    ssh.run("cd sdxl-train && git submodule update --init --recursive")
    ssh.run("cd sdxl-train && uv sync --all-groups")

    checkpoint_path = Path(args.checkpoint)
    data_path = Path(args.data)
    checkpoint_key = f"checkpoints/{checkpoint_path.name}"
    data_key = f"data/{data_path.name}"

    # 원격 다운로드 (멀티파트 병렬)
    ssh.run(
        "cd sdxl-train && "
        f"if [ -f '{checkpoint_path.name}' ]; then "
        f"  echo 'Checkpoint {checkpoint_path.name} already exists. Skipping download.'; "
        f"else "
        f"  uv run python s3.py "
        f"  --action download "
        f"  --endpoint-url '{conf.s3.endpoint_url}' "
        f"  --bucket '{conf.s3.bucket}' "
        f"  --access-key '{conf.s3.access_key}' "
        f"  --secret-key '{conf.s3.secret_key}' "
        f"  --region-name '{conf.s3.region_name}' "
        f"  --key '{checkpoint_key}' "
        f"  --dst '{checkpoint_path.name}' "
        f"  --multipart-chunksize-mb 32 --concurrency 32; "
        f"fi"
    )
    ssh.run(
        "cd sdxl-train && "
        f"uv run python s3.py "
        f"--action download "
        f"--endpoint-url '{conf.s3.endpoint_url}' "
        f"--bucket '{conf.s3.bucket}' "
        f"--access-key '{conf.s3.access_key}' "
        f"--secret-key '{conf.s3.secret_key}' "
        f"--region-name '{conf.s3.region_name}' "
        f"--key '{data_key}' "
        f"--dst '{data_path.name}' "
        f"--multipart-chunksize-mb 32 --concurrency 32"
    )

    # 학습
    ssh.run(
        f"cd sdxl-train && rm -rf train && uv run train.py --data {data_path.name} --checkpoint {checkpoint_path.name}"
    )

    # 결과 아카이브
    ssh.run(f"cd sdxl-train && tar -cf {data_path.stem}-out.tar train/output")

    # 원격 업로드 (멀티파트 병렬)
    ssh.run(
        "cd sdxl-train && "
        f"uv run python s3.py "
        f"--action upload "
        f"--endpoint-url '{conf.s3.endpoint_url}' "
        f"--bucket '{conf.s3.bucket}' "
        f"--access-key '{conf.s3.access_key}' "
        f"--secret-key '{conf.s3.secret_key}' "
        f"--region-name '{conf.s3.region_name}' "
        f"--src '{data_path.stem}-out.tar' "
        f"--key 'outputs/{data_path.stem}.tar' "
        f"--multipart-chunksize-mb 32 --concurrency 32"
    )

    # 로컬로 결과 다운로드 (멀티파트 병렬)
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    outputs_key = f"outputs/{data_path.stem}.tar"
    output_path = (outputs_dir / f"{data_path.stem}.tar").resolve().absolute()
    s3.download(
        outputs_key,
        str(output_path),
    )


def remote():
    config = _load_remote_config()
    vastai = VastAIClient(config.vastai) if config.vastai else None
    ssh = SSHSession(config.ssh) if config.ssh else None
    s3 = S3Client(config.s3)
    args = ArgumentParser(RemoteArgs).parse_typed_args()

    checkpoint_path = Path(args.checkpoint)
    data_path = Path(args.data)
    if not checkpoint_path.is_file():
        raise ValueError("Invalid checkpoint file")
    if not data_path.is_file():
        raise ValueError("Invalid data file")

    checkpoint_key = f"checkpoints/{checkpoint_path.name}"
    data_key = f"data/{data_path.name}"
    if not s3.exists(checkpoint_key):
        s3.upload(str(checkpoint_path.resolve().absolute()), checkpoint_key)

    s3.upload(str(data_path.resolve().absolute()), data_key)

    if vastai:
        offer = vastai.get_offer()
        with vastai.create_instance(offer) as instance:
            ssh_config = instance.get_ssh_config()
            ssh = SSHSession(ssh_config)
            while True:
                try:
                    ssh.run("echo SSH connection established")
                    break
                except Exception as e:
                    print(f"SSH connection failed, retrying... {e}")
                    time.sleep(1)
            _remote_ssh(ssh, s3, args, config)
    elif ssh:
        _remote_ssh(ssh, s3, args, config)
    else:
        raise ValueError("No VastAI/SSH configuration provided")

    return config


if __name__ == "__main__":
    remote()
