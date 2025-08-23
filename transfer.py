from __future__ import annotations

import os
from typing import Literal, Optional

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from pydantic.v1 import BaseModel
from pydantic_argparse import ArgumentParser
from tqdm import tqdm


class S3CliArgs(BaseModel):
    action: Literal["upload", "download"]
    endpoint_url: str
    bucket: str
    access_key: str
    secret_key: str
    region_name: str = "auto"
    src: Optional[str] = None
    dst: Optional[str] = None
    key: Optional[str] = None
    multipart_chunksize_mb: int = 32
    concurrency: int = 0
    use_threads: bool = True


def _tqdm_cb(total: int, desc: str):
    class _CB:
        def __init__(self):
            self._bar = tqdm(
                total=total, unit="B", unit_scale=True, desc=desc, ascii=True
            )
            self._seen = 0

        def __call__(self, n: int):
            self._seen += n
            self._bar.update(n)

        def close(self):
            self._bar.close()

    return _CB()


def _default_concurrency(given: int) -> int:
    if given and given > 0:
        return given
    cpu = os.cpu_count() or 8
    return min(64, max(8, cpu * 4))


def _client(args: S3CliArgs):
    region = None if args.region_name == "auto" else args.region_name
    cfg = BotoConfig(
        signature_version="s3v4",
        s3={"addressing_style": "virtual"},
        retries={"max_attempts": 10, "mode": "adaptive"},
        connect_timeout=20,
        read_timeout=300,
        tcp_keepalive=True,
        max_pool_connections=max(32, _default_concurrency(args.concurrency) * 2),
    )
    return boto3.client(
        "s3",
        endpoint_url=args.endpoint_url,
        aws_access_key_id=args.access_key,
        aws_secret_access_key=args.secret_key,
        region_name=region,
        config=cfg,
    )


def main():
    args = ArgumentParser(S3CliArgs).parse_typed_args()
    cli = _client(args)
    conc = _default_concurrency(args.concurrency)
    mchunk = max(5, args.multipart_chunksize_mb) * 1024 * 1024

    tcfg = TransferConfig(
        multipart_threshold=mchunk,
        multipart_chunksize=mchunk,
        max_concurrency=conc,
        use_threads=args.use_threads,
        max_io_queue=conc * 2,
        io_chunksize=2 * 1024 * 1024,
    )

    if args.action == "upload":
        if args.src is None or args.key is None:
            raise SystemExit("upload에는 --src 와 --key 가 필요합니다.")
        total = os.path.getsize(args.src)
        cb = _tqdm_cb(total, "Upload")
        try:
            cli.upload_file(
                Filename=args.src,
                Bucket=args.bucket,
                Key=args.key,
                Callback=cb,
                Config=tcfg,
            )
        finally:
            cb.close()
    else:
        if args.key is None or args.dst is None:
            raise SystemExit("download에는 --key 와 --dst 가 필요합니다.")
        head = cli.head_object(Bucket=args.bucket, Key=args.key)
        total = int(head["ContentLength"])
        cb = _tqdm_cb(total, "Download")
        try:
            cli.download_file(
                Bucket=args.bucket,
                Key=args.key,
                Filename=args.dst,
                Callback=cb,
                Config=tcfg,
            )
        finally:
            cb.close()


if __name__ == "__main__":
    main()
