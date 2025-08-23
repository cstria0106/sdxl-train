import os
from typing import Dict, Literal, Optional

import boto3
import boto3.session
import botocore
import botocore.exceptions
from boto3.s3.transfer import TransferConfig
from botocore.client import BaseClient
from botocore.config import Config as BotoConfig
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic_argparse import ArgumentParser
from tqdm import tqdm


class S3Config(BaseModel):
    bucket: str
    access_key: str
    secret_key: str
    endpoint_url: str
    region_name: str = "auto"


class _TqdmCallback:
    """
    boto3 Transfer API의 Callback 인터페이스에 맞춘 tqdm 진행률 콜백
    """

    def __init__(self, total: int, desc: str):
        self._bar = tqdm(total=total, unit="B", unit_scale=True, desc=desc, ascii=True)
        self._seen_so_far = 0

    def __call__(self, bytes_amount: int):
        self._seen_so_far += bytes_amount
        self._bar.update(bytes_amount)

    def close(self):
        self._bar.close()


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _default_concurrency() -> int:
    cpu = os.cpu_count() or 8
    return min(64, max(8, cpu * 4))


def _default_chunk() -> int:
    # 32MiB 기본, 필요 시 환경변수로 조정
    return _env_int("S3_MULTIPART_CHUNKSIZE", 32 * 1024 * 1024)


class S3Client:
    def __init__(self, conf: S3Config):
        self.conf = conf
        self.client: BaseClient = self._create_client(conf)

    @staticmethod
    def _create_client(conf: S3Config) -> BaseClient:
        region = None if conf.region_name == "auto" else conf.region_name
        boto_conf = BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "virtual"},
            retries={"max_attempts": 10, "mode": "adaptive"},
            connect_timeout=20,
            read_timeout=300,
            tcp_keepalive=True,
            max_pool_connections=_env_int("S3_MAX_POOL", 64),
        )
        session = boto3.session.Session()
        client = session.client(
            "s3",
            endpoint_url=conf.endpoint_url,
            aws_access_key_id=conf.access_key,
            aws_secret_access_key=conf.secret_key,
            region_name=region,
            config=boto_conf,
        )
        return client

    # ---------- 존재 확인 ----------
    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.conf.bucket, Key=key)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    # ---------- boto3 Transfer API로 직접 업/다운로드 (멀티파트 자동) ----------
    def upload(
        self,
        src_path: str,
        key: str,
        content_type: Optional[str] = None,
        storage_class: Optional[str] = None,
        multipart_chunksize: int | None = None,
        multipart_threshold: int | None = None,
        max_concurrency: int | None = None,
        use_threads: bool = True,
        extra_args: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        boto3 고수준 전송으로 업로드 (멀티파트 자동)
        """
        file_size = os.path.getsize(src_path)
        mchunk = (
            _default_chunk() if multipart_chunksize is None else multipart_chunksize
        )
        mth = mchunk if multipart_threshold is None else multipart_threshold
        conc = _default_concurrency() if max_concurrency is None else max_concurrency

        cfg = TransferConfig(
            multipart_threshold=mth,
            multipart_chunksize=mchunk,
            max_concurrency=conc,
            use_threads=use_threads,
            max_io_queue=conc * 2,
            io_chunksize=2 * 1024 * 1024,
        )

        ea: Dict[str, str] = {} if extra_args is None else dict(extra_args)
        if content_type is not None:
            ea["ContentType"] = content_type
        if storage_class is not None:
            ea["StorageClass"] = storage_class

        cb = _TqdmCallback(total=file_size, desc="Upload")
        try:
            self.client.upload_file(
                Filename=src_path,
                Bucket=self.conf.bucket,
                Key=key,
                ExtraArgs=ea if len(ea) > 0 else None,
                Callback=cb,
                Config=cfg,
            )
        finally:
            cb.close()

    def download(
        self,
        key: str,
        dst_path: str,
        multipart_chunksize: int | None = None,
        max_concurrency: int | None = None,
        use_threads: bool = True,
    ) -> None:
        """
        boto3 고수준 전송으로 다운로드 (멀티파트 자동)
        """
        head = self.client.head_object(Bucket=self.conf.bucket, Key=key)
        total = int(head["ContentLength"])

        mchunk = (
            _default_chunk() if multipart_chunksize is None else multipart_chunksize
        )
        conc = _default_concurrency() if max_concurrency is None else max_concurrency

        cfg = TransferConfig(
            multipart_threshold=mchunk,
            multipart_chunksize=mchunk,
            max_concurrency=conc,
            use_threads=use_threads,
            max_io_queue=conc * 2,
            io_chunksize=2 * 1024 * 1024,
        )

        cb = _TqdmCallback(total=total, desc="Download")
        try:
            self.client.download_file(
                Bucket=self.conf.bucket,
                Key=key,
                Filename=dst_path,
                Callback=cb,
                Config=cfg,
            )
        finally:
            cb.close()


class S3CliArgs(BaseModelV1):
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


def _cli_default_concurrency(given: int) -> int:
    if given and given > 0:
        return min(64, given)
    return _default_concurrency()


def main():
    args = ArgumentParser(S3CliArgs).parse_typed_args()

    conf = S3Config(
        bucket=args.bucket,
        access_key=args.access_key,
        secret_key=args.secret_key,
        endpoint_url=args.endpoint_url,
        region_name=args.region_name,
    )
    s3 = S3Client(conf)

    # 멀티파트/동시성 파라미터 계산
    mchunk_mb = max(5, args.multipart_chunksize_mb)
    mchunk = mchunk_mb * 1024 * 1024
    conc = _cli_default_concurrency(args.concurrency)

    if args.action == "upload":
        if args.src is None or args.key is None:
            raise SystemExit("upload에는 --src 와 --key 가 필요합니다.")
        s3.upload(
            src_path=args.src,
            key=args.key,
            multipart_chunksize=mchunk,
            multipart_threshold=mchunk,
            max_concurrency=conc,
            use_threads=args.use_threads,
        )
    else:  # download
        if args.key is None or args.dst is None:
            raise SystemExit("download에는 --key 와 --dst 가 필요합니다.")
        s3.download(
            key=args.key,
            dst_path=args.dst,
            multipart_chunksize=mchunk,
            max_concurrency=conc,
            use_threads=args.use_threads,
        )


if __name__ == "__main__":
    main()
