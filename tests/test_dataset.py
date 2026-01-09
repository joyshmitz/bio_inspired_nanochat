from __future__ import annotations

from pathlib import Path

from bio_inspired_nanochat.dataset import parquet_paths_for_split


def test_parquet_paths_for_split_single_file(tmp_path: Path) -> None:
    (tmp_path / "shard_00000.parquet").write_bytes(b"")

    assert parquet_paths_for_split("train", data_dir=str(tmp_path)) == [
        str(tmp_path / "shard_00000.parquet")
    ]
    assert parquet_paths_for_split("val", data_dir=str(tmp_path)) == [
        str(tmp_path / "shard_00000.parquet")
    ]


def test_parquet_paths_for_split_two_files(tmp_path: Path) -> None:
    (tmp_path / "shard_00000.parquet").write_bytes(b"")
    (tmp_path / "shard_00001.parquet").write_bytes(b"")

    assert parquet_paths_for_split("train", data_dir=str(tmp_path)) == [
        str(tmp_path / "shard_00000.parquet")
    ]
    assert parquet_paths_for_split("val", data_dir=str(tmp_path)) == [
        str(tmp_path / "shard_00001.parquet")
    ]

