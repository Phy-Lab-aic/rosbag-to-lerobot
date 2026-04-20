"""Writers for meta/aic/{task,scoring,scene,tf_snapshots}.parquet."""

from pathlib import Path
from typing import Any, Dict, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from . import schemas


def _write_rows(
    target: Path, rows: Iterable[Dict[str, Any]], schema: pa.Schema
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, target)


def write_task_parquet(target: Path, rows: Iterable[Dict[str, Any]]) -> None:
    _write_rows(target, rows, schemas.TASK_SCHEMA)


def write_scoring_parquet(target: Path, rows: Iterable[Dict[str, Any]]) -> None:
    _write_rows(target, rows, schemas.SCORING_SCHEMA)
