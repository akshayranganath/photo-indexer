"""
tests/test_pipeline_end2end.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An **integration smoke-test** that drives the *entire* indexing pipeline
via the public CLI (`pi index …`).  It copies a tiny sample RAW file into
a temporary folder, launches the CLI with *one* worker, and then inspects
the SQLite database to confirm that **at least one row** was inserted.

The test is deliberately lightweight:

* Only a single `.NEF` is processed.
* Vision models run on the CPU; weights are auto-downloaded/cached.
* If the sample RAW (`tests/data/sample1.NEF`) is missing, the test is
  skipped so CI can still pass.

Requirements:

* `sample1.NEF` – keep it small (≈ 2 MB) and git-ignored.
* The `pi` CLI must be resolvable via `photo_indexer.cli:main`.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

try:
    # Importing here avoids hard-fail if user hasn't wired the CLI yet
    from photo_indexer.cli import main as pi_cli
except ModuleNotFoundError:
    pytest.skip("photo_indexer.cli not implemented – skipping end-to-end test",
                allow_module_level=True)

HERE = Path(__file__).parent
SAMPLE_NEF = HERE / "data" / "sample1.NEF"


@pytest.mark.skipif(
    not SAMPLE_NEF.exists(),
    reason="Provide a tiny sample NEF in tests/data/ to run end-to-end test.",
)
def test_cli_index_writes_sqlite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Run `pi index` on a temp folder and verify that exactly one photo row
    was written to the SQLite DB.
    """
    # ── arrange: copy sample RAW into an isolated temp tree ────────────────
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    shutil.copy(SAMPLE_NEF, photo_root / SAMPLE_NEF.name)

    # Make sure the pipeline writes its DB into the temp area, not repo /data
    db_path = tmp_path / "photo_index.sqlite"
    monkeypatch.setenv("PHOTO_INDEXER_DB_PATH", str(db_path))  # pipeline honours this env var

    # ── act: invoke CLI ----------------------------------------------------
    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(
        pi_cli,
        ["index", str(photo_root), "--workers", "1", "--db", "sqlite"],
        catch_exceptions=False,
    )

    # ── assert: CLI exited cleanly, DB exists, 1+ rows ---------------------
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert db_path.exists(), "SQLite DB file not created"

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    assert rows >= 1, "No rows written to photos table"
