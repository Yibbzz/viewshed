"""Every recorded golden output must be reproduced exactly (see conformance/golden.py)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformance import golden  # noqa: E402
from conformance.adapters import Rust  # noqa: E402


@pytest.mark.skipif(not Rust("rviewshed").available(), reason="Rust toolchain not available")
def test_golden_outputs_are_unchanged():
    problems, _ = golden.check()
    assert not problems, "\n".join(problems)
