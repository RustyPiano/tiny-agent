from __future__ import annotations

import os
from contextlib import contextmanager
from tempfile import TemporaryDirectory


@contextmanager
def sandbox_cwd():
    original = os.getcwd()
    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        try:
            yield tmp_dir
        finally:
            os.chdir(original)
