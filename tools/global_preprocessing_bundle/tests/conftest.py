from pathlib import Path
from tempfile import TemporaryDirectory
import pytest


@pytest.fixture
def tmp_path():
    # Unique OS temporary directory; no project scratch folders or stale shared
    # pytest-of-User ACLs. Remove on normal success/failure.
    with TemporaryDirectory(prefix='era5_portable_test_') as root:
        yield Path(root)
