"""Keep pytest scratch data outside this checkout, with native cleanup."""
import hashlib
import os
from pathlib import Path
import tempfile


def pytest_configure(config):
    # Respect explicit caller overrides. Never touch a pre-existing generic
    # pytest-of-User directory or any project data/model/result directory.
    if config.option.basetemp or 'PYTEST_DEBUG_TEMPROOT' in os.environ:
        return
    identity = hashlib.sha256(str(config.rootpath.resolve()).encode()).hexdigest()[:12]
    root = Path(tempfile.gettempdir()) / f'gev-use-nn-pytest-{identity}'
    root.mkdir(parents=True, exist_ok=True)
    os.environ['PYTEST_DEBUG_TEMPROOT'] = str(root)
    config.add_cleanup(lambda: os.environ.pop('PYTEST_DEBUG_TEMPROOT', None))
