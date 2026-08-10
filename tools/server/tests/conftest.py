import pytest
from utils import *


# ref: https://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test-in-py-test
@pytest.fixture(autouse=True)
def stop_server_after_each_test():
    # do nothing before each test
    yield
    # stop all servers after each test
    instances = set(
        server_instances
    )  # copy the set to prevent 'Set changed size during iteration'
    for server in instances:
        server.stop()


_server_presets_loaded = False


@pytest.fixture(scope="module", autouse=True)
def load_server_presets(request):
    global _server_presets_loaded

    # Local-model suites validate and provide their own immutable fixture and
    # must not download or launch the unrelated preset inventory.
    if getattr(request.module, "NO_PRELOAD_SERVER_PRESETS", False):
        return
    if not _server_presets_loaded:
        ServerPreset.load_all()
        _server_presets_loaded = True
