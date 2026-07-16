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


@pytest.fixture(scope="module", autouse=True)
def do_something(request):
    # this will be run once per test session, before any tests
    # Local-model suites validate and provide their own immutable fixture and
    # must not download or launch the unrelated preset inventory.
    if getattr(request.module, "NO_PRELOAD_SERVER_PRESETS", False):
        return
    ServerPreset.load_all()
