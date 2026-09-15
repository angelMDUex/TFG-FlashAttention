import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--seq-len",
        action="store",
        type=int,
        default=65536,
    )


@pytest.fixture
def seq_len(request):
    return request.config.getoption("--seq-len")
