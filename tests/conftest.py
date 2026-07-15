import pytest

from lqr.controller import LqrController
from lqr.linearize import linearize
from lqr.model import build_model


@pytest.fixture(scope="session")
def rm():
    return build_model()


@pytest.fixture(scope="session")
def lin(rm):
    return linearize(rm)


@pytest.fixture(scope="session")
def ctrl(rm, lin):
    return LqrController(rm, lin)
