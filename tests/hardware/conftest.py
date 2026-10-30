'''
Paradigm tests that play through an RME Fireface looped back on itself.

These need the hardware, so they are skipped unless pytest is run with
`--fireface`:

    pytest tests/hardware --fireface

Each run writes its data, psi's log, and a report with plots under
`tests/hardware/results/<date time>/` (or `--fireface-results`); open
`index.html` there to look through them. The options, the routing probe and
the runner are shared with other psi programs in `psi.loopback`.
'''
import pytest

from psi.loopback.pytest_plugin import *  # noqa: F403

from rig import RIG


@pytest.fixture(scope='session')
def rig():
    return RIG
