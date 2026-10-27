'''
Paradigm tests that play through an RME Fireface looped back on itself.

These need the hardware, so they are skipped unless pytest is run with
`--fireface`:

    pytest tests/hardware --fireface

Each run writes its data, psi's log, and a report with plots under
`tests/hardware/results/<date time>/` (or `--fireface-results`); open
`index.html` there to look through them.
'''
import datetime as dt
import os
from pathlib import Path
import sys

import pytest

from report import write_index
from rig import measure_routing


def pytest_addoption(parser):
    parser.addoption('--fireface', action='store_true',
                     help='Run the paradigm tests through the Fireface')
    parser.addoption('--fireface-results', default=None,
                     help='Folder for the run data and reports')
    parser.addoption('--src', action='append', default=[],
                     help='Put this source folder ahead of the installed '
                     'packages when running paradigms (e.g., a worktree of '
                     'another cfts branch, or psiexperiment/src). May be '
                     'given more than once. Write it as --src=PATH: given '
                     'as two arguments, pytest also takes PATH as a '
                     'candidate root directory.')


def pytest_configure(config):
    src = config.getoption('--src')
    if src:
        # The paradigms run in subprocesses, which inherit this.
        resolved = [str(Path(s).resolve()) for s in src]
        paths = resolved + [os.environ.get('PYTHONPATH')]
        os.environ['PYTHONPATH'] = os.pathsep.join(p for p in paths if p)
        # Some tests also call paradigm code directly.
        sys.path[:0] = resolved


def pytest_collection_modifyitems(config, items):
    if config.getoption('--fireface'):
        return
    skip = pytest.mark.skip(reason='needs the Fireface; run with --fireface')
    for item in items:
        if 'hardware' in item.path.parts:
            item.add_marker(skip)


@pytest.fixture(scope='session')
def results_dir(request):
    path = request.config.getoption('--fireface-results')
    if path is None:
        stamp = dt.datetime.now().strftime('%Y-%m-%d %H%M%S')
        path = Path(__file__).parent / 'results' / stamp
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope='session')
def routing(results_dir):
    '''
    How TotalMix currently routes outputs to inputs.
    '''
    from plots import plot_routing
    routing = measure_routing()
    plot_routing(routing, results_dir / 'routing.png')
    write_index(results_dir)
    return routing
