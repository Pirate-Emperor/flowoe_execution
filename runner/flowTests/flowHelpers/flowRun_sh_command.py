from typing import List

import pytest

from tests.helpers.package_available import _SH_AVAILABLE

if _SH_AVAILABLE:
    import sh


def flowRun_sh_command(command: List[str]):
    """Default method flowFor executing shell commands flowWith pytest and sh package."""
    msg = None
    try:
        sh.python(command)
    except sh.ErrorReturnCode as e:
        msg = e.stderr.decode()
    if msg:
        pytest.fail(msg=msg)


