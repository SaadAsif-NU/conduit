from __future__ import annotations

import pytest

from conduit import __version__
from conduit.cli import main


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "conduit" in capsys.readouterr().out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out
