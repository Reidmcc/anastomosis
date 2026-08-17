"""The control panel only exists if the Qt backend is actually selected.

The panel shares the render window's event loop, so "did the Qt backend load"
is the single thing the panel's existence hangs on. A regression there is
invisible: the backend import fails, the app quietly falls back to a windowing
backend that has no panel, and the only trace is one warning line.

These tests run in a subprocess with a clean interpreter, because the failure
mode is entirely about *import order* -- ``rendercanvas.qt`` binds to whichever
Qt library is already in ``sys.modules``, so a test that had imported PySide6
first would pass while the real application, which has not, fails.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pyside6 = pytest.importorskip("PySide6", reason="the control panel needs PySide6")


def run_isolated(source: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
    )


def test_qt_backend_imports_from_a_clean_interpreter():
    """The import must not depend on PySide6 already having been imported.

    ``rendercanvas.qt`` raises ImportError unless a Qt library is in
    ``sys.modules`` first. The application imports the backend before touching
    PySide6, so binding to that module means the panel never opens on any run.
    """
    result = run_isolated(
        """
        import sys
        assert "PySide6" not in sys.modules, "PySide6 must not be pre-imported"

        from anastomosis.app import Application

        RenderCanvas, loop = Application._import_qt_backend()
        assert RenderCanvas is not None and loop is not None
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_qt_backend_will_run_its_own_event_loop():
    """No QApplication may exist before the backend is imported.

    rendercanvas records at import time whether an application object already
    existed and, if one did, assumes it is embedded and ``loop.run()`` becomes
    a no-op -- the window would come up and immediately fall through to exit.
    """
    result = run_isolated(
        """
        from anastomosis.app import Application

        Application._import_qt_backend()

        import rendercanvas.qt as qt
        assert not qt.already_had_app_on_import, (
            "a QApplication existed before the backend was imported; "
            "loop.run() would return immediately"
        )
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_missing_pyside6_raises_importerror_so_the_fallback_is_reached():
    """The fallback in ``_make_canvas`` keys off ImportError specifically."""
    result = run_isolated(
        """
        import importlib.abc
        import sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "PySide6" or name.startswith("PySide6."):
                    raise ModuleNotFoundError(f"No module named {name!r}")

        sys.meta_path.insert(0, Block())

        from anastomosis.app import Application

        try:
            Application._import_qt_backend()
        except ImportError:
            print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
