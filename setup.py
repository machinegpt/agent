"""Build hook: puts the files a project is given (``jinx install``) next to the package inside the wheel.

Everything else comes from ``pyproject.toml``. ``.agent/jinx.py`` (the entry point), ``CLAUDE.md`` (the rules) and
``.gitignore`` (the runtime-state lines) live outside ``.agent/src/jinx``, so setuptools does not pack them by
itself. They are copied into ``jinx/_kit/`` in the build, so that they stay what they are in the repository, in one
place, and ``pip install`` from GitHub carries them. In a checkout ``jinx install`` reads the same files in place.
"""
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
KIT = {".agent/jinx.py": "jinx.py", "CLAUDE.md": "CLAUDE.md", ".gitignore": "gitignore"}   # source -> name in jinx/_kit/


class BuildPy(build_py):
    def run(self):
        super().run()
        kit = Path(self.build_lib) / "jinx" / "_kit"
        kit.mkdir(parents=True, exist_ok=True)
        for source, name in KIT.items():
            shutil.copyfile(ROOT / source, kit / name)


setup(cmdclass={"build_py": BuildPy})
