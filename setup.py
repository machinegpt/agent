"""Build hook: puts the files a project is given (``jinx install``) next to the package inside the wheel.

Everything else comes from ``pyproject.toml``. ``.agent/jinx.py`` (the entry point), ``CLAUDE.md`` (the rules),
``.gitignore`` (the runtime-state lines) and the text of the license (``LICENSE.md``, ``LICENSE_RU.md``,
``LICENSE_ZH.md``) live outside ``.agent/src/jinx``, so setuptools does not pack them by itself. They are copied into
``jinx/_kit/`` in the build, so that they stay what they are in the repository, in one place, and ``pip install`` from
GitHub carries them. In a checkout ``jinx install`` reads the same files in place.
"""
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
KIT = {                                                    # source -> name in jinx/_kit/
    ".agent/jinx.py": "jinx.py", "CLAUDE.md": "CLAUDE.md", ".gitignore": "gitignore",
    "LICENSE.md": "LICENSE.md", "LICENSE_RU.md": "LICENSE_RU.md", "LICENSE_ZH.md": "LICENSE_ZH.md",
}


class BuildPy(build_py):
    """``build_py`` that also copies the kit (``KIT``) into ``jinx/_kit/`` of the build."""

    def run(self):
        """Builds as usual, then copies the files of the kit next to the package."""
        super().run()
        kit = Path(self.build_lib) / "jinx" / "_kit"
        kit.mkdir(parents=True, exist_ok=True)
        for source, name in KIT.items():
            shutil.copyfile(ROOT / source, kit / name)


setup(cmdclass={"build_py": BuildPy})
