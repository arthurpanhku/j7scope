"""Include the shared, repo-authored viewer in built wheels."""
from pathlib import Path
from shutil import copytree
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        copytree(Path(__file__).parent / "apps/site",
                 Path(self.build_lib) / "j7scope_serve/static", dirs_exist_ok=True)
        from shutil import copy2
        target = Path(self.build_lib) / "j7scope_serve/data"
        target.mkdir(parents=True, exist_ok=True)
        for name in ("probe_prompts_en.jsonl", "probe_prompts_zh.jsonl"):
            copy2(Path(__file__).parent / "data" / name, target / name)


setup(cmdclass={"build_py": BuildPy})
