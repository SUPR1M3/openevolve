"""Tests for seeding islands from per-island initial programs (issue #55)."""

import tempfile
import unittest
from pathlib import Path

import yaml

from openevolve.config import Config


class TestIslandInitialProgramsConfig(unittest.TestCase):
    def test_defaults_to_empty(self):
        self.assertEqual(Config().database.island_initial_programs, [])
        self.assertEqual(Config.from_dict({}).database.island_initial_programs, [])

    def test_accepts_null_entries(self):
        config = Config.from_dict(
            {"database": {"num_islands": 3, "island_initial_programs": [None, "b.py", None]}}
        )
        self.assertEqual(config.database.island_initial_programs, [None, "b.py", None])

    def test_rejects_more_entries_than_islands(self):
        with self.assertRaisesRegex(ValueError, "island_initial_programs.*num_islands"):
            Config.from_dict(
                {"database": {"num_islands": 2, "island_initial_programs": [None, "b.py", "c.py"]}}
            )

    def test_rejects_non_null_entry_zero(self):
        for entry in ("a.py", ""):
            with self.subTest(entry=entry):
                with self.assertRaisesRegex(ValueError, r"island_initial_programs\[0\]"):
                    Config.from_dict(
                        {"database": {"num_islands": 2, "island_initial_programs": [entry]}}
                    )

    def test_relative_paths_resolve_against_config_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config_subdir"
            config_dir.mkdir()
            config_file = config_dir / "config.yaml"
            absolute_seed = str(Path(tmpdir) / "absolute.py")
            with open(config_file, "w") as f:
                yaml.dump(
                    {
                        "database": {
                            "num_islands": 4,
                            "island_initial_programs": [
                                None,
                                "seeds/one.py",
                                absolute_seed,
                                None,
                            ],
                        }
                    },
                    f,
                )

            config = Config.from_yaml(config_file)

            self.assertEqual(
                config.database.island_initial_programs,
                [None, str((config_dir / "seeds/one.py").resolve()), absolute_seed, None],
            )


if __name__ == "__main__":
    unittest.main()
