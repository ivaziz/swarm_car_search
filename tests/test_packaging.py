#!/usr/bin/env python3
"""
Unit and integration tests for system packaging, CLI entry points,
and documentation integrity.
"""

import os
import sys
import unittest
from io import StringIO
from unittest.mock import patch

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestPackagingMetadata(unittest.TestCase):
    """Tests package structure, packaging files, and importability."""

    def test_packaging_files_exist(self):
        """Verifies setup.py, pyproject.toml, requirements.txt, and LICENSE exist."""
        required_files = ["setup.py", "pyproject.toml", "requirements.txt", "LICENSE", "README.md"]
        for fname in required_files:
            fpath = os.path.join(PROJECT_ROOT, fname)
            self.assertTrue(os.path.isfile(fpath), f"Missing required file: {fname}")

    def test_core_package_imports(self):
        """Ensures all top-level packages can be imported without error."""
        packages = [
            "interfaces",
            "slam_providers",
            "perception",
            "swarm",
            "simulation",
            "visualization",
            "evaluation",
            "scripts",
        ]
        for pkg_name in packages:
            __import__(pkg_name)
            self.assertIn(pkg_name, sys.modules)

    def test_docs_exist_and_nonempty(self):
        """Verifies all referenced thesis documentation files exist and are populated."""
        doc_files = [
            "docs/thesis_chapter_swarm.md",
            "docs/slam_integration_guide.md",
            "docs/user_guide.md",
            "docs/architecture.md",
            "docs/aco_algorithm.md",
            "docs/slam_interface_spec.md",
        ]
        for doc in doc_files:
            fpath = os.path.join(PROJECT_ROOT, doc)
            self.assertTrue(os.path.isfile(fpath), f"Missing documentation file: {doc}")
            size = os.path.getsize(fpath)
            self.assertGreater(size, 500, f"Documentation file '{doc}' appears truncated ({size} bytes).")


class TestUnifiedCLI(unittest.TestCase):
    """Tests the unified CLI launcher in scripts/main.py."""

    def test_cli_banner_output(self):
        """Verifies the system banner prints properly."""
        from scripts.main import print_system_banner
        captured = StringIO()
        with patch("sys.stdout", captured):
            print_system_banner()
        output = captured.getvalue()
        self.assertIn("AUTONOMOUS MULTI-ROBOT SWARM", output)
        self.assertIn("Mohammed & Ibrahim", output)
        self.assertIn("Saqr & Mahmoud", output)

    def test_cli_info_command(self):
        """Verifies 'info' command prints architecture and CLI guidance."""
        from scripts.main import run_info_command
        captured = StringIO()
        with patch("sys.stdout", captured):
            run_info_command([])
        output = captured.getvalue()
        self.assertIn("Multi-Objective ACO Task Scorer", output)
        self.assertIn("Multi-Layer Stigmergic Pheromone Field", output)
        self.assertIn("swarm sim", output)
        self.assertIn("swarm eval", output)

    def test_cli_help_handling(self):
        """Verifies help menu exit behavior."""
        from scripts.main import main
        with patch("sys.argv", ["main.py", "--help"]), patch("sys.stdout", StringIO()):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)

    def test_cli_invalid_command(self):
        """Verifies graceful rejection of unknown subcommands."""
        from scripts.main import main
        with patch("sys.argv", ["main.py", "invalid_cmd"]), patch("sys.stdout", StringIO()):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
