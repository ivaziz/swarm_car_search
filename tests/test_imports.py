"""
Smoke tests verifying that all packages and modules in the repository are cleanly importable.
"""

import unittest
import importlib


class TestPackageImports(unittest.TestCase):
    """Verifies that all defined packages and skeleton modules import without error."""

    def test_import_core_packages(self):
        packages = [
            "interfaces",
            "swarm",
            "swarm.core",
            "swarm.aco",
            "swarm.task_allocation",
            "swarm.communication",
            "swarm.states",
            "slam_providers",
            "perception",
            "simulation",
            "visualization",
            "evaluation",
            "scripts",
        ]
        for pkg in packages:
            with self.subTest(package=pkg):
                mod = importlib.import_module(pkg)
                self.assertIsNotNone(mod)

    def test_import_interfaces(self):
        import interfaces
        self.assertTrue(hasattr(interfaces, "ISLAMProvider"))
        self.assertTrue(hasattr(interfaces, "IPerceptionProvider"))
        self.assertTrue(hasattr(interfaces, "IServerReporter"))
        self.assertTrue(hasattr(interfaces, "SLAMState"))
        self.assertTrue(hasattr(interfaces, "Pose2D"))


if __name__ == "__main__":
    unittest.main()
