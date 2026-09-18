"""
Comprehensive Unit Tests for Tri-Layer Pheromone Map, Evaporation, and Diffusion.
"""

import unittest
from swarm.pheromone.evaporation_model import PheromoneLayerType, EvaporationConfig, EvaporationModel
from swarm.pheromone.pheromone_map import PheromoneMap
from swarm.pheromone.pheromone_updater import PheromoneUpdater


class TestEvaporationModel(unittest.TestCase):
    """Tests for discrete decay factors and bounds clamping."""

    def setUp(self):
        self.config = EvaporationConfig(min_pheromone=0.01, max_pheromone=50.0)
        self.model = EvaporationModel(self.config)

    def test_decay_factors(self):
        decay_exp = self.model.get_decay_factor(PheromoneLayerType.EXPLORATION, dt=1.0)
        decay_suc = self.model.get_decay_factor(PheromoneLayerType.SUCCESS, dt=1.0)
        decay_avo = self.model.get_decay_factor(PheromoneLayerType.AVOIDANCE, dt=1.0)

        self.assertAlmostEqual(decay_exp, 0.99)
        self.assertAlmostEqual(decay_suc, 0.95)
        self.assertAlmostEqual(decay_avo, 0.70)

        # Avoidance decays faster (lower retention factor) than exploration
        self.assertLess(decay_avo, decay_suc)
        self.assertLess(decay_suc, decay_exp)

    def test_clamping(self):
        self.assertEqual(self.model.clamp(-5.0), 0.01)
        self.assertEqual(self.model.clamp(0.001), 0.01)
        self.assertEqual(self.model.clamp(100.0), 50.0)
        self.assertEqual(self.model.clamp(25.0), 25.0)


class TestPheromoneMap(unittest.TestCase):
    """Tests for spatial indexing, multi-layer separation, and aggregate queries."""

    def setUp(self):
        self.pmap = PheromoneMap(width=20.0, height=20.0, resolution=1.0)

    def test_initialization(self):
        self.assertEqual(self.pmap.grid_width, 20)
        self.assertEqual(self.pmap.grid_height, 20)
        self.assertEqual(self.pmap.total_cells, 400)

        # Baseline value must be min_pheromone
        val = self.pmap.get_value(PheromoneLayerType.EXPLORATION, 5.0, 5.0)
        self.assertEqual(val, self.pmap.config.min_pheromone)

    def test_discrete_and_world_coordinates(self):
        gx, gy = self.pmap.world_to_grid(5.2, 8.7)
        self.assertEqual((gx, gy), (5, 8))
        wx, wy = self.pmap.grid_to_world(gx, gy)
        self.assertAlmostEqual(wx, 5.5)
        self.assertAlmostEqual(wy, 8.5)

        # Out of bounds returns None or min_pheromone
        self.assertIsNone(self.pmap.world_to_grid(-1.0, 5.0))
        self.assertEqual(self.pmap.get_value(PheromoneLayerType.EXPLORATION, -5.0, 10.0), 0.01)

    def test_layer_independence(self):
        # Setting a value on SUCCESS layer should not affect EXPLORATION layer
        self.pmap.set_value_grid(PheromoneLayerType.SUCCESS, 5, 5, 20.0)
        self.assertEqual(self.pmap.get_value_grid(PheromoneLayerType.SUCCESS, 5, 5), 20.0)
        self.assertEqual(self.pmap.get_value_grid(PheromoneLayerType.EXPLORATION, 5, 5), 0.01)
        self.assertEqual(self.pmap.get_value_grid(PheromoneLayerType.AVOIDANCE, 5, 5), 0.01)

    def test_box_aggregates(self):
        # Populate a 3x3 patch
        for gx in range(5, 8):
            for gy in range(5, 8):
                self.pmap.set_value_grid(PheromoneLayerType.EXPLORATION, gx, gy, 10.0)

        # Box query covering patch (5 to 7.9 meters)
        mean_val = self.pmap.get_box_aggregate(
            PheromoneLayerType.EXPLORATION, min_x=5.0, min_y=5.0, max_x=7.5, max_y=7.5, mode="mean"
        )
        max_val = self.pmap.get_box_aggregate(
            PheromoneLayerType.EXPLORATION, min_x=5.0, min_y=5.0, max_x=7.5, max_y=7.5, mode="max"
        )
        self.assertAlmostEqual(mean_val, 10.0)
        self.assertAlmostEqual(max_val, 10.0)


class TestPheromoneUpdater(unittest.TestCase):
    """Tests for deposition, kernel diffusion, and evaporation execution."""

    def setUp(self):
        self.pmap = PheromoneMap(width=30.0, height=30.0, resolution=1.0)
        self.updater = PheromoneUpdater(self.pmap)

    def test_point_deposit(self):
        self.updater.deposit(PheromoneLayerType.EXPLORATION, 15.0, 15.0, amount=5.0, radius=0.0)
        val = self.pmap.get_value(PheromoneLayerType.EXPLORATION, 15.0, 15.0)
        self.assertAlmostEqual(val, 5.01)  # 0.01 baseline + 5.0

    def test_gaussian_radial_kernel_deposit(self):
        # Deposit with 3 meter radius
        self.updater.deposit(PheromoneLayerType.SUCCESS, 15.0, 15.0, amount=10.0, radius=3.0)

        center_val = self.pmap.get_value(PheromoneLayerType.SUCCESS, 15.0, 15.0)
        neighbor_1m = self.pmap.get_value(PheromoneLayerType.SUCCESS, 16.0, 15.0)
        neighbor_2m = self.pmap.get_value(PheromoneLayerType.SUCCESS, 17.0, 15.0)
        outside_4m = self.pmap.get_value(PheromoneLayerType.SUCCESS, 19.5, 15.0)

        # Peak at center
        self.assertGreater(center_val, neighbor_1m)
        self.assertGreater(neighbor_1m, neighbor_2m)
        # Outside radius should remain at baseline
        self.assertAlmostEqual(outside_4m, 0.01)

    def test_evaporation_dynamics(self):
        # Deposit equal amounts on exploration and avoidance
        self.updater.deposit_exploration(10.0, 10.0, amount=10.0, radius=0.0)
        self.updater.deposit_avoidance(10.0, 10.0, amount=10.0, radius=0.0)

        exp_initial = self.pmap.get_value(PheromoneLayerType.EXPLORATION, 10.0, 10.0)
        avo_initial = self.pmap.get_value(PheromoneLayerType.AVOIDANCE, 10.0, 10.0)
        self.assertAlmostEqual(exp_initial, avo_initial)

        # Step evaporation 5 cycles
        for _ in range(5):
            self.updater.step_evaporation(dt=1.0)

        exp_after = self.pmap.get_value(PheromoneLayerType.EXPLORATION, 10.0, 10.0)
        avo_after = self.pmap.get_value(PheromoneLayerType.AVOIDANCE, 10.0, 10.0)

        # Both must have decayed
        self.assertLess(exp_after, exp_initial)
        self.assertLess(avo_after, avo_initial)

        # Avoidance must decay much faster than exploration:
        # After 5 steps: (0.99)^5 approx 0.95 vs (0.70)^5 approx 0.168
        self.assertGreater(exp_after, avo_after * 4.0)

    def test_stigmergic_peer_sync(self):
        self.updater.sync_from_peer_update(PheromoneLayerType.SUCCESS, 20.0, 20.0, amount=15.0, radius=2.0)
        val = self.pmap.get_value(PheromoneLayerType.SUCCESS, 20.0, 20.0)
        self.assertGreater(val, 10.0)


if __name__ == "__main__":
    unittest.main()
