"""
Unit and Integration Tests for Ant Colony Optimization (ACO) Engine and Task Scorer.
"""

import unittest
import random
from interfaces.slam_interface import Pose2D
from swarm.task_allocation.frontier_task import SearchTask, TaskStatus
from swarm.pheromone.pheromone_map import PheromoneMap
from swarm.pheromone.pheromone_updater import PheromoneUpdater
from swarm.aco.aco_config import ACOConfig, ACOWeights, StochasticParams
from swarm.aco.task_scorer import TaskScorer
from swarm.aco.aco_engine import ACOEngine


class TestACOConfig(unittest.TestCase):
    """Tests for ACO parameter configurations and YAML loading."""

    def test_default_config(self):
        config = ACOConfig()
        self.assertEqual(config.weights.alpha, 1.2)
        self.assertEqual(config.weights.beta, 1.5)
        self.assertEqual(config.stochastic.temperature, 1.0)

    def test_load_from_yaml(self):
        config = ACOConfig.from_yaml("configs/aco_params.yaml")
        self.assertEqual(config.weights.gamma, 1.8)
        self.assertEqual(config.weights.zeta, 2.0)
        self.assertEqual(config.stochastic.temperature, 1.0)


class TestTaskScorer(unittest.TestCase):
    """Tests for mathematical sensitivity of multi-objective scoring formula."""

    def setUp(self):
        self.config = ACOConfig()
        self.scorer = TaskScorer(self.config)
        self.robot_pose = Pose2D(x=10.0, y=10.0, theta=0.0)

    def _make_task(self, task_id="t1", x=15.0, y=10.0, size=10, p_det=0.5, priority=1.0):
        return SearchTask(
            task_id=task_id,
            cluster_id=f"c_{task_id}",
            centroid_x=x,
            centroid_y=y,
            bounding_box=(x - 1, y - 1, x + 1, y + 1),
            size=size,
            detection_probability=p_det,
            priority=priority,
        )

    def test_information_gain_sensitivity(self):
        # Larger cluster size should yield higher score
        small_task = self._make_task("small", size=5)
        large_task = self._make_task("large", size=50)

        score_small = self.scorer.compute_score(small_task, self.robot_pose)
        score_large = self.scorer.compute_score(large_task, self.robot_pose)
        self.assertGreater(score_large, score_small)

    def test_distance_penalty_sensitivity(self):
        # Farther task should yield lower score
        near_task = self._make_task("near", x=12.0, y=10.0)  # Dist = 2
        far_task = self._make_task("far", x=30.0, y=10.0)    # Dist = 20

        score_near = self.scorer.compute_score(near_task, self.robot_pose)
        score_far = self.scorer.compute_score(far_task, self.robot_pose)
        self.assertGreater(score_near, score_far)

    def test_detection_prior_sensitivity(self):
        # Higher prior probability of vehicle should yield higher score
        low_prior = self._make_task("low_p", p_det=0.2)
        high_prior = self._make_task("high_p", p_det=0.9)

        score_low = self.scorer.compute_score(low_prior, self.robot_pose)
        score_high = self.scorer.compute_score(high_prior, self.robot_pose)
        self.assertGreater(score_high, score_low)

    def test_congestion_penalty_sensitivity(self):
        # Task targeted by another robot should be heavily penalized
        task = self._make_task("congested")

        score_uncongested = self.scorer.compute_score(task, self.robot_pose, congestion_count=0)
        score_congested_1 = self.scorer.compute_score(task, self.robot_pose, congestion_count=1)
        score_congested_2 = self.scorer.compute_score(task, self.robot_pose, congestion_count=2)

        self.assertGreater(score_uncongested, score_congested_1)
        self.assertGreater(score_congested_1, score_congested_2)

    def test_pheromone_influence(self):
        pmap = PheromoneMap(width=30.0, height=30.0, resolution=1.0)
        updater = PheromoneUpdater(pmap)

        task_a = self._make_task("a", x=15.0, y=10.0)
        task_b = self._make_task("b", x=10.0, y=15.0)

        # Deposit success pheromone on task A
        updater.deposit_success(15.0, 10.0, amount=20.0, radius=2.0)

        score_a = self.scorer.compute_score(task_a, self.robot_pose, pheromone_map=pmap)
        score_b = self.scorer.compute_score(task_b, self.robot_pose, pheromone_map=pmap)

        self.assertGreater(score_a, score_b)


class TestACOEngine(unittest.TestCase):
    """Tests for ranking, deterministic selection, and Boltzmann Softmax sampling."""

    def setUp(self):
        self.config = ACOConfig()
        self.engine = ACOEngine(self.config)
        self.robot_pose = Pose2D(x=10.0, y=10.0, theta=0.0)

    def _make_task(self, task_id, x, y, size=10, status=TaskStatus.PENDING):
        task = SearchTask(
            task_id=task_id,
            cluster_id=f"c_{task_id}",
            centroid_x=x,
            centroid_y=y,
            bounding_box=(x - 1, y - 1, x + 1, y + 1),
            size=size,
            status=status,
        )
        return task

    def test_ranking(self):
        task_near = self._make_task("near", 12.0, 10.0, size=20)
        task_far = self._make_task("far", 30.0, 30.0, size=5)

        ranked = self.engine.score_tasks([task_far, task_near], self.robot_pose)
        self.assertEqual(ranked[0][0].task_id, "near")
        self.assertEqual(ranked[1][0].task_id, "far")

    def test_deterministic_selection(self):
        self.engine.config.stochastic.deterministic = True

        task_near = self._make_task("near", 12.0, 10.0, size=20)
        task_far = self._make_task("far", 30.0, 30.0, size=5)

        chosen = self.engine.select_task([task_far, task_near], self.robot_pose)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.task_id, "near")

    def test_filter_only_pending_tasks(self):
        self.engine.config.stochastic.deterministic = True

        task_assigned = self._make_task("assigned", 11.0, 10.0, size=30, status=TaskStatus.ASSIGNED)
        task_completed = self._make_task("completed", 11.0, 11.0, size=30, status=TaskStatus.COMPLETED)
        task_pending = self._make_task("pending", 20.0, 20.0, size=5, status=TaskStatus.PENDING)

        chosen = self.engine.select_task([task_assigned, task_completed, task_pending], self.robot_pose)
        self.assertIsNotNone(chosen)
        # Even though assigned and completed were closer/larger, only pending can be selected
        self.assertEqual(chosen.task_id, "pending")

    def test_congestion_dispersion(self):
        # Two identical tasks equidistant from robot
        task_1 = self._make_task("task_1", 15.0, 10.0, size=15)
        task_2 = self._make_task("task_2", 10.0, 15.0, size=15)

        # Peer robot is already targeting task_1
        peer_commitments = {"robot_2": "task_1"}

        self.engine.config.stochastic.deterministic = True
        chosen = self.engine.select_task(
            [task_1, task_2],
            self.robot_pose,
            peer_commitments=peer_commitments,
        )

        # Swarm avoids congestion: picks task_2
        self.assertEqual(chosen.task_id, "task_2")

    def test_softmax_distribution(self):
        self.engine.config.stochastic.deterministic = False
        self.engine.config.stochastic.temperature = 0.5
        self.engine.config.stochastic.epsilon_greedy = 0.0

        task_best = self._make_task("best", 11.0, 10.0, size=30)
        task_mediocre = self._make_task("mediocre", 16.0, 10.0, size=10)

        rng = random.Random(42)
        selections = {"best": 0, "mediocre": 0}

        # Run 100 trials
        for _ in range(100):
            selected = self.engine.select_task([task_best, task_mediocre], self.robot_pose, rng=rng)
            selections[selected.task_id] += 1

        # The best task should be chosen much more frequently, but mediocre is still occasionally explored
        self.assertGreater(selections["best"], 80)
        self.assertGreater(selections["mediocre"], 0)


if __name__ == "__main__":
    unittest.main()
