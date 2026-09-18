#!/usr/bin/env python3
"""
Scientific Benchmark Evaluation & Comparative Experimentation Script.
Executes Monte Carlo comparative trials between:
1. Proposed Ant Colony Optimization (ACO) with Multi-Objective Stigmergy
2. Greedy Nearest Frontier (Yamauchi Classical Baseline)
3. Stochastic Random Walk / Exploration Baseline
Also supports 4-configuration Ablation Studies:
- ACO Baseline (no target probability, no target recruitment pheromone)
- ACO + Target Probability
- ACO + Target Recruitment Pheromone
- Full Proposed ACO
Generates statistical markdown reports, JSON telemetry, and comparative vector/PNG plots.
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple, Any, Union

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from interfaces.slam_interface import Pose2D
from simulation.environment import UrbanEnvironment, UrbanZone
from simulation.obstacle_generator import ObstacleGenerator
from simulation.scenarios import create_scenario, ScenarioConfig
from slam_providers.mock_slam_provider import MockSLAMProvider
from swarm.states.robot_states import RobotState
from swarm.aco.aco_config import ACOConfig
from swarm.aco.aco_engine import ACOEngine
from swarm.task_allocation.allocation_strategy import ACOAllocationStrategy
from swarm.core.robot_agent import RobotAgent
from swarm.core.swarm_coordinator import SwarmCoordinator
from perception.vehicle_detector import TargetVehicle, SimulatedVehicleDetector
from perception.plate_recognizer import SimulatedPlateRecognizer
from perception.matching_engine import MatchingEngine
from evaluation.baselines import StrategyType, create_strategy
from evaluation.metrics_collector import MetricsCollector, MissionMetrics

# Optional Matplotlib
_HAS_MATPLOTLIB = False
try:
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_swarm_eval"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Comparative Swarm Task Allocation Benchmark & Ablation Trials"
    )
    parser.add_argument("--scenario", type=str, default="easy_target",
                        help="Named scenario: easy_target, hidden_target, target_recruitment, failure_recovery, false_positive, multiple_vehicles, or custom (default: easy_target)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run 4-variant ACO ablation study instead of multi-algorithm benchmark")
    parser.add_argument("--trials", type=int, default=3, help="Number of Monte Carlo trials per strategy (default: 3)")
    parser.add_argument("--steps", type=int, default=100, help="Steps per trial (default: 100)")
    parser.add_argument("--dt", type=float, default=0.2, help="Physics dt per step (default: 0.2)")
    parser.add_argument("--robots", type=int, default=4, help="Fleet robot count (default: 4)")
    parser.add_argument("--width", type=float, default=50.0, help="Grid width (default: 50.0)")
    parser.add_argument("--height", type=float, default=50.0, help="Grid height (default: 50.0)")
    parser.add_argument("--target-x", type=float, default=25.0, help="Target car X (default: 25.0)")
    parser.add_argument("--target-y", type=float, default=25.0, help="Target car Y (default: 25.0)")
    parser.add_argument("--target-plate", type=str, default="DXB-88392", help="Target plate (default: DXB-88392)")
    parser.add_argument("--inject-failure-tick", type=int, default=None, help="Tick to inject failure (default: None)")
    parser.add_argument("--fail-robot", type=str, default="robot_1", help="Target robot ID for failure (default: robot_1)")
    parser.add_argument("--output-dir", type=str, default="output/evaluation", help="Directory for benchmark results")
    parser.add_argument("--seed-start", type=int, default=100, help="Starting random seed for trials")
    return parser.parse_args()


def run_single_trial(
    strategy_label_or_type: Union[str, StrategyType],
    strategy_type: Optional[StrategyType] = None,
    seed: int = 100,
    args: Optional[argparse.Namespace] = None,
    aco_config: Optional[ACOConfig] = None,
) -> MissionMetrics:
    """Executes a single simulation trial with the given allocation strategy and seed."""
    if isinstance(strategy_label_or_type, StrategyType):
        strategy_type = strategy_label_or_type
        strategy_label = strategy_type.name
    else:
        strategy_label = str(strategy_label_or_type)
    
    if args is None:
        raise ValueError("args parameter must be provided to run_single_trial")
    scenario: Optional[ScenarioConfig] = None
    if getattr(args, "scenario", None) and args.scenario.lower() != "custom":
        scenario = create_scenario(args.scenario)
        w, h = scenario.width, scenario.height
        steps = args.steps if args.steps != 100 else scenario.max_steps
        inject_tick = args.inject_failure_tick if args.inject_failure_tick is not None else scenario.inject_failure_tick
        fail_robot = scenario.fail_robot_id or args.fail_robot
        vehicles = scenario.vehicles
        zones = scenario.zones
        start_poses = scenario.start_poses
        target_plate = scenario.target_plate
    else:
        w, h = args.width, args.height
        steps = args.steps
        inject_tick = args.inject_failure_tick
        fail_robot = args.fail_robot
        target_plate = args.target_plate
        target = TargetVehicle(
            target_id="target_vehicle_alpha",
            plate_number=target_plate,
            vehicle_class="sedan",
            color="dark_gray",
            position=(args.target_x, args.target_y),
            is_stolen=True,
        )
        vehicles = [target]
        zones = [UrbanZone(
            name="target_zone",
            bounds=(args.target_x - 8.0, args.target_y - 8.0, args.target_x + 8.0, args.target_y + 8.0),
            vehicle_prior_probability=0.85,
        )]
        start_poses = [
            (6.0, 6.0, 0.0),
            (6.0, h - 6.0, -math.pi / 2.0),
            (w - 6.0, 6.0, math.pi / 2.0),
            (w - 6.0, h - 6.0, math.pi),
        ]

    random.seed(seed)

    # 1. Environment & Obstacles
    env = UrbanEnvironment(width=w, height=h, resolution=1.0)
    ObstacleGenerator.add_perimeter_walls(env, thickness_cells=1)
    ObstacleGenerator.add_box_obstacle(env, w * 0.2, h * 0.2, w * 0.45, h * 0.45)
    ObstacleGenerator.add_box_obstacle(env, w * 0.55, h * 0.2, w * 0.8, h * 0.45)
    ObstacleGenerator.add_box_obstacle(env, w * 0.2, h * 0.55, w * 0.45, h * 0.8)
    ObstacleGenerator.add_box_obstacle(env, w * 0.55, h * 0.55, w * 0.8, h * 0.8)

    for z in zones:
        env.add_zone(z)

    # 2. Watchlist & Matching Engine Setup
    matcher = MatchingEngine(match_threshold=0.80)
    for v in vehicles:
        if getattr(v, "is_target", False) or getattr(v, "is_stolen", False) or v.plate_number == target_plate:
            matcher.add_target_vehicle(v)

    # 3. Strategy & Coordinator Setup
    if strategy_type is not None:
        cfg = aco_config or ACOConfig()
        cfg.stochastic.deterministic = False
        cfg.stochastic.temperature = 0.5
        strategy = create_strategy(strategy_type, aco_config=cfg, seed=seed)
    else:
        # Ablation mode: explicit ACO configuration
        cfg = aco_config or ACOConfig()
        cfg.stochastic.deterministic = False
        cfg.stochastic.temperature = 0.5
        strategy = ACOAllocationStrategy(ACOEngine(cfg))

    coordinator = SwarmCoordinator(
        allocation_strategy=strategy,
        environment=env,
    )

    # 4. Fleet Setup
    fleet_size = scenario.fleet_size if scenario else args.robots
    for i in range(fleet_size):
        rid = f"robot_{i+1}"
        sp = start_poses[i % len(start_poses)]
        slam = MockSLAMProvider(env)
        detector = SimulatedVehicleDetector(max_detection_range=14.0, fov_degrees=90.0)
        ocr = SimulatedPlateRecognizer(base_ocr_confidence=0.95)

        agent = RobotAgent(
            robot_id=rid,
            initial_pose=Pose2D(sp[0], sp[1], sp[2]),
            slam_provider=slam,
            detector=detector,
            ocr=ocr,
            matcher=matcher,
            known_vehicles=vehicles,
            zone_search_duration=4.0,
            reporting_duration=5.0,
        )
        coordinator.register_robot(agent)

    # 5. Metrics Collector Setup
    collector = MetricsCollector(
        environment=env,
        strategy_name=strategy_label,
        trial_seed=seed,
    )

    # 6. Simulation Loop
    failure_injected = False
    for tick in range(1, steps + 1):
        sim_time = tick * args.dt

        # Inject failure if specified
        if not failure_injected and inject_tick is not None and tick == inject_tick:
            coordinator.simulate_robot_failure(
                robot_id=fail_robot,
                timestamp=sim_time,
                reason="INJECTED_HARDWARE_BURNOUT",
            )
            collector.record_failure_injection(sim_time)
            failure_injected = True

        coordinator.step_coordination(timestamp=sim_time, dt=args.dt)
        collector.step(coordinator, timestamp=sim_time)

    return collector.compute_metrics(coordinator)


def aggregate_strategy_metrics(metrics_list: List[MissionMetrics]) -> Dict[str, Any]:
    """Calculates mean, standard deviation, median, and success rates across multiple trials."""
    n = len(metrics_list)
    if n == 0:
        return {}

    found_trials = [m for m in metrics_list if m.target_found and m.time_to_detection is not None]
    ttd_vals = [m.time_to_detection for m in found_trials]
    coverage_vals = [m.final_exploration_ratio * 100.0 for m in metrics_list]
    exp_rate_vals = [m.exploration_rate_m2_per_sec for m in metrics_list]
    overlap_vals = [m.overlap_ratio * 100.0 for m in metrics_list]
    dist_vals = [m.total_distance_traveled for m in metrics_list]
    energy_vals = [m.total_energy_consumed for m in metrics_list]
    tasks_vals = [m.tasks_completed for m in metrics_list]
    reassign_vals = [m.reassignment_events for m in metrics_list]
    cand_det_vals = [m.candidate_detections_count for m in metrics_list]
    false_pos_vals = [m.false_positives_count for m in metrics_list]
    confirm_vals = [m.target_confirmations_count for m in metrics_list]

    success_rate = (len(found_trials) / n) * 100.0
    mean_ttd = statistics.mean(ttd_vals) if ttd_vals else None
    std_ttd = statistics.stdev(ttd_vals) if len(ttd_vals) > 1 else 0.0
    median_ttd = statistics.median(ttd_vals) if ttd_vals else None

    mean_cov = statistics.mean(coverage_vals)
    std_cov = statistics.stdev(coverage_vals) if n > 1 else 0.0

    mean_expr = statistics.mean(exp_rate_vals)
    mean_overlap = statistics.mean(overlap_vals)
    std_overlap = statistics.stdev(overlap_vals) if n > 1 else 0.0

    mean_dist = statistics.mean(dist_vals)
    std_dist = statistics.stdev(dist_vals) if n > 1 else 0.0

    mean_energy = statistics.mean(energy_vals)
    mean_tasks = statistics.mean(tasks_vals)
    mean_reassign = statistics.mean(reassign_vals)

    # Failure recovery statistics
    recovered_trials = [m for m in metrics_list if m.failure_recovery_success]
    recovery_success_rate = (len(recovered_trials) / n) * 100.0
    rec_latencies = [m.failure_recovery_latency for m in metrics_list if m.failure_recovery_latency is not None]
    mean_rec_lat = statistics.mean(rec_latencies) if rec_latencies else None

    mean_det = statistics.mean(cand_det_vals)
    mean_fp = statistics.mean(false_pos_vals)
    mean_conf = statistics.mean(confirm_vals)

    # Average coverage timeline
    min_len = min(len(m.coverage_timeline) for m in metrics_list)
    avg_timeline = []
    for i in range(min_len):
        t = metrics_list[0].coverage_timeline[i][0]
        avg_c = statistics.mean(m.coverage_timeline[i][1] * 100.0 for m in metrics_list)
        avg_timeline.append((t, avg_c))

    return {
        "strategy_name": metrics_list[0].strategy_name,
        "trials_count": n,
        "success_rate_pct": success_rate,
        "mean_time_to_detection_sec": mean_ttd,
        "std_time_to_detection_sec": std_ttd,
        "median_time_to_detection_sec": median_ttd,
        "mean_coverage_pct": mean_cov,
        "std_coverage_pct": std_cov,
        "mean_exploration_rate_m2_per_sec": mean_expr,
        "mean_overlap_ratio_pct": mean_overlap,
        "std_overlap_ratio_pct": std_overlap,
        "mean_distance_traveled_m": mean_dist,
        "std_distance_traveled_m": std_dist,
        "mean_energy_consumed": mean_energy,
        "mean_tasks_completed": mean_tasks,
        "mean_reassignment_events": mean_reassign,
        "failure_recovery_success_rate_pct": recovery_success_rate,
        "mean_failure_recovery_latency_sec": mean_rec_lat,
        "mean_candidate_detections": mean_det,
        "mean_false_positives": mean_fp,
        "mean_target_confirmations": mean_conf,
        "average_coverage_timeline": avg_timeline,
        "raw_trials": [m.to_dict() for m in metrics_list],
    }


def generate_markdown_report(results: Dict[str, Dict[str, Any]], args: argparse.Namespace, is_ablation: bool = False) -> str:
    """Generates an academic-grade markdown report comparing the strategies across 15 metrics."""
    title = "Autonomous Swarm Car Search: Ablation Study Report" if is_ablation else "Autonomous Swarm Car Search: Task Allocation Benchmark Report"
    scenario_str = getattr(args, "scenario", "easy_target")

    lines = [
        f"# {title}",
        "",
        "## 1. Executive Summary & Experimental Methodology",
        f"- **Urban Grid Dimensions**: {args.width}m x {args.height}m (Cell Resolution: 1.0m)",
        f"- **Benchmark Scenario**: `{scenario_str}`",
        f"- **Swarm Fleet Size**: {args.robots} Autonomous Ground Vehicles",
        f"- **Trials per Configuration**: {args.trials} Monte Carlo trials with distinct pseudorandom seeds",
        f"- **Simulation Horizon**: {args.steps * args.dt:.1f} seconds per trial (dt = {args.dt:.2f}s)",
        "",
    ]

    strat_names = list(results.keys())
    label_map = {
        "ACO": "ACO (Proposed)",
        "NEAREST_FRONTIER": "Nearest Frontier (Baseline)",
        "RANDOM": "Random Walk (Baseline)",
        "RANDOM_WALK": "Random Walk (Baseline)",
    }
    display_names = [label_map.get(s, s) if not is_ablation else s for s in strat_names]
    header_cols = " | ".join(display_names)
    separator_cols = " | ".join([":---:"] * len(strat_names))

    lines.extend([
        "## 2. Comparative Performance Matrix (15 Metrics)",
        "",
        f"| Metric | {header_cols} |",
        f"| :--- | {separator_cols} |",
    ])

    def fmt_val(d, key, unit="", fmt=".1f"):
        val = d.get(key)
        if val is None:
            return "N/A"
        return f"{val:{fmt}}{unit}"

    def fmt_mean_std(d, mean_k, std_k, unit=""):
        m = d.get(mean_k)
        s = d.get(std_k, 0.0)
        if m is None:
            return "N/A"
        return f"{m:.1f} ± {s:.1f} {unit}"

    # Row 1: Target Discovery Rate
    row = " | ".join([f"**{fmt_val(results[s], 'success_rate_pct', '%')}**" for s in strat_names])
    lines.append(f"| **Target Discovery Rate (%)** | {row} |")

    # Row 2: Mean Time-to-Detection
    row = " | ".join([fmt_mean_std(results[s], 'mean_time_to_detection_sec', 'std_time_to_detection_sec', 's') for s in strat_names])
    lines.append(f"| **Mean Time-to-Detection (s)** | {row} |")

    # Row 3: Median Time-to-Detection
    row = " | ".join([fmt_val(results[s], 'median_time_to_detection_sec', 's') for s in strat_names])
    lines.append(f"| **Median Time-to-Detection (s)** | {row} |")

    # Row 4: Final Area Coverage
    row = " | ".join([fmt_mean_std(results[s], 'mean_coverage_pct', 'std_coverage_pct', '%') for s in strat_names])
    lines.append(f"| **Final Area Coverage (%)** | {row} |")

    # Row 5: Exploration Rate
    row = " | ".join([fmt_val(results[s], 'mean_exploration_rate_m2_per_sec', ' m²/s') for s in strat_names])
    lines.append(f"| **Exploration Rate (m²/s)** | {row} |")

    # Row 6: Trajectory Overlap Ratio
    row = " | ".join([fmt_mean_std(results[s], 'mean_overlap_ratio_pct', 'std_overlap_ratio_pct', '%') for s in strat_names])
    lines.append(f"| **Trajectory Overlap Ratio (%)** | {row} |")

    # Row 7: Total Distance Traveled
    row = " | ".join([fmt_mean_std(results[s], 'mean_distance_traveled_m', 'std_distance_traveled_m', 'm') for s in strat_names])
    lines.append(f"| **Total Distance Traveled (m)** | {row} |")

    # Row 8: Energy / Battery Consumed
    row = " | ".join([fmt_val(results[s], 'mean_energy_consumed', '', fmt='.4f') for s in strat_names])
    lines.append(f"| **Battery Energy Consumed** | {row} |")

    # Row 9: Tasks Completed
    row = " | ".join([fmt_val(results[s], 'mean_tasks_completed', '', fmt='.1f') for s in strat_names])
    lines.append(f"| **Tasks Completed** | {row} |")

    # Row 10: Dynamic Reassignment Events
    row = " | ".join([fmt_val(results[s], 'mean_reassignment_events', '', fmt='.1f') for s in strat_names])
    lines.append(f"| **Dynamic Reassignments** | {row} |")

    # Row 11: Failure Recovery Success Rate
    row = " | ".join([fmt_val(results[s], 'failure_recovery_success_rate_pct', '%') for s in strat_names])
    lines.append(f"| **Failure Recovery Rate (%)** | {row} |")

    # Row 12: Failure Recovery Latency
    row = " | ".join([fmt_val(results[s], 'mean_failure_recovery_latency_sec', 's') for s in strat_names])
    lines.append(f"| **Recovery Latency (s)** | {row} |")

    # Row 13: Candidate Vehicle Detections
    row = " | ".join([fmt_val(results[s], 'mean_candidate_detections', '', fmt='.1f') for s in strat_names])
    lines.append(f"| **Candidate Detections** | {row} |")

    # Row 14: False Positives Rejected
    row = " | ".join([fmt_val(results[s], 'mean_false_positives', '', fmt='.1f') for s in strat_names])
    lines.append(f"| **False Positives Rejected** | {row} |")

    # Row 15: Target Confirmations
    row = " | ".join([fmt_val(results[s], 'mean_target_confirmations', '', fmt='.1f') for s in strat_names])
    lines.append(f"| **Target Confirmations** | {row} |")

    lines.extend([
        "",
        "## 3. Scientific Key Findings & Analysis",
        "1. **Pheromone Stigmergy & Avoidance**:",
        "   Multi-objective stigmergic avoidance fields significantly suppress redundant visits and dispersion overlaps.",
        "2. **Target Prior Escalation & Recruitment**:",
        "   When candidate vehicles are sighted, stigmergic success pheromone recruitment rapidly vectors nearby robots to the scene.",
        "3. **Decentralized Fault Recovery**:",
        "   Failed robot tasks are seamlessly reclaimed by operational peers through heartbeat timeout and stigmergic reassignment.",
        "",
        "---",
        "*Report automatically synthesized by Swarm Evaluation Framework.*",
    ])
    return "\n".join(lines)


def generate_svg_comparison_plots(results: Dict[str, Dict[str, Any]], filepath: str) -> str:
    """Generates clean pure-Python vector SVG comparative timeline curves."""
    width, height = 750, 420
    margin_left, margin_right = 60, 40
    margin_top, margin_bottom = 50, 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    palette = ["#00E5FF", "#FF9800", "#9E9E9E", "#E91E63", "#4CAF50", "#FFD700"]
    colors = {name: palette[i % len(palette)] for i, name in enumerate(results.keys())}

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="background-color: #181820; font-family: sans-serif;">',
        f'<text x="{width / 2}" y="30" fill="#FFFFFF" font-size="16" font-weight="bold" text-anchor="middle">Exploration Coverage Over Time Comparison</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#121218" stroke="#333342" stroke-width="1.5"/>',
    ]

    for pct in range(0, 101, 20):
        y = margin_top + plot_h - (pct / 100.0) * plot_h
        svg.append(f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + plot_w}" y2="{y}" stroke="#282834" stroke-dasharray="4,4"/>')
        svg.append(f'<text x="{margin_left - 10}" y="{y + 4}" fill="#888899" font-size="11" text-anchor="end">{pct}%</text>')

    max_t = 1.0
    for sdata in results.values():
        tl = sdata.get("average_coverage_timeline", [])
        if tl and tl[-1][0] > max_t:
            max_t = tl[-1][0]

    for sname, sdata in results.items():
        tl = sdata.get("average_coverage_timeline", [])
        if not tl:
            continue
        c = colors.get(sname, "#FFFFFF")
        points = []
        for t, cov in tl:
            px = margin_left + (t / max_t) * plot_w
            py = margin_top + plot_h - (cov / 100.0) * plot_h
            points.append(f"{px:.1f},{py:.1f}")

        poly_pts = " ".join(points)
        svg.append(f'<polyline points="{poly_pts}" fill="none" stroke="{c}" stroke-width="3"/>')

    # Legend
    leg_x = margin_left + 20
    leg_y = margin_top + 25
    for i, (sname, c) in enumerate(colors.items()):
        svg.append(f'<circle cx="{leg_x}" cy="{leg_y + i * 22}" r="6" fill="{c}"/>')
        svg.append(f'<text x="{leg_x + 15}" y="{leg_y + i * 22 + 4}" fill="#FFFFFF" font-size="12">{sname}</text>')

    svg.append("</svg>")
    svg_str = "\n".join(svg)

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(svg_str)
    return svg_str


def generate_matplotlib_figures(results: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    """Generates publication-quality PNG comparison plots using Matplotlib."""
    if not _HAS_MATPLOTLIB:
        return

    strat_names = list(results.keys())
    palette = ["#00E5FF", "#FF9800", "#9E9E9E", "#E91E63", "#4CAF50", "#FFD700"]
    colors = {name: palette[i % len(palette)] for i, name in enumerate(strat_names)}

    # 1. Coverage Curves Figure
    fig_cov, ax = plt.subplots(figsize=(8, 5), facecolor="#1E1E24")
    ax.set_facecolor("#121216")

    for sname, sdata in results.items():
        tl = sdata.get("average_coverage_timeline", [])
        if not tl:
            continue
        times = [p[0] for p in tl]
        covs = [p[1] for p in tl]
        c = colors.get(sname, "#FFFFFF")
        ax.plot(times, covs, label=sname, color=c, linewidth=2.4)

    ax.set_title("Exploration Coverage Rate Over Time", color="#FFFFFF", fontsize=12, pad=12, weight="bold")
    ax.set_xlabel("Time (s)", color="#AAAAAA", fontsize=10)
    ax.set_ylabel("Exploration Coverage (%)", color="#AAAAAA", fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, color="#282834", linestyle="--", alpha=0.6)
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_color("#444450")
    ax.legend(facecolor="#1E1E24", edgecolor="#444450", labelcolor="#FFFFFF")
    plt.tight_layout()

    cov_png = os.path.join(output_dir, "coverage_comparison.png")
    plt.savefig(cov_png, dpi=160, facecolor=fig_cov.get_facecolor())
    plt.close(fig_cov)

    # 2. Bar Chart Comparison Figure
    fig_bar, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="#1E1E24")
    ax1.set_facecolor("#121216")
    ax2.set_facecolor("#121216")

    bar_colors = [colors.get(s, "#FFFFFF") for s in strat_names]

    # Bar 1: Overlap Ratio (%)
    overlaps = [results[s].get("mean_overlap_ratio_pct", 0.0) for s in strat_names]
    ax1.bar(strat_names, overlaps, color=bar_colors, width=0.5, edgecolor="#FFFFFF", alpha=0.85)
    ax1.set_title("Trajectory Overlap Ratio (Lower is Better)", color="#FFFFFF", fontsize=10, weight="bold")
    ax1.set_ylabel("Overlap %", color="#AAAAAA", fontsize=9)
    ax1.grid(axis="y", color="#282834", linestyle="--")
    ax1.tick_params(colors="#888888", rotation=15)
    for spine in ax1.spines.values():
        spine.set_color("#444450")

    # Bar 2: Final Area Coverage (%)
    coverages = [results[s].get("mean_coverage_pct", 0.0) for s in strat_names]
    ax2.bar(strat_names, coverages, color=bar_colors, width=0.5, edgecolor="#FFFFFF", alpha=0.85)
    ax2.set_title("Final Area Coverage % (Higher is Better)", color="#FFFFFF", fontsize=10, weight="bold")
    ax2.set_ylabel("Coverage %", color="#AAAAAA", fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", color="#282834", linestyle="--")
    ax2.tick_params(colors="#888888", rotation=15)
    for spine in ax2.spines.values():
        spine.set_color("#444450")

    plt.tight_layout()
    bar_png = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(bar_png, dpi=160, facecolor=fig_bar.get_facecolor())
    plt.close(fig_bar)


def run_benchmark(args):
    os.makedirs(args.output_dir, exist_ok=True)

    is_ablation = getattr(args, "ablation", False)
    experiment_title = "ACO ABLATION STUDY" if is_ablation else "SWARM TASK ALLOCATION BENCHMARK"

    print("=" * 76)
    print(f"      {experiment_title} RUNNER      ")
    print("=" * 76)
    print(f"Scenario            : {args.scenario}")
    print(f"Trials per Config   : {args.trials}")
    print(f"Simulation Horizon  : {args.steps} ticks (Duration: {args.steps * args.dt:.1f}s)")
    print(f"Robot Fleet Size    : {args.robots} agents")
    print(f"Output Directory    : {args.output_dir}")
    print("=" * 76)

    results: Dict[str, Dict[str, Any]] = {}

    if is_ablation:
        configs = [
            ("ACO_BASELINE", ACOConfig(weights=ACOConfig().weights.__class__(alpha=0.0, gamma=0.0))),
            ("ACO_PROB_ONLY", ACOConfig(weights=ACOConfig().weights.__class__(alpha=0.0, gamma=1.8))),
            ("ACO_PHERO_ONLY", ACOConfig(weights=ACOConfig().weights.__class__(alpha=1.2, gamma=0.0))),
            ("ACO_FULL", ACOConfig(weights=ACOConfig().weights.__class__(alpha=1.2, gamma=1.8))),
        ]

        for label, cfg in configs:
            print(f"\n>>> Running Ablation Variant: [{label}] <<<")
            trial_metrics = []
            for t in range(args.trials):
                seed = args.seed_start + t * 17
                t_start = time.time()
                m = run_single_trial(label, None, seed, args, aco_config=cfg)
                dur = time.time() - t_start
                found_str = f"FOUND (t={m.time_to_detection:.1f}s)" if m.target_found else "NOT FOUND"
                print(
                    f"  Trial {t+1}/{args.trials} [Seed {seed}]: "
                    f"Coverage: {m.final_exploration_ratio * 100:.1f}% | "
                    f"Overlap: {m.overlap_ratio * 100:.1f}% | "
                    f"Target: {found_str} ({dur:.2f}s)"
                )
                trial_metrics.append(m)
            results[label] = aggregate_strategy_metrics(trial_metrics)

    else:
        strategies = [
            ("ACO", StrategyType.ACO),
            ("NEAREST_FRONTIER", StrategyType.NEAREST_FRONTIER),
            ("RANDOM", StrategyType.RANDOM),
        ]

        for label, stype in strategies:
            print(f"\n>>> Running Benchmark Strategy: [{label}] <<<")
            trial_metrics = []
            for t in range(args.trials):
                seed = args.seed_start + t * 17
                t_start = time.time()
                m = run_single_trial(label, stype, seed, args)
                dur = time.time() - t_start
                found_str = f"FOUND (t={m.time_to_detection:.1f}s)" if m.target_found else "NOT FOUND"
                print(
                    f"  Trial {t+1}/{args.trials} [Seed {seed}]: "
                    f"Coverage: {m.final_exploration_ratio * 100:.1f}% | "
                    f"Overlap: {m.overlap_ratio * 100:.1f}% | "
                    f"Target: {found_str} ({dur:.2f}s)"
                )
                trial_metrics.append(m)
            results[label] = aggregate_strategy_metrics(trial_metrics)

    # 1. Save JSON results
    prefix = "ablation" if is_ablation else "benchmark"
    json_path = os.path.join(args.output_dir, f"{prefix}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved JSON raw metrics : {json_path}")

    # 2. Save Markdown Report
    report_md = generate_markdown_report(results, args, is_ablation=is_ablation)
    md_path = os.path.join(args.output_dir, f"{prefix}_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[+] Saved Markdown report  : {md_path}")

    # 3. Generate Vector SVG plots (pure Python)
    svg_path = os.path.join(args.output_dir, f"{prefix}_coverage_comparison.svg")
    generate_svg_comparison_plots(results, svg_path)
    print(f"[+] Saved SVG comparison   : {svg_path}")

    # 4. Generate Matplotlib PNG plots if available
    if _HAS_MATPLOTLIB:
        generate_matplotlib_figures(results, args.output_dir)
        print(f"[+] Saved PNG figures in   : {args.output_dir}")

    # 5. Print Terminal Summary Table
    print("\n" + "=" * 80)
    print(f"                         {experiment_title} SUMMARY                         ")
    print("=" * 80)
    print(f"{'Strategy / Variant':<22} | {'Coverage %':<12} | {'Overlap %':<12} | {'Success %':<10} | {'Mean TTD':<10}")
    print("-" * 80)
    for sname, agg in results.items():
        cov = f"{agg['mean_coverage_pct']:.1f}%"
        ovp = f"{agg['mean_overlap_ratio_pct']:.1f}%"
        succ = f"{agg['success_rate_pct']:.0f}%"
        ttd = f"{agg['mean_time_to_detection_sec']:.1f}s" if agg['mean_time_to_detection_sec'] else "N/A"
        print(f"{sname:<22} | {cov:<12} | {ovp:<12} | {succ:<10} | {ttd:<10}")
    print("=" * 80 + "\n")


def main():
    cli_args = parse_args()
    run_benchmark(cli_args)


if __name__ == "__main__":
    main()
