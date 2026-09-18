"""
Swarm Robotics Urban Search & Target Identification Visualizer.
Supports multi-panel Matplotlib figures, pure-Python SVG vector rendering,
and terminal ASCII dashboards for maximum portability and publication-grade figures.
"""

from __future__ import annotations
import math
import os
from typing import Dict, List, Optional, Tuple, Any

from interfaces.slam_interface import Pose2D
from simulation.environment import UrbanEnvironment
from swarm.states.robot_states import RobotState
from swarm.task_allocation.frontier_task import SearchTask, TaskStatus
from swarm.pheromone.pheromone_map import PheromoneMap
from swarm.pheromone.evaporation_model import PheromoneLayerType
from swarm.core.robot_agent import RobotAgent
from swarm.core.swarm_coordinator import SwarmCoordinator
from perception.vehicle_detector import TargetVehicle

# Optional Matplotlib import with headless Agg backend configuration
_HAS_MATPLOTLIB = False
try:
    # Set headless backend and safe cache directory
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_swarm_cache"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle, Rectangle, Wedge
    import numpy as np
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# Optional PIL import for animated GIF generation
_HAS_PIL = False
try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


STATE_COLOR_MAP: Dict[RobotState, str] = {
    RobotState.IDLE: "#9E9E9E",                # Neutral Gray
    RobotState.NAVIGATING: "#00BCD4",           # Cyan
    RobotState.SEARCHING: "#4CAF50",            # Vibrant Green
    RobotState.VEHICLE_DETECTED: "#FFEB3B",     # Yellow
    RobotState.VERIFYING: "#FF9800",            # Orange
    RobotState.REPORTING: "#FFD700",            # Gold
    RobotState.REASSIGNING: "#03A9F4",          # Light Blue
    RobotState.RETURNING: "#9C27B0",            # Purple
    RobotState.COMMUNICATION_LOST: "#E91E63",   # Deep Pink
    RobotState.COMPLETED: "#8BC34A",            # Light Green
    RobotState.FAILED: "#F44336",               # Bright Red
}


class SwarmVisualizer:
    """
    Visualization engine rendering urban search progress:
    - 2D grid buildings, streets, and exploration boundary
    - Multi-layer pheromone intensity heatmaps
    - Directional robot vectors, status badges, and FOV cones
    - Discovered and pending SLAM frontier tasks
    - Stolen/missing vehicle targets and identification events
    - Real-time diagnostic mission HUD
    """

    def __init__(
        self,
        environment: UrbanEnvironment,
        title: str = "Swarm Urban Search & Target Identification",
    ) -> None:
        self.env = environment
        self.title = title
        self.trajectories: Dict[str, List[Tuple[float, float]]] = {}
        self.exploration_history: List[Tuple[float, float]] = []

    def update_trajectories(self, coordinator: SwarmCoordinator, timestamp: float) -> None:
        """Records current robot coordinates for path history rendering."""
        for robot_id, agent in coordinator.robots.items():
            if robot_id not in self.trajectories:
                self.trajectories[robot_id] = []
            self.trajectories[robot_id].append((agent.pose.x, agent.pose.y))

        # Record exploration ratio history
        status = coordinator.get_swarm_status(timestamp)
        avg_exp = status.get("average_exploration_ratio", 0.0)
        self.exploration_history.append((timestamp, avg_exp))

    # =========================================================================
    # Matplotlib High-Fidelity Rendering
    # =========================================================================

    def render_matplotlib_dashboard(
        self,
        coordinator: SwarmCoordinator,
        timestamp: float,
        target_vehicles: Optional[List[TargetVehicle]] = None,
        filepath: Optional[str] = None,
        dpi: int = 140,
    ) -> Optional[Any]:
        """
        Generates a 3-panel publication-quality Matplotlib dashboard:
        1. Main Urban Map with robots, FOVs, trails, and tasks.
        2. Exploration & Avoidance Pheromone Overlay.
        3. Real-time Telemetry & Search Progress Timeline.
        """
        if not _HAS_MATPLOTLIB:
            return None

        self.update_trajectories(coordinator, timestamp)

        fig = plt.figure(figsize=(18, 9), facecolor="#1E1E24")
        gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.0])

        ax_map = fig.add_subplot(gs[:, 0])
        ax_phero = fig.add_subplot(gs[0, 1])
        ax_telemetry = fig.add_subplot(gs[1, 1])

        # Panel 1: Main Urban Search Field
        self._plot_urban_map(ax_map, coordinator, timestamp, target_vehicles)

        # Panel 2: Pheromone Density
        self._plot_pheromone_field(ax_phero, coordinator)

        # Panel 3: Exploration Progress & Telemetry
        self._plot_telemetry_metrics(ax_telemetry, coordinator, timestamp)

        plt.tight_layout()

        if filepath:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            plt.savefig(filepath, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close(fig)
            return filepath

        return fig

    def _plot_urban_map(
        self,
        ax: Any,
        coordinator: SwarmCoordinator,
        timestamp: float,
        target_vehicles: Optional[List[TargetVehicle]] = None,
    ) -> None:
        """Plots the urban grid, obstacles, robot trajectories, FOVs, and tasks."""
        ax.set_facecolor("#121216")
        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        ax.set_aspect("equal")

        # 1. Draw obstacles / buildings
        for gy in range(self.env.grid_height):
            for gx in range(self.env.grid_width):
                if self.env.get_cell(gx, gy) == 100:  # Blocked / Wall
                    wx = gx * self.env.resolution
                    wy = gy * self.env.resolution
                    rect = Rectangle(
                        (wx, wy),
                        self.env.resolution,
                        self.env.resolution,
                        facecolor="#373742",
                        edgecolor="#282830",
                        linewidth=0.5,
                    )
                    ax.add_patch(rect)

        # 2. Draw SLAM Frontier Tasks
        for task in coordinator.task_manager.tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED):
                color = "#00E5FF" if task.status == TaskStatus.ASSIGNED else "#FFD600"
                ax.scatter(
                    task.centroid_x,
                    task.centroid_y,
                    s=task.size * 8 + 30,
                    facecolor="none",
                    edgecolor=color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax.text(
                    task.centroid_x,
                    task.centroid_y + 0.8,
                    task.task_id.split("_")[1] if "_" in task.task_id else task.task_id,
                    color=color,
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    weight="bold",
                )

        # 3. Draw Target Vehicles
        if target_vehicles:
            for tv in target_vehicles:
                tx, ty = tv.position
                found = coordinator.target_found and coordinator.confirmed_target_location == (tx, ty)
                badge_color = "#00E676" if found else "#FF3D00"
                ax.scatter(tx, ty, marker="s", s=140, facecolor=badge_color, edgecolor="#FFFFFF", linewidth=2.0, zorder=5)
                ax.text(
                    tx,
                    ty - 1.2,
                    f"TARGET: {tv.plate_number}\n({'FOUND' if found else 'WANTED'})",
                    color=badge_color,
                    fontsize=8,
                    ha="center",
                    va="top",
                    weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#1E1E24", edgecolor=badge_color, alpha=0.9),
                )

        # 4. Draw Robot Trails and Poses
        for robot_id, agent in coordinator.robots.items():
            state = agent.current_state
            color = STATE_COLOR_MAP.get(state, "#FFFFFF")
            failed = coordinator.failure_detector.is_failed(robot_id)

            # Historical trail
            trail = self.trajectories.get(robot_id, [])
            if len(trail) > 1:
                tx_coords = [p[0] for p in trail]
                ty_coords = [p[1] for p in trail]
                ax.plot(tx_coords, ty_coords, color=color, alpha=0.45, linewidth=1.8, linestyle=":")

            # Robot body & heading
            rx, ry = agent.pose.x, agent.pose.y
            heading = agent.pose.theta

            # FOV Wedge if active
            if not failed and state in (RobotState.SEARCHING, RobotState.NAVIGATING):
                fov_deg = 90.0
                fov_rad = math.radians(fov_deg)
                wedge = Wedge(
                    (rx, ry),
                    r=12.0,
                    theta1=math.degrees(heading - fov_rad / 2.0),
                    theta2=math.degrees(heading + fov_rad / 2.0),
                    facecolor=color,
                    alpha=0.12,
                    edgecolor=color,
                    linestyle="--",
                    linewidth=1.0,
                )
                ax.add_patch(wedge)

            # Robot glyph
            if failed:
                ax.scatter(rx, ry, marker="X", s=220, color="#F44336", edgecolor="#FFFFFF", linewidth=2.0, zorder=6)
                ax.text(rx, ry + 1.2, f"{robot_id}\n[FAILED]", color="#F44336", fontsize=8, ha="center", weight="bold")
            else:
                ax.scatter(rx, ry, marker="o", s=160, facecolor=color, edgecolor="#FFFFFF", linewidth=1.5, zorder=6)
                # Heading direction arrow
                arrow_len = 1.8
                ax.arrow(
                    rx,
                    ry,
                    arrow_len * math.cos(heading),
                    arrow_len * math.sin(heading),
                    head_width=0.6,
                    head_length=0.6,
                    fc="#FFFFFF",
                    ec="#FFFFFF",
                    zorder=7,
                )
                ax.text(
                    rx,
                    ry - 1.2,
                    f"{robot_id}\n{state.value}",
                    color="#FFFFFF",
                    fontsize=8,
                    ha="center",
                    va="top",
                    weight="bold",
                )

        ax.set_title(f"Urban Search Map (Time: {timestamp:.1f}s)", color="#FFFFFF", fontsize=12, pad=10, weight="bold")
        ax.tick_params(colors="#888888")
        for spine in ax.spines.values():
            spine.set_color("#444450")

    def _plot_pheromone_field(self, ax: Any, coordinator: SwarmCoordinator) -> None:
        """Renders discrete heatmap overlay of stigmergic exploration and avoidance layers."""
        pmap = coordinator.pheromone_map
        ax.set_facecolor("#121216")
        ax.set_xlim(0, pmap.width)
        ax.set_ylim(0, pmap.height)
        ax.set_aspect("equal")

        # Build combined 2D array: exploration (blue/green) + avoidance (red) + success (gold)
        field_img = np.zeros((pmap.grid_height, pmap.grid_width, 3), dtype=np.float32)

        exp_layer = pmap.layers.get(PheromoneLayerType.EXPLORATION, [])
        avoid_layer = pmap.layers.get(PheromoneLayerType.AVOIDANCE, [])
        succ_layer = pmap.layers.get(PheromoneLayerType.SUCCESS, [])

        max_exp = max(exp_layer) if exp_layer and max(exp_layer) > 0.01 else 1.0
        max_avoid = max(avoid_layer) if avoid_layer and max(avoid_layer) > 0.01 else 1.0
        max_succ = max(succ_layer) if succ_layer and max(succ_layer) > 0.01 else 1.0

        for gy in range(pmap.grid_height):
            for gx in range(pmap.grid_width):
                idx = gy * pmap.grid_width + gx
                e = (exp_layer[idx] / max_exp) if idx < len(exp_layer) else 0.0
                a = (avoid_layer[idx] / max_avoid) if idx < len(avoid_layer) else 0.0
                s = (succ_layer[idx] / max_succ) if idx < len(succ_layer) else 0.0

                # R = avoidance + success, G = exploration + success, B = exploration
                r = min(1.0, a * 0.9 + s * 1.0)
                g = min(1.0, e * 0.8 + s * 0.9)
                b = min(1.0, e * 0.6)
                field_img[gy, gx] = [r, g, b]

        ax.imshow(
            field_img,
            origin="lower",
            extent=[0, pmap.width, 0, pmap.height],
            interpolation="gaussian",
            alpha=0.9,
        )
        ax.set_title("Stigmergic Pheromone Layers (G: Explore, R: Avoid, Gold: Target)", color="#FFFFFF", fontsize=10, weight="bold")
        ax.tick_params(colors="#888888")
        for spine in ax.spines.values():
            spine.set_color("#444450")

    def _plot_telemetry_metrics(
        self,
        ax: Any,
        coordinator: SwarmCoordinator,
        timestamp: float,
    ) -> None:
        """Plots timeline curves and status badges."""
        ax.set_facecolor("#121216")

        status = coordinator.get_swarm_status(timestamp)

        if self.exploration_history:
            times = [h[0] for h in self.exploration_history]
            exps = [h[1] * 100.0 for h in self.exploration_history]
            ax.plot(times, exps, color="#00E5FF", linewidth=2.2, label="Exploration %")
            ax.fill_between(times, exps, color="#00E5FF", alpha=0.15)

        ax.set_ylim(0, 105)
        ax.set_xlabel("Simulation Time (s)", color="#AAAAAA", fontsize=9)
        ax.set_ylabel("Coverage (%)", color="#AAAAAA", fontsize=9)
        ax.set_title("Mission Telemetry & Search Progress", color="#FFFFFF", fontsize=10, weight="bold")
        ax.grid(True, color="#282832", linestyle="--", alpha=0.6)
        ax.tick_params(colors="#888888")
        for spine in ax.spines.values():
            spine.set_color("#444450")

        # Telemetry Text HUD
        hud_lines = [
            f"Active Robots: {status.get('robot_count', 0) - status.get('failed_robots_count', 0)} / {status.get('robot_count', 0)}",
            f"Failed Robots: {status.get('failed_robots_count', 0)}",
            f"Recovered Tasks: {status.get('recovered_tasks_count', 0)}",
            f"Completed Tasks: {status.get('completed_tasks_count', 0)}",
            f"Pending Tasks: {status.get('pending_tasks_count', 0)}",
            f"Target Car Found: {'YES (CONFIRMED)' if status.get('target_found') else 'SEARCHING...'}",
        ]
        hud_str = "\n".join(hud_lines)
        ax.text(
            0.05,
            0.40,
            hud_str,
            transform=ax.transAxes,
            color="#FFFFFF",
            fontsize=8.5,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1E1E24", edgecolor="#00E5FF", alpha=0.85),
        )

    # =========================================================================
    # Pure-Python Standalone Renderers (Zero External Dependencies)
    # =========================================================================

    def render_ascii_map(
        self,
        coordinator: SwarmCoordinator,
        timestamp: float,
        cols: int = 50,
        rows: int = 20,
        target_vehicles: Optional[List[TargetVehicle]] = None,
    ) -> str:
        """
        Renders an ASCII text grid representation of the environment, robots, tasks,
        and targets. Guaranteed to work on ANY machine/interpreter without external libraries.
        """
        grid = [["." for _ in range(cols)] for _ in range(rows)]

        # Map scale factors
        sx = cols / self.env.width
        sy = rows / self.env.height

        # Draw obstacles '#'
        for gy in range(self.env.grid_height):
            for gx in range(self.env.grid_width):
                if self.env.get_cell(gx, gy) == 100:
                    wx = gx * self.env.resolution
                    wy = gy * self.env.resolution
                    cx = min(cols - 1, int(wx * sx))
                    cy = min(rows - 1, int(wy * sy))
                    # Invert Y for terminal display
                    ty = rows - 1 - cy
                    grid[ty][cx] = "#"

        # Draw tasks '?'
        for task in coordinator.task_manager.tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED):
                cx = min(cols - 1, int(task.centroid_x * sx))
                cy = min(rows - 1, int(task.centroid_y * sy))
                grid[rows - 1 - cy][cx] = "?"

        # Draw vehicles 'V' (candidate/unconfirmed) and '!' (confirmed target)
        if target_vehicles:
            for tv in target_vehicles:
                cx = min(cols - 1, int(tv.position[0] * sx))
                cy = min(rows - 1, int(tv.position[1] * sy))
                is_found = coordinator.target_found and coordinator.confirmed_target_location == tv.position
                grid[rows - 1 - cy][cx] = "!" if is_found else "V"
        elif coordinator.target_found and coordinator.confirmed_target_location:
            tx, ty = coordinator.confirmed_target_location
            cx = min(cols - 1, int(tx * sx))
            cy = min(rows - 1, int(ty * sy))
            grid[rows - 1 - cy][cx] = "!"

        # Draw robots 'R' (or 'X' if failed)
        for robot_id, agent in coordinator.robots.items():
            cx = min(cols - 1, int(agent.pose.x * sx))
            cy = min(rows - 1, int(agent.pose.y * sy))
            glyph = "X" if coordinator.failure_detector.is_failed(robot_id) else "R"
            grid[rows - 1 - cy][cx] = glyph

        status = coordinator.get_swarm_status(timestamp)
        header = f"=== Swarm Urban Search Map [t={timestamp:.1f}s | Active: {status['robot_count'] - status['failed_robots_count']} | Target: {'FOUND' if status['target_found'] else 'SEARCHING'}] ==="
        body = "\n".join("".join(row) for row in grid)
        footer = "Legend: [#] Obstacle  [.] Free Space  [R] Robot  [X] Failed Robot  [?] Frontier Task  [V] Vehicle  [!] Target Confirmed"
        return f"{header}\n{body}\n{footer}\n"

    def render_svg_map(
        self,
        coordinator: SwarmCoordinator,
        timestamp: float,
        filepath: Optional[str] = None,
        target_vehicles: Optional[List[TargetVehicle]] = None,
        width_px: int = 900,
        height_px: int = 600,
    ) -> str:
        """
        Generates clean, self-contained SVG vector graphic.
        Pure Python; requires zero external libraries.
        """
        scale_x = (width_px - 60) / self.env.width
        scale_y = (height_px - 60) / self.env.height

        def to_svg(x: float, y: float) -> Tuple[float, float]:
            return (30 + x * scale_x, height_px - 30 - y * scale_y)

        svg_parts: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width_px} {height_px}" style="background-color: #16161D; font-family: sans-serif;">',
            f'<text x="30" y="25" fill="#FFFFFF" font-size="16" font-weight="bold">{self.title} (Time: {timestamp:.1f}s)</text>',
        ]

        # Draw urban boundary
        bx, by = to_svg(0, self.env.height)
        svg_parts.append(
            f'<rect x="{bx}" y="{by}" width="{self.env.width * scale_x}" height="{self.env.height * scale_y}" fill="#1F1F28" stroke="#333340" stroke-width="2"/>'
        )

        # Draw Obstacles
        for gy in range(self.env.grid_height):
            for gx in range(self.env.grid_width):
                if self.env.get_cell(gx, gy) == 100:
                    wx = gx * self.env.resolution
                    wy = gy * self.env.resolution
                    sx, sy = to_svg(wx, wy + self.env.resolution)
                    svg_parts.append(
                        f'<rect x="{sx}" y="{sy}" width="{self.env.resolution * scale_x}" height="{self.env.resolution * scale_y}" fill="#3C3C4D"/>'
                    )

        # Draw Tasks
        for task in coordinator.task_manager.tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED):
                cx, cy = to_svg(task.centroid_x, task.centroid_y)
                color = "#00E5FF" if task.status == TaskStatus.ASSIGNED else "#FFD600"
                svg_parts.append(
                    f'<circle cx="{cx}" cy="{cy}" r="6" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="3,3"/>'
                )

        # Draw Targets
        if target_vehicles:
            for tv in target_vehicles:
                cx, cy = to_svg(tv.position[0], tv.position[1])
                is_found = coordinator.target_found and coordinator.confirmed_target_location == tv.position
                color = "#00E676" if is_found else "#FF3D00"
                svg_parts.append(
                    f'<rect x="{cx - 8}" y="{cy - 8}" width="16" height="16" fill="{color}" stroke="#FFFFFF" stroke-width="2"/>'
                )
                svg_parts.append(
                    f'<text x="{cx}" y="{cy + 18}" fill="{color}" font-size="10" text-anchor="middle" font-weight="bold">{tv.plate_number}</text>'
                )

        # Draw Robots
        for robot_id, agent in coordinator.robots.items():
            rx, ry = to_svg(agent.pose.x, agent.pose.y)
            failed = coordinator.failure_detector.is_failed(robot_id)
            color = "#F44336" if failed else STATE_COLOR_MAP.get(agent.current_state, "#00BCD4")

            if failed:
                svg_parts.append(
                    f'<line x1="{rx - 7}" y1="{ry - 7}" x2="{rx + 7}" y2="{ry + 7}" stroke="#F44336" stroke-width="3"/>'
                )
                svg_parts.append(
                    f'<line x1="{rx - 7}" y1="{ry + 7}" x2="{rx + 7}" y2="{ry - 7}" stroke="#F44336" stroke-width="3"/>'
                )
            else:
                # Heading arrow
                hx = rx + 14 * math.cos(-agent.pose.theta)
                hy = ry + 14 * math.sin(-agent.pose.theta)
                svg_parts.append(f'<circle cx="{rx}" cy="{cy}" r="8" fill="{color}" stroke="#FFFFFF" stroke-width="1.5"/>')
                svg_parts.append(f'<line x1="{rx}" y1="{ry}" x2="{hx}" y2="{hy}" stroke="#FFFFFF" stroke-width="2"/>')

            svg_parts.append(
                f'<text x="{rx}" y="{ry - 12}" fill="#FFFFFF" font-size="10" text-anchor="middle">{robot_id}</text>'
            )

        svg_parts.append("</svg>")
        svg_content = "\n".join(svg_parts)

        if filepath:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(svg_content)

        return svg_content
