"""
Kinematic Simulation of Autonomous Differential-Drive Ground Robot.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any
import math
from interfaces.slam_interface import Pose2D
from simulation.environment import UrbanEnvironment


@dataclass
class Twist2D:
    """Robot velocity command containing linear and angular components."""
    linear: float = 0.0   # Forward speed in meters/second
    angular: float = 0.0  # Yaw rate in radians/second


class SimulatedRobot:
    """
    Simulates continuous unicycle / differential-drive robot physics,
    waypoint tracking controllers, boundary collision detection, and battery discharge.
    """

    def __init__(
        self,
        robot_id: str,
        initial_pose: Pose2D,
        max_linear_speed: float = 1.5,
        max_angular_speed: float = 1.5,
        battery_capacity: float = 1.0,
    ) -> None:
        self.robot_id = robot_id
        self.pose = initial_pose
        self.velocity = Twist2D()
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.battery = battery_capacity
        self.total_distance = 0.0

        # Kinematic tuning gains for proportional waypoint controller
        self.kp_linear = 1.0
        self.kp_angular = 2.5

    def set_velocity(self, linear: float, angular: float) -> None:
        """Sets target velocities clamped to operational limits."""
        clamped_lin = max(min(linear, self.max_linear_speed), -self.max_linear_speed)
        clamped_ang = max(min(angular, self.max_angular_speed), -self.max_angular_speed)
        self.velocity = Twist2D(linear=clamped_lin, angular=clamped_ang)

    def stop(self) -> None:
        """Commands zero velocity."""
        self.velocity = Twist2D(linear=0.0, angular=0.0)

    def step(self, dt: float, env: Optional[UrbanEnvironment] = None) -> Pose2D:
        """
        Integrates kinematics forward by time step dt:
          theta(t + dt) = theta(t) + omega * dt
          x(t + dt) = x(t) + v * cos(theta) * dt
          y(t + dt) = y(t) + v * sin(theta) * dt
        """
        if self.battery <= 0.0:
            self.stop()
            return self.pose

        # Angular integration
        new_theta = self.pose.theta + self.velocity.angular * dt
        # Wrap heading to [-pi, pi]
        new_theta = (new_theta + math.pi) % (2.0 * math.pi) - math.pi

        # Linear integration
        dx = self.velocity.linear * math.cos(new_theta) * dt
        dy = self.velocity.linear * math.sin(new_theta) * dt

        candidate_x = self.pose.x + dx
        candidate_y = self.pose.y + dy

        # Collision check against environment
        if env is not None:
            if not env.is_free(candidate_x, candidate_y):
                # Collision with wall or obstacle: halt forward movement
                self.stop()
                self.pose = Pose2D(x=self.pose.x, y=self.pose.y, theta=new_theta)
                return self.pose

        # Update pose
        self.pose = Pose2D(x=candidate_x, y=candidate_y, theta=new_theta)
        self.total_distance += math.hypot(dx, dy)

        # Battery drain (slight base idle drain + motion-induced drain)
        drain = (0.00005 + 0.0002 * abs(self.velocity.linear) + 0.0001 * abs(self.velocity.angular)) * dt
        self.battery = max(0.0, self.battery - drain)

        return self.pose

    def drive_towards_waypoint(
        self,
        target_x: float,
        target_y: float,
        tolerance: float = 0.5,
        env: Optional[Any] = None,
    ) -> bool:
        """
        Proportional closed-loop steering controller towards a target waypoint
        with reactive obstacle avoidance lookahead.
        Returns True if robot has reached the waypoint within tolerance.
        """
        dx = target_x - self.pose.x
        dy = target_y - self.pose.y
        distance = math.hypot(dx, dy)

        if distance <= tolerance:
            self.stop()
            return True

        desired_heading = math.atan2(dy, dx)
        effective_heading = desired_heading

        # Reactive obstacle avoidance lookahead:
        # If moving directly towards desired heading would encounter an obstacle,
        # sweep candidate lateral deflections to navigate along free streets.
        if env is not None:
            lookahead = 1.6
            probe_x = self.pose.x + lookahead * math.cos(desired_heading)
            probe_y = self.pose.y + lookahead * math.sin(desired_heading)
            if not env.is_free(probe_x, probe_y):
                candidate_offsets = [
                    math.pi / 4.0, -math.pi / 4.0,
                    math.pi / 2.0, -math.pi / 2.0,
                    3.0 * math.pi / 4.0, -3.0 * math.pi / 4.0,
                ]
                for offset in candidate_offsets:
                    cand_h = desired_heading + offset
                    cx = self.pose.x + lookahead * math.cos(cand_h)
                    cy = self.pose.y + lookahead * math.sin(cand_h)
                    if env.is_free(cx, cy):
                        if env.is_free(self.pose.x + 0.8 * math.cos(cand_h), self.pose.y + 0.8 * math.sin(cand_h)):
                            effective_heading = cand_h
                            break

        heading_error = (effective_heading - self.pose.theta + math.pi) % (2.0 * math.pi) - math.pi

        # If heading error is large, prioritize rotating in place
        if abs(heading_error) > 0.6:
            cmd_linear = 0.1
            cmd_angular = self.kp_angular * heading_error
        else:
            cmd_linear = min(self.kp_linear * distance, self.max_linear_speed)
            cmd_angular = self.kp_angular * heading_error

        self.set_velocity(cmd_linear, cmd_angular)
        return False
