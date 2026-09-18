#!/usr/bin/env python3
"""
Autonomous Swarm Outdoor Car Search - Unified Command-Line Interface (CLI).
Provides a single launcher for running simulations, benchmark evaluations,
executing unit tests, and displaying system metadata.
"""

import argparse
import os
import sys
import unittest

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def print_system_banner():
    banner = """
================================================================================
   AUTONOMOUS MULTI-ROBOT SWARM: URBAN CAR SEARCH & IDENTIFICATION SYSTEM   
================================================================================
  Graduation Project: AI Swarm Robotics Subsystem
  Developers        : Mohammed & Ibrahim (Swarm Coordination & ACO Task Allocation)
  External Interface: Saqr & Mahmoud (SLAM, Odometry, & Frontier Mapping)
  Version           : 1.0.0 (Production Release)
  Architecture      : Decentralized Ant Colony Optimization (ACO) & Stigmergy
================================================================================
"""
    print(banner)


def run_info_command(args):
    print_system_banner()
    info_text = """
Core Subsystem Architecture:
  1. Multi-Objective ACO Task Scorer:
     - Balances frontier information gain, detection priors, and urgency against
       distance costs, robot congestion, and negative pheromone repulsion.
  2. Multi-Layer Stigmergic Pheromone Field:
     - Exploration layer (tau_e): Maps covered territory with slow evaporation.
     - Target recruitment layer (tau_s): Attracts nearby robots to vehicle sightings.
     - Collision & reservation layer (tau_a): Repels robots to avoid search overlap.
  3. Finite State Machine (FSM) & Kinematics:
     - Differential drive non-holonomic motion model with obstacle collision halting.
  4. Fault-Tolerant Dynamic Reallocation:
     - Autonomous failure detection via heartbeat timeout, battery, and stall monitors.
     - Orphaned tasks immediately released and reclaimed by healthy swarm members.
  5. Perception & Target Identification:
     - Field-of-view (FOV) raycast occlusion detection.
     - License plate OCR simulation with Levenshtein fuzzy string similarity.

Available CLI Commands:
  - swarm sim   : Run multi-agent urban simulation with live ASCII HUD / SVG / GIF.
  - swarm eval  : Run Monte Carlo benchmark trials comparing ACO against baselines.
  - swarm test  : Run the full automated verification test suite.
  - swarm info  : Display this system specification and architecture overview.

Run 'python3 scripts/main.py <command> --help' for command-specific flags and options.
================================================================================
"""
    print(info_text)


def run_sim_command(args, remaining_args):
    from scripts.run_simulation import parse_args as parse_sim_args, run_simulation
    # Re-parse arguments using simulation parser
    sim_args = parse_sim_args()
    run_simulation(sim_args)


def run_eval_command(args, remaining_args):
    from scripts.run_evaluation import parse_args as parse_eval_args, run_benchmark
    # Re-parse arguments using evaluation parser
    eval_args = parse_eval_args()
    run_benchmark(eval_args)


def run_test_command(args, remaining_args):
    print_system_banner()
    print("[*] Discovering and executing all unit tests in 'tests/'...\n")
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=os.path.join(PROJECT_ROOT, "tests"), pattern="test_*.py")
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    sys.exit(0 if result.wasSuccessful() else 1)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print_system_banner()
        print("Usage: python3 scripts/main.py <command> [options]\n")
        print("Commands:")
        print("  sim     Run autonomous multi-robot swarm simulation")
        print("  eval    Run Monte Carlo benchmarking and baseline comparison")
        print("  test    Execute the complete 96+ automated test suite")
        print("  info    Display architectural specifications and project details")
        print("\nUse 'python3 scripts/main.py <command> --help' for detailed parameters.")
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "info":
        run_info_command(sys.argv[2:])
    elif cmd == "test":
        run_test_command(None, sys.argv[2:])
    elif cmd == "sim":
        # Shift sys.argv so subcommand parser reads remaining options
        sys.argv.pop(1)
        from scripts.run_simulation import main as sim_main
        sim_main()
    elif cmd == "eval":
        # Shift sys.argv so subcommand parser reads remaining options
        sys.argv.pop(1)
        from scripts.run_evaluation import main as eval_main
        eval_main()
    else:
        print(f"[-] Unknown command: '{cmd}'")
        print("Available commands: sim, eval, test, info")
        sys.exit(1)


if __name__ == "__main__":
    main()
