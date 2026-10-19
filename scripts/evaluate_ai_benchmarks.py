#!/usr/bin/env python3
"""
SwarmX AI Benchmark Evaluation CLI.
Executes scenarios A through H, benchmarks perception ML models,
and prints before/after statistical comparisons.
"""

import argparse
import os
import sys
import json
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.ai_benchmarks import evaluate_all_scenarios, run_scenario_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="SwarmX AI Benchmark Evaluation CLI")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for benchmark report")
    parser.add_argument("--output-file", type=str, default="ai_benchmark_report.json", help="Output JSON filename")
    parser.add_argument("--scenario", type=str, default=None, help="Run single scenario (e.g. scenario_a)")
    args = parser.parse_args()

    print("=" * 80)
    print("        SWARMX INTELLIGENT AI BENCHMARK EVALUATION & ML VERIFICATION       ")
    print("================================================================================")
    if args.scenario:
        print(f"Evaluating Single Scenario: [{args.scenario}]...")
    else:
        print("Evaluating Scenarios A through H (Multi-Stage Perception & Adaptive ACO)...")
    print("")

    t0 = time.time()
    if args.scenario:
        res = run_scenario_benchmark(args.scenario)
        from evaluation.ai_benchmarks import ComprehensiveAIBenchmarkReport
        report = ComprehensiveAIBenchmarkReport(
            timestamp=time.time(),
            scenarios_evaluated=[res],
            avg_coverage_pct=res.final_coverage_pct,
            avg_time_to_target_s=res.time_to_detection or 0.0,
            avg_distance_traveled_m=res.total_distance_m,
            avg_redundant_exploration_pct=res.redundancy_ratio * 100.0,
            avg_inter_robot_overlap_pct=res.inter_robot_overlap_ratio * 100.0 if res.inter_robot_overlap_ratio is not None else None,
            avg_shared_cells_pct=res.shared_cells_ratio * 100.0 if res.shared_cells_ratio is not None else None,
            avg_temporal_revisit_pct=res.temporal_revisit_rate * 100.0 if res.temporal_revisit_rate is not None else None,
            avg_intra_robot_self_revisit_pct=res.intra_robot_self_revisit_rate * 100.0 if res.intra_robot_self_revisit_rate is not None else None,
        )
    else:
        report = evaluate_all_scenarios()
    total_time = time.time() - t0

    print("--------------------------------------------------------------------------------")
    print("SCENARIO BENCHMARK RESULTS:")
    print("--------------------------------------------------------------------------------")
    header = f"{'Scenario':<22} | {'Found':<6} | {'TTD (s)':<8} | {'Cov %':<6} | {'Dist (m)':<8} | {'Conf %':<7} | {'Wall (s)':<8}"
    print(header)
    print("-" * len(header))

    for res in report.scenarios_evaluated:
        ttd_str = f"{res.time_to_detection:5.1f}s" if res.time_to_detection is not None else "N/A"
        found_str = "YES 🎯" if res.target_found else "NO"
        print(
            f"{res.scenario_name:<22} | {found_str:<6} | {ttd_str:<8} | {res.final_coverage_pct:5.1f}% | "
            f"{res.total_distance_m:7.1f}m | {res.target_confidence*100:5.1f}% | {res.wall_clock_sec:6.3f}s"
        )

    print("=" * 80)
    print("                      PERCEPTION ML & CV BENCHMARKS                             ")
    print("================================================================================")
    print(f"Target Vehicle Watchlist   : DXB-88392 (Dark Gray Sedan)")
    print(f"Perception Precision       : {report.precision * 100:.2f}%")
    print(f"Perception Recall          : {report.recall * 100:.2f}%")
    print(f"Perception F1-Score        : {report.f1_score * 100:.2f}%")
    print(f"False Positive Rate (FPR)  : {report.false_positive_rate * 100:.2f}%")
    print(f"False Negative Rate (FNR)  : {report.false_negative_rate * 100:.2f}%")
    print(f"OCR Character Accuracy     : {report.average_ocr_accuracy * 100:.2f}%")
    print(f"Target Verification Acc.   : {report.target_verification_accuracy * 100:.2f}%")
    print(f"Average Inference Latency  : {report.avg_inference_latency_ms:.3f} ms / frame")
    print(f"Perception Inference Speed : {report.inference_fps:.1f} FPS")
    print("================================================================================")
    print("                   SWARM COORDINATION & EXPLORATION METRICS                     ")
    print("================================================================================")
    print(f"Average Search Coverage    : {report.avg_coverage_pct:.2f}%")
    print(f"Average Time-to-Target     : {report.avg_time_to_target_s:.2f} s")
    print(f"Average Fleet Distance     : {report.avg_distance_traveled_m:.2f} m")
    print(f"Redundant Trajectory %     : {report.avg_redundant_exploration_pct:.2f}% (Inter-Robot Footprint)")
    if report.avg_inter_robot_overlap_pct is not None:
        print(f"Inter-Robot Overlap %      : {report.avg_inter_robot_overlap_pct:.2f}%")
    if report.avg_shared_cells_pct is not None:
        print(f"Inter-Robot Shared Area %  : {report.avg_shared_cells_pct:.2f}%")
    if report.avg_temporal_revisit_pct is not None:
        print(f"Temporal Revisit Rate %    : {report.avg_temporal_revisit_pct:.2f}%")
    if report.avg_intra_robot_self_revisit_pct is not None:
        print(f"Intra-Robot Self-Revisit % : {report.avg_intra_robot_self_revisit_pct:.2f}%")
    print(f"Robot Failures Recovered   : {report.total_failures_recovered}")
    print(f"ACO Dynamic Reassignments  : {report.total_reassignments}")
    print(f"Total Evaluation Runtime   : {total_time:.2f} seconds")
    print("================================================================================")

    # Save benchmark json artifact
    out_dir = os.path.isabs(args.output_dir) and args.output_dir or os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": report.timestamp,
                "precision": report.precision,
                "recall": report.recall,
                "f1_score": report.f1_score,
                "fpr": report.false_positive_rate,
                "fnr": report.false_negative_rate,
                "ocr_accuracy": report.average_ocr_accuracy,
                "target_verification_accuracy": report.target_verification_accuracy,
                "avg_inference_latency_ms": report.avg_inference_latency_ms,
                "inference_fps": report.inference_fps,
                "avg_coverage_pct": report.avg_coverage_pct,
                "avg_time_to_target_s": report.avg_time_to_target_s,
                "avg_distance_traveled_m": report.avg_distance_traveled_m,
                "avg_redundant_exploration_pct": report.avg_redundant_exploration_pct,
                "avg_inter_robot_overlap_pct": report.avg_inter_robot_overlap_pct,
                "avg_shared_cells_pct": report.avg_shared_cells_pct,
                "avg_temporal_revisit_pct": report.avg_temporal_revisit_pct,
                "avg_intra_robot_self_revisit_pct": report.avg_intra_robot_self_revisit_pct,
                "total_failures_recovered": report.total_failures_recovered,
                "total_reassignments": report.total_reassignments,
                "scenarios": [s.to_dict() for s in report.scenarios_evaluated],
            },
            f,
            indent=2,
        )
    print(f"[+] Saved AI Benchmark Artifact: {out_path}")


if __name__ == "__main__":
    main()
