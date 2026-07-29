#!/usr/bin/env python3
"""Task-level success statistics for the laundry manipulation study.

The default counts reproduce the current manuscript:
  ACT:              24/50
  Ours single-shot: 39/50
  Ours recoverable: 46/50

Independent groups are compared with two-sided Fisher's exact tests. If trials
were truly paired, aggregate counts are insufficient; use --paired_discordant
with the two discordant cell counts for an exact McNemar test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scipy.stats import beta, binomtest, fisher_exact, norm


def clopper_pearson(successes: int, total: int, alpha: float = 0.05):
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial count")
    lower = (
        0.0 if successes == 0
        else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    )
    upper = (
        1.0 if successes == total
        else float(beta.ppf(
            1.0 - alpha / 2.0, successes + 1, total - successes))
    )
    return lower, upper


def wilson_interval(successes: int, total: int, alpha: float = 0.05):
    z = float(norm.ppf(1.0 - alpha / 2.0))
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = (
        z * math.sqrt(
            p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - margin, center + margin


def risk_difference_interval(
    success_a: int,
    total_a: int,
    success_b: int,
    total_b: int,
    alpha: float = 0.05,
):
    """Newcombe hybrid-score interval for p_a - p_b."""
    p_a = success_a / total_a
    p_b = success_b / total_b
    low_a, high_a = wilson_interval(success_a, total_a, alpha)
    low_b, high_b = wilson_interval(success_b, total_b, alpha)
    difference = p_a - p_b
    lower = difference - math.sqrt(
        (p_a - low_a) ** 2 + (high_b - p_b) ** 2)
    upper = difference + math.sqrt(
        (high_a - p_a) ** 2 + (p_b - low_b) ** 2)
    return lower, upper


def compare_independent(
    name_a: str,
    success_a: int,
    total_a: int,
    name_b: str,
    success_b: int,
    total_b: int,
):
    failure_a = total_a - success_a
    failure_b = total_b - success_b
    odds_ratio, p_value = fisher_exact(
        [[success_a, failure_a], [success_b, failure_b]],
        alternative="two-sided",
    )
    p_a = success_a / total_a
    p_b = success_b / total_b
    rd_ci = risk_difference_interval(
        success_a, total_a, success_b, total_b)
    return {
        "comparison": f"{name_a} vs {name_b}",
        "risk_difference": p_a - p_b,
        "risk_difference_95ci": list(rd_ci),
        "risk_ratio": p_a / p_b if p_b > 0 else None,
        "odds_ratio": float(odds_ratio),
        "fisher_exact_two_sided_p": float(p_value),
    }


def required_sample_size_per_group(
    rate_a: float,
    rate_b: float,
    alpha: float = 0.05,
    power: float = 0.80,
):
    """Normal-approximation sample size for two independent proportions."""
    if not 0.0 < rate_a < 1.0 or not 0.0 < rate_b < 1.0:
        raise ValueError("rates must be strictly between zero and one")
    difference = abs(rate_a - rate_b)
    pooled = (rate_a + rate_b) / 2.0
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    z_power = float(norm.ppf(power))
    numerator = (
        z_alpha * math.sqrt(2.0 * pooled * (1.0 - pooled))
        + z_power * math.sqrt(
            rate_a * (1.0 - rate_a) + rate_b * (1.0 - rate_b))
    ) ** 2
    return int(math.ceil(numerator / (difference * difference)))


def method_result(name: str, successes: int, total: int):
    lower, upper = clopper_pearson(successes, total)
    return {
        "name": name,
        "successes": successes,
        "total": total,
        "rate": successes / total,
        "clopper_pearson_95ci": [lower, upper],
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--act", nargs=2, type=int, default=(24, 50),
                        metavar=("SUCCESS", "TOTAL"))
    parser.add_argument("--single", nargs=2, type=int, default=(39, 50),
                        metavar=("SUCCESS", "TOTAL"))
    parser.add_argument("--recoverable", nargs=2, type=int, default=(46, 50),
                        metavar=("SUCCESS", "TOTAL"))
    parser.add_argument(
        "--paired_discordant",
        nargs=2,
        type=int,
        default=None,
        metavar=("SINGLE_FAIL_RECOVERY_SUCCESS",
                 "SINGLE_SUCCESS_RECOVERY_FAIL"),
        help="Run an exact McNemar test when valid paired outcomes exist",
    )
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    counts = {
        "ACT": tuple(args.act),
        "Ours single-shot": tuple(args.single),
        "Ours recoverable": tuple(args.recoverable),
    }
    for name, (successes, total) in counts.items():
        if total <= 0 or successes < 0 or successes > total:
            parser.error(f"invalid count for {name}: {successes}/{total}")

    methods = [
        method_result(name, successes, total)
        for name, (successes, total) in counts.items()
    ]
    comparisons = [
        compare_independent(
            "Ours recoverable", *counts["Ours recoverable"],
            "Ours single-shot", *counts["Ours single-shot"]),
        compare_independent(
            "Ours single-shot", *counts["Ours single-shot"],
            "ACT", *counts["ACT"]),
        compare_independent(
            "Ours recoverable", *counts["Ours recoverable"],
            "ACT", *counts["ACT"]),
    ]
    single_rate = counts["Ours single-shot"][0] / counts["Ours single-shot"][1]
    recovery_rate = (
        counts["Ours recoverable"][0] / counts["Ours recoverable"][1])
    sample_size = required_sample_size_per_group(
        single_rate, recovery_rate, power=args.power)

    report: dict[str, Any] = {
        "assumption": "independent_groups",
        "methods": methods,
        "comparisons": comparisons,
        "sample_size_planning": {
            "target_rates": [single_rate, recovery_rate],
            "two_sided_alpha": 0.05,
            "power": args.power,
            "approx_required_per_group": sample_size,
        },
    }
    if args.paired_discordant is not None:
        discordant_a, discordant_b = args.paired_discordant
        if discordant_a < 0 or discordant_b < 0:
            parser.error("paired discordant counts must be non-negative")
        discordant_total = discordant_a + discordant_b
        paired_p = (
            1.0 if discordant_total == 0
            else float(binomtest(
                discordant_a, discordant_total, 0.5,
                alternative="two-sided").pvalue)
        )
        report["paired_exact_mcnemar"] = {
            "single_fail_recovery_success": discordant_a,
            "single_success_recovery_fail": discordant_b,
            "two_sided_p": paired_p,
        }

    print("Method                 Success      95% exact CI")
    for result in methods:
        lower, upper = result["clopper_pearson_95ci"]
        print(
            f"{result['name']:<22} "
            f"{result['successes']:>2}/{result['total']:<2} "
            f"({pct(result['rate']):>6})  "
            f"[{pct(lower)}, {pct(upper)}]")
    print()
    for result in comparisons:
        lower, upper = result["risk_difference_95ci"]
        print(
            f"{result['comparison']}: "
            f"RD={pct(result['risk_difference'])} "
            f"[{pct(lower)}, {pct(upper)}], "
            f"OR={result['odds_ratio']:.3f}, "
            f"Fisher p={result['fisher_exact_two_sided_p']:.6g}")
    print(
        f"\nApproximate sample size for {pct(single_rate)} vs "
        f"{pct(recovery_rate)}, alpha=.05, power={args.power:.2f}: "
        f"{sample_size} trials/group")
    if "paired_exact_mcnemar" in report:
        paired = report["paired_exact_mcnemar"]
        print(
            "Exact McNemar: "
            f"discordant={paired['single_fail_recovery_success']}/"
            f"{paired['single_success_recovery_fail']}, "
            f"p={paired['two_sided_p']:.6g}")

    output = (
        Path(args.output).expanduser()
        if args.output else Path("task_statistics_report.json"))
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"[Done] {output.resolve()}")


if __name__ == "__main__":
    main()
