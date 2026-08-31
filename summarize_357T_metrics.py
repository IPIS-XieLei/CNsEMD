import argparse
import re
from pathlib import Path

import numpy as np


METRIC_NAMES = [
    "Dice_avg",
    "Jac_avg",
    "ASD_avg",
    "AHD_avg"
]


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize mix, 3T, 5T and 7T metrics "
            "from ensemble_metrics_summary.txt"
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to ensemble_metrics_summary.txt"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path. If omitted, the result is saved "
            "beside the input file as "
            "ensemble_metrics_mix_3T_5T_7T.txt"
        )
    )

    return parser.parse_args()


def get_field_strength(case_name):
    """
    根据病例名称识别场强。

    支持：
        3_100206
        5_242722
        7_102816
        NO.3_100206
        NO.5_242722
        NO.7_102816
        HCP3T_xxx
        Diff5T_xxx
        HCP7T_xxx
    """
    name = case_name.upper().strip()

    patterns = {
        "3T": [
            r"^3[_-]",
            r"NO[._-]?3(?:T)?[_-]",
            r"(?<!\d)3T(?!\d)"
        ],
        "5T": [
            r"^5[_-]",
            r"NO[._-]?5(?:T)?[_-]",
            r"(?<!\d)5T(?!\d)"
        ],
        "7T": [
            r"^7[_-]",
            r"NO[._-]?7(?:T)?[_-]",
            r"(?<!\d)7T(?!\d)"
        ]
    }

    for field_strength, field_patterns in patterns.items():
        for pattern in field_patterns:
            if re.search(pattern, name):
                return field_strength

    raise ValueError(
        f"Cannot determine field strength from "
        f"case name: {case_name}"
    )


def read_case_metrics(summary_path):
    """
    读取原始ensemble_metrics_summary.txt。

    返回：
        [
            (
                case_name,
                [Dice_avg, Jac_avg, ASD_avg, AHD_avg]
            ),
            ...
        ]
    """
    case_metrics = []

    with open(
        summary_path,
        "r",
        encoding="utf-8"
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1
        ):
            line = line.strip()

            if not line:
                continue

            # 忽略分隔线
            if line.startswith("="):
                continue

            # 忽略表头和总汇总行
            if line.startswith("Case\t"):
                continue

            if line.startswith("MEAN"):
                continue

            columns = line.split("\t")

            if len(columns) < 5:
                print(
                    f"Warning: skipped line {line_number}, "
                    f"insufficient columns."
                )
                continue

            case_name = columns[0].strip()

            try:
                metrics = [
                    float(columns[1]),  # Dice_avg
                    float(columns[2]),  # Jac_avg
                    float(columns[3]),  # ASD_avg
                    float(columns[4])   # AHD_avg
                ]
            except ValueError:
                print(
                    f"Warning: skipped line {line_number}, "
                    f"invalid metric values."
                )
                continue

            case_metrics.append(
                (case_name, metrics)
            )

    if len(case_metrics) == 0:
        raise RuntimeError(
            "No valid case-level metrics were found in "
            f"{summary_path}"
        )

    return case_metrics


def build_groups(case_metrics):
    """
    构建mix、3T、5T和7T四个分组。
    """
    groups = {
        "mix": [],
        "3T": [],
        "5T": [],
        "7T": []
    }

    for case_name, metrics in case_metrics:
        field_strength = get_field_strength(
            case_name
        )

        metric_array = np.asarray(
            metrics,
            dtype=np.float64
        )

        groups["mix"].append(metric_array)
        groups[field_strength].append(metric_array)

    return groups


def summarize_group(group_metrics):
    """
    计算组内mean和std。

    np.nanstd默认ddof=0，与原测试代码一致。
    """
    if len(group_metrics) == 0:
        return {
            "count": 0,
            "mean": np.full(4, np.nan),
            "std": np.full(4, np.nan)
        }

    metric_array = np.stack(
        group_metrics,
        axis=0
    )

    return {
        "count": metric_array.shape[0],
        "mean": np.nanmean(
            metric_array,
            axis=0
        ),
        "std": np.nanstd(
            metric_array,
            axis=0,
            ddof=0
        )
    }


def save_wide_summary(groups, output_path):
    group_order = [
        "mix",
        "3T",
        "5T",
        "7T"
    ]

    summaries = {
        group_name: summarize_group(
            groups[group_name]
        )
        for group_name in group_order
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        # 第一行：mix、3T、5T、7T
        group_header = []

        for group_name in group_order:
            group_header.extend([
                group_name,
                "",
                "",
                ""
            ])

        file.write(
            "\t".join(group_header) + "\n"
        )

        # 第二行：每组的四个指标名称
        metric_header = []

        for _ in group_order:
            metric_header.extend(METRIC_NAMES)

        file.write(
            "\t".join(metric_header) + "\n"
        )

        # 第三行：mean±std
        result_row = []

        for group_name in group_order:
            summary = summaries[group_name]

            for mean_value, std_value in zip(
                summary["mean"],
                summary["std"]
            ):
                if (
                    np.isnan(mean_value) or
                    np.isnan(std_value)
                ):
                    result_row.append("NA")
                else:
                    result_row.append(
                        f"{mean_value:.4f}±"
                        f"{std_value:.4f}"
                    )

        file.write(
            "\t".join(result_row) + "\n"
        )

        # 第四行：各组样本数量
        count_row = []

        for group_name in group_order:
            count_row.extend([
                f"N={summaries[group_name]['count']}",
                "",
                "",
                ""
            ])

        file.write(
            "\t".join(count_row) + "\n"
        )

    return summaries


def print_summary(summaries):
    print("\n" + "=" * 105)
    print(
        " MIX / 3T / 5T / 7T SUMMARY ".center(
            105
        )
    )
    print("=" * 105)

    print(
        f"{'Group':<10}"
        f"{'N':>6}"
        f"{'Dice_avg':>22}"
        f"{'Jac_avg':>22}"
        f"{'ASD_avg':>22}"
        f"{'AHD_avg':>22}"
    )

    print("-" * 105)

    for group_name in [
        "mix",
        "3T",
        "5T",
        "7T"
    ]:
        summary = summaries[group_name]

        formatted_metrics = []

        for mean_value, std_value in zip(
            summary["mean"],
            summary["std"]
        ):
            if (
                np.isnan(mean_value) or
                np.isnan(std_value)
            ):
                formatted_metrics.append("NA")
            else:
                formatted_metrics.append(
                    f"{mean_value:.4f}±"
                    f"{std_value:.4f}"
                )

        print(
            f"{group_name:<10}"
            f"{summary['count']:>6}"
            f"{formatted_metrics[0]:>22}"
            f"{formatted_metrics[1]:>22}"
            f"{formatted_metrics[2]:>22}"
            f"{formatted_metrics[3]:>22}"
        )

    print("=" * 105)


def check_case_counts(summaries):
    expected_counts = {
        "mix": 40,
        "3T": 20,
        "5T": 10,
        "7T": 10
    }

    for group_name, expected_count in (
        expected_counts.items()
    ):
        actual_count = summaries[
            group_name
        ]["count"]

        if actual_count != expected_count:
            print(
                f"Warning: {group_name} contains "
                f"{actual_count} cases, expected "
                f"{expected_count}."
            )


def main():
    args = get_parser()

    input_path = Path(args.input).resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    if args.output is None:
        output_path = (
            input_path.parent /
            "ensemble_metrics_mix_3T_5T_7T.txt"
        )
    else:
        output_path = Path(
            args.output
        ).resolve()

    case_metrics = read_case_metrics(
        input_path
    )

    groups = build_groups(
        case_metrics
    )

    summaries = save_wide_summary(
        groups,
        output_path
    )

    check_case_counts(
        summaries
    )

    print_summary(
        summaries
    )

    print(
        f"\nSummary saved to:\n{output_path}"
    )


if __name__ == "__main__":
    main()