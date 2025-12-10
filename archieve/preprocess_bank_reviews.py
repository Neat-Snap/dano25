#!/usr/bin/env python3
"""
Utility script that prepares bank product review data for hypothesis testing.

The script reads dropped_shi.csv and produces:
1. bank_reviews_preprocessed.csv – row-level dataset focused on bank products
2. bank_product_dissatisfaction_summary.csv – aggregated dissatisfaction metrics by product/month
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

INPUT_FILE = Path("dropped_shi.csv")
CLEANED_FILE = Path("bank_reviews_preprocessed.csv")
SUMMARY_FILE = Path("bank_product_dissatisfaction_summary.csv")


def parse_datetime(value: str) -> datetime | None:
    """Parse timestamps with optional microseconds."""
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def normalize_category(value: str) -> Tuple[str, str]:
    """Return original fallback label and lowercase token for grouping."""
    if not value:
        pretty = "Не указано"
        token = "не указано"
    else:
        pretty = value.strip() or "Не указано"
        token = pretty.lower()
    return pretty, token


def parse_mark(value: str) -> int | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def sentiment_label(emotion_raw: str) -> str:
    if not emotion_raw:
        return "unknown"
    emotion_raw = emotion_raw.strip()
    if emotion_raw == "1":
        return "positive"
    if emotion_raw == "0":
        return "negative"
    return "unknown"


def dissatisfaction_bucket(mark: int | None) -> Tuple[str, int]:
    if mark is None:
        return "unknown", 0
    if mark <= 2:
        return "dissatisfied", 1
    if mark == 3:
        return "neutral", 0
    return "satisfied", 0


@dataclass
class AggregationRow:
    product_group: str
    product_group_token: str
    product_detail: str
    product_detail_token: str
    review_month: str
    total_reviews: int = 0
    dissatisfied_reviews: int = 0


def main() -> None:
    if not INPUT_FILE.exists():
        raise SystemExit(f"Input file {INPUT_FILE} not found")

    fieldnames = [
        "record_id",
        "client_id",
        "review_datetime",
        "finish_datetime",
        "review_source",
        "product_group",
        "product_group_token",
        "product_detail",
        "product_detail_token",
        "review_mark",
        "review_emotion",
        "sentiment_label",
        "dissatisfaction_bucket",
        "is_dissatisfied",
        "reason",
        "review_theme",
        "subtheme",
        "solution_flag",
        "age_segment",
        "segment_name",
        "review_month",
        "response_time_hours",
        "review_text",
    ]

    aggregations: Dict[Tuple[str, str, str], AggregationRow] = {}
    record_id = 0

    with INPUT_FILE.open(newline="", encoding="utf-8") as src, CLEANED_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as cleaned:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(cleaned, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            company = (row.get("company") or "").strip().lower()
            if company != "банк":
                continue  # Focus on bank product reviews only

            mark_value = parse_mark(row.get("review_mark", ""))
            if mark_value is None:
                continue

            review_dt = parse_datetime(row.get("review_dttm", ""))
            finish_dt = parse_datetime(row.get("finish_dttm", ""))
            review_month = (
                review_dt.strftime("%Y-%m") if review_dt is not None else "unknown"
            )
            response_hours = (
                round((finish_dt - review_dt).total_seconds() / 3600, 2)
                if review_dt and finish_dt
                else ""
            )

            product_group, product_group_token = normalize_category(
                row.get("business_line", "")
            )
            product_detail, product_detail_token = normalize_category(row.get("product", ""))

            bucket, dissatisfied_flag = dissatisfaction_bucket(mark_value)
            sent_label = sentiment_label(row.get("review_emotion", ""))

            record_id += 1

            cleaned_row = {
                "record_id": record_id,
                "client_id": row.get("id_client", ""),
                "review_datetime": row.get("review_dttm", ""),
                "finish_datetime": row.get("finish_dttm", ""),
                "review_source": row.get("review_source", ""),
                "product_group": product_group,
                "product_group_token": product_group_token,
                "product_detail": product_detail,
                "product_detail_token": product_detail_token,
                "review_mark": mark_value,
                "review_emotion": row.get("review_emotion", ""),
                "sentiment_label": sent_label,
                "dissatisfaction_bucket": bucket,
                "is_dissatisfied": dissatisfied_flag,
                "reason": row.get("reason", ""),
                "review_theme": row.get("review_theme", ""),
                "subtheme": row.get("subtheme", ""),
                "solution_flag": row.get("solution_flg", ""),
                "age_segment": row.get("age_segment", ""),
                "segment_name": row.get("segment_name", ""),
                "review_month": review_month,
                "response_time_hours": response_hours,
                "review_text": (row.get("review_text") or "").strip(),
            }
            writer.writerow(cleaned_row)

            key = (product_group_token, product_detail_token, review_month)
            if key not in aggregations:
                aggregations[key] = AggregationRow(
                    product_group=product_group,
                    product_group_token=product_group_token,
                    product_detail=product_detail,
                    product_detail_token=product_detail_token,
                    review_month=review_month,
                )
            agg_row = aggregations[key]
            agg_row.total_reviews += 1
            agg_row.dissatisfied_reviews += dissatisfied_flag

    # Write aggregated summary covering dissatisfaction share per product and month
    with SUMMARY_FILE.open("w", newline="", encoding="utf-8") as summary_file:
        summary_fields = [
            "product_group",
            "product_detail",
            "product_group_token",
            "product_detail_token",
            "review_month",
            "total_reviews",
            "dissatisfied_reviews",
            "dissatisfied_share",
        ]
        writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        writer.writeheader()
        for agg in sorted(
            aggregations.values(),
            key=lambda row: (row.product_group_token, row.product_detail_token, row.review_month),
        ):
            share = (
                round(agg.dissatisfied_reviews / agg.total_reviews, 4)
                if agg.total_reviews
                else 0
            )
            writer.writerow(
                {
                    "product_group": agg.product_group,
                    "product_detail": agg.product_detail,
                    "product_group_token": agg.product_group_token,
                    "product_detail_token": agg.product_detail_token,
                    "review_month": agg.review_month,
                    "total_reviews": agg.total_reviews,
                    "dissatisfied_reviews": agg.dissatisfied_reviews,
                    "dissatisfied_share": share,
                }
            )


if __name__ == "__main__":
    main()
