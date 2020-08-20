import os
import os.path as osp
from typing import *
import pandas as pd

from hyper_params import HyperParams


def _save_comparison_table(comparison_table: pd.DataFrame,
                           comparison_dir: str):
    os.makedirs(comparison_dir, exist_ok=True)
    save_path = osp.join(comparison_dir, "comparison_table.xlsx")
    comparison_table.to_excel(save_path)


def create_comparison_table(metrics_per_model: Dict[HyperParams, Dict[str, List[float]]],
                            comparison_dir: str):
    comparison_table = _create_comparison_table(metrics_per_model)
    _save_comparison_table(comparison_table, comparison_dir)


def _create_comparison_table(metrics_per_model: Dict[HyperParams, Dict[str, List[float]]]
                             ) -> pd.DataFrame:
    table_rows = []
    for model_key, model_metrics in metrics_per_model.items():
        interesting_metrics = _extract_interesting_metrics(model_metrics)
        hyper_params = model_key.as_dict()
        row = {**hyper_params, **interesting_metrics}
        table_rows.append(row)
    comparison_table = pd.DataFrame(table_rows)
    return comparison_table


def _extract_interesting_metrics(model_metrics: Dict):
    train_test_keys = ["train_classification_metrics",
                       "test_classification_metrics"]
    interesting_metrics = {k: v for k, v in model_metrics.items() if k not in train_test_keys}

    for stage in ("train", "test"):
        stage_classification_metrics = model_metrics[stage + "_classification_metrics"]
        for class_name in stage_classification_metrics.keys():
            f1_score = stage_classification_metrics[class_name]["f1-score"]
            interesting_metrics[f"{stage}_{class_name}_f1_score"] = f1_score

    return interesting_metrics
