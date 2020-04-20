import os
import os.path as osp
from typing import *
import numpy as np
import pandas as pd

from hyper_params import HyperParams


def create_comparison_tables(metrics_per_model: Dict[HyperParams, Dict[str, List[float]]],
                             comparison_dir: str,
                             hyper_param_names_to_compare: List[str]):
    best_metrics_per_model = {model_key: _best_epoch_metrics(metrics)
                              for model_key, metrics in metrics_per_model.items()}
    comparison_table = _create_model_comparison_table(best_metrics_per_model)
    _save_comparison_table(comparison_table, comparison_dir)
    hyper_param_statistics = _create_hyper_param_statistics(comparison_table,
                                                            hyper_param_names_to_compare)
    _save_hyper_param_statistics(hyper_param_statistics, comparison_dir)


def _best_epoch_metrics(metrics: Dict[str, List[float]]
                        ) -> Dict[str, float]:
    best_epoch = np.argmax(metrics["test_accuracy"])
    best_epoch_metrics = {metric_name: metrics[metric_name][best_epoch]
                          for metric_name in metrics.keys()}
    return best_epoch_metrics


def _create_model_comparison_table(
        best_metrics_per_model: Dict[HyperParams, Dict[str, float]]) -> pd.DataFrame:
    table_rows = []
    for model_key, best_epoch_metrics in best_metrics_per_model.items():
        hyper_params = model_key.as_dict()
        row = {**hyper_params, **best_epoch_metrics}
        table_rows.append(row)
    comparison_table = pd.DataFrame(table_rows)

    comparison_table = comparison_table.sort_values(by="test_accuracy", ascending=False)
    comparison_table.index = range(1, len(comparison_table) + 1)

    hyper_param_names = list(hyper_params.keys())
    primary_metric_names = ["test_accuracy", "train_accuracy", "test_loss", "train_loss"]
    minory_metric_names = list(set(best_epoch_metrics.keys()).difference(primary_metric_names))
    column_order = hyper_param_names + primary_metric_names + minory_metric_names
    comparison_table = comparison_table[column_order]

    return comparison_table


def _save_comparison_table(comparison_table: pd.DataFrame,
                           comparison_dir: str):
    os.makedirs(comparison_dir, exist_ok=True)
    save_path = osp.join(comparison_dir, "comparison_table.xlsx")
    comparison_table.to_excel(save_path)


def _create_hyper_param_statistics(comparison_table: pd.DataFrame,
                                   hyper_param_names_to_compare: List[str]):
    stats_per_param = []
    for hyper_param_name in hyper_param_names_to_compare:
        groupby_param = comparison_table.groupby(hyper_param_name)
        test_stats = groupby_param[["test_accuracy"]].agg(["median", "min", "max"])
        train_stats = groupby_param[["train_accuracy"]].agg(["median", "min", "max"])
        param_stats = pd.concat([test_stats, train_stats], axis="columns")
        stats_per_param.append(param_stats)

    stats_per_param = [_named_index_to_multi_index(stats) for stats in stats_per_param]
    hyper_param_statistics = pd.concat(stats_per_param)
    return hyper_param_statistics


def _named_index_to_multi_index(df: pd.DataFrame):
    named_index = df.index
    multi_index = pd.MultiIndex.from_tuples((named_index.name, value) for value in named_index)
    df = df.set_index(multi_index)
    return df


def _save_hyper_param_statistics(hyper_param_statistics: pd.DataFrame,
                                 comparison_dir: str):
    os.makedirs(comparison_dir, exist_ok=True)
    save_path = osp.join(comparison_dir, "hyper_param_statistics.xlsx")
    hyper_param_statistics.to_excel(save_path)
