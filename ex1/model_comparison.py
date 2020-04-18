import os
import os.path as osp
import json
from glob import glob
from typing import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hyper_params import HyperParams
from line_style_chooser import LineStyleChooser


def compare_models(models_dir: str,
                   hyper_param_names_to_compare: List[str]):
    plot_metrics(models_dir, hyper_param_names_to_compare)
    create_comparison_tables(models_dir, hyper_param_names_to_compare)


def plot_metrics(models_dir: str,
                 hyper_param_names_to_compare: List[str]):
    metrics_per_model = _gather_metrics(models_dir, hyper_param_names_to_compare)
    _plot_metrics_by_type(metrics_per_model, models_dir)


def create_comparison_tables(models_dir: str,
                             hyper_param_names_to_compare: List[str]):
    metrics_per_model = _gather_metrics(models_dir, hyper_param_names_to_compare)
    best_metrics_per_model = {model_key: _best_epoch_metrics(metrics)
                              for model_key, metrics in metrics_per_model.items()}
    comparison_table = _create_model_comparison_table(best_metrics_per_model)
    _save_comparison_table(comparison_table, models_dir)
    hyper_param_statistics = _create_hyper_param_statistics(comparison_table,
                                                            hyper_param_names_to_compare)
    _save_hyper_param_statistics(hyper_param_statistics, models_dir)


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
                           models_dir: str):
    save_dir = osp.join(models_dir, "model_comparison")
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, "comparison_table.xlsx")
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
                                 models_dir: str):
    save_dir = osp.join(models_dir, "model_comparison")
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, "hyper_param_statistics.xlsx")
    hyper_param_statistics.to_excel(save_path)


def _gather_metrics(models_dir: str,
                    hyper_param_names_for_label: List[str]):
    """
    returns metrics_per_model- dict that maps a model label to the model metrics dict
    """
    metrics_per_model = {}
    metric_files = _find_metric_files(models_dir)
    for metrics_file in metric_files:
        model_key = _build_model_key(metrics_file, hyper_param_names_for_label)
        metrics = _load_metrics(metrics_file)
        metrics_per_model[model_key] = metrics
    return metrics_per_model


def _find_metric_files(models_dir: str) -> List[str]:
    file_pattern = osp.join(models_dir, "**", "metrics.json")
    metric_files = glob(file_pattern)
    return metric_files


def _find_matching_hyper_param_file(metric_file: str) -> str:
    model_dir = osp.dirname(metric_file)
    hyper_param_file = osp.join(model_dir, "hyper_params.json")
    return hyper_param_file


def _extract_hyper_params(hyper_param_file: str,
                          to_extract: List[str]
                          ) -> Dict[str, Any]:
    with open(hyper_param_file, 'r') as f:
        hyper_params = json.load(f)
    hyper_params = {param_name: hyper_params[param_name] for param_name in to_extract}
    return hyper_params


def _build_model_key(metrics_file: str,
                     hyper_param_names_for_label: List[str]
                     ) -> HyperParams:
    hyper_param_file = _find_matching_hyper_param_file(metrics_file)
    hyper_params = _extract_hyper_params(hyper_param_file,
                                         to_extract=hyper_param_names_for_label)
    model_key = HyperParams(hyper_params)
    return model_key


def _load_metrics(metrics_file: str):
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    return metrics


def _plot_metrics_by_type(metrics_per_model,
                          models_dir: str):
    metrics_by_type = _rearrange_metrics_by_type(metrics_per_model)
    for metric_name, specific_metric_per_model in metrics_by_type.items():
        figures_dir = osp.join(models_dir, "model_comparison")
        os.makedirs(figures_dir, exist_ok=True)
        save_path = osp.join(figures_dir, metric_name + ".png")
        _plot_metric_per_model(specific_metric_per_model, title=metric_name, save_path=save_path)


def _rearrange_metrics_by_type(metrics_per_model):
    metrics_by_type = {}
    metric_names = _metric_names(metrics_per_model)
    for metric_name in metric_names:
        specific_metric_per_model = _extract_specific_metric(metrics_per_model, metric_name)
        metrics_by_type[metric_name] = specific_metric_per_model
    return metrics_by_type


def _metric_names(metrics_per_model):
    metric_names = []
    for metrics in metrics_per_model.values():
        for name in metrics.keys():
            metric_names.append(name)
    metric_names = list(set(metric_names))
    return metric_names


def _extract_specific_metric(metrics_per_model,
                             metric_name: str):
    specific_metric_per_model = {}
    for model_key, metrics in metrics_per_model.items():
        if metric_name in metrics.keys():
            specific_metric_per_model[model_key] = metrics[metric_name]
    return specific_metric_per_model


def _plot_metric_per_model(
        metric_per_model: Dict[HyperParams, List[float]],
        title: str,
        save_path: str = None):
    unique_hyper_params = _unique_hyper_params(metric_per_model)
    line_style_chooser = LineStyleChooser(unique_hyper_params)
    keys_and_styles = [(model_key, line_style_chooser.choose(model_key))
                       for model_key in metric_per_model.keys()]
    keys_and_styles = sorted(keys_and_styles, key=lambda tup: tup[1])

    W, H = plt.rcParamsDefault["figure.figsize"]
    fig, ax = plt.subplots(figsize=(2.5 * W, 2.5 * H))
    ax.set_title(title, fontsize=24)
    ax.set_xlabel("epoch", fontsize=20)
    ax.tick_params(labelsize=18)
    for model_key, line_style in keys_and_styles:
        values = metric_per_model[model_key]
        epochs = list(range(1, len(values) + 1))
        line, = ax.plot(epochs, values, line_style,
                        markevery=int(np.ceil(len(values) / 10)), markersize=6)
        model_label = model_key.build_model_label()
        line.set_label(model_label)

    num_plots = len(metric_per_model)
    ncol = 1 if num_plots <= 10 else 3 if num_plots % 3 == 0 else 2
    legend_title = line_style_chooser.legend()
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=ncol,
              title=legend_title, fontsize=18, title_fontsize=20)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def _unique_hyper_params(metric_per_model: Dict[HyperParams, List[float]]
                         ) -> Dict[str, List[float]]:
    model_keys = list(metric_per_model.keys())
    hyper_param_names = list(model_keys[0].as_dict().keys())
    all_hyper_params = {name: [] for name in hyper_param_names}
    for model_key in model_keys:
        for name, value in model_key.as_dict().items():
            all_hyper_params[name].append(value)
    unique_hyper_params = {name: list(set(values)) for name, values in all_hyper_params.items()}
    return unique_hyper_params


def _compare_all():
    # grid_search
    save_dir = osp.join("models", "fully_connected", "grid_search")
    hyper_param_names_for_label = ["init_gaussian_std", "learning_rate", "sgd_momentum"]
    compare_models(save_dir, hyper_param_names_for_label)

    # optimization
    save_dir = osp.join("models", "fully_connected", "optimization")
    hyper_param_names_for_label = ["optimizer_type"]
    compare_models(save_dir, hyper_param_names_for_label)

    # initialization
    save_dir = osp.join("models", "fully_connected", "initialization")
    hyper_param_names_for_label = ["init_type"]
    compare_models(save_dir, hyper_param_names_for_label)

    # pca
    save_dir = osp.join("models", "fully_connected", "pca")
    hyper_param_names_for_label = ["model_name"]
    compare_models(save_dir, hyper_param_names_for_label)

    # regularization
    save_dir = osp.join("models", "fully_connected", "regularization")
    hyper_param_names_for_label = ["dropout_drop_probability", "weight_decay"]
    compare_models(save_dir, hyper_param_names_for_label)

    # width
    save_dir = osp.join("models", "fully_connected", "width")
    hyper_param_names_for_label = ["hidden_size"]
    compare_models(save_dir, hyper_param_names_for_label)

    # depth
    save_dir = osp.join("models", "fully_connected", "depth")
    hyper_param_names_for_label = ["num_hidden_layers"]
    compare_models(save_dir, hyper_param_names_for_label)


if __name__ == '__main__':
    _compare_all()
