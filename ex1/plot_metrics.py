import os
import os.path as osp
import json
from glob import glob
from typing import *
import matplotlib.pyplot as plt
from itertools import cycle


def plot_metrics(models_dir: str,
                 hyper_param_names_for_label: List[str]):
    metrics_per_model = _gather_metrics(models_dir, hyper_param_names_for_label)
    _plot_metrics_by_type(metrics_per_model, models_dir)


def _gather_metrics(models_dir: str,
                    hyper_param_names_for_label: List[str]):
    """
    returns metrics_per_model- dict that maps a model label to the model metrics dict
    """
    metrics_per_model = {}
    metric_files = _find_metric_files(models_dir)
    for metrics_file in metric_files:
        hyper_param_file = _find_matching_hyper_param_file(metrics_file)
        model_label = _build_model_label(hyper_param_file, hyper_param_names_for_label)
        metrics = _load_metrics(metrics_file)
        metrics_per_model[model_label] = metrics
    return metrics_per_model


def _find_metric_files(models_dir: str) -> List[str]:
    file_pattern = osp.join(models_dir, "**", "metrics.json")
    metric_files = glob(file_pattern)
    return metric_files


def _find_matching_hyper_param_file(metric_file: str) -> str:
    model_dir = osp.dirname(metric_file)
    hyper_param_file = osp.join(model_dir, "hyper_params.json")
    return hyper_param_file


def _build_model_label(hyper_param_file: str,
                       hyper_param_names_for_label: List[str]):
    with open(hyper_param_file, 'r') as f:
        hyper_params = json.load(f)
    model_label = '\n'.join([f"{name}={hyper_params[name]}"
                             for name in hyper_param_names_for_label])
    return model_label


def _load_metrics(metrics_file: str):
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    return metrics


def _plot_metrics_by_type(metrics_per_model,
                          models_dir: str):
    metrics_by_type = _rearrange_metrics_by_type(metrics_per_model)
    for metric_name, specific_metric_per_model in metrics_by_type.items():
        figures_dir = osp.join(models_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        save_path = osp.join(figures_dir, metric_name + ".png")
        _plot_several(specific_metric_per_model, title=metric_name, save_path=save_path)


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
    for model_label, metrics in sorted(metrics_per_model.items()):
        if metric_name in metrics.keys():
            specific_metric_per_model[model_label] = metrics[metric_name]
    return specific_metric_per_model


def _plot_several(
        to_plot: Dict[str, List[float]],
        title: str,
        save_path: str = None):
    line_styles = _line_style_iterator()

    plt.figure()
    plt.title(title)
    for name, values in to_plot.items():
        line, = plt.plot(values, next(line_styles))
        line.set_label(name)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")


def _line_style_iterator():
    colors = ['b', 'g', 'r', 'c', 'm', 'k']
    line_shapes = ["-", "--", ":"]
    line_styles = [color + shape for color in colors for shape in line_shapes]
    line_styles = cycle(line_styles)
    return line_styles


if __name__ == '__main__':
    plot_metrics(models_dir=r"models\fully_connected",
                 hyper_param_names_for_label=["init_gaussian_std", "learning_rate", "momentum"])
