import os
import os.path as osp
from typing import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from hyper_params import HyperParams
from line_style_chooser import LineStyleChooser


def plot_metrics_by_type(metrics_per_model: Dict[HyperParams, Dict[str, List[float]]],
                         comparison_dir: str):
    metrics_by_type = _rearrange_metrics_by_type(metrics_per_model)
    for metric_name, specific_metric_per_model in metrics_by_type.items():
        os.makedirs(comparison_dir, exist_ok=True)
        save_path = osp.join(comparison_dir, metric_name + ".png")
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
        save_path: str = None,
        smooth_metric: bool = True):
    unique_hyper_params = _unique_hyper_params(metric_per_model)
    line_style_chooser = LineStyleChooser(unique_hyper_params)
    keys_and_styles = [(model_key, line_style_chooser.choose(model_key))
                       for model_key in metric_per_model.keys()]
    keys_and_styles = sorted(keys_and_styles, key=lambda tup: tup[1])

    W, H = plt.rcParamsDefault["figure.figsize"]
    fig, ax = plt.subplots(figsize=(2.5 * W, 2.5 * H))
    ax.set_title(title, fontsize=24)
    ax.set_xlabel("batch", fontsize=20)
    ax.tick_params(labelsize=18)
    for model_key, line_style in keys_and_styles:
        metric_values = metric_per_model[model_key]
        if smooth_metric:
            metric_values = _smooth_metric(metric_values)

        epochs = list(range(1, len(metric_values) + 1))
        line, = ax.plot(epochs, metric_values, line_style,
                        markevery=int(np.ceil(len(metric_values) / 10)), markersize=6)
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


def _smooth_metric(metric_values: Iterable[float]):
    smooth_metric_values = gaussian_filter1d(metric_values, sigma=20)
    return smooth_metric_values


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
