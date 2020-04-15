import os
import os.path as osp
import json
from glob import glob
from typing import *
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np


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


class _HyperParams:
    def __init__(self, hyper_params: Dict[str, Any]):
        self.hyper_params = tuple(hyper_params.items())

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.hyper_params)

    def build_model_label(self) -> str:
        hyper_params = self.as_dict()
        if list(hyper_params.keys()) == ["model_name"]:
            model_label = hyper_params["model_name"]
        else:
            model_label = '\n'.join([f"{param_name}={hyper_params[param_name]}"
                                     for param_name in hyper_params.keys()])
        return model_label

    def __hash__(self):
        return hash(self.hyper_params)

    def __eq__(self, other):
        return self.hyper_params == other.hyper_params


def _build_model_key(metrics_file: str,
                     hyper_param_names_for_label: List[str]
                     ) -> _HyperParams:
    hyper_param_file = _find_matching_hyper_param_file(metrics_file)
    hyper_params = _extract_hyper_params(hyper_param_file,
                                         to_extract=hyper_param_names_for_label)
    model_key = _HyperParams(hyper_params)
    return model_key


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


class _LineStyleChooser:
    def __init__(self, unique_hyper_params: Dict[str, List[float]]):
        ordered_param_names = _order_param_names(unique_hyper_params)
        self._build_line_styles(unique_hyper_params, ordered_param_names)
        self._build_legend(ordered_param_names)

    def choose(self, model_key: _HyperParams) -> str:
        line_style = ''
        for name_and_value in model_key.as_dict().items():
            line_style += self._name_and_value_to_style[name_and_value]
        return line_style

    def legend(self):
        return self._legend

    def _build_legend(self, ordered_param_names: List[str]):
        element_names = ["color", "line_shape", "marker"]
        num_elements = min(len(element_names), len(ordered_param_names))
        legend = ' | '.join([f"{element}: {param}" for param, element in
                             zip(ordered_param_names[:num_elements], element_names[:num_elements])])
        self._legend = legend

    @staticmethod
    def _style_elements():
        colors = ['b', 'g', 'r', 'm', 'k', 'c']
        line_shapes = ['-', '--', ':', '-.']
        markers = ['o', 's', '^', '*']
        style_elements = [cycle(colors), cycle(line_shapes), cycle(markers)]
        return style_elements

    def _build_line_styles(self,
                           unique_hyper_params: Dict[str, List[float]],
                           ordered_param_names: List[str]):
        self._name_and_value_to_style = {}
        style_elements = self._style_elements()
        for param_name, param_styles_iter in zip(ordered_param_names, style_elements):
            param_values = unique_hyper_params[param_name]
            param_styles = sorted([next(param_styles_iter) for _ in range(len(param_values))])
            for param_value, param_style in zip(param_values, param_styles):
                self._name_and_value_to_style[(param_name, param_value)] = param_style


def _plot_metric_per_model(
        metric_per_model: Dict[_HyperParams, List[float]],
        title: str,
        save_path: str = None):
    unique_hyper_params = _unique_hyper_params(metric_per_model)
    line_style_chooser = _LineStyleChooser(unique_hyper_params)
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


def _unique_hyper_params(metric_per_model: Dict[_HyperParams, List[float]]
                         ) -> Dict[str, List[float]]:
    model_keys = list(metric_per_model.keys())
    hyper_param_names = list(model_keys[0].as_dict().keys())
    all_hyper_params = {name: [] for name in hyper_param_names}
    for model_key in model_keys:
        for name, value in model_key.as_dict().items():
            all_hyper_params[name].append(value)
    unique_hyper_params = {name: list(set(values)) for name, values in all_hyper_params.items()}
    return unique_hyper_params


def _order_param_names(unique_hyper_params: Dict[str, List[float]]) -> List[str]:
    unique_param_counts = [(name, len(values)) for name, values in unique_hyper_params.items()]
    ordered_param_counts = sorted(unique_param_counts, key=lambda tup: tup[1], reverse=True)
    ordered_param_names = [name for name, count in ordered_param_counts]
    return ordered_param_names


def _plot_all():
    # grid_search
    save_dir = osp.join("models", "fully_connected", "grid_search")
    hyper_param_names_for_label = ["init_gaussian_std", "learning_rate", "sgd_momentum"]
    plot_metrics(save_dir, hyper_param_names_for_label)

    # optimization
    save_dir = osp.join("models", "fully_connected", "optimization")
    hyper_param_names_for_label = ["optimizer_type"]
    plot_metrics(save_dir, hyper_param_names_for_label)

    # initialization
    save_dir = osp.join("models", "fully_connected", "initialization")
    hyper_param_names_for_label = ["init_type"]
    plot_metrics(save_dir, hyper_param_names_for_label)

    # pca
    save_dir = osp.join("models", "fully_connected", "pca")
    hyper_param_names_for_label = ["model_name"]
    plot_metrics(save_dir, hyper_param_names_for_label)

    # regularization
    save_dir = osp.join("models", "fully_connected", "regularization")
    hyper_param_names_for_label = ["dropout_drop_probability", "weight_decay"]
    plot_metrics(save_dir, hyper_param_names_for_label)

    # width
    save_dir = osp.join("models", "fully_connected", "width")
    hyper_param_names_for_label = ["hidden_size"]
    plot_metrics(save_dir, hyper_param_names_for_label)

    # depth
    save_dir = osp.join("models", "fully_connected", "depth")
    hyper_param_names_for_label = ["num_hidden_layers"]
    plot_metrics(save_dir, hyper_param_names_for_label)


if __name__ == '__main__':
    _plot_all()
