import os.path as osp
import json
from glob import glob
from typing import *

from hyper_params import HyperParams
from plot_metrics import plot_metrics_by_type
from create_comparison_tables import create_comparison_tables


def compare_models(models_dir: str,
                   hyper_param_names_to_compare: List[str]):
    metrics_per_model = _gather_metrics(models_dir, hyper_param_names_to_compare)
    comparison_dir = osp.join(models_dir, "model_comparison")
    plot_metrics_by_type(metrics_per_model, comparison_dir)
    create_comparison_tables(metrics_per_model, comparison_dir, hyper_param_names_to_compare)


def _gather_metrics(models_dir: str,
                    hyper_param_names_to_compare: List[str]
                    ) -> Dict[HyperParams, Dict[str, List[float]]]:
    """
    returns metrics_per_model- dict that maps a model label to the model metrics dict
    """
    metrics_per_model = {}
    metric_files = _find_metric_files(models_dir)
    for metrics_file in metric_files:
        model_key = _build_model_key(metrics_file, hyper_param_names_to_compare)
        metrics = _load_metrics(metrics_file)
        metrics_per_model[model_key] = metrics
    return metrics_per_model


def _find_metric_files(models_dir: str) -> List[str]:
    file_pattern = osp.join(models_dir, "**", "metrics.json")
    metric_files = glob(file_pattern)
    return metric_files


def _build_model_key(metrics_file: str,
                     hyper_param_names_for_label: List[str]
                     ) -> HyperParams:
    hyper_param_file = _find_matching_hyper_param_file(metrics_file)
    hyper_params = _extract_hyper_params(hyper_param_file,
                                         to_extract=hyper_param_names_for_label)
    model_key = HyperParams(hyper_params)
    return model_key


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


def _load_metrics(metrics_file: str):
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    return metrics


def _compare_all():
    # grid_search
    save_dir = osp.join("models", "fully_connected", "grid_search")
    hyper_param_names_for_label = ["init_gaussian_std", "learning_rate", "sgd_momentum"]
    compare_models(save_dir, hyper_param_names_for_label)

    # optimization
    save_dir = osp.join("models", "fully_connected", "optimization")
    hyper_param_names_for_label = ["optimizer_type", "learning_rate"]
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
