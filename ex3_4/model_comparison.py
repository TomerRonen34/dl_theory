import os.path as osp
import json
from glob import glob
from typing import *

from hyper_params import HyperParams
from plot_metrics import plot_metrics_by_type
from comparison_table import create_comparison_table


def compare_models(models_dir: str,
                   hyper_param_names_to_compare: List[str]):
    comparison_dir = osp.join(models_dir, "model_comparison")

    final_model_metrics_per_model = _gather_metrics(models_dir, hyper_param_names_to_compare,
                                                    metric_file_name="final_model_metrics.json")
    create_comparison_table(final_model_metrics_per_model, comparison_dir)

    training_metrics_per_model = _gather_metrics(models_dir, hyper_param_names_to_compare,
                                                 metric_file_name="training_metrics.json")
    plot_metrics_by_type(training_metrics_per_model, comparison_dir)


def _gather_metrics(models_dir: str,
                    hyper_param_names_to_compare: List[str],
                    metric_file_name: str
                    ) -> Dict[HyperParams, Dict[str, List[float]]]:
    """
    returns metrics_per_model- dict that maps a model label to the model metrics dict
    """
    metrics_per_model = {}
    metric_files = _find_metric_files(models_dir, metric_file_name)
    for metrics_file in metric_files:
        model_key = _build_model_key(metrics_file, hyper_param_names_to_compare)
        metrics = _load_metrics(metrics_file)
        metrics_per_model[model_key] = metrics
    return metrics_per_model


def _find_metric_files(models_dir: str,
                       metric_file_name: str) -> List[str]:
    file_pattern = osp.join(models_dir, "**", metric_file_name)
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


if __name__ == '__main__':
    compare_models(models_dir="models/mobilenet_v2",
                   hyper_param_names_to_compare=["model_name"])
