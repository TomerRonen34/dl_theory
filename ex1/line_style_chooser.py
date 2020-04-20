from typing import *
from itertools import cycle
from hyper_params import HyperParams


class LineStyleChooser:
    def __init__(self, unique_hyper_params: Dict[str, List[float]]):
        ordered_param_names = self._order_param_names(unique_hyper_params)
        self._build_line_styles(unique_hyper_params, ordered_param_names)
        self._build_legend(ordered_param_names)

    def choose(self, model_key: HyperParams) -> str:
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
    def _order_param_names(unique_hyper_params: Dict[str, List[float]]) -> List[str]:
        unique_param_counts = [(name, len(values)) for name, values in unique_hyper_params.items()]
        ordered_param_counts = sorted(unique_param_counts, key=lambda tup: tup[1], reverse=True)
        ordered_param_names = [name for name, count in ordered_param_counts]
        return ordered_param_names

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
