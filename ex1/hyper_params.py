from typing import *


class HyperParams:
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
