from typing import Tuple, Dict, Union

from dataset_utils import get_cifar_data_loaders, AdversarialTargetReplacementParams
from mini_mobilenet import get_mini_mobilenet, print_mini_mobilenet_shapes
from training import fit_classifier
from utils import save_model
from model_comparison import compare_models


def train_and_eval_mobilenet(model_name: str,
                             epochs: int,
                             random_train_targets_fraction: float,
                             adversarial_target_replacement_params: Union[AdversarialTargetReplacementParams, None],
                             models_dir: str = "models",
                             data_root: str = "data",
                             classes_to_keep: Tuple[str, ...] = (
                                     "airplane", "automobile", "horse", "ship")):
    fraction_to_keep_train = 1
    fraction_to_keep_test = 1

    print(model_name)
    print('=======================================')

    trainloader, testloader = get_cifar_data_loaders(
        data_root=data_root,
        classes_to_keep=classes_to_keep,
        random_train_targets_fraction=random_train_targets_fraction,
        fraction_to_keep_train=fraction_to_keep_train,
        fraction_to_keep_test=fraction_to_keep_test,
        adversarial_target_replacement_params=adversarial_target_replacement_params)

    num_classes = len(trainloader.dataset.classes)
    net = get_mini_mobilenet(num_classes=num_classes,
                             remove_batchnorm_layers=True)
    print_mini_mobilenet_shapes(net)

    final_model_metrics, training_metrics = fit_classifier(net, trainloader, testloader, epochs)

    save_model(models_dir, model_name, net, final_model_metrics, training_metrics)
    compare_models(models_dir=models_dir,
                   hyper_param_names_to_compare=["model_name"])


def perform_experiments(epochs=10):
    adversarial_target_replacement_params = AdversarialTargetReplacementParams(
        replace_from=["horse", "ship"],
        replace_to=["ship", "horse"],
        fraction_to_replace=[0.5, 0.5]
    )
    train_and_eval_mobilenet(model_name="adversarial_targets",
                             epochs=epochs,
                             random_train_targets_fraction=0.,
                             adversarial_target_replacement_params=adversarial_target_replacement_params)
    train_and_eval_mobilenet(model_name="half_random_targets",
                             epochs=epochs,
                             random_train_targets_fraction=0.5,
                             adversarial_target_replacement_params=None)
    train_and_eval_mobilenet(model_name="all_random_targets",
                             epochs=epochs,
                             random_train_targets_fraction=1.,
                             adversarial_target_replacement_params=None)
    train_and_eval_mobilenet(model_name="standard",
                             epochs=epochs,
                             random_train_targets_fraction=0.,
                             adversarial_target_replacement_params=None)


if __name__ == '__main__':
    perform_experiments(epochs=10)
