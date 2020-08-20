from typing import Tuple, Union
import os.path as osp

from dataset_utils import get_cifar_data_loaders, AdversarialTargetReplacementParams
from mini_mobilenet import get_mini_mobilenet_v2, print_mini_mobilenet_shapes
from mini_vgg16 import get_mini_vgg16, print_mini_vgg_shapes
from training import fit_classifier
from utils import save_model_metrics, save_model_weights
from model_comparison import compare_models


def train_and_eval_model(model_name: str,
                         epochs: int,
                         random_train_targets_fraction: float,
                         adversarial_target_replacement_params: Union[AdversarialTargetReplacementParams, None],
                         net_architecture: str,
                         models_dir: str = "models",
                         data_root: str = "data",
                         classes_to_keep: Tuple[str, ...] = (
                                 "airplane", "automobile", "horse", "ship"),
                         save_trained_weights: bool = False,
                         fraction_to_keep_train=1.,
                         fraction_to_keep_test=1.
                         ):
    print()
    print('=======================================')
    print(model_name)
    print('=======================================')

    trainloader, testloader = get_cifar_data_loaders(
        data_root=data_root,
        classes_to_keep=classes_to_keep,
        random_train_targets_fraction=random_train_targets_fraction,
        fraction_to_keep_train=fraction_to_keep_train,
        fraction_to_keep_test=fraction_to_keep_test,
        adversarial_target_replacement_params=adversarial_target_replacement_params,
        batch_size=4)

    num_classes = len(trainloader.dataset.classes)
    net = _get_mini_net(net_architecture, num_classes)

    final_model_metrics, training_metrics = fit_classifier(net, trainloader, testloader, epochs)

    save_model_metrics(models_dir, model_name, final_model_metrics, training_metrics)
    if save_trained_weights:
        save_model_weights(models_dir, model_name, net)

    compare_models(models_dir=models_dir,
                   hyper_param_names_to_compare=["model_name"])


def _get_mini_net(net_architecture: str,
                  num_classes: int):
    if net_architecture.lower() == "mobilenet_v2":
        net = get_mini_mobilenet_v2(num_classes=num_classes,
                                    remove_regularization_layers=True)
        print_mini_mobilenet_shapes(net)
        return net
    elif net_architecture == "vgg16":
        net = get_mini_vgg16(num_classes=num_classes,
                             remove_regularization_layers=True)
        print_mini_vgg_shapes(net)
        return net
    else:
        raise ValueError("Invalid net architecture")


def perform_experiments():
    fraction_to_keep_train = 1.

    # for net_architecture, epochs in [("vgg16", 10), ("mobilenet_v2", 20)]:
    for net_architecture, epochs in [("mobilenet_v2", 10)]:
        models_dir = osp.join("models", net_architecture)

        adversarial_params_2_classes = AdversarialTargetReplacementParams(
            replace_from=["horse", "ship"],
            replace_to=["ship", "horse"],
            fraction_to_replace=[0.5, 0.5])
        train_and_eval_model(model_name="adversarial_targets_2_classes",
                             epochs=epochs,
                             random_train_targets_fraction=0.,
                             adversarial_target_replacement_params=adversarial_params_2_classes,
                             net_architecture=net_architecture,
                             models_dir=models_dir,
                             fraction_to_keep_train=fraction_to_keep_train)

        adversarial_params_4_classes = AdversarialTargetReplacementParams(
            replace_from=["horse", "ship", "automobile", "airplane"],
            replace_to=["ship", "horse", "airplane", "automobile"],
            fraction_to_replace=[0.5, 0.5, 0.5, 0.5])
        train_and_eval_model(model_name="adversarial_targets_4_classes",
                             epochs=epochs,
                             random_train_targets_fraction=0.,
                             adversarial_target_replacement_params=adversarial_params_4_classes,
                             net_architecture=net_architecture,
                             models_dir=models_dir,
                             fraction_to_keep_train=fraction_to_keep_train)

        train_and_eval_model(model_name="half_random_targets",
                             epochs=epochs,
                             random_train_targets_fraction=0.5,
                             adversarial_target_replacement_params=None,
                             net_architecture=net_architecture,
                             models_dir=models_dir,
                             fraction_to_keep_train=fraction_to_keep_train)

        train_and_eval_model(model_name="quarter_random_targets",
                             epochs=epochs,
                             random_train_targets_fraction=0.25,
                             adversarial_target_replacement_params=None,
                             net_architecture=net_architecture,
                             models_dir=models_dir,
                             fraction_to_keep_train=fraction_to_keep_train)

        train_and_eval_model(model_name="all_random_targets",
                             epochs=epochs,
                             random_train_targets_fraction=1.,
                             adversarial_target_replacement_params=None,
                             net_architecture=net_architecture,
                             models_dir=models_dir,
                             fraction_to_keep_train=fraction_to_keep_train)

        train_and_eval_model(model_name="standard",
                             epochs=epochs,
                             random_train_targets_fraction=0.,
                             adversarial_target_replacement_params=None,
                             net_architecture=net_architecture,
                             models_dir=models_dir,
                             fraction_to_keep_train=fraction_to_keep_train)

    print("Done")


if __name__ == '__main__':
    perform_experiments()
