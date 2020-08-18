from typing import Tuple, Dict, Union
from dataset_utils import get_cifar_data_loaders
from mini_mobilenet import get_mini_mobilenet, print_mini_mobilenet_shapes
from training import fit_classifier
from utils import save_model
from model_comparison import compare_models


def train_and_eval_mobilenet(model_name: str,
                             epochs: int,
                             random_train_targets: bool,
                             specific_adversarial_class_fractions: Union[Dict[str, float], None],
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
        random_train_targets=random_train_targets,
        fraction_to_keep_train=fraction_to_keep_train,
        fraction_to_keep_test=fraction_to_keep_test,
        specific_adversarial_class_fractions=specific_adversarial_class_fractions)

    num_classes = len(trainloader.dataset.classes)
    net = get_mini_mobilenet(num_classes=num_classes,
                             remove_batchnorm_layers=True)
    print_mini_mobilenet_shapes(net)

    final_model_metrics, training_metrics = fit_classifier(net, trainloader, testloader, epochs)

    save_model(models_dir, model_name, net, final_model_metrics, training_metrics)
    compare_models(models_dir=models_dir,
                   hyper_param_names_to_compare=["model_name"])


def perform_experiments(epochs=10):
    train_and_eval_mobilenet(model_name="adversarial_targets",
                             epochs=epochs,
                             random_train_targets=False,
                             specific_adversarial_class_fractions={
                                 "horse": 0.5, "ship": 0.5
                             })
    train_and_eval_mobilenet(model_name="random_targets",
                             epochs=epochs,
                             random_train_targets=True,
                             specific_adversarial_class_fractions=None)
    train_and_eval_mobilenet(model_name="standard",
                             epochs=epochs,
                             random_train_targets=False,
                             specific_adversarial_class_fractions=None)


if __name__ == '__main__':
    perform_experiments(epochs=1)
