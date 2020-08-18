from dataset_utils import get_cifar_data_loaders
from mini_mobilenet import get_mini_mobilenet, print_mini_mobilenet_shapes
from training import fit_classifier
from utils import save_model
from model_comparison import compare_models


def main(epochs=10):
    model_name = "standard"
    models_dir = "models"
    hyper_param_names_for_label = ["model_name"]

    classes_to_keep = ["airplane", "automobile", "horse", "ship"]
    trainloader, testloader = get_cifar_data_loaders(data_root="./data",
                                                     classes_to_keep=classes_to_keep,
                                                     fraction_to_keep_train=0.01)

    num_classes = len(trainloader.dataset.classes)
    net = get_mini_mobilenet(num_classes=num_classes)
    print_mini_mobilenet_shapes(net)

    final_model_metrics, training_metrics = fit_classifier(net, trainloader, testloader, epochs)

    save_model(models_dir, model_name, net, final_model_metrics, training_metrics)
    compare_models(models_dir=models_dir,
                   hyper_param_names_to_compare=hyper_param_names_for_label)


if __name__ == '__main__':
    main(epochs=10)
