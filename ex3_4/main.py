from dataset_utils import get_cifar_data_loaders
from mini_mobilenet import get_mini_mobilenet, print_mini_mobilenet_shapes
from training import fit_classifier
from utils import save_model


def main():
    model_name = "standard"
    epochs = 1
    save_dir = f"./models/{model_name}"

    classes_to_keep = ["airplane", "automobile", "horse", "ship"]
    trainloader, testloader = get_cifar_data_loaders(data_root="./data",
                                                     classes_to_keep=classes_to_keep)

    num_classes = len(trainloader.dataset.classes)
    net = get_mini_mobilenet(num_classes=num_classes)
    print_mini_mobilenet_shapes(net)

    final_model_metrics, training_metrics = fit_classifier(net, trainloader, testloader, epochs)
    save_model(save_dir, net, final_model_metrics, training_metrics)


if __name__ == '__main__':
    main()
