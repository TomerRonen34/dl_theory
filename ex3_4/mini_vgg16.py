import torch
from torch import nn
from torchvision.models import vgg16, VGG

from utils import remove_regularization_layers_in_place


def get_mini_vgg16(input_shape=(3, 32, 32),
                   num_feature_layers=23,
                   num_classes=10,
                   remove_regularization_layers: bool = False
                   ) -> VGG:
    net = vgg16(pretrained=False)
    net.features = net.features[:num_feature_layers]

    if remove_regularization_layers:
        remove_regularization_layers_in_place(net.features)
        remove_regularization_layers_in_place(net.classifier)

    net.avgpool = nn.Identity()

    dummy_inputs = torch.zeros((1,) + input_shape)
    feature_vector_length = net.features(dummy_inputs).numel()

    out_features_first_fc_layer = net.classifier[0].out_features
    net.classifier[0] = nn.Linear(feature_vector_length, out_features_first_fc_layer)
    in_features_last_fc_layer = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_features_last_fc_layer, num_classes)

    return net


def print_mini_vgg_shapes(net: VGG,
                          input_shape=(3, 32, 32)
                          ) -> None:
    dummy_inputs = torch.zeros((1,) + input_shape)
    print()
    print("VGG shapes:")
    print("input shape:               ", dummy_inputs.shape)
    print("net features shape:        ", net.features(dummy_inputs).shape)
    print("net outputs shape:         ", net(dummy_inputs).shape)
    print()
