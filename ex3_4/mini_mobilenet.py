import torch
from torch import nn
from torchvision.models import mobilenet_v2, MobileNetV2


def get_mini_mobilenet(input_shape=(3, 32, 32),
                       num_feature_layers=14,
                       num_classes=10
                       ) -> MobileNetV2:
    net = mobilenet_v2(pretrained=False)
    net.features = net.features[:num_feature_layers]
    net.features.add_module("reshape_to_mobilenet_convention",
                            ReshapeFeaturesToMobilenetConvention())

    dummy_inputs = torch.zeros((1,) + input_shape)
    feature_vector_length = net.features(dummy_inputs).numel()
    net.classifier[-1] = nn.Linear(feature_vector_length, num_classes)
    return net


class ReshapeFeaturesToMobilenetConvention(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view((batch_size, -1, 1, 1))


def print_mini_mobilenet_shapes(net: MobileNetV2,
                                input_shape=(3, 32, 32)
                                ) -> None:
    dummy_inputs = torch.zeros((1,) + input_shape)
    print()
    print("input shape:               ", dummy_inputs.shape)
    print("net features shape:        ", net.features[:-1](dummy_inputs).shape)
    print("'flat' net features shape: ", net.features(dummy_inputs).shape)
    print("net outputs shape:         ", net(dummy_inputs).shape)
    print()
