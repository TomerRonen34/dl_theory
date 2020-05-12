import os.path as osp
from cifar_loader import prepare_cifar_data_for_classifier, PCA_whitened_images, ZCA_whitened_images
from training import train_and_eval_cnn_model
from model_comparison import compare_models
from cnn import ConvLayerParams
conv1 = ConvLayerParams(num_kernels=64, kernel_size=3, padding_size=0)
conv2 = ConvLayerParams(num_kernels=16, kernel_size=3, padding_size=0)
basic_conv_layers_params = [conv1, conv2]

dataset_dir = "cifar-10-batches-py"
cache_dir = "data_cache"
subsample_fraction = 0.1

def grid_search(epochs=10):
    save_dir = osp.join("models", "cnn", "grid_search")
    hyper_param_names_for_label = ["init_gaussian_std", "learning_rate", "sgd_momentum"]

    grid_init_gaussian_std = [1e-1, 1e-2, 1e-3]
    grid_learning_rate = [1e-2, 1e-3, 1e-4]
    grid_momentum = [0., 0.5, 0.9]

    X_train, y_train, X_test, y_test, class_names = prepare_cifar_data_for_classifier(dataset_dir, cache_dir,
                                                                                      subsample_fraction,
                                                                                      keep_as_image=True)

    num_models = len(grid_init_gaussian_std) * len(grid_learning_rate) * len(grid_momentum)
    init_type = "gaussian"
    i_model = 1

    for init_gaussian_std in grid_init_gaussian_std:
        for learning_rate in grid_learning_rate:
            for sgd_momentum in grid_momentum:
                if i_model==1 or i_model==2 or i_model==3:
                    i_model+=1
                    continue
                print(f"\nmodel {i_model}/{num_models}")
                print(f"init_gaussian_std: {init_gaussian_std}"
                      f"  learning_rate: {learning_rate}"
                      f"  sgd_momentum: {sgd_momentum}")
                model_name = f"cnn_{i_model}"
                train_and_eval_cnn_model(X_train, y_train, X_test, y_test, class_names,
                                         save_dir, model_name,
                                         basic_conv_layers_params,
                                         init_type=init_type,
                                         learning_rate=learning_rate,
                                         sgd_momentum=sgd_momentum,
                                         init_gaussian_std=init_gaussian_std,
                                         epochs=epochs)
                i_model += 1

    compare_models(models_dir=save_dir,
                   hyper_param_names_to_compare=hyper_param_names_for_label)


def optimization(epochs=10):
    save_dir = osp.join("models", "cnn", "optimization")
    hyper_param_names_for_label = ["optimizer_type"]

    dataset_dir = "cifar-10-batches-py"
    cache_dir = "data_cache"
    subsample_fraction = 0.1

    X_train, y_train, X_test, y_test, class_names = (
        prepare_cifar_data_for_classifier(dataset_dir, cache_dir, subsample_fraction, keep_as_image=True))

    init_type = "gaussian"
    init_gaussian_std = 0.1
    learning_rate = 0.0001
    sgd_momentum = 0.9
    for optimizer_type in ["adam", "sgd"]:
        model_name = optimizer_type
        print('\n', model_name)
        train_and_eval_cnn_model(X_train, y_train, X_test, y_test, class_names,
                                 save_dir, model_name,
                                 basic_conv_layers_params,
                                 init_type=init_type,
                                 learning_rate=learning_rate,
                                 sgd_momentum=sgd_momentum,
                                 init_gaussian_std=init_gaussian_std,
                                 epochs=epochs,
                                 optimizer_type=optimizer_type)

    compare_models(models_dir=save_dir,
                   hyper_param_names_to_compare=hyper_param_names_for_label)


def initialization(epochs=10):
    save_dir = osp.join("models", "cnn", "initialization")
    hyper_param_names_for_label = ["init_type"]

    X_train, y_train, X_test, y_test, class_names = prepare_cifar_data_for_classifier(dataset_dir, cache_dir,
                                                                                      subsample_fraction,
                                                                                      keep_as_image=True)

    for init_type in ["xavier", "gaussian"]:
        model_name = f"cnn_{init_type}"
        print('\n', init_type)
        train_and_eval_cnn_model(X_train, y_train, X_test, y_test, class_names, save_dir, model_name,
                                 basic_conv_layers_params,
                                 init_type=init_type,
                                 epochs=epochs)

    compare_models(models_dir=save_dir,
                   hyper_param_names_to_compare=hyper_param_names_for_label)


def regularization(epochs=10):
    save_dir = osp.join("models", "cnn", "regularization")
    hyper_param_names_for_label = ["dropout_drop_probability", "weight_decay"]

    X_train, y_train, X_test, y_test, class_names = prepare_cifar_data_for_classifier(dataset_dir, cache_dir,
                                                                                      subsample_fraction,
                                                                                      keep_as_image=True)

    for weight_decay in [0., 0.1, 0.05, 0.01, 0.001]:
        for dropout_drop_probability in [0., 0.3, 0.5, 0.8]:
            model_name = f"decay_{weight_decay}_dropout_{dropout_drop_probability}"
            print('\n', model_name, '\n', '=' * len(model_name))
            train_and_eval_cnn_model(X_train, y_train, X_test, y_test,
                                     class_names, save_dir, model_name,
                                     basic_conv_layers_params,
                                     weight_decay=weight_decay,
                                     dropout_drop_probability=dropout_drop_probability,
                                     epochs=epochs)

    compare_models(models_dir=save_dir,
                   hyper_param_names_to_compare=hyper_param_names_for_label)

def preprocessing(epochs=100):
    save_dir = osp.join("models", "cnn", "preprocessing")
    hyper_param_names_for_label = ["model_name"]

    X_train, y_train, X_test, y_test, class_names = prepare_cifar_data_for_classifier(dataset_dir, cache_dir,
                                                                                      keep_as_image=False)

    X_train_pca, X_test_pca = PCA_whitened_images(X_train, X_test)

    for n_components in [3072]:
        model_name = f"pca_whitened_dim_{n_components}"
        train_and_eval_cnn_model(X_train_pca, y_train, X_test_pca, y_test,
                                 class_names, save_dir, model_name,
                                 basic_conv_layers_params,
                                 dropout_drop_probability = 0.3,
                                 epochs=epochs)

    # for n_components in [10,20,100,500,1500,3072]:
    #     X_train_zca, X_test_zca = ZCA_whitened_images(X_train, X_test, num_components=n_components)
    #     model_name = f"zca_whitened_dim_{n_components}"
    #     train_and_eval_cnn_model(X_train_zca, y_train, X_test_zca, y_test,
    #                              class_names, save_dir, model_name,
    #                              basic_conv_layers_params,
    #                              dropout_drop_probability=0.3,
    #                              epochs=epochs)

    compare_models(models_dir=save_dir,
                   hyper_param_names_to_compare=hyper_param_names_for_label)


def kernel_size(epochs=100):
    save_dir = osp.join("models", "cnn", "kernel_size")
    hyper_param_names_for_label = ["model_name"]

    X_train, y_train, X_test, y_test, class_names = prepare_cifar_data_for_classifier(dataset_dir, cache_dir,
                                                                                      subsample_fraction,
                                                                                      keep_as_image=True)

    conv1_new = ConvLayerParams(num_kernels=64, kernel_size=5, padding_size=1)
    conv2_new = ConvLayerParams(num_kernels=16, kernel_size=5, padding_size=1)
    conv_layers_params=[conv1_new,conv2_new]
    model_name = f"cnn_kernel5"
    train_and_eval_cnn_model(X_train, y_train, X_test, y_test, class_names, save_dir, model_name,
                                 conv_layers_params,
                                 dropout_drop_probability=0.3,
                                 epochs=epochs)

    compare_models(models_dir=save_dir,
                   hyper_param_names_to_compare=hyper_param_names_for_label)

def width(epochs=100):
    save_dir = osp.join("models", "cnn", "width")
    hyper_param_names_for_label = ["model_name"]

    X_train, y_train, X_test, y_test, class_names = prepare_cifar_data_for_classifier(dataset_dir, cache_dir,
                                                                                      subsample_fraction,
                                                                                      keep_as_image=True)

    conv1_1 = ConvLayerParams(num_kernels=256, kernel_size=3, padding_size=0)
    conv1_2 = ConvLayerParams(num_kernels=64, kernel_size=3, padding_size=0)
    conv_layers_params_1=[conv1_1,conv1_2]
    conv2_1 = ConvLayerParams(num_kernels=512, kernel_size=3, padding_size=0)
    conv2_2 = ConvLayerParams(num_kernels=256, kernel_size=3, padding_size=0)
    conv_layers_params_2 = [conv2_1, conv2_2]

    for conv_layers_params in [conv_layers_params_1, conv_layers_params_2]:
        model_name = f"kernel_width_{conv_layers_params[0].num_kernels}_{conv_layers_params[1].num_kernels}"
        print('\n', model_name, '\n', '=' * len(model_name))
        train_and_eval_cnn_model(X_train, y_train, X_test, y_test, class_names, save_dir, model_name,
                                 conv_layers_params,
                                 dropout_drop_probability=0.3,
                                 epochs=epochs)

    compare_models(models_dir=save_dir,
                   hyper_param_names_to_compare=hyper_param_names_for_label)



def depth(epochs=100):
    save_dir = osp.join("models", "cnn", "depth")
    hyper_param_names_for_label = ["model_name"]

    X_train, y_train, X_test, y_test, class_names = prepare_cifar_data_for_classifier(dataset_dir, cache_dir,
                                                                                      subsample_fraction,
                                                                                      keep_as_image=True)
    additional_conv = ConvLayerParams(num_kernels=64, kernel_size=3, padding_size=1)
    bigger_net = [additional_conv,additional_conv,additional_conv,conv1,conv2]
    for num_cnn_layers in [3,4,5]:
        model_name = f"num_layers_{num_cnn_layers}"
        print('\n', model_name, '\n', '=' * len(model_name))
        cur_cnn = bigger_net[5-num_cnn_layers:]
        train_and_eval_cnn_model(X_train, y_train, X_test, y_test, class_names, save_dir, model_name,
                                 cur_cnn,
                                 dropout_drop_probability=0.3,
                                 epochs=epochs)

    compare_models(models_dir=save_dir,
                   hyper_param_names_to_compare=hyper_param_names_for_label)


def residuals(epochs=100):
    save_dir = osp.join("models", "cnn", "residuals")
    hyper_param_names_for_label = ["model_name"]

    X_train, y_train, X_test, y_test, class_names = prepare_cifar_data_for_classifier(dataset_dir, cache_dir,
                                                                                      subsample_fraction,
                                                                                      keep_as_image=True)
    additional_conv = ConvLayerParams(num_kernels=64, kernel_size=3, padding_size=1)
    bigger_net = [additional_conv,additional_conv,additional_conv,conv1,conv2]
    for num_cnn_layers in [4,5,3]:
        model_name = f"residual_num_layers_{num_cnn_layers}"
        print('\n', model_name, '\n', '=' * len(model_name))
        cur_cnn = bigger_net[5-num_cnn_layers:]
        train_and_eval_cnn_model(X_train, y_train, X_test, y_test, class_names, save_dir, model_name,
                                 cur_cnn,
                                 dropout_drop_probability=0.3,
                                 epochs=epochs,
                                 residual_net=True)

    compare_models(models_dir=save_dir,
                   hyper_param_names_to_compare=hyper_param_names_for_label)

if __name__ == '__main__':
    epochs = 25
    # grid_search(epochs)
    # optimization(epochs)
    # initialization(epochs)
    # regularization(epochs)
    # kernel_size(epochs)
    # width(epochs)
    # depth(epochs)
    # preprocessing(epochs)
    residuals(epochs)
