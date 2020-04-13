import os.path as osp
from cifar_loader import prepare_cifar_data_for_vector_classifier
from training import train_and_eval_fully_connected_model
from sklearn.decomposition import PCA


def grid_search():
    save_dir = osp.join("models", "fully_connected")
    dataset_dir = "cifar-10-batches-py"
    cache_dir = "data_cache"

    subsample_fraction = 0.1
    grid_init_gaussian_std = [1e-1, 1e-2, 1e-3]
    grid_learning_rate = [1e-2, 1e-3, 1e-4]
    grid_momentum = [0., 0.5, 0.9]

    X_train, y_train, X_test, y_test, class_names = (
        prepare_cifar_data_for_vector_classifier(dataset_dir,
                                                 cache_dir,
                                                 subsample_fraction))

    num_models = len(grid_init_gaussian_std) * len(grid_learning_rate) * len(grid_momentum)
    i_model = 1

    for init_gaussian_std in grid_init_gaussian_std:
        for learning_rate in grid_learning_rate:
            for momentum in grid_momentum:
                print(f"\nmodel {i_model}/{num_models}")
                print(f"init_gaussian_std: {init_gaussian_std}  learning_rate: {learning_rate}"
                      f"  momentum: {momentum}")
                model_name = f"fully_connected_{i_model}"
                train_and_eval_fully_connected_model(X_train, y_train, X_test, y_test, class_names, save_dir,
                                                     model_name, learning_rate=learning_rate, momentum=momentum,
                                                     init_gaussian_std=init_gaussian_std)
                i_model += 1


def compare_inits():
    save_dir = osp.join("models", "fully_connected_compare_inits")
    dataset_dir = "cifar-10-batches-py"
    cache_dir = "data_cache"

    subsample_fraction = 0.1
    init_gaussian_std = 0.001
    learning_rate = 0.001
    momentum = 0.9

    X_train, y_train, X_test, y_test, class_names = (
        prepare_cifar_data_for_vector_classifier(dataset_dir,
                                                 cache_dir,
                                                 subsample_fraction))

    for init_type in ["xavier", "gaussian"]:
        model_name = f"fully_connected_{init_type}"
        train_and_eval_fully_connected_model(X_train, y_train, X_test, y_test, class_names, save_dir, model_name,
                                             learning_rate=learning_rate, momentum=momentum, init_type=init_type,
                                             init_gaussian_std=init_gaussian_std)


def compare_with_and_without_pca():
    save_dir = osp.join("models", "fully_connected_PCA")
    dataset_dir = "cifar-10-batches-py"
    cache_dir = "data_cache"
    subsample_fraction = 0.1

    X_train, y_train, X_test, y_test, class_names = (
        prepare_cifar_data_for_vector_classifier(dataset_dir,
                                                 cache_dir,
                                                 subsample_fraction))

    pca_model = PCA(whiten=True)
    pca_model.fit(X_train)
    X_train_pca = pca_model.transform(X_train)
    X_test_pca = pca_model.transform(X_test)

    for n_components in [5, 10, 20, 100, 3072]:
        model_name = f"pca_whitened_dim_{n_components}"
        train_and_eval_fully_connected_model(X_train_pca[:, :n_components], y_train,
                                             X_test_pca[:, :n_components], y_test,
                                             class_names, save_dir, model_name)

    model_name = "original_data"
    train_and_eval_fully_connected_model(X_train, y_train, X_test, y_test,
                                         class_names, save_dir, model_name)


if __name__ == '__main__':
    compare_with_and_without_pca()
