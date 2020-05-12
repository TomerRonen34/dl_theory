import os.path as osp
from cifar_loader import prepare_cifar_data
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from utils import fit_and_save, eval_and_plot


def linear_svm():
    save_dir = osp.join("models", "svm", "linear_svm")
    seed = 34

    dataset_dir = "cifar-10-batches-py"
    cache_dir = "data_cache"
    subsample_fraction = 0.1

    X_train, y_train, X_test, y_test, class_names = \
        prepare_cifar_data(dataset_dir,
                           cache_dir,
                           subsample_fraction,
                           keep_as_image=False)
    model = SVC(kernel="linear", random_state=seed)
    model_name = "linear_svm"
    model_display_name = "Linear SVM"

    fit_and_save(model, X_train, y_train,
                 save_dir, model_name)

    eval_and_plot(model, X_test, y_test,
                  class_names, save_dir, model_name, model_display_name)


def rbf_svm():
    save_dir = osp.join("models", "svm", "rbf_svm")
    seed = 34

    dataset_dir = "cifar-10-batches-py"
    cache_dir = "data_cache"
    subsample_fraction = 0.1

    X_train, y_train, X_test, y_test, class_names = \
        prepare_cifar_data(dataset_dir,
                           cache_dir,
                           subsample_fraction,
                           keep_as_image=False)
    model = SVC(kernel="rbf", random_state=seed)
    model_name = "rbf_svm"
    model_display_name = "SVM with RBF Kernel"

    fit_and_save(model, X_train, y_train,
                 save_dir, model_name)

    eval_and_plot(model, X_test, y_test,
                  class_names, save_dir, model_name, model_display_name)


def rbf_svm_with_pca():
    save_dir = osp.join("models", "svm", "rbf_svm_with_pca")
    seed = 34

    dataset_dir = "cifar-10-batches-py"
    cache_dir = "data_cache"
    subsample_fraction = 0.1

    X_train, y_train, X_test, y_test, class_names = \
        prepare_cifar_data(dataset_dir,
                           cache_dir,
                           subsample_fraction,
                           keep_as_image=False)

    pca_model = PCA(whiten=True)
    pca_model.fit(X_train)
    X_train_pca = pca_model.transform(X_train)
    X_test_pca = pca_model.transform(X_test)

    for n_components in [20, 100]:
        model = SVC(kernel="rbf", random_state=seed)
        model_name = f"rbf_svm_with_pca_dim_{n_components}"
        model_display_name = "SVM with RBF Kernel"

        fit_and_save(model, X_train_pca[:, :n_components], y_train,
                     save_dir, model_name)

        eval_and_plot(model, X_test_pca[:, :n_components], y_test,
                      class_names, save_dir, model_name, model_display_name)


if __name__ == '__main__':
    linear_svm()
    rbf_svm()
    rbf_svm_with_pca()
