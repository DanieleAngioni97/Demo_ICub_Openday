from secml.ml.classifiers.sklearn.c_classifier_svm import CClassifierSVM
from secml.array import CArray
from secml.ml.classifiers.clf_utils import convert_binary_labels
from sklearn.svm import SVC, LinearSVC

# LinearSVC(C=self.C, class_weight=self.class_weight)

def _fit_one_ova(tr_class_idx, svm, x, y, svc_kernel, verbose):
    """Fit a OVA classifier.

    Parameters
    ----------
    tr_class_idx : int
        Index of the label against which the classifier should be trained.
    svm : CClassifierSVM
        Instance of the multiclass SVM classifier.
    x : CArray
        Array to be used for training with shape (n_samples, n_features).
    y : CArray
        Array of shape (n_samples,) containing the class labels.
    verbose : int
        Verbosity level of the logger.

    """
    # Resetting verbosity level. This is needed as objects
    # change id  when passed to subprocesses and our logging
    # level is stored per-object looking to id
    svm.verbose = verbose

    svm.logger.info(
        "Training against class: {:}".format(tr_class_idx))

    # Binarize labels
    y_ova = CArray(y == svm.classes[tr_class_idx])

    # Training the one-vs-all classifier
    if svc_kernel == "linear":
        svc = LinearSVC(C=svm.C, class_weight=svm.class_weight)
    else:
        svc = SVC(C=svm.C, kernel=svc_kernel, class_weight=svm.class_weight)
    svc.fit(x.get_data(), y_ova.get_data())

    # Assign output based on kernel type
    w = CArray(svc.coef_.ravel()) if svm.kernel is None else None
    sv_idx = CArray(svc.support_).ravel() if svm.kernel is not None else None
    alpha = CArray(svc.dual_coef_) if svm.kernel is not None else None

    # Intercept is always available
    b = CArray(svc.intercept_[0])[0]

    return w, sv_idx, alpha, b


class CClassifierSVM_(CClassifierSVM):

    def _fit_binary(self, x, y, svc_kernel):
        if svc_kernel == "linear":
            svc = LinearSVC(C=self.C, class_weight=self.class_weight)
        else:
            svc = SVC(C=self.C, kernel=svc_kernel,
                      class_weight=self.class_weight)
        if svc_kernel == 'precomputed':
            # training on sparse precomputed kernels is not supported
            svc.fit(x.tondarray(), y.get_data())
        else:
            svc.fit(x.get_data(), y.get_data())
        if self.kernel is None:
            self._w = CArray(svc.coef_)
        else:
            sv_idx = CArray(svc.support_).ravel()
            self._alpha[sv_idx] = CArray(svc.dual_coef_)
        self._b = CArray(svc.intercept_[0])[0]
