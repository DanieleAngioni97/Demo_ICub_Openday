from secml.utils import fm
from secml.utils.pickle_utils import *
from torchvision.models import alexnet
from secml.ml.features import CNormalizerMeanStd
from secml.ml.classifiers import CClassifierPyTorch
from secml.ml.features import CNormalizerDNN
from c_classifier_svm_ import CClassifierSVM_
from secml.array import CArray
from secml.data import CDataset
from secml.ml.classifiers.loss import CSoftmax
import numpy as np
import cv2
import time
from threading import Thread
import torch

plot_colors = [(65, 47, 213), (122, 79, 58), (154, 208, 72),
               (71, 196, 249), (224, 176, 96), (22, 118, 238)]
icub7_names = ['tazza', 'detersivo piatti', 'detersivo lavatrice', 'piatto',
               'saponetta', 'spugna', 'spray']


class InputHandler(Thread):

    def __init__(self):
        self.cmd = None
        super(InputHandler, self).__init__()

    def run(self):
        while True:
            cmd = input()
            self.cmd = cmd


class ICubDemoHandler(Thread):

    def __init__(self, clf_rate=5, n_acquisition=30, n_jobs=1):
        self._clf_rate = clf_rate
        self._n_acquisition = n_acquisition
        self._n_jobs = n_jobs
        self._class_names = None
        self._n_classes = None
        self._buffer = None
        self._tr_set = None
        self._dnn = None
        self._svm = None
        self._feature_extractor = None
        self._cap = None
        self._load_tr_set()
        self._load_clf()
        self._classifying = True
        self._txt = None
        super(ICubDemoHandler, self).__init__()

    # def run(self):
    #     self._cap = cv2.VideoCapture(0)
    #     cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    #     while True:
    #         frame = self._cap.read()[1]
    #         cv2.imshow('preview', frame)
    #         # cv2.imwrite("frame.png", frame)
    #         cv2.waitKey()
    #     self._cap.release()
    #     cv2.destroyAllWindows()

    def run(self):
        self._input_thread = InputHandler()
        self._input_thread.start()
        self._cap = cv2.VideoCapture(0)
        st_time = 0
        i = 0
        while True:
            if cv2.waitKey(1) == 27:
                break  # esc to quit
            el_time = time.time() - st_time
            img = self._cap.read()[1]
            if img is None:
                continue
            img = cv2.flip(img, 1)
            self._show_img(img, self._txt)
            if self._input_thread.cmd is not None:
                cmd = self._input_thread.cmd
                self._input_thread.cmd = None
                self._classifying = False
                t = Thread(target=self._handle_command, args=(cmd,))
                t.start()
            if el_time > 1. / self._clf_rate and self._classifying:
                img = self._center_crop(img, 224)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = CArray(img.transpose(2, 0, 1).ravel() / 255.)
                self._classify(img, i)
                st_time = time.time()
                i += 1
        self._cap.release()
        cv2.destroyAllWindows()

    def _handle_command(self, cmd):
        if cmd == 'reset':
            self._txt = None
            self._load_tr_set()
            self._load_clf()
            self._class_names = icub7_names.copy()
            self._n_classes = len(self._class_names)
            self._buffer = CArray.zeros(shape=(self._clf_rate,
                                               self._n_classes))
            self._classifying = True
            return
        elif cmd.startswith('train '):
            class_name = cmd.split('train ')[1]
            self._txt = 'looking {:}'.format(class_name)
            if class_name not in self._class_names:
                self._class_names.append(class_name)
            print("looking {:}".format(class_name))
            X = CArray.zeros(shape=(self._n_acquisition,
                                    self._tr_set.num_features))
            acquired = 0
            while acquired < self._n_acquisition:
                img = self._cap.read()[1]
                if img is None:
                    continue
                img = cv2.flip(img, 1)
                img = self._center_crop(img, 224)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = CArray(img.transpose(2, 0, 1).ravel() / 255.)
                X[acquired, :] = self._feature_extractor.transform(img)
                acquired += 1
            Y = CArray.ones(shape=(self._n_acquisition,), dtype=int) * \
                self._class_names.index(class_name)
            self._tr_set = self._tr_set.append(CDataset(X, Y))
            self._txt = 'training'
        elif cmd.startswith('forget '):
            self._txt = None
            class_name = cmd.split('forget ')[1]
            if class_name in self._class_names:
                tr_set = None
                idx = 0
                for c in self._class_names:
                    if c == class_name:
                        continue
                    tr_set = self._tr_set[self._tr_set.Y == idx, :] if tr_set \
                        is None else tr_set.append(
                        self._tr_set[self._tr_set.Y == idx, :])
                    idx += 1
                self._tr_set = tr_set
                self._class_names.remove(class_name)
        else:
            print("unknown command")
            self._classifying = True
            return
        print("training classifier")
        self._svm.fit(self._tr_set.X, self._tr_set.Y)
        self._dnn._model.classifier[-1] = torch.nn.Linear(
            4096, self._svm.n_classes)
        self._dnn._model.classifier[-1].weight = torch.nn.Parameter(
            torch.from_numpy(self._svm.w.tondarray()).type(
                torch.FloatTensor).to(self._dnn._device))
        self._dnn._model.classifier[-1].bias = torch.nn.Parameter(
            torch.from_numpy(self._svm.b.tondarray()).type(
                torch.FloatTensor).to(self._dnn._device))
        self._dnn._classes = self._svm._classes
        self._n_classes = len(self._class_names)
        self._buffer = CArray.zeros(shape=(self._clf_rate, self._n_classes))
        self._classifying = True
        print("classifier trained")

    def _load_clf(self):
        clf_path = 'linear_svm.gz'
        if fm.file_exist(clf_path):
            print("loading classifier")
            self._svm = load(clf_path)
            self._svm.n_jobs = self._n_jobs
            print("classifier loaded")
        else:
            print("training classifier")
            self._svm = CClassifierSVM_(C=0.1, n_jobs=self._n_jobs)
            self._svm.fit(self._tr_set.X, self._tr_set.Y)
            save(clf_path, self._svm)
            print("classifier trained")
        self._class_names = icub7_names.copy()
        self._n_classes = len(self._class_names)
        self._buffer = CArray.zeros(shape=(self._clf_rate,
                                           self._n_classes))
        model = alexnet(pretrained=True)
        model.classifier = torch.nn.Sequential(
            *model.classifier[:3],
            torch.nn.Linear(4096, self._svm.n_classes))
        normalizer = CNormalizerMeanStd(mean=(0.485, 0.456, 0.406),
                                        std=(0.229, 0.224, 0.225))
        self._dnn = CClassifierPyTorch(
            model=model, input_shape=(3, 224, 224),
            pretrained=True, preprocess=normalizer)
        self._feature_extractor = CNormalizerDNN(
            net=self._dnn, out_layer='classifier:2')
        self._dnn._model.classifier[-1].weight = torch.nn.Parameter(
            torch.from_numpy(self._svm.w.tondarray()).type(
                torch.FloatTensor).to(self._dnn._device))
        self._dnn._model.classifier[-1].bias = torch.nn.Parameter(
            torch.from_numpy(self._svm.b.tondarray()).type(
                torch.FloatTensor).to(self._dnn._device))

    def _load_tr_set(self):
        print("loading training set")
        tr_set_path = fm.join('iCubWorld7_day4_deep_features.gz')
        if fm.file_exist(tr_set_path):
            self._tr_set = load(tr_set_path)
            self._tr_set.header = None
        else:
            raise FileNotFoundError("file {:} does not exist".format(
                tr_set_path))
        print("num training samples: {:}".format(self._tr_set.num_samples))
        print("num training classes: {:}".format(self._tr_set.num_classes))

    def _show_img(self, img, txt=None):
        upp_l, bott_r = self._get_crop_points(img, 224)
        cv2.rectangle(img, upp_l, bott_r, (0, 255, 0), 1)
        if txt:
            cv2.putText(img, text=txt, org=(upp_l[0], upp_l[1] - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                        color=(0, 255, 0), thickness=2)
        cv2.imshow('frame', img)

    def _center_crop(self, img, crop):
        h, w, _ = img.shape
        upp_l, bott_r = self._get_crop_points(img, crop)
        return img[upp_l[1]:bott_r[1], upp_l[0]:bott_r[0]]

    def _get_crop_points(self, img, crop):
        h, w, _ = img.shape
        upp_l = (int((w - crop) / 2.), int((h - crop) / 2.))
        bott_r = (upp_l[0] + crop, upp_l[1] + crop)
        return upp_l, bott_r

    def _classify(self, img, i):
        label, scores = self._dnn.predict(img, return_decision_function=True)
        self._buffer[i % self._buffer.shape[0], :] = scores
        scores = self._buffer.mean(axis=0).ravel()
        self._show_results(label, scores)

    def _show_results(self, label, scores):
        self._txt = self._class_names[label.item()]
        plot_size = 400 - 400 % self._n_classes
        scores = CSoftmax().softmax(scores).ravel()
        scores *= plot_size
        bar_w = plot_size / self._n_classes
        plot = np.zeros((plot_size, plot_size, 3), np.uint8)
        txt = np.zeros((plot_size, plot_size, 3), np.uint8)
        for i in range(self._n_classes):
            cv2.rectangle(plot, (int(i * bar_w),
                                 plot_size - int(scores[i].item())),
                          (int((i + 1) * bar_w), plot_size),
                          plot_colors[i % len(plot_colors)], -1)
            cv2.putText(txt, self._class_names[i],
                        (0, int((i + 1) * bar_w - bar_w / 2)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                        color=(255, 255, 255), thickness=2)
        txt = cv2.rotate(txt, cv2.ROTATE_90_COUNTERCLOCKWISE)
        plot[txt != 0] = txt[txt != 0]
        cv2.imshow('scores', plot)


if __name__ == '__main__':
    # runtime.LockOSThread()
    clf_rate = 5
    n_acquisition = 1
    n_jobs = 1
    ICubDemoHandler(clf_rate, n_acquisition, n_jobs).start()
