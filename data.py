import numpy as np
import csv
import itertools
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures


class Data:
    def __init__(self, files, rotates=None, recurrent=False, one_hot=False, mapa=None):
        self.files = files
        self.rotates = rotates
        # print(self.files)
        self.rec = recurrent
        self.one_hot = one_hot
        self.inputs = None
        self.outputs = None
        self.classes = None
        self.mapa = mapa
        self.type = np.int16
        self._load()

    def _load(self):
        dataset = dict()

        for index, file in enumerate(self.files):
            with open(file, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    if not row:
                        continue
                    if row[-1] not in dataset:
                        dataset[row[-1]] = list()
                    if self.rotates is None:
                        dataset[row[-1]].append(row[:-1])
                    else:
                        dataset[row[-1]].append(np.roll(np.array(row[:-1]), shift=self.rotates[index]))

        inputs = list()
        outputs = list()

        cond = lambda cl: cl in ['neutral', 'fist', 'fingerpoint']

        if self.mapa is None:
            self.mapa = dict()

            index = 0

            for cls in dataset:
                if not cond(cls):
                    continue
                inputs.extend(dataset[cls])
                outputs.extend(list(itertools.repeat(index, len(dataset[cls]))))
                self.mapa[cls] = index
                # print('train cls {}: {}'.format(cls, index))
                self.mapa[index] = cls
                index += 1
        else:
            for cls in dataset:
                if not cond(cls):
                    continue
                # if cls == 'tripodpinch':
                #     continue
                inputs.extend(dataset[cls])
                # print('val cls {}: {}'.format(cls, self.mapa[cls]))
                outputs.extend(list(itertools.repeat(self.mapa[cls], len(dataset[cls]))))

        inputs = np.array(inputs, dtype=self.type)
        outputs = np.array(outputs, dtype=self.type)

        self.classes = len(dataset.keys())

        if self.one_hot:
            outputs_ = np.zeros((outputs.shape[0], 10))
            outputs_[np.arange(0, outputs.shape[0]), outputs] = 1
        else:
            # outputs_ = outputs.reshape(outputs.shape[0], 1)
            outputs_ = outputs

        self.inputs, self.outputs = shuffle(inputs, outputs_)

        if self.rec:
            self.inputs = self.inputs.reshape(self.inputs.shape[0], self.inputs.shape[1], 1)

    def split(self):
        self.inputs, self.outputs = shuffle(self.inputs, self.outputs)
        return self.inputs, self.outputs

    def add_poly_features(self):
        poly = PolynomialFeatures(2, interaction_only=True)
        print(self.inputs.shape)
        self.inputs = poly.fit_transform(self.inputs)
        print(self.inputs.shape)

    def split_to_train_and_val(self):
        self.inputs, self.outputs = shuffle(self.inputs, self.outputs)

        split_index = int(self.inputs.shape[0] * 0.3)
        train_x, validation_x = self.inputs[:-split_index], self.inputs[-split_index:]
        train_y, validation_y = self.outputs[:-split_index], self.outputs[-split_index:]

        return train_x, train_y, validation_x, validation_y

    def split_data(self):
        split_index = int(self.inputs.shape[0] * 0.3)
        train_x = self.inputs[:-split_index]
        test_x = self.inputs[-split_index:]
        train_x, validation_x = train_x[:-split_index], train_x[-split_index:]

        train_y = self.outputs[:-split_index]
        test_y = self.outputs[-split_index:]
        train_y, validation_y = train_y[:-split_index], train_y[-split_index:]

        return train_x, train_y, validation_x, validation_y, test_x, test_y
