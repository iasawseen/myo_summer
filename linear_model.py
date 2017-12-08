from data import Data
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import time
import numpy as np
from sklearn import decomposition
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations


def get_accuracy(y, preds):
    accuracy = y == preds
    return accuracy.sum() / y.shape[0]


def print_statistics(name, accuracies):
    print('{0} avg score: {1:.4f}'.format(name, float(np.average(accuracies))))
    print('{0} std: {1:.4f}'.format(name, np.std(accuracies)))
    print('{0} min: {1:.4f}'.format(name, np.min(accuracies)))
    print('{0} max: {1:.4f}'.format(name, np.max(accuracies)))
    print()


def main(input_file, pca=False):

    print(input_file)

    train_data = Data(input_file, recurrent=False, one_hot=False)
    train_x, train_y, val_x, val_y, test_x, test_y = train_data.split_data()

    if pca:
        pca = decomposition.PCA(n_components=3)
        pca.fit(train_x)
        train_x = pca.transform(train_x)
        val_x = pca.transform(val_x)
        test_x = pca.transform(test_x)

    train_acc = []
    val_acc = []
    test_acc = []

    models = {'logistic': LogisticRegression(),
              'linear reg': LinearRegression(),
              # 'sgd': SGDClassifier(),
              'ridge': RidgeClassifier(),
              'dec tree': DecisionTreeClassifier()}

    # linear = LogisticRegression()
    # linear = LinearRegression()
    # linear = SGDClassifier()
    # linear = RidgeClassifier()
    # linear = RandomForestClassifier(n_estimators=10, criterion='entropy', max_features='auto',
    #                                 max_depth=6, min_samples_split=32, min_samples_leaf=16, n_jobs=16)

    # linear = DecisionTreeClassifier()

    # poly = PolynomialFeatures(2)

    # train_x = poly.fit_transform(train_x)
    # val_x = poly.fit_transform(val_x)
    # test_x = poly.fit_transform(test_x)

    for model in models:
        models[model].fit(train_x, train_y)

        total_time = 0

        for i in range(int(val_x.shape[0] * 0.1)):
            start_time = time.time()
            models[model].predict(val_x[i:])
            total_time += time.time() - start_time

        print('{}, average time: {:.6f} ms'.format(model, total_time / val_x.shape[0] * 1000))

    # train_acc.append(linear.score(train_x, train_y))
    # val_acc.append(linear.score(val_x, val_y))
    # test_acc.append(linear.score(test_x, test_y))
    #
    # train_acc = np.array(train_acc)
    # val_acc = np.array(val_acc)
    # test_acc = np.array(test_acc)
    #
    # # print_statistics('train', train_acc)
    # print_statistics('val', val_acc)
    # print_statistics('test', test_acc)


if __name__ == '__main__':
    begin = time.time()
    files = ['data/1.csv', 'data/2.csv', 'data/3.csv', 'data/4.csv']
    main(files)
    end = time.time()
    print('{0:.4f} sec'.format(end - begin))

