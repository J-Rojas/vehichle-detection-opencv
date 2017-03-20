import numpy as np
import sklearn.linear_model as skmodel
import sklearn.model_selection as skselect
import sklearn.svm as sksvm
import sklearn.preprocessing as skprocess
from sklearn.externals import joblib

import time
import os

class Classifier:

    def __init__(self, alpha=0.0001, epochs=10):
        self.scaler = skprocess.StandardScaler()
        if True:
            self.model = skmodel.SGDClassifier(
                learning_rate='optimal',
                loss='hinge',
                penalty='l2',
                alpha=alpha,
                shuffle=True,
                verbose=1,
                n_jobs=-1,
                n_iter=epochs
            )
        else:
            self.model = sksvm.SVC(C=1.0, verbose=1, class_weight='balanced', max_iter=epochs, kernel='rbf')

    def trainAndValidate(self, train, test):

        X_train, y_train = train
        X_test, y_test = test

        # Check the training time for model
        t=time.time()
        self.model.partial_fit(X_train, y_train, [0, 1])
        t2 = time.time()

        if (len(X_test) > 0 and len(y_test) > 0):
            results = self.test(X_test, y_test)
            test_time = results['time']
        else:
            results = {}
            test_time = 0

        results['time'] = t2 - t
        results['test_time'] = test_time

        return results

    def test(self, X, y):

        t=time.time()
        accuracy = self.model.score(X, y)
        t2=time.time()

        # Confidence scores
        y_scores = self.model.decision_function(X)

        return {'time': t2-t, 'accuracy': accuracy, 'scores': y_scores }


    def predict(self, X):

        t=time.time()
        predictions = self.model.predict(X)
        y_scores = self.model.decision_function(X)
        t2 = time.time()

        return {'time': t2-t, 'labels': predictions, 'scores': y_scores }

    def save(self, filename):
        joblib.dump(self.model, filename + '.clf.pkl')
        joblib.dump(self.scaler, filename + '.scaler.pkl')

    def load(self, filename):
        self.model = joblib.load(filename + '.clf.pkl')
        self.scaler = joblib.load(filename + '.scaler.pkl')

    def transform(self, features, standardize='fit'):
        if standardize == 'fit':
            self.scaler.fit(features)
        return self.scaler.transform(features)

class Trainer:

    def __init__(self, featureExtractor, classifier):
        self.fEx = featureExtractor
        self.clf = classifier

    def prepare(self, data=None, classes=None, dump=None, load=None, augment={}):

        features, labels = None, None
        positive, negative = None, None

        if data is None and classes is None and load and os.path.isfile(load):
            features, labels = joblib.load(load)
        else:

            if classes:
                positive = classes[0]
                negative = classes[1]
            elif data:
                positive = data

            flip = augment.get('flip', False)

            positive_features = self.fEx.extract(positive, flip) if positive is not None and len(positive) > 0 else None
            negative_features = self.fEx.extract(negative, flip) if negative is not None and len(negative) > 0 else None

            if positive_features is not None and negative_features is not None:
                features = np.concatenate((positive_features, negative_features))
                labels = np.concatenate((np.ones(len(positive_features)), np.zeros(len(negative_features))))
            else:
                features = positive_features if positive_features is not None else negative_features
                labels = np.ones(len(positive_features)) if positive_features is not None else np.zeros(len(negative_features))

            if data is None:

                pos_len = len(positive) if positive is not None else 0
                neg_len = len(negative) if negative is not None else 0
                print('Loaded ', pos_len, ' positive examples', ', total with augmentation ', pos_len * (2 if flip else 1))
                print('Loaded ', neg_len, ' negative examples', ', total with augmentation ', neg_len * (2 if flip else 1))

            if dump:
                joblib.dump((features, labels), dump)

        return features, labels

    def split(self, X, y, test_ratio=0.2):
        # Split up data into stratified randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = skselect.train_test_split(
            X, y,
            test_size=test_ratio,
            random_state=rand_state,
            stratify=y
        )

        pos_train = len(np.where(y_train == 1.)[0])
        neg_train = len(np.where(y_train == 0.)[0])
        pos_test = len(np.where(y_test == 1.)[0])
        neg_test = len(np.where(y_test == 0.)[0])

        print('Split train/test, train ', len(X_train), ', test ', len(X_test))
        print('   positive: train ', pos_train, ', test ', pos_test)
        print('   negative: train ', neg_train, ', test ', neg_test)

        return (X_train, y_train), (X_test, y_test)

    def train(self, groups, verbose=True):

        train_set, test_set = groups

        train_set = (self.clf.transform(train_set[0], standardize='fit'), train_set[1])
        if (len(test_set[0]) > 0):
            test_set = (self.clf.transform(test_set[0], standardize='transform'), test_set[1])

        results = self.clf.trainAndValidate(train_set, test_set)

        if verbose:
            print('Training time {}, accuracy {:.3f}'.format(results['time'],results['accuracy']))

        return results

    def test(self, X, y):

        X_test, y_test = self.clf.transform(X, standardize='transform'), y

        results = self.clf.test(X_test, y_test)

        print('Test time {}, accuracy {:.3f}'.format(results['time'],results['accuracy']))

        return results

    def predict(self, X):

        X_test = self.clf.transform(X, standardize='transform')

        results = self.clf.predict(X_test)

        return results
