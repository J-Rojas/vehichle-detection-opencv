import numpy as np
import sklearn.linear_model as skmodel
import sklearn.model_selection as skselect
import sklearn.svm as sksvm
import time
import os
from sklearn.externals import joblib

class Classifier:

    def __init__(self, alpha=0.0001, epochs=10):
        if False:
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
        self.model = sksvm.LinearSVC(C=1.0, verbose=1, class_weight='balanced', max_iter=epochs)

    def trainAndValidate(self, train, test):

        X_train, y_train = train
        X_test, y_test = test

        # Check the training time for model
        t=time.time()
        self.model.fit(X_train, y_train)
        t2 = time.time()

        results = self.test(X_test, y_test)
        test_time = results['time']
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
        t2 = time.time()

        return {'time': t2-t, 'labels': predictions }

class Trainer:

    def __init__(self, classes, featureExtractor, classifier, augment=None):
        self.classes = classes
        self.augment = augment
        self.fEx = featureExtractor
        self.clf = classifier
        self.features = None
        self.labels = None

    def prepare(self, dump=None, load=None):

        if load and os.path.isfile(load):
            self.features, self.labels = joblib.load(load)
        else:
            positive = self.classes[0]
            negative = self.classes[1]

            flip = self.augment.get('flip', False)

            positive_features = self.fEx.extract(positive, flip) if len(positive) > 0 else []
            negative_features = self.fEx.extract(negative, flip) if len(negative) > 0 else []

            self.features = np.concatenate((positive_features, negative_features))
            self.labels = np.concatenate((np.ones(len(positive_features)), np.zeros(len(negative_features))))

            print('Loaded ', len(positive), ' positive examples', ', total with augmentation ', len(positive) * (2 if flip else 1))
            print('Loaded ', len(negative), ' negative examples', ', total with augmentation ', len(negative) * (2 if flip else 1))

            if dump:
                joblib.dump((self.features, self.labels), dump)

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

    def train(self, groups=None, test_ratio=0.2):

        train_set, test_set = groups

        if groups is None:
            train_set, test_set = self.split(self.features, self.labels, test_ratio=test_ratio)

        train_set = (self.fEx.transform(train_set[0], standardize='fit'), train_set[1])
        test_set = (self.fEx.transform(test_set[0], standardize='transform'), test_set[1])

        results = self.clf.trainAndValidate(train_set, test_set)

        print('Training time {}, accuracy {:.3f}'.format(results['time'],results['accuracy']))

        return results

    def test(self, X, y):

        X_test, y_test = self.fEx.transform(X, standardize='transform'), y

        print(X_test, y_test)

        results = self.clf.test(X_test, y_test)

        print('Test time {}, accuracy {:.3f}'.format(results['time'],results['accuracy']))
