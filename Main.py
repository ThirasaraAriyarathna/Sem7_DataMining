import csv
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.externals import joblib

import math

class Main():

    dataset_x = []
    dataset_y = []
    test = []
    training_x = []
    training_y = []
    test_set = []

    def __init__(self):

        with open('Data/dengue_features_train.csv', 'rb') as dataset_x_file:
            dataset_x = csv.reader(dataset_x_file, delimiter=',')
            for row in dataset_x:
                self.dataset_x.append(row)

        with open('Data/dengue_labels_train.csv', 'rb') as dataset_y_file:
            dataset_y = csv.reader(dataset_y_file, delimiter=',')
            for row in dataset_y:
                self.dataset_y.append(row)

        with open('Data/dengue_features_test.csv', 'rb') as test_file:
            test = csv.reader(test_file, delimiter=',')
            for row in test:
                self.test.append(row)

    def processor(self):
        train_set = self.get_training_set()
        train_x_sj = train_set[0]
        train_y_sj = train_set[1]
        train_x_iq = train_set[2]
        train_y_iq = train_set[3]
        test_set = self.get_test_set()
        test_sj = test_set[0]
        test_iq = test_set[1]
        # self.accuracy_checker(train_x_sj, train_y_sj, 'sj')
        # preds = self.predict_values(test_sj, 'sj')
        # self.print_results(preds, 'sj')
        self.accuracy_checker(train_x_iq, train_y_iq, 'iq')
        preds = self.predict_values(test_iq, 'iq')
        self.print_results(preds, 'iq')



    def get_training_set(self):

        self.training_x = self.dataset_x[1:]
        self.training_y = self.dataset_y[1:]

        x_train_sj = []
        y_train_sj = []
        x_train_iq = []
        y_train_iq = []

        for i in range(0, len(self.training_x)):
            row_x = []
            for j in range(4, len(self.training_x[i])):
                if self.training_x[i][j] != '':
                    row_x.append(float(self.training_x[i][j]))
                else:
                    row_x.append(self.get_missing_value_average(j, i, False))

            if self.training_x[i][0] == 'sj':
                x_train_sj.append(row_x)
                y_train_sj.append(int(self.training_y[i][3]))
            else:
                x_train_iq.append(row_x)
                y_train_iq.append(int(self.training_y[i][3]))

        return x_train_sj, y_train_sj, x_train_iq, y_train_iq

    def trainer(self, x_train, y_train, x_test, name):

        clf = linear_model.LinearRegression()
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        filename = name + '.sav'
        joblib.dump(clf, "Models/" + filename)
        return predictions

    def get_test_set(self):
        test_sj = []
        test_iq = []
        self.test_set = self.test[1:]
        for i in range(0, len(self.test_set)):
            row_x = []
            for j in range(4, len(self.test_set[i])):
                if self.test_set[i][j] != '':
                    row_x.append(float(self.test_set[i][j]))
                else:
                    row_x.append(self.get_missing_value_average(j, i, True))

            if self.test_set[i][0] == 'sj':
                test_sj.append(row_x)
            else:
                test_iq.append(row_x)

        return test_sj, test_iq

    def accuracy_checker(self, train_x, train_y, name):

        x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3)
        predictions = self.trainer(train_x, train_y, x_test, name)
        preds = []
        for i in predictions:
            preds.append(round(i))
        mean_abs_err = mean_absolute_error(y_test, preds)
        print name + ': ' + str(mean_abs_err)

    def get_missing_value_average(self, attr_no, raw_no, is_test):
        moving_avg_set = []
        index = raw_no - 1
        while len(moving_avg_set) < 5:
            if index < 0:
                index = attr_no + 1
            if is_test:
                if self.test_set[index][attr_no] != '':
                    moving_avg_set.append(float(self.test_set[index][attr_no]))
            else:
                if self.training_x[index][attr_no] != '':
                    moving_avg_set.append(float(self.training_x[index][attr_no]))
            if index > raw_no:
                index += 1
            else:
                index -= 1
        missing_value = sum(moving_avg_set) / len(moving_avg_set)
        if is_test:
            self.test_set[raw_no][attr_no] = missing_value
        else:
            self.training_x[raw_no][attr_no] = missing_value
        return missing_value

    def predict_values(self, samples, name):

        filename = name + '.sav'
        clf = joblib.load("Models/" + filename)
        predictions = clf.predict(samples)
        preds = []
        for i in predictions:
            preds.append(int(round(i)))
        return preds

    def print_results(self, preds, name):

        filename = 'results_' + name + '.csv'
        with open('Results/' + filename, 'wb') as results_file:
            results = csv.writer(results_file, delimiter=',')
            results.writerow(["city", "year", "weekofyear", "total_cases"])
            if name == 'sj':
                start = 0
                end = sum(x.count('sj') for x in self.test[1:])
            else:
                start = sum(x.count('sj') for x in self.test[1:])
                end = len(self.test) - 1
            for i in range(start, end):
                if self.test[i + 1][0] == name:
                    results.writerow([self.test[i + 1][0], self.test[i + 1][1], self.test[i + 1][2]] + [preds[i - start]])


Main().processor()
