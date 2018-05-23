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
        train_x = self.get_training_set()[0]
        train_y = self.get_training_set()[1]
        test = self.get_test_set()
        x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3)
        predictions = self.trainer(train_x, train_y, test, x_test)
        predictions_1 = list(predictions[0])
        predictions_2 = list(predictions[1])
        preds = []
        for i in predictions_1:
            preds.append(round(i))
        accuracy = accuracy_score(y_test, preds)
        mean_abs_err = mean_absolute_error(y_test, predictions_1)
        print accuracy
        print mean_abs_err
        print mean_absolute_error(y_test, preds)
        preds_2 = []
        for i in predictions_2:
            preds_2.append(int(round(i)))
        with open('Results/results.csv', 'wb') as results_file:
            results = csv.writer(results_file, delimiter=',')
            results.writerow(["city", "year", "weekofyear", "total_cases"])
            for i in range(0, len(self.test) - 1):
                results.writerow([self.test[i + 1][0], self.test[i + 1][1], self.test[i + 1][2]] + [preds_2[i]])


    def get_training_set(self):

        training_x = self.dataset_x[1:]
        training_y = self.dataset_y[1:]

        x_train_sj = []
        y_train_sj = []
        x_train_iq = []
        y_train_iq = []

        for i in range(0, len(training_x)):
            if '' not in training_x[i]:
                row_x = []
                row_x.append(int(training_x[i][1]))
                row_x.append(int(training_x[i][2]))
                for element in training_x[i][4:]:
                    row_x.append(float(element))

                if training_x[i][0] == 'sj':
                    x_train_sj.append(row_x)
                    y_train_sj.append(int(training_y[i][3]))
                else:
                    x_train_iq.append(row_x)
                    y_train_iq.append(int(training_y[i][3]))

        return x_train_sj, y_train_sj, x_train_iq, y_train_iq

    def trainer(self, x_train, y_train, test, x_test):

        clf = linear_model.LinearRegression()
        clf.fit(x_train, y_train)
        predictions_1 = clf.predict(x_test)
        predictions_2 = clf.predict(test)
        filename = 'sj.sav'
        joblib.dump(clf, "ModelData/" + filename)
        return predictions_1, predictions_2

    def get_test_set(self):
        test = []
        test_set = self.test[1:]
        for row in test_set:
            if '' not in row:
                new_row = []
                new_row.append(0 if row[0] == "sj" else 1)
                new_row.append(int(row[1]))
                new_row.append(int(row[2]))
                for element in row[4:]:
                    new_row.append(float(element))
                test.append(new_row)

            else:
                for i in range(0, len(row)):
                    if row[i] == '':
                        row[i] = 0
                new_row = []
                new_row.append(0 if row[0] == "sj" else 1)
                new_row.append(int(row[1]))
                new_row.append(int(row[2]))
                for element in row[4:]:
                    new_row.append(float(element))
                test.append(new_row)

        return test

    def accuracy_checker(self):
        filename = 'sj.sav'
        clf = joblib.load("ModelData/" + filename)

    def get_missing_value_average(self, attr_no, raw_no):






Main().processor()
