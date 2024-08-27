import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures


def main():
    type_of_model = input('Type Of Model: ')
    power_of_features = eval(input('Power Of Features : '))
    input_means = input('Input manually or through csv_file: ')
    input_features = re.findall(r'\b\w+\b', input('Input name of features (follow format x y z): '))
    input_label = input('Label (Y-axis) name: ')

    if input_means.lower() == 'manually':
        x_train, y_train = manual_input(type_of_model)
    else:
        file_path = input('File Path Name: ')
        dataset = pd.read_csv(file_path)
        x_train = dataset.iloc[:, :-1].values
        y_train = dataset.iloc[:, -1].values

    ml = Machine_Learning_Model(type_of_model, power_of_features, input_means, x_train, y_train, input_features,
                                input_label)
    ml.run()


def manual_input(type_of_model):
    x_input = input('X axis (Provide values as a list [x0, x1, ..., xn]): ')
    y_input = input('Y axis (Provide polynomial like ): ')
    x_train = np.array(eval(x_input))

    if type_of_model.lower() == 'linear':
        y_train = np.array(eval(y_input))
    else:
        y_train = np.array(re.findall(r'\b\w+\b', y_input))

    return x_train, y_train


# the class that contains every function contribute to machine learning model
class Machine_Learning_Model:

    def __init__(self, type_of_Model, power_of_features, input_means, x_train, y_train, input_features, input_label):
        self.type_of_Model = type_of_Model
        self.power_of_features = power_of_features
        self.input_means = input_means
        self.x_train = x_train
        self.y_train = y_train
        self.input_features = input_features
        self.input_label = input_label

    # Will initiate the model caller
    def run(self):
        if self.type_of_Model.lower() == 'linear':
            self.linear_regression()
        else:
            self.logistic_regression()

    # Code for implementing linear_regression
    def linear_regression(self):
        model, y_pred = self.train_linear_model(self.x_train)
        self.plot_function(model, y_pred)
        if len(self.input_features) == 1:
            self.predict_function(model, np.array(5).reshape(-1, 1))
        else:
            self.predict_function(model, np.array([[200, 4, 4]]).reshape(1, -1))

    def predict_function(self, model, value_wanna_predict):
        print(model.predict(value_wanna_predict))

    def train_linear_model(self, x_train):
        model = make_pipeline(PolynomialFeatures(degree=self.power_of_features, include_bias=False),
                              StandardScaler(), LinearRegression())
        if len(self.input_features) == 1:
            x_train = x_train.reshape(-1, 1)
        model.fit(x_train, self.y_train)
        y_pred = model.predict(x_train)
        return model, y_pred

    def plot_function(self, model, y_pred):
        fig, ax = plt.subplots(1, len(self.input_features), figsize=(12, 12), sharey=True)
        # For single_variable plot
        if len(self.input_features) == 1:
            x_range = np.linspace(self.x_train.min(), self.x_train.max(), 100).reshape(-1, 1)
            y_range_pred = model.predict(x_range)
            ax.scatter(self.x_train, self.y_train, label='Actual Value')
            ax.scatter(self.x_train, y_pred, color='yellow', label='Predicted')
            ax.set_xlabel(self.input_features[0])
            ax.set_ylabel(self.input_label)
            ax.plot(x_range, y_range_pred, color='green', label='Regression Line/Curve')
            plt.legend()
            plt.show()
        # For multiple_variables plot
        else:
            x_range = np.zeros((100, len(self.input_features)))
            for i in range(len(self.input_features)):
                x_range[:, i] = np.linspace(self.x_train[:, i].min(), self.x_train[:, i].max(), 100)
            y_range_pred = model.predict(x_range)
            for j in range(len(self.input_features)):
                ax[j].scatter(self.x_train[:, j], self.y_train, label='Actual Value')
                ax[j].scatter(self.x_train[:, j], y_pred, color='green', label='Predicted')
                ax[j].set_xlabel(self.input_features[j])
                ax[j].set_ylabel(self.input_label)
                ax[j].plot(x_range[:, j].reshape(-1, 1), y_range_pred, color='red')
                ax[j].legend()
            plt.show()

    # code for implementing logistic_regression
    def logistic_regression(self):
        model, pos, neg, label_encoder = self.train_linear_logistic_model()
        if len(self.input_features) == 1:
            self.single_variable_plot(model, pos, neg, label_encoder)
            self.predict_logistic_function(model, np.array([11]).reshape(-1, 1))
        else:
            self.mutiple_variables_plot(model, pos, neg, label_encoder)
            self.predict_logistic_function(model, np.array([17, 17]).reshape(1, -1))

    def predict_logistic_function(self, model, value_wanna_predict):
        print(model.predict(value_wanna_predict))

    def train_linear_logistic_model(self):
        label_encoder = LabelEncoder()
        self.y_train = label_encoder.fit_transform(self.y_train)
        pos = self.y_train == 1
        neg = self.y_train == 0

        model = make_pipeline(PolynomialFeatures(degree=self.power_of_features), StandardScaler(),
                              LogisticRegression())
        if len(self.input_features) == 1:
            self.x_train = self.x_train.reshape(-1, 1)
        model.fit(self.x_train, self.y_train)
        return model, pos, neg, label_encoder

    def single_variable_plot(self, model, pos, neg, label_encoder):
        # Plotting
        x_min, x_max = self.x_train.min() - 1, self.x_train.max() + 1
        x_values = np.linspace(x_min, x_max, 100).reshape(-1, 1)

        # Generate the mesh grid
        y_values = model.predict_proba(x_values)[:, 1]

        fig, ax = plt.subplots(1, 1, figsize=(5, 6))
        ax.scatter(self.x_train[pos], self.y_train[pos], marker='X', color='red',
                   label=f'y=1 ({label_encoder.inverse_transform([1])[0]})')
        ax.scatter(self.x_train[neg], self.y_train[neg], marker='o', color='blue',
                   label=f'y=0 ({label_encoder.inverse_transform([0])[0]})')
        h_line_y = 0.5
        ax.plot(x_values, y_values, color='green', label=f'y=1 ({label_encoder.inverse_transform([1])[0]})')
        ax.axhline(y=0.5, color='red', linewidth=2, label='Threshold')

        ax.set_ylim(0, 1)
        ax.set_xlabel(self.input_features[0])
        ax.set_ylabel(self.input_label)
        ax.legend()
        plt.show()

    def mutiple_variables_plot(self, model, pos, neg, label_encoder):
        # Encode target labels
        x_min, x_max = self.x_train[:, 0].min() - 1, self.x_train[:, 0].max() + 1
        y_min, y_max = self.x_train[:, 1].min() - 1, self.x_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        fig, ax = plt.subplots(1, 1, figsize=(5, 6))
        # ax.scatter(self.x_train[:, 0], self.x_train[:, 1], c=self.y_train, edgecolors='k')
        scatter1 = ax.scatter(self.x_train[pos, 0], self.x_train[pos, 1], marker='X', color='red',
                              label=f'({label_encoder.inverse_transform([1])[0]})')
        scatter2 = ax.scatter(self.x_train[neg, 0], self.x_train[neg, 1], marker='o', color='blue',
                              label=f'({label_encoder.inverse_transform([0])[0]})')
        # Plotting the decision boundary line using ax.plot
        contour = ax.contourf(xx, yy, Z, alpha=0.5)
        handles = [
            Patch(color='yellow', label=f'Predicted: {label_encoder.inverse_transform([1])[0]}'),
            Patch(color='purple', label=f'Predicted: {label_encoder.inverse_transform([0])[0]}')
        ]
        ax.set_xlabel(self.input_features[0])
        ax.set_ylabel(self.input_features[1])
        ax.legend(handles=handles + [scatter1, scatter2], loc='best')
        plt.show()


main()
