import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def main():
    type_of_model = input('Type Of Model: ')
    power_of_features = eval(input('Power Of Features : '))
    input_means = input('Input manually or through csv_file: ')
    input_features = re.findall(r'\b\w+\b', input('Input name of features (follow format x y z): '))
    input_label = input('Label (Y-axis) name: ')

    if input_means.lower() == 'manually':
        x_input = input('X axis (Provide values as a list [x0, x1, ..., xn]): ')
        y_input = input('Y axis (Provide polynomial like "x**2 + x + 1"): ')

        x_train = np.array(eval(x_input))  # Parse x values as a list
        y_train = np.array(eval(y_input))  # Parse y values as a list
    else:
        file_path = input('File Path Name: ')
        dataset = pd.read_csv(file_path)

        x_train = dataset.iloc[:, :-1].values
        y_train = dataset.iloc[:, -1].values

    ml = Machine_Learning_Model(type_of_model, power_of_features, input_means, x_train, y_train, input_features,
                                input_label)
    ml.run()


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
        if len(self.input_features) > 1:
            self.mutiple_variables_linear_regression()
        else:
            self.single_linear_regression()

    def single_linear_regression(self):
        x_poly = (self.x_train ** self.power_of_features).reshape(-1, 1)
        poly = 0
        # Generate polynomial features
        if isinstance(self.power_of_features, int):
            poly = PolynomialFeatures(degree=self.power_of_features, include_bias=False)
            x_poly = poly.fit_transform(self.x_train.reshape(-1, 1))

        # Standardize the polynomial features
        scaler = StandardScaler()
        x_norm = scaler.fit_transform(x_poly)

        # Perform linear regression on polynomial features
        sgdr = LinearRegression()
        sgdr.fit(x_norm, self.y_train)
        b_norm = sgdr.intercept_
        w_norm = sgdr.coef_

        # Compute predictions
        y_pred = sgdr.predict(x_norm)

        # Plotting
        fig, ax = plt.subplots(1, len(self.input_features))
        values = np.zeros(50)
        for i in range(50):
            if isinstance(self.power_of_features, int):
                x_elements = poly.fit_transform(np.array([i]).reshape(-1, 1))
                x_value = scaler.transform(x_elements)
                values[i] = sgdr.predict(x_value)[0]
            else:
                x_value = scaler.transform((np.array([i]) ** self.power_of_features).reshape(-1, 1))
                values[i] = sgdr.predict(x_value)[0]
        ax.scatter(self.x_train, self.y_train, label='Actual Value')
        ax.scatter(self.x_train, y_pred, color='yellow', label='Predicted')
        ax.set_xlabel(self.input_features[0])
        ax.set_ylabel(self.input_label)
        ax.plot(values, color='green')
        plt.legend()
        plt.show()

        # Example prediction for a new data point
        x_test = np.array([20]).reshape(-1, 1)
        prediction = 0
        if isinstance(self.power_of_features, int):
            x_test_poly = poly.fit_transform(x_test)
            x_test_transformed = scaler.transform(x_test_poly)
            prediction = sgdr.predict(x_test_transformed)
        else:
            x_test_transformed = scaler.transform(x_test ** self.power_of_features)
            prediction = sgdr.predict(x_test_transformed)
        print(prediction)

    def mutiple_variables_linear_regression(self):
        x_poly = self.x_train ** self.power_of_features
        poly = 0
        # Generate polynomial features
        if isinstance(self.power_of_features, int):
            poly = PolynomialFeatures(degree=self.power_of_features, include_bias=False)
            x_poly = poly.fit_transform(self.x_train)

        # Standardize the polynomial features
        scaler = StandardScaler()
        x_norm = scaler.fit_transform(x_poly)

        # Perform linear regression on polynomial features
        sgdr = LinearRegression()
        sgdr.fit(x_norm, self.y_train)
        b_norm = sgdr.intercept_
        w_norm = sgdr.coef_

        # Plotting
        fig, ax = plt.subplots(1, len(self.input_features), figsize=(12, 12), sharey=True)
        m, n = self.x_train.shape
        m = x_norm.shape[0]
        yp = np.zeros(m)
        for i in range(m):
            yp[i] = np.dot(x_norm[i], w_norm) + b_norm

        for j in range(len(self.input_features)):
            ax[j].scatter(self.x_train[:, j], self.y_train, label='Actual Value')
            ax[j].scatter(self.x_train[:, j], yp, color='green', label='Predicted')
            ax[j].set_xlabel(self.input_features[j])
            ax[j].set_ylabel(self.input_label)
            ax[j].plot(self.x_train[:, j], yp, color='red')
            ax[j].legend()
        plt.show()
        x_test = np.array([[200, 4, 4]]).reshape(1, -1)
        prediction = 0
        if isinstance(self.power_of_features, int):
            x_test_poly = poly.transform(x_test)
            x_test_transformed = scaler.transform(x_test_poly)
            prediction = np.dot(x_test_transformed, w_norm) + b_norm
        else:
            x_test_transformed = scaler.transform(x_test ** self.power_of_features)
            prediction = sgdr.predict(x_test_transformed)
        print(prediction)

    # code for implementing logistic_regression
    def logistic_regression(self):
        return 0


main()
