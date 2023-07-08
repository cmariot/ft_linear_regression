# predict.py

# This program will be used to predict the price of a car for a given mileage
# When you launch the program, it should prompt you for a mileage,
# and then give you back the estimated price for that mileage.
# The program will use the following hypothesis to predict the price:
# estimatePrice(mileage) = θ0 + (θ1 ∗ mileage)
# Before the run of the training program, theta0 and theta1 will be set to 0.

# The program will be run as follows:
# > python3 predict.py

from linear_regression import LinearRegression
from train import get_data
import numpy as np


def intro():
    """
    Print an introduction message to the user.
    """
    print("Welcome to the car price prediction program ! 🚗")
    print("This program will predict the price of a car " +
          "for a given mileage.")
    print()


def prompt_mileage():
    """
    Prompt the user for a mileage.
    """
    while True:
        try:
            mileage = float(input("Please enter a mileage in kilometers : "))
            if mileage < 0:
                raise ValueError
            print()
            return mileage
        except ValueError:
            print("Please enter a valid mileage")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye !")
            exit()


def get_thetas():
    """
    Get thetas from the model file.
    If the model has not been trained, return the default thetas.
    """
    try:
        thetas = np.zeros((2, 1))
        with open('model.npy', 'rb') as f:
            thetas = np.load(f)
            if not isinstance(thetas, np.ndarray) \
                    or thetas.shape != (2, 1) \
                    or not isinstance(thetas[0, 0], (int, float)) \
                    or not isinstance(thetas[1, 0], (int, float)):
                raise TypeError
    except TypeError:
        print("Thetas are invalid.\n" +
              "Default thetas will be used.\n")
    except Exception:
        print("It looks like you didn't train the model.\n" +
              "Default thetas will be used.\n")
    return thetas


def pedict_price(mileage, thetas):
    """
    Predict the price of a car for a given mileage.
    """
    try:
        linear_regression = LinearRegression(thetas)
        price = linear_regression.predict(np.array([mileage]))
        return price[0, 0]
    except Exception:
        return None


def print_prediction(mileage, price, feature, target):
    """
    Print the predicted price.
    """

    if mileage < feature.min() or mileage > feature.max():
        print("Warning: The input data is outside the training range, " +
              "which may result in inaccurate predictions.\n" +
              "Please exercise caution when interpreting the results.\n")

    if price < 0:
        print("The predicted price is negative, which is not possible.\n" +
              "Please try again with a lower mileage.")
        exit()

    else:
        def format_value(value):
            """
            Format a value to a string with a space as thousands separator
            and a comma as decimal separator.
            """
            return "{:,.2f}".format(value).replace(",", " ").replace(".", ",")

        print("For a mileage of {} km, the estimated price is {} €"
              .format(format_value(mileage), format_value(price)))


def plot_prediction(mileage, price, thetas, feature, target):
    """
    Plot the data, the prediction line and the prediction point
    """
    try:
        linear_regression = LinearRegression(thetas)
        y_hat = linear_regression.predict(feature)
        predict = np.array([[mileage, price]])
        linear_regression.plot_prediction(feature, target,
                                          y_hat, predict=predict)
    except Exception:
        return None


if __name__ == "__main__":
    intro()
    mileage = prompt_mileage()
    thetas = get_thetas()
    price = pedict_price(mileage, thetas)
    feature, target = get_data("data.csv")
    print_prediction(mileage, price, feature, target)
    plot_prediction(mileage, price, thetas, feature, target)
