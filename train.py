# train.py

# The second program will be used to train your model.
# It will read your dataset file and perform a linear regression on the data.
# Once the linear regression has completed, you will save the
# variables theta0 and theta1 for use in the first program.

from linear_regression import LinearRegression
import numpy as np
import pandas


def intro():
    print("Welcome to the car price training program ! 🦾\n" +
          "Let's train a linear regression model to predict " +
          "the price of a car depending on mileage.\n")


def get_data(path: str):
    """
    Get the data from the csv file and return the feature and the target.
    - Check the shape of the dataframe
    - Check if the file is well formatted
    - Check if the file contains only numeric values

    (the dataset is too small to split it into train and test sets,
     and as we already know the hypothesis equation,
     we don't need to split it to avoid overfitting)
    """
    try:
        df = pandas.read_csv(path)
        if df.shape[0] < 2 or df.shape[1] != 2:
            raise IndexError(
                "Invalid dataframe shape.")
        elif df.columns[0] != "km" or df.columns[1] != "price":
            raise ValueError(
                "The file data.csv is not well formatted.")
        elif df.dtypes[0] != (int, float) \
                or df.dtypes[1] != (int, float):
            raise TypeError(
                "The file data.csv contains non-numeric values.")
        df = df.to_numpy()
        feature = df[:, 0].reshape(-1, 1)
        target = df[:, 1].reshape(-1, 1)
        return feature, target

    except Exception:
        exit()


def save_thetas(thetas, filename):
    """
    Save the thetas in a file model.pkl for the first program.
    """
    try:
        with open(filename, "wb") as f:
            np.save(f, thetas)
    except Exception:
        exit()


if __name__ == "__main__":

    intro()

    # feature = mileage, target = price
    feature, target = get_data("data.csv")

    # Create the linear regression object
    linear_regression = LinearRegression()

    # Normalize the feature
    normalized_feature = linear_regression.normalize(feature)

    # Train the model
    linear_regression.fit(normalized_feature, target)

    # Denormalize the thetas
    thetas = linear_regression.denormalize()
    if thetas is not None:
        linear_regression.thetas = thetas
        print(f"theta0 = {thetas[0, 0]}")
        print(f"theta1 = {thetas[1, 0]}")

    # Save the thetas in a file model.npy for the first program
    save_thetas(linear_regression.thetas, "model.npy")

    # Calculate the loss
    y_hat = linear_regression.predict(feature)
    loss = linear_regression.loss(target, y_hat)
    if loss is not None:
        print(f"loss   = {loss}")

    # Plot the evolution of the gradient descent algorithm
    linear_regression.plot_progress(feature, target)

    # A program that calculates the precision of your algorithm
    linear_regression.plot_losses()

    # Plot the data to see the distribution and the line resulting
    # from your linear regression on the same graph
    linear_regression.plot(feature, target, y_hat)
