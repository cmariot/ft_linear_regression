import shutil
import time
import numpy as np
import matplotlib.pyplot as plt


def ft_progress(iterable,
                length=shutil.get_terminal_size().columns - 2,
                fill='█',
                empty='░',
                print_end='\r'):
    """
    Progress bar generator.
    """

    def get_eta_str(eta):
        """
        Return the Estimed Time Arrival as str.
        """
        if eta == 0.0:
            return '[DONE]    '
        elif eta < 60:
            return f'[ETA {eta:.0f} s]'
        elif eta < 3600:
            return f'[ETA {eta / 60:.0f} m]'
        else:
            return f'[ETA {eta / 3600:.0f} h]'

    def get_elapsed_time_str(elapsed_time):
        """
        Return the elapsed time as str.
        """
        if elapsed_time < 60:
            return f' [Elapsed-time {elapsed_time:.2f} s]'
        elif elapsed_time < 3600:
            return f' [Elapsed-time {elapsed_time / 60:.0f} m]'
        else:
            return f' [Elapsed-time {elapsed_time / 3600:.0f} h]'
    try:
        print()
        total = len(iterable)
        start = time.time()
        for i, item in enumerate(iterable, start=1):
            elapsed_time = time.time() - start
            et_str = get_elapsed_time_str(elapsed_time)
            eta_str = get_eta_str(elapsed_time * (total / i - 1))
            filled_length = int(length * i / total)
            percent_str = f'[{(i / total) * 100:6.2f} %] '
            progress_str = str(fill * filled_length
                               + empty * (length - filled_length))
            counter_str = f' [{i:>{len(str(total))}}/{total}] '
            bar = ("\033[F\033[K " + progress_str + "\n"
                   + et_str
                   + counter_str
                   + percent_str
                   + eta_str)
            print(bar, end=print_end)
            yield item
        print()
    except Exception:
        return None


class LinearRegression:
    """
    Linear regression class.
    """

    def __init__(self, thetas=np.zeros((2, 1)), alpha=10e-4, n_cycle=10_000):
        """
        Init the class.
        """
        self.thetas = thetas
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.x_mean = None
        self.x_std = None
        self.losses = []
        self.thetas_plot = []

    def normalize(self, x):
        """
        Return the normalized feature.
        """
        try:
            self.x_mean = np.mean(x, axis=0)
            self.x_std = np.std(x, axis=0)
            return (x - self.x_mean) / self.x_std
        except Exception:
            return None

    def predict(self, x):
        """
        Return the prediction vector.
        """
        try:
            X_prime = np.c_[np.ones(x.shape[0]), x]
            y_hat = np.dot(X_prime, self.thetas)
            return y_hat
        except Exception:
            return None

    def gradient(self, x, y):
        """
        Return the gradient vector.
        """
        try:
            m = x.shape[0]
            X_prime = np.c_[np.ones(m), x]
            return (X_prime.T @ (X_prime @ self.thetas - y)) / m
        except Exception:
            return None

    def fit(self, x, y):
        """
        Train the model.
        """
        try:
            # Get 10 points to plot the progress of the gradient descent
            points = np.logspace(0, np.log10(self.n_cycle - 1), 10, dtype=int)
            for _ in ft_progress(range(self.n_cycle)):
                gradient = self.gradient(x, y)
                self.thetas = self.thetas - self.alpha * gradient
                y_hat = self.predict(x)
                self.losses.append(self.loss(y, y_hat))
                if _ in points:
                    denormalized_thetas = self.denormalize()
                    self.thetas_plot.append(denormalized_thetas)
            print("\nTraining completed !\n")
            return self.thetas
        except Exception:
            return None

    def denormalize(self):
        """
        Return the denormalized thetas.
        """
        try:
            denormalized_thetas = np.zeros((2, 1))
            denormalized_thetas[0] = self.thetas[0, 0] - \
                (self.thetas[1, 0] * self.x_mean / self.x_std)
            denormalized_thetas[1] = self.thetas[1, 0] / self.x_std
            return denormalized_thetas
        except Exception:
            return None

    def loss(self, y, y_hat):
        """
        Return the loss value.
        """
        try:
            loss = np.square(y_hat - y) / (2 * y.shape[0])
            return np.mean(loss)
        except Exception:
            return None

    def plot(self, feature, target, y_hat):
        """
        Plot the data and the prediction line.
        """

        def equation(denormalized_thetas) -> str:
            """
            Return the equation of the prediction line as str.
            """
            sign = "+" if denormalized_thetas[1, 0] > 0 else "-"
            return f"f(x)  = {denormalized_thetas[0, 0]:.2f} {sign} " + \
                f"{abs(denormalized_thetas[1, 0]):.4f} * x\n" + \
                f"loss = {self.loss(target, y_hat):.2f}"

        try:
            plt.scatter(feature, target, color="blue", label="Data")
            plt.plot(feature, y_hat, color="red", label="Prediction")
            for i in range(feature.shape[0]):
                plt.plot([feature[i], feature[i]], [target[i], y_hat[i]],
                         'r--', linewidth=0.75, alpha=0.5)
            plt.text(0.5, 0.10, equation(self.thetas),
                     transform=plt.gca().transAxes,
                     fontsize=11, verticalalignment='top', color="red")
            plt.title("Linear regression on car price depending on mileage")
            plt.xlabel("Mileage (km)")
            plt.ylabel("Price (€)")
            plt.grid(linestyle=':', linewidth=0.5)
            plt.xlim(0, np.max(feature) * 1.05)
            plt.ylim(0, np.max(target) * 1.05)
            plt.legend()
            plt.show()

        except Exception:
            return None

    def plot_progress(self, feature, target):
        """
        Plot the progress of the gradient descent.
        """
        try:
            _lr = LinearRegression()
            for thetas in self.thetas_plot:
                _lr.thetas = thetas
                y_hat = _lr.predict(feature)
                _lr.plot(feature, target, y_hat)

        except Exception:
            return None

    def plot_losses(self):
        """
        Plot the losses.
        """
        try:
            plt.plot(self.losses, color="red")
            plt.title("Evolution of the loss function during training")
            plt.xlabel("Number of iterations")
            plt.ylabel("Loss")
            plt.text(0.6, 0.10, f"final loss = {self.losses[-1]:.2f}",
                     transform=plt.gca().transAxes,
                     fontsize=11, verticalalignment='top', color="red")
            plt.grid(linestyle=':', linewidth=0.5)
            plt.show()
        except Exception:
            return None
