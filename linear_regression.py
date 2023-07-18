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

    def __init__(self, thetas=np.zeros((2, 1)), alpha=10e-5, n_cycle=100_000):
        """
        Init the class.
        """
        try:
            self.thetas = thetas
            self.alpha = alpha
            self.n_cycle = n_cycle
            self.x_mean = None
            self.x_std = None
            self.losses = []
            self.r2_scores = []
            self.thetas_plot = []
            self.points = np.logspace(0, np.log10(self.n_cycle - 1), 10, dtype=int)
            self.points[0] = 0
            self.points[-1] = self.n_cycle - 1
        except Exception:
            return None

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
            for _ in ft_progress(range(self.n_cycle)):
                if _ in self.points:
                    denormalized_thetas = self.denormalize(update=False)
                    self.thetas_plot.append(denormalized_thetas)
                gradient = self.gradient(x, y)
                self.thetas = self.thetas - self.alpha * gradient
                y_hat = self.predict(x)
                self.losses.append(self.loss(y, y_hat))
                self.r2_scores.append(self.r2_score(y, y_hat))
            print("\nTraining completed !\n")
            return self.thetas
        except Exception:
            return None

    def denormalize(self, update=True):
        """
        Return the denormalized thetas.
        """
        try:
            denormalized_thetas = np.zeros((2, 1))
            denormalized_thetas[0] = self.thetas[0, 0] - \
                (self.thetas[1, 0] * self.x_mean / self.x_std)
            denormalized_thetas[1] = self.thetas[1, 0] / self.x_std
            if update:
                self.thetas = denormalized_thetas
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

    def r2_score(self, y, y_hat):
        """
        Return the R2 score.
        """
        try:
            y_mean = np.mean(y)
            return 1 - (np.sum(np.square(y_hat - y)) /
                        np.sum(np.square(y - y_mean)))
        except Exception:
            return None

    def plot_prediction(self, feature, target, y_hat,
                        iteration=None, predict=None):
        """
        Plot the data and the prediction line.
        """

        def equation(denormalized_thetas, iteration) -> str:
            """
            Return the equation of the prediction line as str.
            """
            try:
                sign = "+" if denormalized_thetas[1, 0] > 0 else "-"
                if iteration is not None:
                    return f"Iteration {iteration} :\n" + \
                        f"f(x)  = {denormalized_thetas[0, 0]:.2f} {sign} " + \
                        f"{abs(denormalized_thetas[1, 0]):.4f} * x\n" + \
                        f"loss  = {self.loss(target, y_hat):.2f}\n" + \
                        f"R2   = {self.r2_score(target, y_hat):.2f}"
                return f"f(x)  = {denormalized_thetas[0, 0]:.2f} {sign} " + \
                    f"{abs(denormalized_thetas[1, 0]):.4f} * x\n" + \
                    f"loss = {self.loss(target, y_hat):.2f}\n" + \
                    f"R2   = {self.r2_score(target, y_hat):.2f}"
            except Exception:
                return None

        try:

            plt.scatter(feature, target, color="blue", label="Data")
            plt.plot(feature, y_hat, color="red", label="Prediction")
            for i in range(feature.shape[0]):
                plt.plot([feature[i], feature[i]], [target[i], y_hat[i]],
                         'r--', linewidth=0.75, alpha=0.5)

            if iteration is not None:
                """
                Plot the iteration number, the equation of the predicted line
                and metrics.
                """
                plt.text(0.05, 0.22, equation(self.thetas, iteration),
                         transform=plt.gca().transAxes,
                         fontsize=11, verticalalignment='top',
                         bbox=dict(facecolor='white', alpha=0.8,
                                   edgecolor='grey', boxstyle='round,pad=0.5',
                                   linewidth=0.75))
            else:
                """
                Plot the equation of the predicted line and metrics.
                """
                plt.text(0.05, 0.175, equation(self.thetas, iteration),
                         transform=plt.gca().transAxes,
                         fontsize=11, verticalalignment='top',
                         bbox=dict(facecolor='white', alpha=0.8,
                                   edgecolor='grey', boxstyle='round,pad=0.5',
                                   linewidth=0.75))

            if predict is not None:
                """
                Plot the predicted point and the lines to the x and y axis.
                (Used in the predict.py program)
                """
                predicted_x = predict[0, 0]
                predicted_y = predict[0, 1]
                plt.scatter(predicted_x, predicted_y, color="green",
                            marker='x', label="Predicted point")
                plt.plot([predicted_x, predicted_x], [0, predicted_y],
                         'g--', linewidth=0.75, alpha=0.8)
                plt.plot([0, predicted_x], [predicted_y, predicted_y],
                         'g--', linewidth=0.75, alpha=0.8)
                # Add the predicted point to y axis
                plt.text(0.6, 0.1,
                         f"{predicted_x:.0f} km, {predicted_y:.0f} €",
                         transform=plt.gca().transAxes,
                         fontsize=11, verticalalignment='top',
                         bbox=dict(facecolor='white', alpha=0.8,
                                   edgecolor='grey', boxstyle='round,pad=0.5',
                                   linewidth=0.75))

            plt.title("Linear regression on car price depending on mileage")
            plt.xlabel("Mileage (km)")
            plt.ylabel("Price (€)")
            plt.grid(linestyle=':', linewidth=0.5)
            max_x = np.max(feature)
            max_y = np.max(target)
            if predict is not None:
                if predicted_x > max_x:
                    max_x = predicted_x
                if predicted_y > max_y:
                    max_y = predicted_y
            plt.xlim(0, max_x * 1.05)
            plt.ylim(0, max_y * 1.05)
            plt.legend()
            plt.show()

        except Exception:
            return None

    def plot_gradient_descent(self, feature, target):
        """
        Plot the progress of the gradient descent.
        """
        try:
            _lr = LinearRegression()
            for i, thetas in enumerate(self.thetas_plot):
                _lr.thetas = thetas
                y_hat = _lr.predict(feature)
                _lr.plot_prediction(feature, target, y_hat,
                                    iteration=self.points[i], predict=None)

        except Exception:
            return None

    def plot_metrics(self):
        """
        Plot the losses.
        """
        try:

            fig = plt.figure()

            # Plot the losses
            ax1 = fig.add_subplot(111)
            ax1.set_xlabel("Iteration")
            ax1.plot(self.losses, color="red", label="Loss")
            ax1.legend(["Loss"], loc="center right",
                       bbox_to_anchor=(0.95, 0.45))
            ax1.set_ylabel("Loss", color="red")
            for tl in ax1.get_yticklabels():
                tl.set_color("red")
                tl.set_fontsize(9)
            plt.text(0.6, 0.10, f"last loss = {self.losses[-1]:.2f}",
                     transform=plt.gca().transAxes,
                     fontsize=11, verticalalignment='top', color="red")

            # Plot the R2 scores
            ax2 = ax1.twinx()
            ax2.plot(self.r2_scores, color="blue", label="R2 score")
            ax2.legend(["R2 score"], loc="center right",
                       bbox_to_anchor=(0.95, 0.55))
            ax2.set_ylabel("R2 score", color="blue")
            for tl in ax2.get_yticklabels():
                tl.set_color("blue")
            plt.text(0.6, 0.925, f"last R2-score = {self.r2_scores[-1]:.2f}",
                     transform=plt.gca().transAxes,
                     fontsize=11, verticalalignment='top', color="blue")

            plt.title("Metrics evolution during training")
            plt.grid(linestyle=':', linewidth=0.5)
            plt.show()
        except Exception:
            return None
