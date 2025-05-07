import numpy as np
import matplotlib.pyplot as plt


def utils_plot_poincare(xk, target, iter, K=-1):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    circle = plt.Circle((0, 0), radius=np.sqrt(-1/K), color='k', linewidth=2,
                        fill=False)
    ax.add_patch(circle)

    plt.scatter(xk[:, 0], xk[:, 1], label="Learned")
    plt.scatter(target[:, 0], target[:, 1], label="Target")

    plt.xlim(-1.01, 1.01)
    plt.ylim(-1.01, 1.01)
    plt.axis("equal")
    plt.title("Iteration "+str(iter))
    plt.legend()
    plt.show()
