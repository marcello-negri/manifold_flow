import matplotlib.pyplot as plt
import numpy as np

def main():
    K_max = 20
    K_min = 0
    L_max = 20
    L_min = 0
    ax = plt.subplot(111)
    x_offset = 7  # tune these
    y_offset = 7  # tune these
    plt.setp(ax, 'frame_on', False)
    ax.set_ylim([0, (K_max - K_min + 1) * y_offset])
    ax.set_xlim([0, (L_max - L_min + 1) * x_offset])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid('off')

    for k in np.arange(K_min, K_max + 1):
        for l in np.arange(L_min, L_max + 1):
            # ax.plot(np.arange(5) + l * x_offset, 5 + np.arange(5) + k * y_offset,
            #         'r-o', ms=1, mew=0, mfc='r')
            ax.hist(np.random.rand(10000), bins=100)
            ax.annotate('K={},L={}'.format(k, l), (2.5 + (k) * x_offset, l * y_offset), size=3, ha='center')

    plt.show()

if __name__ == "__main__":
    main()