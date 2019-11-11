import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

__author__ = 'aGn'


data_set = np.array([
    [.9, .9], [.85, 2.1], [1.2, 1.], [2.1, .95], [3., 1.1],
    [3.9, .7], [4., 1.4], [4.2, 1.8], [2., 2.3], [3., 2.3],
    [1.5, 1.8], [2., 1.5], [2.2, 2.], [2.6, 1.7], [2.7, 1.85]
])
categories = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
color1 = (0.69411766529083252, 0.3490196168422699, 0.15686275064945221, 1.0)
color2 = (0.65098041296005249, 0.80784314870834351, 0.89019608497619629, 1.0)
colormap = np.array([color1, color2])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(
    x=[data_set[:, 0]],
    y=[data_set[:, 1]],
    c=colormap[categories],
    marker='o',
    alpha=0.9
)

margin = .1
min_f0, max_f0 = min(data_set[10:, 0]) - margin, max(data_set[10:, 0]) + margin
min_f1, max_f1 = min(data_set[10:, 1]) - margin, max(data_set[10:, 1]) + margin
width = max_f0 - min_f0
height = max_f1 - min_f1

ax.add_patch(
    patches.Rectangle(
        xy=(min_f0, min_f1),  # point of origin.
        width=width,
        height=height,
        linewidth=1,
        color='red',
        fill=False
    )
)
plt.show()


def test(x: 'feature one', y: 'feature two')->'Test point classification':
    if (min_f0 <= x <= max_f0) and (min_f1 <= y <= max_f1):
        print('(', x, ',', y, ')', ' belongs to the class 1')
    else:
        print('(', x, ',', y, ')', ' belongs to the class 0')


def data_set_test():
    for x, y in data_set:
        test(x, y)


def live_test():
    while True:
        val = input("Enter two value to classify (separate them by a space): ")
        test(float(val[0]), float(val[2]))


print(test.__annotations__)
data_set_test()
live_test()
