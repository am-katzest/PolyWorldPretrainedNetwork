#!/usr/bin/env python3
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import prediction as p

fn = sys.argv[1]
a = p.predict(fn)
img = np.asarray(Image.open(fn).resize((320, 320)))
imgplot = plt.imshow(img)
for e in a:
    plt.plot([x + 10 for x in e["X"]], [y + 10 for y in e["Y"]], marker="o")
axes = plt.gca()
axes.set_xlim([0, 320])
axes.set_ylim([0, 320])

plt.show()
