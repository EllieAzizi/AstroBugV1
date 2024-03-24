import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

print()

a=[]
for i in range(10,1000):
    a=np.random.randint(low=10,high=200,size=i)
    rsio=100/len(a)
    new_a = ndimage.interpolation.zoom(a, rsio)
    if len(new_a) != 100:
        print(len(new_a))
exit()
a=np.array(a)
print(a.shape)
new_c = ndimage.interpolation.zoom(a, 0.5)

plt.plot(range(len(a)),a)

plt.plot(range(len(new_c)),new_c) 
plt.show()