import pylab as pl
from matplotlib import pyplot as plt
import numpy as np
import PIL.Image as Image

plt.rcParams['font.sans-serif'] = ['SimHei']

name_list = ['头', '左手', '左腿', '右手', '右腿']
num_list = np.array([[1.5974808e-01,3.3728755e-04,7.9550268e-03,9.7646207e-01,9.9998558e-01]])
num_list = num_list.reshape(num_list.shape[0]*num_list.shape[1], )

print(num_list)
# print(num_list.shape)
fig = pl.figure(figsize=(5, 5))
pl.barh(num_list,name_list , tick_label=num_list)
# pl.barh(y=range(len(num_list)), height=range(len(name_list)),width=1, tick_label=name_list)
pl.show()

