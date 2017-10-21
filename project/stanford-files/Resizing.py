
# coding: utf-8

# In[1]:

from PIL import Image
import datautils
import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = (5.0, 5.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[2]:

res = 128
datautils.resize_images(res=res)
im_data = datautils.load_image("data/png{}/laptop/9555.png".format(res))
plt.imshow(im_data.reshape(res, res))
plt.show()


# In[ ]:



