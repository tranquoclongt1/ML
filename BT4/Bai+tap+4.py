
# coding: utf-8

# ### Tran Quoc Long - 14520490

# ### Bai tap 4

# File's contents:
# 1. Test functions on an image set: test some functions in need
# 2. Let's get start: homework 4
# 
# Dataset:
# 1. http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php
# 2. http://benchmark.ini.rub.de/?section=gtsrb&subsection=news

# ### Using HOG - sklearn

# In brief, a HOG descriptor is computed by calculating image gradients that capture contour and silhouette information of grayscale images.
# Compute a Histogram of Oriented Gradients (HOG) by:
# 1. (optional) global image normalization
# 2. computing the gradient image in x and y
# 3. computing gradient histograms
# 4. normalizing across blocks
# 5. flattening into a feature vector
# 
# 
# read more and go to details at: https://www.learnopencv.com/histogram-of-oriented-gradients/

# ### Test functions on an image set

# In[12]:


#import libs
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg


# In[13]:


#import scikit-learn
from sklearn import metrics
from sklearn.cluster import KMeans, spectral_clustering, DBSCAN, AgglomerativeClustering
from sklearn.datasets import load_digits
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import load_iris

from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler

from skimage.feature import ORB, hog
import cv2
from skimage import data, color, exposure


# In[1]:


# C:\Users\tranq\Desktop\Thay Duy\GTSRB_Final_Test_Images\GTSRB\Final_Test\Images


# In[15]:



# using traffic test dataset from: http://benchmark.ini.rub.de/?section=gtsrb&subsection=news

import glob
file_list = glob.glob('C:/Users/tranq/Desktop/Thay Duy/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/*.ppm')


# In[16]:


print(len(file_list))


# In[17]:


#load image and convert into gray image
gray_sample = cv2.imread(file_list[0], 0)


# In[18]:


print(gray_sample.shape)
print(gray_sample)


# In[19]:


fd, hog_image = hog(gray_sample, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)


# In[20]:


print(hog_image.shape)


# In[21]:


print(hog_image)


# In[22]:


print(fd)


# In[23]:


print(fd.shape)


# In[24]:


gray_sample = cv2.imread(file_list[1], 0)
fd, hog_image = hog(gray_sample, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
print(fd)


# In[25]:


print(fd.shape)
print(fd)


# In[26]:


gray_sample = cv2.imread(file_list[2], 0)
fd, hog_image = hog(gray_sample, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
print(fd)
print(fd.shape)


# In[28]:


gray_sample = cv2.imread(file_list[3], 0)
fd, hog_image = hog(gray_sample, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
print(fd)
print(cv2.imread(file_list[3], 0).shape)
print(cv2.imread(file_list[4], 0).shape)
print(cv2.imread(file_list[5], 0).shape)
print(cv2.imread(file_list[6], 0).shape)
print(cv2.imread(file_list[7], 0).shape)
print(cv2.imread(file_list[8], 0).shape)
print(cv2.imread(file_list[9], 0).shape)


# ### =>> We need to resize images into the samesize

# ### Let's get start

# In[1]:


#import libs
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg

#import scikit-learn
from sklearn import metrics
from sklearn.cluster import KMeans, spectral_clustering, DBSCAN, AgglomerativeClustering
from sklearn.datasets import load_digits
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler

from skimage.feature import ORB, hog
from skimage import data, color, exposure
import cv2


# ### Get all paths of images

# In[2]:


# using Columbia University Image Library (COIL-20) data from: http://benchmark.ini.rub.de/?section=gtsrb&subsection=news
import glob
file_list = glob.glob('C:/Users/tranq/Desktop/Thay Duy/coil_20_proc/coil_20_proc/*.png')
print('Total images: ', len(file_list))


# In[3]:


#load image, resize them into the same size, and convert into greyscale 
def load_image_and_pre_processing(image_path):
    gray_img = cv2.imread(image_path, 0)
    #gray_img = cv2.resize(gray_img,(20,20))
    return gray_img   


# In[4]:


# input: list of file paths
# output: data - list of hog_vectors
def HOG_data_measurement(file_list_):
    data = []
    for path in file_list:
        grey_img = load_image_and_pre_processing(path)
        hog_data,hog_image = hog(grey_img, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualise=True)
        data.append(hog_data)
    return data


# In[5]:


data = HOG_data_measurement(file_list)


# ## Clustering

# In[6]:


#Kmeans
nClusters = 20
t0 = time()
kmeans_model = KMeans(nClusters)
t_kmeans = time()- t0
labels_kmeans = kmeans_model.fit_predict(data)


# In[7]:


#Spectral_clustering
t0 = time()
graph = cosine_similarity(data)
t_spectral = time()- t0
labels_spectral = spectral_clustering(graph, n_clusters=20)


# In[8]:


#DBSCAN
t0 = time()
data = StandardScaler().fit_transform(data)
labels_dbscan = DBSCAN(eps=0.3, min_samples=1,algorithm='kd_tree').fit_predict(data)
t_dbscan = time()- t0


# In[9]:


#Agglomerative Clustering
t0 = time()
Agglomerative_model = AgglomerativeClustering(n_clusters = nClusters)
labels_AgglomerativeClustering = Agglomerative_model.fit_predict(data)
t_agg = time() - t0


# ## Visulization

# In[10]:


from sklearn.decomposition import PCA
get_ipython().run_line_magic('matplotlib', 'inline')
nComponents = 2
vPCA = PCA(nComponents)
digitData_to_2D = vPCA.fit_transform(data)

fig = plt.figure(figsize=(15,15))
fig.suptitle('Comparition results of methods', fontsize=20)

ax = fig.add_subplot(3,2,1)
plt.scatter(digitData_to_2D[:,0], digitData_to_2D[:,1],  c= labels_spectral, s=20)
ax.set_title('Spectral Clustering')

ax = fig.add_subplot(3,2,2)
plt.scatter(digitData_to_2D[:,0], digitData_to_2D[:,1],  c= labels_dbscan, s=20)
ax.set_title('DBSCAN Clustering')

ax = fig.add_subplot(3,2,3)
plt.scatter(digitData_to_2D[:,0], digitData_to_2D[:,1],  c= labels_kmeans, s=20)
ax.set_title('K-means')

ax = fig.add_subplot(3,2,4)
plt.scatter(digitData_to_2D[:,0], digitData_to_2D[:,1],  c= labels_AgglomerativeClustering, s=20)
ax.set_title('Agglomerative clustering')

# ax = fig.add_subplot(3,2,5)
# plt.scatter(digitData_to_2D[:,0], digitData_to_2D[:,1],  c= lfw_people.target, s=20)
# ax.set_title('Target Result')


# ### References
# 
# 1. http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
# 2. https://www.learnopencv.com/histogram-of-oriented-gradients/
# 
# Dataset:
# 1. http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php
# 2. http://benchmark.ini.rub.de/?section=gtsrb&subsection=news
