
# coding: utf-8

# # Tran Quoc Long - 14520490

# ## Clustering with face datasets

# ## Step in brief of LBP extraction

# 	1. Devide examied window into cells (16x16)
# 	2. For each pixel in a cell, compare to 8 neighbor, Follow along a circle
# 	3. Assign "number" "0" for pixel whose value is greater than the center, and "1" for the others
# 	4. Compute the histogram of frequency of each "number" occuring -> 16*16 = 256-demensional feature vector
# 	5. Optionaly normalize the histogram
# 	6. Concatenate (normalized) histogram of all cells -> Feature vector  for entire window

# ### Content

# Thực hiện các phép cluster trên bộ dữ liệu face lfw_people
# Nội dung bao gồm trong file:
# 1. Chạy thử các hàm cluster và các hàm liên quan
#     - Kmeans
#     - Spectral clustering
#     - DBSCAN
#     - Agglomerative clustering
#     - Cross table
#     - Figure to visualize result
#     - Show centroid of Kmeans
#     - Biểu diễn LBP dưới sơ đồ histogram
# 2. Nội dung thực hành 3

# In[1]:


from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[2]:


# load data set
lfw_people = fetch_lfw_people()


# In[3]:


lfw_people.images.shape


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.matshow(lfw_people.images[0])


# In[5]:


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# ### Test functions on 1 image

# In[6]:


from skimage.feature import local_binary_pattern 


# In[7]:


# settings for LBP
radius = 4
n_points = 8 
METHOD = 'uniform'

image = lfw_people.images[0]

#LBP 
lbp_features = local_binary_pattern(image, n_points, radius)


# In[8]:


print(lbp_features)
print('ccccc')
lbp_features.shape


# In[36]:


plt.imshow(lbp_features)


# In[37]:


import numpy as np
data = np.histogram(lbp_features, bins = range(0,257))
print('data:\n', data)


# In[41]:


fig2 = plt.figure(figsize = [20,5])
fig2.suptitle('Histogram', fontsize = 20)
plt.bar(range(len(data[0])),data[0], align='center')


# # Process for all images in dataset

# ## Extract LBP features of Images

# In[42]:


def get_LBP_feature(mLBP_of_Image):
    return np.histogram(mLBP_of_Image, bins = range(0,257))


# In[43]:


#compute local binary pattern
def pre_Compute(image):
    n_points = 8
    radius = 4
    return local_binary_pattern(image, n_points, radius)


# In[44]:


#Process all images and store to list
list_Features = []
for image in lfw_people.images:
    lbp_value = pre_Compute(image)
    feature_vector = get_LBP_feature(lbp_value)
    list_Features.append(feature_vector[0])
    


# In[46]:


print(len(list_Features)) # to check if list cantained enough element


# In[47]:


print(list_Features[0])


# In[48]:


from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image
from sklearn.metrics.pairwise import cosine_similarity


# ## Using KMEANS to cluster

# In[49]:


from sklearn.cluster import KMeans
import time


# In[50]:


start = time.time()
nClusters = 5749
kmeans_model = KMeans(nClusters)
face_labels = kmeans_model.fit_predict(list_Features)
end = time.time()


# In[25]:


print(len(face_labels))
clustering_time = end - start
print('Time: ', clustering_time, '(s)')


# ### Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.

# In[28]:


from sklearn.decomposition import PCA

nComponents = 3  # 3-dim
vPCA = PCA(nComponents)
digitData_to_3D = vPCA.fit_transform(lfw_people.data)


# In[54]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize = (20,10))
#plt.scatter(digitData_to_3D[:,0], digitData_to_3D[:,1], digitData_to_3D[:,2], c= face_labels, s=10, depthshade=True)
ax = Axes3D(fig)
ax.scatter(digitData_to_3D[:,0], digitData_to_3D[:,1], digitData_to_3D[:,2],
               c=face_labels, edgecolor='k', s = 20)
#set_title('Visulization Result - K-measns')


# In[51]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize = (20,5))
fig.suptitle('Visualization of faces clustering', fontsize=20)

ax = fig.add_subplot(1,2,1)
plt.scatter(digitData_to_3D[:,0], digitData_to_3D[:,1], c= face_labels, s=5)
plt.title('2d representation of clusters')

ax = fig.add_subplot(1,2,2)
plt.scatter(digitData_to_3D[:,1], digitData_to_3D[:,2], c= face_labels, s=5)
plt.title('Another 2d representation of clusters')


# In[62]:


print(lfw_people.target)
print(len(lfw_people.target))


# ## Face clustering on 4 methods

# In[1]:


#import libs
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg


# In[2]:


#import scikit-learn
from sklearn import metrics
from sklearn.cluster import KMeans, spectral_clustering, DBSCAN, AgglomerativeClustering
from sklearn.datasets import load_digits
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler


# In[3]:


# load data set
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
'''
It took lots of time on full dataset, but there're a bunch of clusters just contain 1 face (test above).
So I just select face with min_face = 70
'''


# In[4]:


lfw_people.images.shape


# In[5]:


print(np.unique(lfw_people.target)) #number of faces/clusters expected


# ## Extract LBP features of Images from loaded dataset

# In[6]:


from skimage.feature import local_binary_pattern


# In[7]:


def get_LBP_feature(mLBP_of_Image):
    return np.histogram(mLBP_of_Image, bins = range(0,257))
#compute local binary pattern
def pre_Compute(image):
    n_points = 8
    radius = 4
    return local_binary_pattern(image, n_points, radius)

#Process all images and store to list
data = []
for image in lfw_people.images:
    lbp_value = pre_Compute(image)
    feature_vector = get_LBP_feature(lbp_value)
    data.append(feature_vector[0])
    


# In[8]:


print(data[0])


# In[9]:


#Kmeans
nClusters = 7
t0 = time()
kmeans_model = KMeans(nClusters)
labels_kmeans = kmeans_model.fit_predict(data)
t_kmeans = time()- t0
#Cross table
print('Kmeans:\n')
df1 = pd.DataFrame({'labels':labels_kmeans,'Truth labels':lfw_people.target})
ct2 = pd.crosstab(df1['labels'],df1['Truth labels'])
print(ct2)


# In[10]:


#Spectral_clustering
t0 = time()
graph = cosine_similarity(data)
labels_spectral = spectral_clustering(graph, n_clusters=7)
t_spectral = time()- t0
#Spectral clustering - Crosstable
print('Spectral clustering:\n')
df1 = pd.DataFrame({'labels':labels_spectral,'Truth labels':lfw_people.target})
ct2 = pd.crosstab(df1['labels'],df1['Truth labels'])
print(ct2)


# In[11]:


#DBSCAN
t0 = time()
data1 = StandardScaler().fit_transform(data)
labels_dbscan = DBSCAN(eps=1, min_samples=1, algorithm ='brute').fit_predict(data)
t_dbscan = time()- t0
#DBSCAN - cross table
print('DBSCAN:\n')
df1 = pd.DataFrame({'labels':labels_dbscan,'Truth labels':lfw_people.target})
ct2 = pd.crosstab(df1['labels'],df1['Truth labels'])
print(ct2)


# In[12]:


#Agglomerative Clustering 
t0 = time()
Agglomerative_model = AgglomerativeClustering(n_clusters = nClusters)
labels_AgglomerativeClustering = Agglomerative_model.fit_predict(data)
t_agg = time() - t0
#Agglomerative Clustering - crosstable
print('Agglomerative Clustering:\n')
df1 = pd.DataFrame({'labels':labels_AgglomerativeClustering,'Truth labels':lfw_people.target})
ct2 = pd.crosstab(df1['labels'],df1['Truth labels'])
print(ct2)


# ### Comparison

# In[13]:



n_faces = len(np.unique(lfw_people.target))
#print frame
print(82 * '_')
print('init\t\ttime\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')


data = data
sample_size = 100
#define a function to measure and print out
def bench_clustering(method_name, time_, labels):
    print('%-9s\t%.2fs\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (method_name, time_,
             metrics.homogeneity_score(lfw_people.target, labels),
             metrics.completeness_score(lfw_people.target, labels),
             metrics.v_measure_score(lfw_people.target, labels),
             metrics.adjusted_rand_score(lfw_people.target, labels),
             metrics.adjusted_mutual_info_score(lfw_people.target,  labels),
             metrics.silhouette_score(data, labels,
                                      metric='euclidean',
                                      sample_size=sample_size)))
    

#Kmeans
bench_clustering('K-means', t_kmeans, labels_kmeans)
#Spectral_clustering
bench_clustering('spectral', t_spectral, labels_spectral)
#Agglomerative clustering
bench_clustering('Agg.', t_agg, labels_AgglomerativeClustering)
#DBSCAN ==> Problems with raw data
#bench_clustering('DBSCAN', t_dbscan, labels_dbscan)
print('-----------\nProblems with raw data cause noise with DBSCAN method')


# ### Nhận xét:
# 1. Từ bảng kết quả trên, ta thấy phương pháp spectral clustering cho kết quả có độ chính xác cao nhất trong các phương pháp, với tốc độ nhanh hơn K-means.
# 2. Agglomerative clustering: tốc độ chạy nhanh nhất nhưng kết quả có độ chính xác thấp nhất
# 3. Kmeans: tốc độ chậm nhất với kết quả có độ chính xác ở tầm trung của các phương pháp

# ### Visualization

# In[14]:


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

ax = fig.add_subplot(3,2,5)
plt.scatter(digitData_to_2D[:,0], digitData_to_2D[:,1],  c= lfw_people.target, s=20)
ax.set_title('Target Result')

