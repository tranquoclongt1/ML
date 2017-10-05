
# coding: utf-8

# # Spectral Clustering

# ### Tran Quoc Long - 14520490

# ## Bài tập 2: Handwritting digits

# ### K-means

# In[34]:


#import libs
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[35]:

#import scikit-learn
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits


# In[36]:

digits = load_digits();
print(digits.data.shape);


# In[37]:

get_ipython().run_line_magic('matplotlib', 'inline')
#plt.gray(); 
plt.matshow(digits.images[0]);


# In[38]:

nClusters = 10
model1 = KMeans(nClusters)
labels_kmeans = model1.fit_predict(digits.data)


# In[39]:

df = pd.DataFrame({'labels':labels,'Truth labels':digits.target})
ct = pd.crosstab(df['labels'],df['Truth labels'])
print(ct)


# In[40]:

n = 10
get_ipython().run_line_magic('matplotlib', 'inline')
plt.matshow(digits.images[n])
print('Predict Label:', labels_kmeans[n])
print('Truth: ', digits.target[n])


# ### Visualization - Kmeans

# In[41]:

#import libs
import numpy as np
from sklearn.decomposition import PCA


# ##### PCA

# In[42]:

nComponents = 2
vPCA = PCA(nComponents)
digitData_to_2D = vPCA.fit_transform(digits.data)
plt.scatter(digitData_to_2D[:,0], digitData_to_2D[:,1],  c= labels_kmeans, s=20)
plt.show()


# In[43]:

plt.scatter(digitData_to_2D[:,0], digitData_to_2D[:,1],  c= digits.target, s=20)
plt.show()


# ## Speactral clustering

# In[44]:

# Spectral_clustering

from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image
import numpy as np
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity

# dist = DistanceMetric.get_metric('euclidean')
# graph=dist.pairwise(digits.data) 

graph = cosine_similarity(digits.data)
label_spectral = spectral_clustering(graph, n_clusters=10)


# In[45]:

df1 = pd.DataFrame({'labels':label_spectral,'Truth labels':digits.target})
ct2 = pd.crosstab(df1['labels'],df1['Truth labels'])
print(ct2)


# In[46]:

n = 15
plt.matshow(digits.images[n])
print('lables_predict:',label_spectral[n])
print(' True: ', digits.target[n])


# #### Visualization - Spectral Clustering
# 

# In[15]:


plt.scatter(digitData_to_2D[:,0], digitData_to_2D[:,1],  c= label_spectral, s=20)
plt.show()


# ### Visualize results to compare - Using PCA

# In[58]:


fig = plt.figure(figsize=(15,4))
fig.suptitle('Comparition results of methods', fontsize=20)

ax = fig.add_subplot(1,3,1)
plt.scatter(digitData_to_2D[:,0], digitData_to_2D[:,1],  c= label_spectral, s=20)
ax.set_title('Spectral Clustering')

ax = fig.add_subplot(1,3,2)
plt.scatter(digitData_to_2D[:,0], digitData_to_2D[:,1],  c= labels_kmeans, s=20)
ax.set_title('Kmeans Clustering')

ax = fig.add_subplot(1,3,3)
plt.scatter(digitData_to_2D[:,0], digitData_to_2D[:,1],  c= digits.target, s=20)
ax.set_title('Truth Result')


# ### 
