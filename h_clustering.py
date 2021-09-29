#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pickle as pickle
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


filepath = 'data/similarity_matrix.pickle'
with open(filepath, 'rb') as lf :
        similarity_vector = pickle.load(lf)


# k = 30
kmeans_30 = KMeans(n_clusters= 30, random_state=1004).fit(similarity_vector)
clusters_30 = kmeans_30.labels_


clusters_30 = pd.DataFrame(clusters_30)
a_30 = clusters_30[0].value_counts()
# cluster 별 문서 빈도수
ax_30 = a_30.plot(kind='bar', title='Number of cluster', figsize=(12, 4), legend=None)
ax_30.set_xlabel('Cluster', fontsize=12)
ax_30.set_ylabel('Number of documents', fontsize=12)


# k = 40
kmeans_40 = KMeans(n_clusters= 40, random_state=1004).fit(similarity_vector)
clusters_40 = kmeans_40.labels_

clusters_40 = pd.DataFrame(clusters_40)
a_40 = clusters_40[0].value_counts()
# cluster 별 문서 빈도수
ax_40 = a_40.plot(kind='bar', title='Number of cluster', figsize=(12, 4), legend=None)
ax_40.set_xlabel('Cluster', fontsize=12)
ax_40.set_ylabel('Number of documents', fontsize=12)

# k = 50
kmeans_50 = KMeans(n_clusters= 50, random_state=1004).fit(similarity_vector)
clusters_50 = kmeans_50.labels_

clusters_50 = pd.DataFrame(clusters_50)
a_50 = clusters_50[0].value_counts()
# cluster 별 문서 빈도수
ax_50 = a_50.plot(kind='bar', title='Number of cluster', figsize=(12, 4), legend=None)
ax_50.set_xlabel('Cluster', fontsize=12)
ax_50.set_ylabel('Number of documents', fontsize=12)


clusters_50.to_csv('C:/Users/82109/GitHub/doc2vec/data/cluster_50.csv')


clusters.columns = ['cluster']


# In[105]:


a = clusters['cluster'].value_counts()


# In[106]:


a


# In[117]:


lrg_cluster1 = a.index[0]


# In[118]:


clusters_large = clusters[clusters['cluster'] == lrg_cluster1]


# In[122]:


clusters_large.index


# In[125]:


lrg_cluster1_df = similarity_vector.loc[clusters_large.index, clusters_large.index]
lrg_cluster1_df


# In[ ]:


lrg_cluster1_df


# In[126]:


kmeans2 = KMeans(n_clusters= 5, random_state=1004).fit(lrg_cluster1_df)
clusters = kmeans2.labels_

clusters2 = pd.DataFrame(clusters)
a2 = clusters2[0].value_counts()

ax2 = a2.plot(kind='bar', title='Number of cluster', figsize=(12, 4), legend=None)
ax2.set_xlabel('Cluster', fontsize=12)
ax2.set_ylabel('Number of documents', fontsize=12)


# In[127]:


clusters2


# In[ ]:




