#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_wine


# In[2]:


wine = load_wine()
X = wine.data
     


# In[3]:


from sklearn.ensemble import IsolationForest
clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
clf.fit(X)
y_pred = clf.predict(X)


# In[4]:


import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title("Isolation Forest Outlier Detection on WINE Dataset")
plt.xlabel("alcohol")
plt.ylabel("malic_Acid")
plt.show()


# In[5]:


from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title("Local Outlier Factor Outlier Detection on WINE Dataset")
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.show()


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# In[7]:


# Load the IRIS dataset
wine = load_wine()
X = wine.data
y = wine.target


# In[8]:


# Fit Isolation Forest model
clf_iso = IsolationForest(contamination=0.1, random_state=42)
y_pred_iso = clf_iso.fit_predict(X)


# In[9]:


# Fit Local Outlier Factor model
clf_lof = LocalOutlierFactor(contamination=0.1)
y_pred_lof = clf_lof.fit_predict(X)


# In[10]:


# Plot Isolation Forest outliers
plt.scatter(X[:, 0], X[:, 1], c=np.where(y_pred_iso == -1, 'red', 'green'), label='Isolation Forest')
plt.title("Outlier Detection using Isolation Forest on WINE Dataset")
plt.xlabel("Alcohol")
plt.ylabel("Malic acid")
plt.legend()
plt.show()


# In[11]:


# Plot Local Outlier Factor outliers
plt.scatter(X[:, 0], X[:, 1], c=np.where(y_pred_lof == -1, 'red', 'green'), label='Local Outlier Factor')
plt.title("Outlier Detection using Local Outlier Factor on WINE Dataset")
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.legend()
plt.show()


# In[12]:


wine = load_wine(as_frame=True)
X,y = wine.data,wine.target
df = wine.frame
df.head()


# In[ ]:




