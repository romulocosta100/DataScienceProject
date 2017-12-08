
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np
from numpy import array


# In[58]:


datapath="C:/Users/hespo/OneDrive/Documentos/GitHub/Trab2_DataScience/DataScienceProject/"


# In[59]:


gb_fb = pd.read_csv(datapath+"gabaritoNEW.csv")


# In[60]:


a=gb_fb.groupby('1')['0'].apply(list)
lil=a.tolist()


# In[61]:


print(lil)


# In[64]:


lilnp=np.array(lil)


# In[65]:


print(lilnp)

