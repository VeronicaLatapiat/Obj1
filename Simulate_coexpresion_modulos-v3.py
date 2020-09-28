#!/usr/bin/env python
# coding: utf-8

# In[384]:


#importamos librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import beta, binomial
#cargamos los datos de entrada

A=pd.read_csv('/Users/Vero_Latapiat/Desktop/Dropbox_temp/data_simulada/Matrices_Adyacencia_DS/adjacency_CASOA_red1.txt', header=0, sep='\t', index_col=0, engine='python')
A


# In[385]:


# Now we will convert it into 'int64' type. 
#A = A.astype('float32')
A= A.values
A


# In[386]:


n_zeros = np.sum(A == 0)#matriz de 10*10
n_ones = A.size - n_zeros

print("elementos de la matriz",A.size)
print("nro de no interacciones:",n_zeros) 
print("nro interacciones:",n_ones)


# In[387]:


#cg = sns.clustermap(A, cmap='RdBu', figsize=(18,12))
cg = sns.clustermap(A, cmap='RdBu', vmin=-1, vmax=1, figsize=(18,12))

cg.ax_row_dendrogram.set_visible(False)
cg.ax_col_dendrogram.set_visible(False)
#plt.savefig('heatmap_head_ds.png',dpi=150)


# In[388]:


#%pylab inline

# el numero de 1s el numero de 0s por cada combinacion de listas de modulos, del mismo o del otro
def get_random_corr(size=1, 
                    same_modulo=True,
                    is_null_corr = False,
                    prob_pos_corr=0.8, # aumenta en el mismo modulo
                    corr_neg=[40,10],
                    corr_pos=[10,40],
                    corr_null=[40,40]):
    '''
    size = # of simulated vector
    same_modulo = Boolean. If corresponds to values of the same module
    prob_pos_corr = float 0 -> 1. Prob of positive correlation
    corr_neg = [alpha,beta]. Beta distribution parameters for negative correlations
    corr_pos= [alpha,beta]. Beta distribution parameters for positve correlations
    corr_null=[alpha,beta]. Beta distribution parameters for correlations of nodes not in same module
    '''
    
    def _corr_from_2_betas(size,prob_pos_corr,corr_neg,corr_pos):
        n_pos = int(size*prob_pos_corr)#nro de 1s
        c = np.hstack([beta(corr_neg[0],corr_neg[1],size=size-n_pos),
                       beta(corr_pos[0],corr_pos[1],size=n_pos)])*2 - 1
        np.random.shuffle(c)
        return c
    
    if same_modulo:    
        return _corr_from_2_betas(size, prob_pos_corr, corr_neg, corr_pos)#*
    else:
        
        if is_null_corr:
            return beta(corr_null[0],corr_null[1],size=size)*2 - 1
        else: 
            return _corr_from_2_betas(size,prob_pos_corr,corr_neg,corr_pos)
      


# In[389]:


# https://entrenamiento-python-basico.readthedocs.io/es/latest/leccion3/tipo_diccionarios.html

#myDict = { 'item1' : [ 7, 1, 9], 'item2' : [8, 2, 3], 'item3' : [ 9, 3, 11 ] }

red = {
    'm1':[7, 8, 21, 22, 51, 68, 77, 78, 108, 111],
    'm2':[0, 4, 9, 11, 16, 23, 24, 28, 41, 50, 69, 90, 93, 104],
    'm3':[46, 49, 53, 67, 73, 83, 88, 110, 114],
    'm4':[3, 5, 10, 40, 52, 72, 74, 81, 84, 98, 102, 107],
    'm5':[2, 6, 13, 15, 32, 39, 47, 60, 64, 100, 106],
    'm6':[12, 14, 18, 26, 31, 34, 36, 38, 42, 43, 54, 61, 71, 85, 99],
    'm7':[19, 29, 35, 55, 79, 80, 82, 94, 101],
    'm8':[1, 25, 33, 37, 45, 89, 103, 105, 109],
    'm9':[44, 48, 57, 58, 59, 63, 66, 75, 86, 91, 92, 97, 112],
    'm10':[17, 20, 27, 56, 62, 65, 70, 76, 87, 95, 96, 113],
}




intra_corr_neg=[40,10] #dentro del modulo, corr neg
intra_corr_pos=[10,40] #dentro del modulo, corr pos
inter_corr_neg=[30,20] #fuera del modulo, corr neg
inter_corr_pos=[20,30] #fuera del modulo, corr pos
corr_null= [40,40] # corr -

for i in np.arange(len(red)): # contador de modulos (0-9)
    idx_i = red[list(red.keys())[i]] #lista de modulos
    print ("modulos:",i,idx_i)
    for j in np.arange(i+1): #contador en j (1-10)
        idx_j = red[list(red.keys())[j]] 
        #print ("j",j,idx_j)
        B = A[idx_i,:][:,idx_j]
        #print(B)
        n_ones = int(np.sum(B))# numero de unos en la matriz
        #print("int=",n_ones)
        
        where_1 = np.where(B == 1)# donde estan los 1
        
        is_same_modulo = i==j #si i es igual a j, lista deberia ser la misma, es el mismo modulo
        if is_same_modulo: #si se cumple la condicion
            corr_neg,corr_pos = intra_corr_neg, intra_corr_pos #toma los valores de arriba
        else:
            corr_neg,corr_pos = inter_corr_neg, inter_corr_pos
        #se itera en un indice y en contenido
        for k,val in enumerate(get_random_corr(size=n_ones, same_modulo=is_same_modulo, is_null_corr = False, corr_neg=corr_neg, corr_pos=corr_pos)):
            A[ idx_i[ where_1[0][k] ] , idx_j[ where_1[1][k] ] ] = val # se llena matriz con los valores de la funcion, triangulo sup = triang inf


# In[392]:


# Now we will convert it into 'int64' type. 
A = A.astype('float64')
n_zeros = np.sum(A == 0)
np.putmask(A,A == 0,get_random_corr(size=n_zeros,same_modulo=False, is_null_corr = True, corr_null=corr_null))


# In[393]:


# get the indices of the lowe triangula matrix
tril_i,tril_j = np.tril_indices(A.shape[0],k=-1)
# make the matrix symmetric by making the lowe triangular equal to the upper triangular
A[tril_j,tril_i] = A[tril_i,tril_j]


np.savetxt("DS_A.txt", A, delimiter="\t")


# simulate values based on the covariance
M = np.random.multivariate_normal(mean=np.zeros(A.shape[0]),cov=A,size=500)
print(M)


M_c = np.corrcoef(M.T) #correlacion en columnas
print(M_c)


# In[404]:


df = pd.DataFrame(data=M_c)
df
#df.iloc[7,8]


# In[403]:


# Calculate the correlation between individuals. We have to transpose first, because the corr function calculate the pairwise correlations between columns.
corr_datasim = df.corr()
corr_datasim
# aca es correlacion entre individuos
#corr_m1 = corr_m1.replace(np.nan, 0)
#corr_m1


# In[ ]:


## Plot real versus simulated values
#plot(A.flatten(),M_c.flatten(), 'o')


# In[353]:


# calculate correlation between real and simulated data
corr_np=np.corrcoef(x=A.flatten(),y=M_c.flatten())
print(corr_np)


# In[361]:


sns.distplot(M_c, bins = 115)


# In[405]:


sns.clustermap(M_c, cmap='RdBu', vmin=-1, vmax=1, figsize=(18,12))
plt.savefig('heatmap_head_ds.png',dpi=150)


# In[406]:


# Transform it in a links data frame (3 columns only):

#reset columns and index names 
corr = df.rename_axis(None).rename_axis(None, axis=1)

links = corr_datasim.stack().reset_index()
links.columns = ['gen1', 'gen2','correlation']
links


# In[112]:


# create a mask to identify rows with duplicate features as mentioned above
mask_dups = (links[['gen1', 'gen2']].apply(frozenset, axis=1).duplicated()) | (links['gen1']==links['gen2']) 

# apply the mask to clean the correlation dataframe
links = links[~mask_dups]
links


# In[407]:


# Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
links_filtered=links.loc[(links['correlation'] > 0.5) & (links['gen1'] != links['gen2'])]
links_filtered


# In[408]:


# Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
links_filtered_neg=links.loc[(links['correlation'] < -0.5) & (links['gen1'] != links['gen2'])]
links_filtered_neg


# In[409]:


links_filtrados=pd.concat([links_filtered,links_filtered_neg], ignore_index=True)
links_filtrados


# In[410]:


import networkx as nx
# Build your graph
G=nx.from_pandas_edgelist(links_filtrados, 'gen1', 'gen2')

# Plot the network:
nx.draw(G, with_labels=True, node_color='orange', node_size=100, edge_color='black', linewidths=1, font_size=4)


# In[ ]:




