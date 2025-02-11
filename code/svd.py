##============================================
##============================================
## SVD
##============================================
##============================================
# remplace les valeurs manquantes de façon naïve, puis retourne la matrice de rang k la plus proche_

# fonctions pour la recommandation
# * svd.complete(M_train, k, ___)
# * svd.recommend(M_train, id_user, new=True, k=10, ___)

import numpy as np

##============================================
## utile : replaceNA_with_zeros(M.train)
##============================================
def replaceNA_with_zeros(M_train):
    return res


##============================================
## svd.complete(M_train, k)
##============================================
def complete(M_train, k, replaceNA_fn=replaceNA_with_zeros):
    res = np.zeros(M_train.shape)
    return res


##============================================
## svd.recommend(M_train, id_user, new=True, k=10, ___)
##============================================
def recommend(M_train, id_user, new=True, k=10, replaceNA_fn=replaceNA_with_zeros):
    return 0

