##============================================
##============================================
## Populaire
##============================================
##============================================
# recommande le produit le plus populaire

# * popularity.recommend(M_train, id_user, new) : recommande un film
# * popularity.complete(M_train) : complete la matrice

import numpy as np


##============================================
# popularity.recommend(M_train, id_user, new)
##============================================

'''
la fonction recommend prend en entrée une matrice M_train, un id d'utilisateur id_user et un booléen new. et permet de recommander un film à l'utilisateur id_user. de sorte que
Si new=True : Le code recommande le film non noté par l'utilisateur id_user qui a la moyenne de score la plus élevée parmi les films inconnus pour cet utilisateur.
Si new=False : Le code recommande simplement le produit avec la moyenne de score la plus élevée dans l'ensemble des films, sans se soucier des utilisateurs.
'''

def recommend(M_train, id_user, new=True):  
  scores = np.nanmean(M_train, axis=0) # moyenne des scores de chaque film par tous les utilisateurs en ignorant les valeurs manquantes
  if new: # si new=True
    inds_unknown = np.where(np.isnan(M_train[id_user, :]))[0] # recupèreindices des films non notés par l'utilisateur id_user
    rec_ind_in_unknown = np.nanargmax(scores[inds_unknown]) # recupère l'indice du film non noté par l'utilisateur id_user qui a la moyenne de score la plus élevée parmi les films inconnus pour cet utilisateur
    return inds_unknown[rec_ind_in_unknown]
  else:
    return np.nanargmax(scores) # renvoi l'indice du film avec la moyenne de score la plus élevée dans l'ensemble des films, sans se soucier des utilisateurs

##============================================
# popularity.complete(M_train)
##============================================

'''
Le but de la fonction complete est de compléter la matrice M_train en remplaçant les valeurs manquantes par la moyenne des scores de chaque film par tous les utilisateurs en ignorant les valeurs manquantes.
Donc s'il y a une valeur manquante pour un film, elle sera remplacée par la moyenne des scores de ce film par tous les utilisateurs en ignorant les valeurs manquantes.
'''

def complete(M_train):  #  definition de la fonction complete la matrice
  scores = np.nanmean(M_train, axis=0) # moyenne des scores de chaque film par tous les utilisateurs en ignorant les valeurs manquantes
  scores[np.isnan(scores)] = 0 # remplace les valeurs manquantes par 0
  to_complete = np.ones((M_train.shape[0], 1)) @ scores.reshape((1, -1)) # crée une matrice de taille M_train.shape avec les moyennes des scores de chaque film par tous les utilisateurs en ignorant les valeurs manquantes
  M_completed = M_train.copy() # copie M_train
  M_completed[np.isnan(M_train)] = to_complete[np.isnan(M_train)] # remplace les valeurs manquantes de M_train par les moyennes des scores de chaque film par tous les utilisateurs en ignorant les valeurs manquantes
  return M_completed # retourne M_completed



