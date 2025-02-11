##============================================
##============================================
## kppv
##============================================
##============================================
# Recommande des films en utilisant la stratégie des k plus proches utilisateurs
# La similarité est claculée selon le critère *cosinus*

# * popularity.recommend(M_train, id_user, new) : recommande un film
# * popularity.complete(M_train) : complète la matrice




'''
Il s'agit de la méthode user based collaborative filtering qui est implémentée ici.
'''

# utilise les sous-fonctions
# * cosinus(M_train, u1, u2)
# * complete_a_user_knn(M_train, id_user, k)

import numpy as np

##============================================
## cosinus(M_train, u1, u2)
##============================================

'''
le but de cette fonction est de calculer la similarité cosinus entre deux utilisateurs u1 et u2. de sorte que si aucun film n'est noté par les deux utilisateurs u1 et u2, la fonction retourne 0. sinon, elle retourne la similarité cosinus entre les notes de l'utilisateur u1 et celles de l'utilisateur u2 pour les films notés par les deux utilisateurs u1 et u2.
'''

def cosinus(M_train, u1, u2): # calcule la similarité cosinus entre deux utilisateurs u1 et u2
    # films en commun
    inds_movie = np.where(np.sum(np.isnan(M_train[[u1, u2], ]), axis=0) == 0)[0] # recupère les indices des films notés par les deux utilisateurs u1 et u2
    
    # cosinus
    if len(inds_movie) != 0: # si la longueur de inds_movie est différente de 0 cela veut dire que les deux utilisateurs u1 et u2 ont noté des films en commun
        n1 = M_train[u1, inds_movie] # recupère les notes de l'utilisateur u1 pour les films notés par les deux utilisateurs u1 et u2
        n2 = M_train[u2, inds_movie] # recupère les notes de l'utilisateur u2 pour les films notés par les deux utilisateurs u1 et u2
        cos = sum(n1*n2) / np.sqrt(sum(n1**2)) / np.sqrt(sum(n2**2)) # calcule la similarité cosinus entre les notes de l'utilisateur u1 et celles de l'utilisateur u2 pour les films notés par les deux utilisateurs u1 et u2
        return cos  # retourne la similarité cosinus entre les notes de l'utilisateur u1 et celles de l'utilisateur u2 pour les films notés par les deux utilisateurs u1 et u2

    else:
        return 0 # retourne 0 si les deux utilisateurs u1 et u2 n'ont pas noté de films en commun



##============================================
## complete_a_user_knn(M_train, id_user, k)
##============================================

'''
le but de cette fonction est de compléter les notes de l'utilisateur id_user en utilisant la stratégie des k plus proches utilisateurs. de sorte que si l'utilisateur id_user a déjà noté un film, la fonction retourne la note de l'utilisateur id_user pour ce film. sinon, elle retourne la note prédite de l'utilisateur id_user pour ce film en utilisant la stratégie des k plus proches utilisateurs.
'''

def complete_a_user(M_train, id_user, k):
    scores = np.zeros(M_train.shape[1]) # crée un vecteur de taille le nombre de colonnes rempli de 0
    for id_item in range(M_train.shape[1]): # boucle sur les colonnes de M_train qui representent les films
        inds_known = np.where(~np.isnan(M_train[:, id_item]))[0] # recupère les indices des utilisateurs qui ont noté le film id_item
        
        
        # remove id_user from inds_known
        
        if np.isnan(M_train[id_user,id_item]): # si la note de l'utilisateur id_user pour le film id_item est manquante
            if len(inds_known) > 0 : # si des utilisateurs ont noté le film id_item
                sims = np.array([cosinus(M_train, id_user, u) for u in inds_known]) # calcule la similarité cosinus entre l'utilisateur id_user et les utilisateurs qui ont noté le film id_item

                if len(inds_known) > k : # si le nombre d'utilisateurs qui ont noté le film id_item est supérieur à k
                    ind = np.argsort(-sims) # - for descending order  # trie les similarités cosinus en ordre décroissant
                    inds_known = inds_known[ind][:k] # recupère les indices des k plus proches utilisateurs qui ont noté le film id_item
                    sims = sims[ind][:k] # recupère les similarités cosinus des k plus proches utilisateurs qui ont noté le film id_item

                    if sum(abs(sims)) != 0 : # si la somme des valeurs absolues des similarités cosinus est différente de 0
                        rates = M_train[inds_known, id_item] # recupère les notes des k plus proches utilisateurs qui ont noté le film id_item

                        mean_rates = np.nanmean(M_train[inds_known, :], axis=1) # calcule la moyenne des notes des k plus proches utilisateurs pour tous les films
                        
                        scores[id_item] = np.nanmean(M_train[id_user, :]) + np.sum(sims*(rates-mean_rates))/sum(abs(sims)) # calcule la note prédite de l'utilisateur id_user pour le film id_item en utilisant la stratégie des k plus proches utilisateurs
                    else :
                        scores[id_item] = np.nanmean(M_train[id_user, :]) # si la somme des valeurs absolues des similarités cosinus est égale à 0, la note prédite de l'utilisateur id_user pour le film id_item est la moyenne des notes de l'utilisateur id_user pour tous les films
                        
                else :
                    scores[id_item] = np.nanmean(M_train[id_user, :]) # si le nombre d'utilisateurs qui ont noté le film id_item est inférieur ou égal à k, la note prédite de l'utilisateur id_user pour le film id_item est la moyenne des notes de l'utilisateur id_user pour tous les films
                    
        else: # if np.isnan.... 
            scores[id_item]=M_train[id_user,id_item] # si la note de l'utilisateur id_user pour le film id_item est déjà notée, la note prédite de l'utilisateur id_user pour le film id_item est la note de l'utilisateur id_user pour le film id_item
    return scores # retourne les notes prédites de l'utilisateur id_user pour tous les films en utilisant la stratégie des k plus proches utilisateurs



##============================================
## knn.recommend(M_train, id_user, new=True, k=10)
##============================================
'''
la fonction recommend prend en entrée une matrice M_train, un id d'utilisateur id_user, un booléen new et un entier k. et permet de recommander un film à l'utilisateur id_user. de sorte que si new=True, le code recommande le film non noté par l'utilisateur id_user qui a la moyenne de score la plus élevée 
Sinon l'utiliateur se voit recomandé le film ayant en utilisant la stratégie des k plus proches.
'''
def recommend(M_train, id_user, new=True, k=10):
    scores = complete_a_user(M_train, id_user, k) # complete les notes de l'utilisateur id_user en utilisant la stratégie des k plus proches utilisateurs
  
    if new:
        inds_unknown = np.where(np.isnan(M_train[id_user, :]))[0]
        rec_ind_in_unknown = np.argmax(scores[inds_unknown])
        return inds_unknown[rec_ind_in_unknown]
        # on recommande le film non noté par l'utilisateur id_user qui a la moyenne de score la plus élevée parmi les films inconnus pour cet utilisateur
    else :
        return np.argmax(scores) # on recommande le film sous la base de ses k plus proches voisins



##============================================
## knn.complete
##============================================

'''
Le but de la fonction complete est de compléter la matrice M_train en remplaçant les valeurs manquantes par les notes prédites des utilisateurs en utilisant la stratégie des k plus proches utilisateurs.
'''

def complete(M_train, k):
    #
    M_completed = np.zeros(M_train.shape) # crée une matrice de taille M_train.shape remplie de 0 
    for id_user in range(M_train.shape[0]):
        M_completed[id_user, :] = complete_a_user(M_train, id_user, k) # complete les notes de l'utilisateur id_user en utilisant la stratégie des k plus proches utilisateurs
    return M_completed # retourne M_completed


'''
Implementation de la méthode item based collaborative filtering

le but de cette méthode est de recommander des films en utilisant les preferences de l'utilsateur et en proposant des films similaires à ceux qu'il a déjà noté.
'''

# utilise les sous-fonctions
