import numpy as np

##============================================
## Utilitaire : replaceNA_with_zeros(M_train)
##============================================
def replaceNA_with_zeros(M_train):
    """
    Remplace les valeurs manquantes (NaN) par des zéros.
    
    Cela permet d'utiliser la décomposition SVD sans problème, car np.linalg.svd
    ne peut pas traiter les NaN. On obtient ainsi une matrice entièrement numérique.
    
    Paramètres :
    - M_train : matrice d'entraînement contenant éventuellement des NaN
    
    Retourne :
    - Une matrice identique à M_train mais où les NaN ont été remplacés par 0.
    """
    return np.where(np.isnan(M_train), 0, M_train)


##============================================
## svd.complete(M_train, k, replaceNA_fn)
##============================================
def complete(M_train, k, replaceNA_fn=replaceNA_with_zeros):
    """
    Complète la matrice M_train en utilisant une décomposition en valeurs singulières (SVD)
    tronquée aux k premières valeurs singulières.
    
    Étapes :
    1. Remplacement des NaN : on remplace les valeurs manquantes à l'aide de la fonction
       passée en paramètre (par défaut, replaceNA_with_zeros).
    2. Décomposition SVD : on factorise la matrice complétée en U, s et Vt.
    3. Troncature : on ne garde que les k premières composantes (les k plus grandes valeurs singulières
       et leurs vecteurs associés).
    4. Reconstruction : on reconstruit la matrice approximative en calculant U_k * diag(s_k) * Vt_k.
    
    Paramètres :
    - M_train     : matrice d'entraînement contenant les notes (avec des NaN pour les notes manquantes)
    - k           : nombre de valeurs singulières à retenir pour la troncature
    - replaceNA_fn: fonction pour remplacer les NaN (par défaut replaceNA_with_zeros)
    
    Retourne :
    - La matrice complétée (approximative) obtenue par la troncature SVD.
    """
    # 1. Remplacer les NaN dans la matrice d'entraînement
    M_filled = replaceNA_fn(M_train)
    
    # 2. Calculer la décomposition SVD de la matrice remplie
    #    full_matrices=False permet de réduire la taille des matrices U et Vt
    U, s, Vt = np.linalg.svd(M_filled, full_matrices=False)
    
    # 3. Conserver uniquement les k premières valeurs et vecteurs singuliers
    U_k = U[:, :k]        # Première k colonnes de U
    s_k = s[:k]           # k premières valeurs singulières
    Vt_k = Vt[:k, :]      # Première k lignes de Vt
    
    # 4. Reconstruire la matrice en utilisant les k premières composantes
    #    np.diag(s_k) crée une matrice diagonale avec s_k sur la diagonale
    M_completed = np.dot(np.dot(U_k, np.diag(s_k)), Vt_k)
    
    return M_completed


##============================================
## svd.recommend(M_train, id_user, new=True, k=10, replaceNA_fn)
##============================================
def recommend(M_train, id_user, new=True, k=10, replaceNA_fn=replaceNA_with_zeros):
    """
    Prédit la note d'un utilisateur pour un item en se basant sur la complétion par SVD.
    
    Étapes :
    1. Compléter la matrice M_train avec la fonction complete() en utilisant k valeurs singulières.
    2. Pour un utilisateur donné (id_user) :
       - Si new est True, rechercher parmi les items non évalués par cet utilisateur
         (les NaN dans la ligne correspondante) et retourner l'item ayant la note prédite la plus élevée.
       - Si new est False, retourner simplement l'indice de l'item ayant la note prédite la plus élevée
         sur l'ensemble de la matrice.
    
    Paramètres :
    - M_train     : matrice d'entraînement (avec des NaN pour les notes manquantes)
    - id_user     : indice de l'utilisateur pour lequel on souhaite faire une recommandation
    - new         : booléen, si True, recommander un item non évalué par l'utilisateur
    - k           : nombre de valeurs singulières à utiliser dans la décomposition SVD
    - replaceNA_fn: fonction pour remplacer les NaN (par défaut replaceNA_with_zeros)
    
    Retourne :
    - Si new est True, l'indice de l'item non évalué par l'utilisateur avec la note prédite la plus élevée.
    - Sinon, l'indice de l'item avec la note prédite la plus élevée dans l'ensemble de la matrice.
    """
    # 1. Compléter la matrice avec la factorisation SVD
    M_pred = complete(M_train, k, replaceNA_fn)
    
    if new:
        # Trouver les indices des items que l'utilisateur n'a pas encore notés
        inds_unknown = np.where(np.isnan(M_train[id_user, :]))[0]
        
        # Vérifier s'il existe des items non évalués
        if len(inds_unknown) == 0:
            print("L'utilisateur a déjà noté tous les items.")
            return None
        
        # Parmi ces items, sélectionner celui avec la note prédite la plus élevée
        rec_ind_in_unknown = np.nanargmax(M_pred[id_user, inds_unknown])
        recommended_item = inds_unknown[rec_ind_in_unknown]
    else:
        # Recommander l'item avec la note prédite la plus élevée dans toute la matrice
        recommended_item = np.nanargmax(M_pred)
    
    return recommended_item
