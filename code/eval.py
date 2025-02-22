##============================================
##============================================
## fonctions utiles à l'évaluation des algorithmes
##============================================
##============================================

# * get_train_val(M, prop=0_8)
# * RMSE(M_completed, M_star)
# * quantitative_comparison(scoring_fn, M_star, recommmenders, prop=0_8, nrep=10)



import numpy as np
from time import time
import pandas as pd

##============================================
## get_train_val(M, prop=0_8)
##============================================
def get_train_val(M, prop=0.8):
  n, m = M.shape
  M_train = np.nan * np.ones((n, m), dtype=float)
  M_validation = M.copy()
  
  for id_user in range(n):
    inds_star = np.where(~np.isnan(M[id_user, :]))[0]
    if len(inds_star)==1:
      inds = inds_star
    else:
      inds = np.random.choice(inds_star, max(1, int(prop*len(inds_star))), replace=False)
    M_train[id_user, inds] = M[id_user, inds]
    M_validation[id_user, inds] = np.nan
  
  return (M_train, M_validation)


import numpy as np

#############################
# 1. MAE (Mean Absolute Error)
#############################
def MAE(M_completed, M_star):
    """
    Calcule le MAE (Mean Absolute Error) entre les valeurs prédites (M_completed)
    et les valeurs réelles (M_star).

    Protocole d'évaluation :
    - On considère uniquement les cases où M_star n'est pas NaN (c'est-à-dire
      les valeurs réellement observées dans le set de validation).
    - Pour chacune de ces cases, on calcule la valeur absolue de la différence
      entre la prédiction et la valeur réelle.
    - Le MAE est la moyenne de ces différences absolues.
    
    Paramètres :
    - M_completed : Matrice complétée/predite.
    - M_star      : Matrice de référence (validation) contenant les vraies notes.

    Retourne :
    - Le MAE, c'est-à-dire l'erreur absolue moyenne.
    """
    inds = ~np.isnan(M_star)
    return np.mean(np.abs(M_completed[inds] - M_star[inds]))


#############################
# 2. Recall et Precision
#############################
def Precision(M_completed, M_star, n_recommendations=5, threshold=4.0):
    """
    Calcule la Precision moyenne sur l'ensemble des utilisateurs.
    
    Protocole d'évaluation :
    - Pour chaque utilisateur, on définit les items "pertinents" dans le set de validation 
      comme ceux dont la note réelle (M_star) est supérieure ou égale au seuil (threshold).
    - On génère une liste de n_recommendations items en triant les prédictions (M_completed)
      de l'utilisateur par ordre décroissant.
    - La Precision pour un utilisateur est le rapport du nombre d'items pertinents 
      parmi les n_recommendations sur n_recommendations.
    - La Precision globale est la moyenne sur tous les utilisateurs (ceux pour lesquels 
      il existe au moins un item pertinent dans M_star).

    Paramètres :
    - M_completed       : Matrice prédite par le système.
    - M_star            : Matrice de validation contenant les vraies notes.
    - n_recommendations : Nombre d'items recommandés par utilisateur.
    - threshold         : Seuil de note pour considérer qu'un item est pertinent.

    Retourne :
    - La Precision moyenne.
    """
    n_users = M_star.shape[0]
    precision_sum = 0.0
    count_users = 0
    
    for i in range(n_users):
        # Déterminer les items pertinents pour l'utilisateur i
        # (seules les valeurs non NaN dans M_star sont prises en compte)
        relevant = np.where((~np.isnan(M_star[i, :])) & (M_star[i, :] >= threshold))[0]
        if len(relevant) == 0:
            continue  # on ignore cet utilisateur s'il n'a aucun item pertinent
        
        # Sélectionner les n_recommendations items ayant les plus hautes prédictions
        recommended = np.argsort(M_completed[i, :])[-n_recommendations:]
        
        # Calculer le nombre d'items recommandés qui sont pertinents
        hits = np.intersect1d(recommended, relevant)
        precision_sum += len(hits) / float(n_recommendations)
        count_users += 1
    
    return precision_sum / count_users if count_users > 0 else 0.0


def Recall(M_completed, M_star, n_recommendations=5, threshold=4.0):
    """
    Calcule le Recall moyen sur l'ensemble des utilisateurs.
    
    Protocole d'évaluation :
    - Pour chaque utilisateur, on définit les items "pertinents" dans le set de validation 
      comme ceux dont la note réelle (M_star) est supérieure ou égale au seuil (threshold).
    - On génère une liste de n_recommendations items en triant les prédictions (M_completed)
      de l'utilisateur par ordre décroissant.
    - Le Recall pour un utilisateur est le rapport du nombre d'items pertinents 
      recommandés sur le nombre total d'items pertinents dans M_star.
    - Le Recall global est la moyenne sur tous les utilisateurs (ceux pour lesquels 
      il existe au moins un item pertinent dans M_star).

    Paramètres :
    - M_completed       : Matrice prédite par le système.
    - M_star            : Matrice de validation contenant les vraies notes.
    - n_recommendations : Nombre d'items recommandés par utilisateur.
    - threshold         : Seuil de note pour considérer qu'un item est pertinent.

    Retourne :
    - Le Recall moyen.
    """
    n_users = M_star.shape[0]
    recall_sum = 0.0
    count_users = 0
    
    for i in range(n_users):
        # Déterminer les items pertinents pour l'utilisateur i
        relevant = np.where((~np.isnan(M_star[i, :])) & (M_star[i, :] >= threshold))[0]
        if len(relevant) == 0:
            continue
        
        # Sélectionner les n_recommendations items ayant les plus hautes prédictions
        recommended = np.argsort(M_completed[i, :])[-n_recommendations:]
        
        # Calculer le nombre d'items pertinents recommandés
        hits = np.intersect1d(recommended, relevant)
        recall_sum += len(hits) / float(len(relevant))
        count_users += 1
    
    return recall_sum / count_users if count_users > 0 else 0.0


##============================================
## RMSE(M_completed, M_star)
##============================================
def RMSE(M_completed, M_star):
  inds = ~np.isnan(M_star)
  return np.sqrt(np.mean((M_completed[inds] - M_star[inds])**2))



##============================================
## quantitative_comparison(scoring_fn, M_star, recommmenders, prop=0_8, nrep=10)
##============================================

def quantitative_comparison(scoring_fn, M_star, recommenders, prop=0.8, nrep=10):
  scores = np.zeros((len(recommenders), nrep))
  scores_train = np.zeros((len(recommenders), nrep))
  computation_time = np.zeros((len(recommenders), nrep))
  for id_rep in range(nrep):
    M_train, M_validation = get_train_val(M_star, prop)
    for id_rec in range(len(recommenders)):
      ptm = time()
      M_completed = recommenders[id_rec]['fn'](M_train)
      computation_time[id_rec, id_rep] = (time() - ptm)
      scores[id_rec, id_rep] = scoring_fn(M_completed, M_validation)
      scores_train[id_rec, id_rep] = scoring_fn(M_completed, M_train)


  return pd.DataFrame({
          'recommender': [rec['label'] for rec in recommenders],
          'validation score': np.mean(scores, axis=1),
          'training score': np.mean(scores_train, axis=1),
          'computation time': np.mean(computation_time, axis=1)
          })



