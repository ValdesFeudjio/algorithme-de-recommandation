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



