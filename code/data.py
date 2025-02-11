""" Préparation des données
suppose que les fichiers de données sont stocqués dans le répertoire "./data"

Fournit
-------
* load.data(tiny=False) : retourne une sous-matrice de la matrice de scores ML100k (de taille 500x400 ou 50x40 suivant `tiny`).
* movie.title(id) : retourne le titre du film d'index `id`

Exemples
--------

>>> data = load_data(tiny=True)
>>> data.shape
(50, 40)
>>> data[:5, :5]
array([[nan,  5.,  3.,  4.,  3.],
       [nan,  4., nan, nan, nan],
       [nan, nan, nan, nan, nan],
       [nan,  4.,  3., nan, nan],
       [nan,  4., nan, nan, nan]])
>>> movie_title(42)
'Disclosure (1994)'
"""


##============================================
# bibliothèques utiles
##============================================
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


##============================================
# load.data
##============================================
def load_data(tiny=False):
  # lecture des données brut
  data = pd.read_csv(r"..\data\u.data", names=["user.id", "movie.id", "rate", "date"], sep='\t')

  # mise sous forme de matice
  n_user = max(data['user.id'])+1
  n_movie = max(data['movie.id'])+1
  rate = csr_matrix((data['rate'].astype(float), (data['user.id'], data['movie.id'])), shape=(n_user, n_movie)).toarray()
  rate[rate==0] = np.nan
  
  # réduction
  if tiny:
    rate = rate[:, :40]
    ind_user = np.where(np.sum(~np.isnan(rate), axis=1) != 0)[0][:50]
    rate = rate[ind_user, :]
  else:
    rate = rate[:, :400]
    ind_user = np.where(np.sum(~np.isnan(rate), axis=1) != 0)[0][:500]
    rate = rate[ind_user, :]
  
  return rate

##============================================
# movie.title(id)
##============================================
def movie_title(id):
    data = pd.read_csv("data/u.item", names=["id", "title", "release.date", "video.release.date", "unknown", "Action", "Adventure", "Animation", "Children.s", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film.Noir", "Horror", "Musical", "Mystery", "Romance", "Sci.Fi", "Thriller", "War", "Western"], sep='|', encoding='latin1')
    try:
      return [data['title'][i] for i in id]
    except TypeError as e:
      if str(e).endswith('object is not iterable'):
        return str(data['title'][id])
      else:
        raise e



