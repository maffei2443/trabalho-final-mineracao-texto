import sys


tests = [
    ('k-means++', 'k-means++'),
    ('caseiro', None),
    ('k-means', 'random'),
]

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

dirr = ''

def next(nome: str, results: dict, n_clusters=5, repeat = 7, init_mode=None, verbose=True):
  if not results.get(nome):
    results[nome] = []
  for _ in range(repeat):
    results[nome].append( KMeans(n_clusters=n_clusters, init=quick_init(n_clusters) if not init_mode else init_mode, n_init=1, verbose=verbose) )
    results[nome][-1].fit(transformedData)
    results[nome][-1] = (results[nome][-1].inertia_, results[nome][-1].n_iter_)



def run_bateria(dic, to_test = tests, verbose=False):
  n_cluster, repeat = dic['meta']['n_cluster'], dic['meta']['repeat']
  for nome, init_mode in to_test:
    next(nome, dic, n_clusters=n_cluster, repeat=repeat, init_mode=init_mode, verbose=verbose)

def show_bateria(dic):
  print("meta: ", *dic['meta'])
  for k, v in dic.items():
    if k != 'meta':
      print(k)
      print(*sorted(v), sep='\n', end='\n\n\n')

def run_and_dump(repeat, n_clusters, verbose=False, prefixo='bateria'):
  global dirr
  glb = globals()
  nome_da_bateria = f'corrected_{repeat}rep_{n_clusters}nclust'
  d = glb[nome_da_bateria] = {}

  d['meta'] = {
      'n_cluster': n_clusters,
      'repeat': repeat,
      'step': 1,
      'categories': 'all',
      'subset': suset
    }
  print(d['meta'])
  run_bateria(d, tests, verbose)
  print("dumped: ", nome_da_bateria)
  pickle.dump(d, open(dirr+nome_da_bateria, 'wb'))
  show_bateria(d)


def range_cluster_run_dump(repeat, n_min, n_max, stp, verbose=False):
  try:
    os.mkdir(dirr)
  except:
      if 'n' in input('Diretorio exitente. Prosseguir mesmo assim?').lower():
        return None
  for n_cluster in range(n_min, n_max+1, stp):
    run_and_dump(repeat, n_cluster, verbose)


def main():
    global dirr
    rep, min_clust, max_clust, dirr = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    if not dirr.endswith('/'):
        dirr += '/'
    # range_cluster_run_dump(10, 3, 8, 1)
    # range_cluster_run_dump(10, 9, 12, 1)
    # range_cluster_run_dump(10, 13, 18, 1)
    range_cluster_run_dump(rep, min_clust, max_clust, 1)
    # range_cluster_run_dump(1, 1, 1, 1)

if __name__== '__main__' and len(sys.argv) >= 5:
    import os
    import numpy as np
    import operator
    from numpy.random import rand
    from numpy.random import randint
    from itertools import accumulate
    from numpy import linalg as LA  # norma vetor
    # from numpy.clusters import KMeans
    # Add suporte à matriz esparsa como inicialização dos dados
    import scipy
    from scipy.sparse import csr_matrix
    from scipy.sparse import issparse
    # # # # # # #
    from time import time
    import pickle

    import kmeanspp

    from sklearn.datasets import fetch_20newsgroups
    import kmeanspp
    from importlib import reload as rl
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans



    # suset = 'train' or 'all'
    suset = 'all' or 'train'
    dataset = fetch_20newsgroups(subset=suset, shuffle=True, random_state=42)
    if len(sys.argv) > 5:
      import pre_processor
      print('PRE  PROCESSOR!!!')
      transformedData = TfidfVectorizer().fit_transform( [pre_processor.pre_process_lower(x) for x in dataset.data] )
    else:
      transformedData = TfidfVectorizer().fit_transform(dataset.data)
    quick_init = lambda n_clusters = 7: kmeanspp.KMeansPP(n_clusters, transformedData).fit().getCentroids()
    main()

else:
    print("Should contain 5+ args: repetitions, min_clust, max_clust e o diretório. Em caso de conflito de arquivo, será sobrescrito. Em caso de mais argumentos, usa pre_preocessor")