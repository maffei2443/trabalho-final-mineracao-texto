import numpy as np
import operator
from numpy.random import rand
from numpy.random import randint
from itertools import accumulate
from numpy import linalg as LA  # norma vetor

# Add suporte à matriz esparsa como inicialização dos dados
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
# # # # # # #
from time import time
import pickle
# def sorteio(probab_vet):
#   choosed = rand()
#   print("Rand: ", choosed)
#   accumulated_probab = np.add.accumulate(probab_vet)
#   print("accumulated: ", accumulated_probab)
#   return np.nonzero( accumulated_probab >= choosed )[0][0]

# def sorteio_opt(probab_vet):
#   """Retorna um índice no intervalo [0, len(probab_vet) ),
#   a partir do sorteio baseado em probabilidade acumulada (soma de prefixo).
#   """
#   return np.nonzero(
#     np.add.accumulate(probab_vet) >= rand()
#   )[0][0]

# def sorteio_opt(probab_vet):
#   """Retorna um índice no intervalo [0, len(probab_vet) ),
#   a partir do sorteio baseado em probabilidade acumulada (soma de prefixo).
#   """
#   # return randint(0, 200)
#   prefix_sum = np.add.accumulate(probab_vet)
#   min_idx = np.searchsorted(prefix_sum, 1.0)
#   print(prefix_sum)
#   return randint(min_idx, len(probab_vet)+1)

def sorteio_opt(probab_vet):
  return  np.searchsorted(
        np.add.accumulate(probab_vet), rand() # Com 1.0, por conta de erro de precisao, muitas vezes dava erro
      )

def probab_choice(probab_vet, granularity = 1_000_000_000):
  probab_int = (np.asarray(probab_vet) * granularity).astype(int)
  mapa = np.dstack( (probab_int, range(len(probab_int))) )[0]
  np.random.shuffle(mapa)
  return (
    probab_int[mapa[(np
      .searchsorted( 
        np.cumsum(mapa[:, 0]) , randint(0, granularity)
      ) % len(probab_int))][1] ]
  )

mu, sigma = 0, 0.5 
s = np.random.normal(mu, sigma, 1000000) 

def theHellSorteio(probab_vet):
  i = s[randint(len(probab_vet))]
  if i < 0:
    i = -i
  if i > 1:
    i -= 1
  return np.searchsorted(
        np.add.accumulate(probab_vet), i # Com 1.0, por conta de erro de precisao, muitas vezes dava erro
      )

def simple_sub_rev(data, idx = 0):
    return [*data] - np.array([data[idx]])

normSparseMatrix = lambda x: np.linalg.norm(x.data)
normSparseMatrixRight = lambda x: scipy.sparse.linalg.norm(x)

vect_normSparseMatrix = np.frompyfunc(normSparseMatrixRight, 1, 1)
vect_normSparseMatrixRight = np.frompyfunc(normSparseMatrix, 1, 1)

def getNormsOfSparseMatrixInsideArray(arr):
  return vect_normSparseMatrix(arr)

def getNormsOfSparseMatrixInsideArrayRight(arr):
  return vect_normSparseMatrixRight(arr)


# sorteio_opt = theHellSorteio
# sorteio_opt = theHellSorteio

def toArray(data):
  glb = globals()
  if hasattr(data, 'toarray'):
    glb['toArray'] = lambda x: x.toarray(x)
  else:
    glb['toArray'] = lambda x: np.array(x)
  toArray = glb['toArray']
  return toArray(data)


# TODO : deixar apenas um vetor que representa Dx, atualizando
class KMeansPP:
  """Classe para calcular centróides iniciais do utilizados
  pelo algoritmo kmeans++.

  """


  def __init__(self, k, data: csr_matrix):
    self.__spent_norm_diff = 0
    self.k = k
    if issparse(data) or isinstance(data, np.ndarray):
      self.__toArray = lambda x: x.toarray() if hasattr(data[0], 'toarray') else lambda x: np.array(x)
      try:  # Sempre que pssível, usar arrays pois são absurdamente mas rápidos para operar sobre
        tmp = data.toarray()
        tmp - toArray(tmp[0]) # testar se cabe na memória
        self._data = tmp
        self.isSparse = False
      except BaseException as e:
        print("Warning: sparse data", e)
        self._data = data
        self.isSparse = True
      self._dataLen = dataLen = data.shape[0]
    elif isinstance(data, list) or isinstance(data, tuple):
      self._data = np.array(data)
      self._dataLen = dataLen = len(data)
      self.isSparse = False
    else:
      raise TypeError("""The 'data' argument must be one of the following wypes:
      <scipy_sparse_matrix>, np.ndarray, list os tuple""")
    if not 0 < self.k <= self._dataLen:
      raise ValueError("The 'k' number of centers must be in the range [1, data_sample]")
    self._centroidsIndex = np.zeros( k, dtype=np.uint16 )
    self._computedCentroids = 0
    self._probab = np.full( (k, dataLen), -np.inf )
    self._minDistanceToNearestCentroid = np.full( (1, dataLen), np.inf, dtype=np.float64 ) # ditancias até qqer centroid inicialmente é infinita
  def __reset(self, k, data=None):
    data = self._data if not data else data
    self.__init__(k, data)    
  def __distDebug(self):
    print("LastIndex: ", self._centroidsIndex[self._computedCentroids-1])
    print("distancia: ", self._minDistanceToNearestCentroid)
    print("candidato: ",       LA.norm(
        self._data - self._data[self._centroidsIndex[self._computedCentroids-1]],
        axis = 1
      ))

  def __computeAndSetFirstCentroid(self):
    """Sorteia um número no inervalo [0, tamanho(vetor_de_dados) e seta o primeiro centróide para tal.
    
    Como efeito colateral, altera o contador de centróides computados, bem como
    modifica os arrays de índices de centróides e de centróides.
    """
    self._centroidsIndex[0] = randint(self._dataLen)
    self._computedCentroids += 1
  def __computeAndSetNextCentroidIndex(self):
    t0 = time()
    """Retorna o índice do próximo centro de um cluster, seguindo a descrição do k-means++."""
    lastIndex = self._centroidsIndex[self._computedCentroids-1]
    # t0 = time()
    self._minDistanceToNearestCentroid = np.minimum(
      self._minDistanceToNearestCentroid,
      LA.norm(
        self._data - self._data[lastIndex],
        axis = 1
      )
    )
    self.__spent_norm_diff += (time() - t0)
    # print("diffsized")
    # print(f"Time: {dt1}")
    # t0 = time()
    nextClusterIndex = sorteio_opt( (self._minDistanceToNearestCentroid / np.sum(self._minDistanceToNearestCentroid))[0] )
    # dt2 = time() - t0
    # print(f"Time: {dt2}")
    # raise BaseException(f"dt1: {dt1}, dt2: {dt2}")


    self._centroidsIndex[self._computedCentroids] = nextClusterIndex
    self._computedCentroids += 1
  def __computeAndSetNextCentroidIndex_Sparse(self):
    """Retorna o índice do próximo centro de um cluster, seguindo a descrição do k-means++."""
    t0 = time()
    self._minDistanceToNearestCentroid = np.minimum(
      self._minDistanceToNearestCentroid,
      getNormsOfSparseMatrixInsideArrayRight(
        simple_sub_rev(self._data, [self._centroidsIndex[self._computedCentroids-1]])
      )
    )
    print("(sub + norm) time : ", time() - t0)
    # raise BaseException("fsdafafd")
    nextClusterIndex = sorteio_opt( (self._minDistanceToNearestCentroid / np.sum(self._minDistanceToNearestCentroid))[0] )
    self._centroidsIndex[self._computedCentroids] = nextClusterIndex
    self._computedCentroids += 1

  def __InitCentroids(self):
    # print("__InitCentroids")
    """Seta a lista de centroids iniciais utilizando o procedimento descrito no artigo original do k-means++.
    """
    self.__computeAndSetFirstCentroid()
    # Generate the others centroids
    for _ in range(1,self.k):
      self.__computeAndSetNextCentroidIndex()
  def __InitCentroids_Sparse(self):
    # print("__InitCentroids")
    """Seta a lista de centroids iniciais utilizando o procedimento descrito no artigo original do k-means++.
    """
    self.__computeAndSetFirstCentroid()
    # Generate the others centroids
    for _ in range(1,self.k):
      self.__computeAndSetNextCentroidIndex_Sparse()

  def fit(self):
    if not self.isSparse:
      self.__InitCentroids()
    else:
      self.__InitCentroids_Sparse()
    print("Total spent time on diff_norm: ", self.__spent_norm_diff)
    return self
  f = np.frompyfunc(lambda x: x.toarray(), 1, 1)
  def getCentroids(self):
    if self.isSparse:
      return KMeansPP.f(self._data[self._centroidsIndex])
    else:
      return self._data[self._centroidsIndex]
  def getCentroidsIndex(self):
    return self._centroidsIndex
def main():
  pass

if __name__ == '__main__':
  main()

