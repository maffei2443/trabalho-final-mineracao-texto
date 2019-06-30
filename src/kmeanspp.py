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

def sorteio(probab_vet):
  choosed = rand()
  print("Rand: ", choosed)
  accumulated_probab = np.add.accumulate(probab_vet)
  print("accumulated: ", accumulated_probab)
  return np.nonzero( accumulated_probab >= choosed )[0][0]

def sorteio_opt(probab_vet):
  """Retorna um índice no intervalo [0, len(probab_vet) ),
  a partir do sorteio baseado em probabilidade acumulada (soma de prefixo).
  """
  return np.nonzero(
    np.add.accumulate(probab_vet) >= rand()
  )[0][0]

# TODO : deixar apenas um vetor que representa Dx, atualizando
class KMeansPP:
  """Classe para calcular centróides iniciais do utilizados
  pelo algoritmo kmeans++.

  """
  def __init__(self, k, data: csr_matrix):
    self.k = k
    if issparse(data) or isinstance(data, np.ndarray):
      try:  # Sempre que pssível, usar arrays pois são absurdamente mas rápidos para operar sobre
        self._data = data.toarray()
        self.isSparse = False
      except BaseException:
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
    """Retorna o índice do próximo centro de um cluster, seguindo a descrição do k-means++."""
    self._minDistanceToNearestCentroid = np.minimum(
      self._minDistanceToNearestCentroid,
      LA.norm(
        self._data - self._data[self._centroidsIndex[self._computedCentroids-1]],
        axis = 1
      )
    )
    nextClusterIndex = sorteio_opt( (self._minDistanceToNearestCentroid / np.sum(self._minDistanceToNearestCentroid))[0] )
    self._centroidsIndex[self._computedCentroids] = nextClusterIndex
    self._computedCentroids += 1
  def __computeAndSetNextCentroidIndex_Sparse(self):
    """Retorna o índice do próximo centro de um cluster, seguindo a descrição do k-means++."""
    self._minDistanceToNearestCentroid = np.minimum(
      self._minDistanceToNearestCentroid,
      LA.norm(
        self._data - self._data[self._centroidsIndex[self._computedCentroids-1]],
        axis = 1
      )
    )
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

  def getCentroids(self):
    return np.array([ self._data[i] for i in self._centroidsIndex ])
  def getCentroidsIndex(self):
    return self._centroidsIndex
def main():
  pass

if __name__ == '__main__':
  main()