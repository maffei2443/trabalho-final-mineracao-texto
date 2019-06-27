import numpy as np
import operator
from numpy.random import rand
from numpy.random import randint
from itertools import accumulate
from numpy import linalg as LA  # norma vetor

# Add suporte à matriz esparsa como inicialização dos dados
import scipy
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
  print("probab: ", probab_vet)
  return np.nonzero(
    np.add.accumulate(probab_vet) >= rand()
  )[0][0]

# TODO : deixar apenas um vetor que representa Dx, atualizando
class KMeansPP:
  def __init__(self, k, data):
    self.k = k
    self.isSparseCsr = isinstance(data, scipy.sparse.csr.csr_matrix)
    if self.isSparseCsr:
      self._data = scipy.sparse.csr.csr_matrix.toarray(data)
      self._dataLen = dataLen = data.shape[0]
    else:
      self._data = data
      self._dataLen = dataLen = len(data)
    self._centroidsIndex = np.zeros( k, dtype=np.uint16 )
    self._computedCentroids = 0
    self._probab = np.full( (k, dataLen), -np.inf )
    self._minDistanceToNearestCentroid = np.full( (1, dataLen), np.inf, dtype=np.float64 ) # ditancias até qqer centroid inicialmente é infinita
  def __reset(self, k, data):
    self.__init__(k, data)    
  def __distDebug(self):
    print("LastIndex: ", self._centroidsIndex[self._computedCentroids-1])
    print("distancia: ", self._minDistanceToNearestCentroid)
    print("candidato: ",       LA.norm(
        self._data - self._data[self._centroidsIndex[self._computedCentroids-1]],
        axis = 1
      ))

  def __computeAndSetFirstCentroid(self):
    # print("__computeAndSetFirstCentroid")
    """Sorteia um número no inervalo [0, tamanho(vetor_de_dados) e seta o primeiro centróide para tal.
    
    Como efeito colateral, altera o contador de centróides computados, bem como
    modifica os arrays de índices de centróides e de centróides.
    """
    # index = randint(self._dataLen)
    # print("First Centroid: ", index)
    self._centroidsIndex[0] = randint(self._dataLen)
    self._computedCentroids += 1

  def __computeNextCentroid(self):
    # print("__computeNextCentroid")
    """Retorna o índice do próximo centro de um cluster, seguindo a descrição do k-means++."""
    self.__distDebug()
    self._minDistanceToNearestCentroid = np.minimum(
      self._minDistanceToNearestCentroid,
      LA.norm(
        self._data - self._data[self._centroidsIndex[self._computedCentroids-1]],
        axis = 1
      )
    )
    # self._probab[self._computedCentroids] = (self._minDistanceToNearestCentroid / np.sum(self._minDistanceToNearestCentroid))
    # nextClusterIndex = sorteio_opt( self._probab[self._computedCentroids] )
    nextClusterIndex = sorteio_opt( (self._minDistanceToNearestCentroid / np.sum(self._minDistanceToNearestCentroid))[0] )
    # nextClusterIndex = sorteio_opt( self._minDistanceToNearestCentroid / np.sum(self._minDistanceToNearestCentroid) )  # Por algum motivo, n funciona direto assim.
    # while nextClusterIndex in self._centroidsIndex:  # Garante que n terá dois centroides sobrepostos. Questões de precisão do float
    #   nextClusterIndex = sorteio_opt(self._probab[self._computedCentroids])
    self._computedCentroids += 1
    return nextClusterIndex

  def __computeNextCentroidSparseData(self):
    # print("__computeNextCentroidSparseData")
    """Retorna o índice do próximo centro de um cluster, seguindo a descrição do k-means++."""
    self.__distDebug()
    self._minDistanceToNearestCentroid = np.minimum(
      self._minDistanceToNearestCentroid,
      LA.norm(
        self._data - self._data[self._centroidsIndex[self._computedCentroids-1]],
        axis = 1
      )
    )
    # self._probab[self._computedCentroids] = (self._minDistanceToNearestCentroid / np.sum(self._minDistanceToNearestCentroid))
    # nextClusterIndex = sorteio_opt( self._probab[self._computedCentroids] )
    nextClusterIndex = sorteio_opt( (self._minDistanceToNearestCentroid / np.sum(self._minDistanceToNearestCentroid))[0] )
    # nextClusterIndex = sorteio_opt( self._minDistanceToNearestCentroid / np.sum(self._minDistanceToNearestCentroid) )  # Por algum motivo, n funciona direto assim.
    # while nextClusterIndex in self._centroidsIndex:  # Garante que n terá dois centroides sobrepostos. Questões de precisão do float
    #   nextClusterIndex = sorteio_opt(self._probab[self._computedCentroids])
    self._computedCentroids += 1
    return nextClusterIndex

  def __InitCentroids(self):
    # print("__InitCentroids")
    """Seta a lista de centroids iniciais utilizando o procedimento descrito no artigo original do k-means++.
    """
    self.__computeAndSetFirstCentroid()
    # Generate the others centroids
    for idx in range(1,self.k):
      self._centroidsIndex[idx] = self.__computeNextCentroid()

  def __InitCentroidsSparseData(self):
    # print("__InitCentroids")
    """Seta a lista de centroids iniciais utilizando o procedimento descrito no artigo original do k-means++.
    """
    self.__computeAndSetFirstCentroid()
    # Generate the others centroids
    for idx in range(1,self.k):
      self._centroidsIndex[idx] = self.__computeNextCentroidSparseData()

  def fit(self):
    if not self.isSparseCsr:
      self.__InitCentroids()
    else:
      self.__InitCentroidsSparseData()
  def getCentroids(self):
    return np.array([ self._data[i] for i in self._centroidsIndex ])

def main():
  pass

if __name__ == '__main__':
  main()