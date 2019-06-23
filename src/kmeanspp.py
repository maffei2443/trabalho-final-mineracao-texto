import numpy as np
import operator
from numpy.random import rand
from numpy.random import randint
from itertools import accumulate
from numpy import linalg as LA  # norma vetor

def sorteio(probab_vet):
  choosed = rand()
  print("Rand: ", choosed)
  accumulated_probab = np.add.accumulate(probab_vet)
  return np.nonzero( accumulated_probab >= choosed )[0][0]

def sorteio_opt(probab_vet):
  return np.nonzero(
    np.add.accumulate( 
        np.array(probab_vet, copy=False) >= rand()
  ))[0][0]

class KMeansPP:
  def __init__(self, k, data):
    self.k = k
    self._data = data
    self._dataLen = dataLen= len(data)
    self._centroidsIndex = np.empty( k, dtype=np.uint16 )
    self._centroids = np.empty( dataLen, dtype=(np.float64, len(self._data[0])))
    self._computedCentroids = 0
    self._probab = np.full( (k, dataLen), -np.inf )
    self._distanceToCentroid = np.full( (k, dataLen), np.inf, dtype=np.float64 ) # ditancias até qqer centroid inicialmente é infinita
  def _computeAndSetFirstCentroid(self):
    """Sorteia um número no inervalo [0, tamanho(vetor_de_dados) e seta o primeiro centróide para tal.
    
    Como efeito colateral, altera o contador de centróides computados, bem como
    modifica os arrays de índices de centróides e de centróides.
    """
    index = randint(self._dataLen)
    self._centroidsIndex[0] = index
    self._centroids[0] = self._data[index]
    self._computedCentroids += 1

  def InitCentroids(self):
    """Seta a lista de centroids iniciais utilizando o procedimento descrito no artigo original do k-means++.
    """
    if not self._computedCentroids:
      self._computeAndSetFirstCentroid()
      # Generate the others centroids
      for idx in range(1,self.k):
        pos = self.computeNextCentroid()
        self._centroidsIndex[idx] = pos
        self._centroids[pos] = self._data[pos]

  def computeNextCentroid(self):
    """Computes distances from all points to last centroid"""
    self._distanceToCentroid[self._computedCentroids-1] = LA.norm(
      self._data - self._data[self._centroidsIndex[self._computedCentroids-1]],
      axis = 1
    )
    vetDx = self._distanceToCentroid.min(axis = 1)  # mihnima distancia
    self._probab[self._computedCentroids] = vetDx / np.sum(vetDx)
    nextClusterIndex = sorteio_opt(self._probab)
    while nextClusterIndex in self._centroidsIndex:  # Garante que n terá dois centroides sobrepostos
      nextClusterIndex = sorteio_opt(self._probab)
    self._computedCentroids += 1
    return nextClusterIndex
