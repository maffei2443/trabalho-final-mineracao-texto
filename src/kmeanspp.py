import numpy as np
import operator
from numpy.random import rand
from numpy.random import random_integers
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
        np.array(probab_vet) >= rand()
  ))[0][0]

class KMeansPP:
  def __init__(self, k, data):
    self.k = k
    self._data = data
    self._dataLen = len(data)
    self._lastCentroidIndex = None
    self._centroidsIndex = np.empty( k )
    self._centroids = np.empty( self._dataLen )
    self._computedCentroids = 0
    self._probab = np.full( (k, self._dataLen), -np.inf )
    self._distanceToCentroid = np.full( (k, self._dataLen), np.inf ) # ditancias até qqer centroid inicialmente é infinita
  def InitCentroids(self):
    """Seta a lista de centroids iniciais utilizando o procedimento descrito no artigo original do k-means++.
    """
    if not self._lastCentroidIndex:
      self._lastCentroidIndex = random_integers(len(self._data)) - 1
      
      self._centroidsIndex[0] = self._lastCentroidIndex
      self._centroids[0] = self._data[self._lastCentroidIndex]

      self._computedCentroids += 1
      # Generate the others centroids
      for idx in range(1,self.k):
        pos = self.computeNextCentroid()
        self._centroidsIndex[idx] = pos
        self._centroids[pos] = self._data[pos]


  def computeNextCentroid(self):
    """Computes distances from all points to last centroid"""
    self._distanceToCentroid[self._computedCentroids-1] = LA.norm(
      self._data - self._data[self._lastCentroidIndex],
      axis = 1
    )
    vetDx = self._distanceToCentroid.min(axis = 1)  # mihnima distancia
    self._probab[self._computedCentroids] = vetDx / np.sum(vetDx)
    nextClusterIndex = sorteio_opt(self._probab)
    while nextClusterIndex in self._centroidsIndex:  # Garante que n terá dois centroides sobrepostos
      nextClusterIndex = sorteio_opt(self._probab)
    self._computedCentroids += 1
    return nextClusterIndex
