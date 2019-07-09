import re
import numpy as np

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.datasets.samples_generator import make_blobs

import sys
import os
import kmeanspp

from aux import MyPlot, DIRR
from numpy.random import randint
from matplotlib import pyplot as plt
import glob
BUFFER_FILE = '__dont_use_a_file_like_this__'
def ParseInertia(path):
    """"O arquivo em 'path' deve ter os dados de inércia
    ao longo das iterações da execução do kmeans."""
    lines = open(path).readlines() 
    raw = ''.join(lines).replace('\n', '').lower() 
    logs = raw.split('initialization complete') 
    inercias = [] 
    for log in logs: 
      aux = re.findall(r'(?<=inertia )(\d+[.]\d+)', log) 
      if aux: 
          inercias.append(np.array([float(i) for i in aux])) 
    return np.array(inercias)

def Mean(inercias) -> dict:
    """"Recebe np.array de np.array, tal que os elementos desse último são
    escalares. Retorna array com as média no i-ésimos elementos"""
    # print("type(Inercias): ", type(inercias))
    # print("type(Inercias[0]): ", [k.shape for k in inercias] )
    # print("Inercias.shape: ", inesrcias.shape)
    # print("Inercias[0]: ", inercias[0])
    tamanhos = np.array([k.shape for k in inercias])[:, 1]
    # print("tamanhos: ", tamanhos)
    max_size = max(tamanhos)
    min_size = min(tamanhos)
    inercias_filled = []
    for idx, i in enumerate(inercias):
        z = np.zeros(max_size)
        # print('z: ', z)
        # print(f'inercia[{idx}]: {i}')
        z[:len(i[0])] = i[0]
        inercias_filled.append(z)
    inercias_filled = np.array(inercias_filled)
    return {
        'mean': np.mean(inercias_filled, axis = 0),
        'max_iter': max_size,
        'min_iter': min_size}

def runSingleTest(n_clusters=0, max_iter=0, data=np.array(0), init=None):
    if not n_clusters or not max_iter or not data.any():
        raise ValueError("Nenhum parâmetro pode ser nulo: são todos obrigatórios.")
    sout = sys.stdout
    # TEVE DE SER FEITO DESSA MANEIRA POIS O Kmeans NAO ACEITA
    # ARQUIVO PARA GUARDAR O LOG DO MODO VERBOSO
    sys.stdout = open(BUFFER_FILE, 'w')
    km = KMeans(
        n_clusters=n_clusters, init=init, verbose=True,
        tol=1e-40, max_iter=max_iter, n_init=1)
    km.fit(data)
    sys.stdout.close()
    inercias = ParseInertia(BUFFER_FILE)
    
    sys.stdout = sout
    os.remove(BUFFER_FILE)
    return inercias

def runMultipleTest(n_clusters=0, max_iter=0, data=np.array(0), init=None, n_init=0, rep=30):
    return [runSingleTest(n_clusters=n_clusters, max_iter=max_iter, data=data, init=init) for _ in range(rep)]
def runMultipleCustomTest(n_clusters=0, max_iter=0, data=np.array(0), n_init=0, rep=30):
    global kmppDict
    res = []
    for _ in range(rep):
         res.append( runSingleTest(n_clusters=n_clusters, max_iter=max_iter, data=data, init=kmeanspp.Old( n_clusters, data ).fit().getCentroids() ) )
         kmppDict[n_clusters].UnFit()
    return res

def zeroPad(arr, num):
    print('num: ', num)
    if num > len(arr):
        zeros = np.zeros(num)
        zeros[:len(arr)] = arr[:]
        return zeros
    else:
        return arr

min_clusters = 3
max_clusters = 25 + 1
n_samples = 10000
random_state=randint(2**32)
# random_state=744592
cluster_std = 1.0

folder_result = DIRR + "{}_{}_seed{}_{}_std{}".format(min_clusters, max_clusters, random_state, n_samples, cluster_std)
try:
    os.mkdir(DIRR)
except:
    pass

if glob.glob(folder_result):
    if not input("Warning: já foi executada uma bateria com esses parâmetros. Prosseguirm mesmo assim?").lower().startswith('n'):
        pass
    else:
        raise BaseException("Abortar experimentos.")

    

max_iter = 10
data , _ = make_blobs(
    n_samples=n_samples,
    n_features=3,
    cluster_std=cluster_std,
    random_state=random_state
)

results = []
kmppDict = {}

# for n_clusters in range(min_clusters, max_clusters):
    
print("min_clusters: ", min_clusters)
print("max_clusters: ", max_clusters)
print("random_state: ", random_state)
print("n_samples: ", n_samples)



for n_clusters in range(min_clusters, max_clusters):
    kmppDict[n_clusters] = kmeanspp.Old( n_clusters, data )
    results.append( {} )
    results[-1]['random'] = Mean(runMultipleTest(n_clusters=n_clusters, max_iter=max_iter,data=data, n_init=1, init='random'))
    results[-1]['k-means++'] = Mean(runMultipleTest(n_clusters=n_clusters, max_iter=max_iter,data=data, n_init=1, init='k-means++'))
    results[-1]['new++'] = Mean(runMultipleCustomTest(n_clusters=n_clusters, max_iter=max_iter,data=data, n_init=1))
    results[-1]['n_clusters'] = n_clusters
    results[-1]['n_samples'] = n_samples

os.mkdir(folder_result)
print("Neww folder! Saving experiments on ===> {}".format(folder_result))
for r in results: 
    num = max([len(r['random']['mean']) , len(r['k-means++']['mean']), len(r['new++']['mean']) ] ) 
    title = f"{r['n_clusters']} clusters, {n_samples}_samples, {random_state}_seed"
    MyPlot(title, { 
            'rand': zeroPad(r['random']['mean'], num), 
            'new++': zeroPad(r['new++']['mean'], num), 
            'kmeans++':zeroPad(r['k-means++']['mean'], num), 
            },
            ylabel="Inércia",
            xlabel="Iteração"
    )
    plt.savefig("{}/{}.png".format(folder_result, title.replace(' ', '_') ))
    # plt.show()
     
     
    

# print(results)
