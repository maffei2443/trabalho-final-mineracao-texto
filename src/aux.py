from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt

def ShowOrderedDictList(d, option = 'values', smallTime = False):
    items = d.items() if not smallTime else [(k, [min(v)]) for k, v in d.items() if smallTime and option]
    if option == 'values':
        ordered = sorted(items, key = lambda x: sum(x[1])/len(x[1]) if x[1] else 0)        
    else:
        ordered = sorted(items)
    for k, v in ordered:
        print(f"{k:20s} => {(sum(v) / len(v) if v else 0):3.7f}")


def ShowOrderedDict(d, option = 'values'):
    if option == 'values':
        ordered = sorted(d.items(), key = lambda x: x[1])
    else:
        ordered = sorted(d.items())
    for k, v in ordered:
        print(f"{k:20s} => {v:3.7f}")

def csr_nonzero_tuple_array(arg: csr_matrix, verbose=False):  
    """Recebe uma matriz 'arg' do tipo scipy.sparse.csr_matrix e retorna
    uma tupla de tamanho arg.shape[0], contendo, na ordem original, os
    valores não nulos de 'arg'. 
    O n-ésimo elemento da tupla retornada corresponde aos dados da i-ésima
    linha de 'arg'.
    """
    if verbose: 
        print('[verbose=True] arg: ', arg ) 
    idx = arg.indptr   
    return ( 
        tuple( 
            arg.data[ idx[i] : idx[i+1] ] for i in range(0, len(arg.indptr)-1)
        )      
    )


def read_magic_csv(path):
    labels = []
    data = []
    trans = {'g': 0, 'h': 1}
    with open(path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split(',')
            data.append( [float(i) for i in l[:-1]] )
            labels.append(trans[l[-1]])
    return data, labels

def MyPlot(title, dic, colors=['#ff0000', '#00ff00', '#0000ff'],
    markers=[6, 7, '.'],linestyle=[':', '--', '-'],
    dpi=300,
    xlabel = 'xlabel',
    ylabel='ylabel',
    top=0.99,
    bottom=0.0):
    if not linestyle:
        linestyle = []
    keys = list( dic.keys() )
    if not title:
        top = 0.99
    if 'n_samples' in keys:
        keys.remove('n_samples')
    else:  # Informar qual o range em dic
        dic['n_samples'] = range(len( dic[keys[0]] ))
    assert 'n_samples' in dic
    plt.figure(title, dpi=dpi)
    plt.title(title)
    plt.xlabel(xlabel, fontsize='x-small')
    plt.ylabel(ylabel, fontsize='large')
    plt.subplots_adjust(top=top, bottom=bottom)
    for i, col in enumerate(keys):
        plt.plot( 'n_samples',col, data=dic,
            marker=markers[i] if i in range(len(markers)) else None,
            color=colors[i] if i in range(len(colors)) else None,
            linestyle=linestyle[i] if i in range(len(linestyle)) else ''
        )
    plt.legend(keys)
    return title

def PlotDictInertia(dic, linestyle=[]):
    title = MyPlot('Inertia', dic, 
        xlabel='Tentativas (fora de ordem)', ylabel='Inércia', linestyle=linestyle)
    plt.show()
    plt.close(title)
def ls2dict(ls,  cvt=float): 
    cp = [i.strip() for i in ls]      
    keys = [i.strip() for i in cp[0].strip().split(',')] 
    dic = {} 
    for k in keys: 
        dic[k] = [] 
    for ll in cp[1:]: 
        for ii, k in enumerate(keys):             
            dic[k] += [cvt(ll.split(',')[ii])] 
    return dic   
