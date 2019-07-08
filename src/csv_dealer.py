import csv
import glob

import matplotlib.pyplot as plt
import numpy as np

def get_csv(path, delimiter=',', cast=float):
    with open(path) as f:
        rows = list(csv.reader(f, delimiter=delimiter))
        titles, data = rows[0], rows[1:]
        dic = {}
        for i, t in enumerate(titles):
            t = t.strip()
            dic[t] = []
            for d in data:
                dic[t].append( cast(d[i].strip()) )
    return dic

def getCsvNames(prefix='', sufix=''):
    return glob.glob(f'{prefix}*{sufix}')

def showCsv(path, delimiter=',', cast=float):
    dict_data = get_csv(path,delimiter=delimiter, cast=cast)
    for k, v in dict_data.items():
        plt.plot(v)
        plt.xlabel( list( range(len(v)) ) )
        plt.ylabel( k )
        plt.show()

def showCsvInPlot( path, delimiter=',', cast=float ):
    dict_data = get_csv(path,delimiter=delimiter, cast=cast)    
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    for k, v in dict_data.items():
        ax1.plot(np.array( v ), label=k)
    
    colormap = plt.cm.gist_ncar
    colors = [colormap(i) for i in range(len(dict_data))]
    for i, j in enumerate(ax1.lines):
        j.set_color(colors[i])
    ax1.legend(loc=2)
    plt.show()