import pickle
import glob
tipo_inicializacao = ['k-means++', 'caseiro', 'k-means']
precision = 2
def to_csv_inertia(bat, tipo = tipo_inicializacao, dirr='csv/',precision=precision):
    """Recebe dicionário de bateria de testes já convertido para
    o formato que cada chava leva para tuple(float, int), representando
    a inércia e a quantidade de clusters."""
    meta = bat['meta']
    n, rep  = meta['n'], meta['k']
    t0, t1, t2 = tipo
    with open(f'{dirr}{rep}rep_{n}center_inertia.csv', 'w') as f:
        f.write(f'inertia_{t0}, inertia_{t1}, inertia_{t2}\n')
        for (k, ca, kpp) in zip(bat[t0], bat[t1], bat[t2]):
            f.write(f'{k[0]:.{precision}f}, {ca[0]:.{precision}f}, {kpp[0]:.{precision}f}\n')

def to_csv_n_iter(bat, tipo = tipo_inicializacao, dirr='csv/', precision=precision):
    """Recebe dicionário de bateria de testes já convertido para
    o formato que cada chava leva para tuple(float, int), representando
    a inércia e a quantidade de clusters."""
    meta = bat['meta']
    n, rep  = meta['n'], meta['k']
    t0, t1, t2 = tipo
    with open(f'{dirr}{rep}rep_{n}center_n_iter.csv', 'w') as f:
        f.write(f'n_iter_{t0}, n_iter_{t1}, n_iter_{t2}\n')
        for (k, ca, kpp) in zip(bat[t0], bat[t1], bat[t2]):
            f.write(f'{k[1]}, {ca[1]}, {kpp[1]}\n')

def to_csv_full(bat, tipo = tipo_inicializacao, dirr='csv/', precision=precision):
    """Recebe dicionário de bateria de testes já convertido para
    o formato que cada chava leva para tuple(float, int), representando
    a inércia e a quantidade de clusters."""
    meta = bat['meta']
    n, rep  = meta['n'], meta['k']
    t0, t1, t2 = tipo
    with open(f'{dirr}{rep}rep_{n}center_full.csv', 'w') as f:
        f.write(f'inertia_{t0}, n_iter_{t0}, inertia_{t1}, n_iter_{t1},inertia_{t2}, n_iter_{t2}\n')
        for (k, ca, kpp) in zip(bat[t0], bat[t1], bat[t2]):
            f.write(f'{k[0]:.{precision}f}, {k[1]}, {ca[0]:.{precision}f}, {ca[1]}, {kpp[0]:.{precision}f}, {kpp[1]}\n')


def convert(bat, init = tipo_inicializacao):
    """Converte dicionário da forma
    str -> list<KMeans>
    para
    str -> list<float, int>,
    onde float guarda a inércia e int a quantidade de iterações
    até que se fosse atingida a convergễncia.
    """
    if len(init) != 3:
        return "init must have exactly 3 elements"
    for (n, (k, ca, kpp)) in enumerate(zip(bat[init[0]], bat[init[1]], bat[init[2]])): 
        print(n) 
        bat[init[0]][n] = (k.inertia_, k.n_iter_) 
        bat[init[1]][n] = (ca.inertia_, ca.n_iter_)  
        bat[init[2]][n] = (kpp.inertia_, kpp.n_iter_)  
        print(f'Inertia: {k.inertia_} {ca.inertia_} {kpp.inertia_}' ) 

def getBaterias(k = 10):
    return sorted(
            glob.glob(f'bateria{k}*'),
            key=lambda x: int(str.replace(x[-2:],'_', '')),
            )

def allBat2csv(k = 10, precision=3):
    for fname in getBaterias(k):
        bat = pickle.load(open(fname, 'rb'))
        convert(bat)
        to_csv_inertia(bat, precision=precision)
        to_csv_n_iter(bat, precision=precision)
        to_csv_full(bat, precision=precision)

# allBat2csv()

def main():
    allBat2csv()

if __name__ == '__main__':
    main()