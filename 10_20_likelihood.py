import numpy as np
import matplotlib.pyplot as plt

"""Dada la densidad de una exponencial p(x|tita) = tita * exp(-tita * x) para x >= 0"""

# Punto a: Plot p(x|tita) vs x para tita = 1. Plot p(x|tita) vs tita para 0 =< tita <= 5 para x = 2.

def exponential(x, tita):
    return tita * np.exp(-tita * x)

x = np.linspace(0, 10, 1000)
tita = np.linspace(0, 5, 1000)

def plot_exponential_vs_x(x, tita):
    plt.plot(x, exponential(x, tita))
    plt.xlabel('x')
    plt.ylabel('p(x|tita)')
    plt.grid()
    plt.show()

def plot_exponential_vs_tita(x, tita):
    plt.plot(tita, exponential(x, tita))
    plt.xlabel('tita')
    plt.ylabel('p(x|tita)')
    plt.grid()
    plt.show()



# Punto b: Suponga n muestras x1, x2, ..., xn de la distribución exponencial son dibujadas independientemente.
# Mostrar que el máximo likelihood estimator para tita es tita_mle = 1 / (1/n * sum(xi))

# L(tita) = prod( tita * exp(-tita * xi) ) = tita^n * exp(-tita * sum(xi))
# log(L(tita)) = n * log(tita) - tita * sum(xi)
# d(log(L(tita))) / d(tita) = n / tita - sum(xi)
# n / tita - sum(xi) = 0
# tita_mle = n / sum(xi)

mustras_exponencial = np.random.exponential(2, 1000) # tita = 2

def tita_mle(x):
    return len(x) / np.sum(x)

# plotear como da el likelihood a medida que aumenta la cantidad de muestras
def plot_likelihood_exponential(x):
    step = 10
    tita_mle_list = []
    tita_ideal = []
    for i in range(4, len(x), step):
        tita_mle_list.append(tita_mle(x[:i]))
        tita_ideal.append(tita_mle([2]))
    plt.plot([i for i in range(4, len(x), step)], tita_mle_list)
    plt.plot([i for i in range(4, len(x), step)], tita_ideal)
    plt.legend(['tita_mle', 'tita_ideal'])
    plt.title('tita_mle vs n')
    plt.xlabel('n')
    plt.ylabel('tita_mle')
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid()
    plt.show()

plot_likelihood_exponential(mustras_exponencial)
#plot_exponential_vs_x(x, 1)
#plot_exponential_vs_tita(2, tita)

"""Punto 2: dada una Unif(tita) = 1/tita para 0 <= x <= tita"""
# Punto a: suponga que n muestras x1, x2, ..., xn de la distribución uniforme son dibujadas independientemente.
# mostrar que el maximum likelihood estimator para tita es tita_mle = max(x1, x2, ..., xn)

def tita_mle_uniforme(x):
    return np.max(x)

# plotear como da el likelihood a medida que aumenta la cantidad de muestras
def plot_likelihood_uniforme(x, tita, step = 10):
    tita_mle_list = []
    tita_ideal = []
    for i in range(0, len(x), step):
        if i == 0:
            tita_mle_list.append(x[i])
        else:
            tita_mle_list.append(tita_mle_uniforme(x[:i]))
        tita_ideal.append(tita_mle_uniforme([tita]))
    plt.plot([i for i in range(0, len(x), step)], tita_mle_list)
    plt.plot([i for i in range(0, len(x), step)], tita_ideal)
    plt.legend(['tita_mle', 'tita_ideal'])
    plt.title('tita_mle vs n')
    plt.xlabel('n')
    plt.ylabel('tita_mle')
    plt.yticks(np.arange(0, tita + 1, (tita + 1) / 6))
    plt.grid()
    plt.show()

tita = 2
mustras_uniformes = np.random.uniform(0, tita, 30) # tita = 2
plot_likelihood_uniforme(mustras_uniformes, tita, 1)
