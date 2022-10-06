# este documento contém as funções das métricas de caracterização do reservatório
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray, linalg as LA
import pandas as pd
from reservoirpy.nodes import Reservoir, Ridge, FORCE, ESN




def get_KR(matriz_estados):
    # essa função recebe uma matriz contendo os estados do reservatório
    u, s, vh = np.linalg.svd(matriz_estados, full_matrices=True)   #Singular value decomposition da matriz de estados
    #calculando o rank da matriz com 90% de confidência
    valor_maximo = 0.9 * np.cumsum(s)  
    for i in range(0,s.size):
        if s[i] >= valor_maximo:
            i = i - 1
            break
        if s[i] == valor_maximo:
            break
    return i


def get_GR(matriz_estados):
    # essa função recebe uma matriz contendo os estados do reservatório
    u, s, vh = np.linalg.svd(matriz_estados, full_matrices=True)   #Singular value decomposition da matriz de estados
    #calculando o rank da matriz com 90% de confidência
    valor_maximo = 0.9 * np.cumsum(s)  
    for i in range(0,s.size):
        if s[i] >= valor_maximo:
            i = i - 1
            break
        if s[i] == valor_maximo:
            break
    return i

def get_MC(matriz_estados):
    #criando o nó de saída
    ridge = Ridge(ridge=1e-7)

    
    #criando o sinal, que vai ser um ruído. O sinal de input será a variável 'sinal'
    noise = np.random.normal(0,1,1000).reshape(-1,1)
    sinal = noise[300:500]

    #listas para guardar os valores de mc e do delay a cada iteração para plotarmos esses valores depois
    mc =[]
    delay = []
    for k in range(0,101):
        delay.append(k)

        #criando o sinal com delay, que é o 'sinal' só que atrasado por 'k'
        sinal_atrasado = noise[300 - k : 500 - k]

        #treinando o nó de saída. O 'sinal' é colocado no reservatório para coletar seus estados 'x(t)'. O 'sinal_atrasado' é usado
        #para treinar o nó de saída com a tarefa de prever o sinal atrasado.
        ridge.fit(sinal, sinal_atrasado, warmup=50)
        
        #colocamos o 'sinal' sem atraso na rede para coletar o output 'y_k'. Temos que colocar o 'sinal' pois o nó de saída está
        #treinado para prever o 'sinal_atrasado'. Queremos ver como nossa rede, treinada num sinal atrasado, ao ser apresentada um 
        #sinal sem atraso, consiga prever os atrasos dele.
        y_k = ridge.run(matriz_estados)

        #fórmula para calcular o Mc
        cov_squared = np.cov(sinal_atrasado.reshape(1,-1), y_k.reshape(1,-1))[0][1] ** 2
        var_u = np.var(sinal)
        var_y = np.var(y_k)
        var_total = var_y * var_u
        mc_k = cov_squared/var_total
        mc.append(mc_k)
