import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray, linalg as LA
import struct
import pandas as pd

#função para abrir os dados binários do arduíno
def openbin(path):
    word =[]
    with open(path,'rb') as f:
        while chunk := f.read(31):
            word.append(chunk)
    databin = []
    for i in range(len(word)):
        databin.append(list((struct.unpack('<LHLHLHLHLHc',word[i])))[0:-1])
    data = pd.DataFrame(databin,columns=['T0','ADC0','T1','ADC1','T2','ADC2','T3','ADC3','T_sinal', 'Sinal'])  
    return data

#função para transformar os dados do arduíno(estados do resevatório) em uma matriz de estados
def get_reservoir_states(dados, path):    

    dados = openbin("path")
    dados = dados.drop(labels=['T0','T1','T2','T3','T_sinal','Sinal'], axis =1)
    dados.columns = ['x0','x1','x2','x3']


    #criando a matrix de estados 
    estados = pd.concat([dados],axis =1 )
    estados = estados.to_numpy()
    return estados
