import os
import cv2
import numpy as np


def criaArquivoDeRotulo(diretorio):
    label = 0
    f = open("TRAIN", "w+")
    for dir_principal, nome_dirs, nome_arqs in os.walk(diretorio):
        for sub_dir in nome_dirs:
            caminho_dir = os.path.join(dir_principal, sub_dir)
            for arq in os.listdir(caminho_dir):
                caminho_abs = caminho_dir + "\\" + arq
                f.write(caminho_abs + ";" + str(label) + "\n")
            label += 1
    f.close()
    print("\n1.Arquivo de rótulo criado.")


def criaDicionarioDeImagens(f):
    #cria um dicionário que na posição 0 tem uma lista
    #contendo todas as fotos da pessoa A que estão na base
    #e na posição 1 todas as fotos da pessoa B
    linhas = f.readlines()
    dicionario = {}
    for linha in linhas:
        arq, label = linha.rstrip().split(";")
        if int(label) in dicionario.keys():
            dicionario.get(int(label)).append(cv2.imread(arq, 0))
        else:
            dicionario[int(label)] = [cv2.imread(arq, 0)]
    print("2.Dicionário de imagens criado.")
    return dicionario


def treinaModelo(dicionario):
    modelo = cv2.face.EigenFaceRecognizer_create()
    chaves = []
    valores = []
    for chave in dicionario.keys():
        for valor in dicionario[chave]:
            chaves.append(chave)
            valores.append(valor)
    modelo.train(np.array(valores), np.array(chaves))
    modelo.write("classificadorEigen-CamilaPirulla.yml")
    print("3.Modelo treinado.")


def main():
    #cria um arquivo que associa uma pessoa a uma foto
    criaArquivoDeRotulo("data")
    #lê o arquivo e constrói um dicionário dos dados
    f = open("TRAIN", "r")
    dicionarioDeFotos = criaDicionarioDeImagens(f)
    treinaModelo(dicionarioDeFotos)



if __name__ == "__main__":
    main()
