import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import os
import sys


def obter_imagens_teste(caminho_imagens_teste, forma_img, tam_lote):
    """
    Cria um gerador de imagens para o conjunto de teste.
    """
    gerador_imagens = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    imagens_teste = gerador_imagens.flow_from_directory(
        caminho_imagens_teste, 
        target_size=forma_img[:2], 
        batch_size=tam_lote, 
        class_mode='categorical',
        shuffle=False
    )
    return imagens_teste


def carregar_modelo_resnet(caminho_modelo_resnet):
    """
    Carrega o modelo ResNet treinado a partir do arquivo salvo.
    """
    if not os.path.exists(caminho_modelo_resnet):
        print(f"Arquivo não encontrado: {caminho_modelo_resnet}")
        sys.exit("Execução interrompida: Arquivo do modelo não encontrado.")
        
    modelo = keras.models.load_model(caminho_modelo_resnet)
    return modelo

def testar_modelo(modelo, imagens_teste):
    """
    Gera as previsões do modelo e converte para rótulos de classe.
    """
    previsoes = modelo.predict(imagens_teste, verbose=2)
    rotulos_previstos = np.argmax(previsoes, axis=1)
    return rotulos_previstos

def matriz_confusao(rotulos_verdadeiros, rotulos_previstos, nomes_das_classes, caminho_resultados):
    """
    Calcula e salva a matriz de confusão do modelo.
    """
    conf_matrix = confusion_matrix(rotulos_verdadeiros, rotulos_previstos)
    acuracia = accuracy_score(rotulos_verdadeiros, rotulos_previstos) * 100

    plt.figure(figsize=(10,8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.xticks(np.arange(len(nomes_das_classes)) + 0.5, nomes_das_classes, rotation=0, ha='right')
    plt.yticks(np.arange(len(nomes_das_classes)) + 0.5, nomes_das_classes, rotation=0, va='center')
    plt.xlabel('Rótulos Previstos')
    plt.ylabel('Rótulos Reais')
    plt.title('Matriz de Confusão', fontsize=18, weight='bold', x=0.5, y=1.05)
    plt.suptitle(f'Acurácia do Modelo: {acuracia:.2f}%', fontsize=14, x=0.435, y=0.92)
    plt.savefig(caminho_resultados + 'Matriz_Confusao', dpi=300)
    print(f'\nMatriz salva na pasta: {caminho_resultados}\n')
    plt.show()

def plot_metrica(rotulos_verdadeiros, rotulos_previstos, nome_metrica, nomes_das_classes, caminho_resultados):
    """
    Cria e salva um gráfico para a métrica escolhida.
    """
    report_dict = classification_report(
        rotulos_verdadeiros, rotulos_previstos, 
        target_names=nomes_das_classes, output_dict=True
    )
    
    metric_data = {
        class_name: metrics[nome_metrica]
        for class_name, metrics in report_dict.items() 
        if class_name in nomes_das_classes
    }
    
    sns.barplot(x=list(metric_data.keys()), y=list(metric_data.values()))
    plt.xlabel('Classes')
    plt.ylabel(nome_metrica.capitalize())
    plt.title(f'{nome_metrica.capitalize()} por Classe')
    plt.xticks(rotation=0)
    for index, value in enumerate(metric_data.values()):
        plt.text(index, value, str(round(value, 2)), ha='center', va='bottom')
    plt.savefig(caminho_resultados + nome_metrica, dpi=300)
    print(f'\nMétrica salva na pasta: {caminho_resultados}\n')
    plt.show()










