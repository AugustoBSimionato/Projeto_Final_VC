import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def obter_imagens_treino(caminho_imagens_treino, forma_img, tam_lote):
    gerador_imagens = keras.preprocessing.image.ImageDataGenerator(
        vertical_flip=True,
        horizontal_flip=True,
        rescale=1./255
    ) 
    imagens_treino = gerador_imagens.flow_from_directory(
        caminho_imagens_treino,
        target_size=forma_img[:2], 
        batch_size=tam_lote,
        class_mode='categorical', 
        shuffle=True,
        seed=42
    ) 

    return imagens_treino


def obter_imagens_validacao(caminho_imagens_validacao, forma_img, tam_lote):
    gerador_imagens = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )

    imagens_validacao = gerador_imagens.flow_from_directory(
        caminho_imagens_validacao, 
        target_size=forma_img[:2], 
        batch_size=tam_lote,
        class_mode='categorical',
        shuffle=True
    )

    return imagens_validacao


def obter_modelo_resnet(num_classes):
    # Define um seed para garantir reprodutibilidade
    tf.random.set_seed(42)
    # Define a forma de entrada
    forma_img = (224, 224, 3)
    # Carrega a ResNet50 pré-treinada com pesos do ImageNet, sem a camada de classificação final
    base_model = keras.applications.ResNet50(
        input_shape=forma_img,
        weights='imagenet',
        include_top=False
    )
    base_model.trainable = False

    # Define a camada de entrada
    camada_entrada = keras.Input(shape=forma_img)
    # Aplica o pré-processamento específico para ResNet50
    x = keras.applications.resnet.preprocess_input(camada_entrada)
    # Extrai as características usando a base do modelo
    x = base_model(x)
    # Normaliza os valores com BatchNormalization
    x = keras.layers.BatchNormalization()(x)
    # Reduz as dimensões com GlobalAveragePooling2D
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Camadas adicionais para aprimorar a capacidade de generalização
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, kernel_regularizer=keras.regularizers.L2(0.001), activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    
    # Camada de saída com 'num_classes' neurônios e ativação softmax
    camada_saida = keras.layers.Dense(num_classes, activation='softmax')(x)

    # Cria o modelo final
    modelo = keras.Model(camada_entrada, camada_saida)
    modelo.summary()
    return modelo


def compilar_modelo_resnet(modelo, metrica):
    otimizador = keras.optimizers.Adam(learning_rate=0.0005)
    loss = keras.losses.CategoricalCrossentropy()
    modelo.compile(optimizer=otimizador, loss=loss, metrics=[metrica])
    return modelo


def treinar_modelo(modelo, imagens_treino, imagens_validacao, num_epocas, caminho_checkpoints, lista_callbacks):
    historico = modelo.fit(
        imagens_treino,
        epochs=num_epocas,
        verbose=1,
        validation_data=imagens_validacao,
        callbacks=lista_callbacks
    )
    modelo.load_weights(caminho_checkpoints)
    return modelo, historico
    

def plot_historico(historico, metrica, caminho_resultados):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(historico.history[metrica])
    plt.plot(historico.history['val_'+metrica])
    plt.title('Acurácia do Modelo')
    plt.ylabel('Acurácia')
    plt.xlabel('Época')
    plt.legend(['Treinamento', 'Validação'], loc='lower right')
    plt.subplot(212)
    plt.plot(historico.history['loss'])
    plt.plot(historico.history['val_loss'])
    plt.title('Perda do Modelo')
    plt.ylabel('Perda')
    plt.xlabel('Época')
    plt.legend(['Treinamento', 'Validação'], loc='upper right')
    plt.tight_layout()
    plt.savefig(caminho_resultados+'Historico_Treinamento', dpi=300)
    print(f'\nHistórico salvo na pasta: {caminho_resultados}\n')
    plt.show()


def salvar_modelo(modelo, caminho_dest_modelo):
    modelo.save(caminho_dest_modelo)
    print(f'\nO modelo treinado foi salvo na pasta {caminho_dest_modelo}\n')


def obter_checkpoint_callback(caminho_checkpoints):
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=caminho_checkpoints,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    return checkpoint_callback


def obter_log_callback(caminho_log):
    log_callback = keras.callbacks.CSVLogger(caminho_log, separator=',', append=False)
    return log_callback


def obter_reduce_lr_callback():
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=1, #quantidade de épocas que espera para reduzir a tx de aprend
        verbose=1
    )
    return reduce_lr_callback


def obter_early_stop_callback():
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )
    return early_stop






