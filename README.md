# 🧠 Classificador de Lixo – Rede Neural (TensorFlow + TFLite)

> Este repositório contém os códigos de **treinamento, avaliação e conversão** da rede neural usada no aplicativo **RecycleApp**.

> O objetivo do modelo é **classificar imagens de lixo** em múltiplas categorias (ex.: garrafa de vidro, copo plástico, papel amassado, etc.), gerando um **arquivo `.tflite` otimizado para rodar localmente no Android**, sem necessidade de internet.

---

## ⚙️ Tecnologias utilizadas

- **Linguagem:** Python (3.x)
- **Deep Learning:** TensorFlow 2 / Keras
- **Pré-processamento de imagens:** Pillow (PIL)
- **Métricas e avaliação:** scikit-learn
- **Visualização:** Matplotlib + Seaborn
- **Outros:**
  - Camadas de aumento de dados (data augmentation) do Keras
  - Callbacks de treinamento (`EarlyStopping`, `ReduceLROnPlateau`)
  - Conversão para TensorFlow Lite (`tf.lite.TFLiteConverter`)

---

## 🧱 Estrutura do projeto

```text
TCC/
├─ images/
│  ├─ train/                # Conjunto de treino + validação (subpastas por classe)
│  │  ├─ glass_bottle/
│  │  ├─ glass_cup/
│  │  ├─ metal_can/
│  │  ├─ paper_bag/
│  │  ├─ paper_ball/
│  │  ├─ paper_milk_package/
│  │  ├─ paper_package/
│  │  ├─ plastic_bottle/
│  │  ├─ plastic_cup/
│  │  └─ plastic_transparent_cup/
│  └─ test/                 # Conjunto de teste (mesmos nomes de pastas/classes)
│
├─ venv/                    # (Opcional) Ambiente virtual Python
│
├─ trainer_final_version.py # Script principal de treinamento da rede neural
├─ evaluate.py              # Avaliação em conjunto de teste + matriz de confusão
├─ resize_images.py         # Utilitário para padronizar tamanho das imagens
├─ tflite_converter.py      # Conversão do modelo Keras (.keras) para TFLite (.tflite)
└─ trash_classifier_model_finetuned.keras
                            # Modelo treinado salvo em formato Keras
```

Obs.: O dataset não é versionado no GitHub por questões de tamanho/licença.
O repositório assume que você já tem as pastas images/train e images/test organizadas por classe.

---

## 🧪 Pipeline do modelo

A pipeline da rede neural é dividida em 4 etapas principais:
1. Preparação do dataset
2. Treinamento da CNN com focal loss
3. Avaliação em conjunto de teste
4. Conversão para TensorFlow Lite (.tflite)


## 1️⃣ Preparação do dataset

O TensorFlow usa a função image_dataset_from_directory, que espera a seguinte estrutura de pastas:

```text
images/
├─ train/
│  ├─ classe_1/
│  ├─ classe_2/
│  └─ ...
└─ test/
   ├─ classe_1/
   ├─ classe_2/
   └─ ...
```

Cada subpasta representa uma classe e contém apenas imagens daquele tipo.

🔧 Padronização opcional do tamanho das imagens

O script resize_images.py é um utilitário que:
1. Abre todas as imagens da pasta images/train;
2. Corrige rotação com base no EXIF;
3. Converte para RGB;
4. Redimensiona mantendo proporção (thumbnail);
5. Faz padding para um tamanho fixo (TARGET_SIZE);
6. Sobrescreve os arquivos originais.

Trecho central:

```text
DATA_DIR = "images/train"
TARGET_SIZE = (299, 299)

img = Image.open(filepath)
img = ImageOps.exif_transpose(img)
img = img.convert("RGB")
img.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)
img_padded = ImageOps.pad(img, TARGET_SIZE, color="white")
img_padded.save(filepath, quality=90)
```

⚠️ No treinamento atual o modelo usa IMAGE_SIZE = (256, 256).
O resize_images.py pode ser ajustado para o mesmo tamanho, se necessário.

---

###2️⃣ Treinamento da rede neural (trainer_final_version.py)

- 📥 Carregamento do dataset

  - O script separa automaticamente treino e validação a partir da pasta images/train:

    ```text

    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 24
    VALIDATION_SPLIT_CF = 0.1  # 10% para validação
    DATA_DIR = "./images/train"

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
       DATA_DIR,
       validation_split=VALIDATION_SPLIT_CF,
       subset="training",
       seed=123,
       image_size=IMAGE_SIZE,
       batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
       DATA_DIR,
       validation_split=VALIDATION_SPLIT_CF,
       subset="validation",
       seed=123,
       image_size=IMAGE_SIZE,
       batch_size=BATCH_SIZE
    )
    ```

Depois o pipeline é otimizado com:

- cache() – cache em memória;
- shuffle() – embaralhamento do treino;
- map(..., num_parallel_calls=AUTOTUNE) – processamento em múltiplas threads;
- prefetch(AUTOTUNE) – sobreposição de I/O e computação.

- 🎛 Aumento de dados (data augmentation)

  - Para melhorar a generalização, o modelo aplica várias transformações aleatórias apenas no treino:

    ```text
    data_augmentation = tf.keras.Sequential([
       layers.RandomFlip("horizontal_and_vertical"),
       layers.RandomRotation(0.2),
       layers.RandomTranslation(0.1, 0.1),
       layers.RandomZoom(0.2),
       layers.RandomContrast(0.2),
       layers.RandomBrightness(0.2),
       layers.GaussianNoise(0.05)
    ])
    ```

---

- 🧩 Arquitetura da CNN

  - A rede é uma CNN customizada, com 5 blocos convolucionais e pooling global:

    ```text
    model = models.Sequential([
       data_augmentation,
       layers.Rescaling(1./255, input_shape=(IMAGE_SIZE, 3)),  # Normalização

       layers.Conv2D(32, (3, 3), activation='relu'),
       layers.BatchNormalization(),
       layers.MaxPooling2D(2, 2),

       layers.Conv2D(64, (3, 3), activation='relu'),
       layers.MaxPooling2D(2, 2),

       layers.Conv2D(128, (3, 3), activation='relu'),
       layers.MaxPooling2D(2, 2),

       layers.Conv2D(256, (3, 3), activation='relu'),
       layers.MaxPooling2D(2, 2),

       layers.Conv2D(512, (3, 3), activation='relu'),
       layers.MaxPooling2D(2, 2),

       layers.GlobalAveragePooling2D(),

       layers.Dense(512, activation='relu',
                    kernel_regularizer=regularizers.l2(1e-4)),
       layers.Dropout(0.4),

       layers.Dense(len(class_names), activation='softmax')
    ])
    ```

Conceitualmente, a entrada é uma imagem 256×256×3 (RGB normalizada para [0,1]).

- 🎯 Função de perda: Focal Loss multiclasse

  - Em vez da entropia cruzada padrão, o projeto usa Focal Loss, mais robusta em cenários com classes desbalanceadas:

    ```text
    def focal_loss_multiclass(y_true, y_pred, alpha=0.25, gamma=3.0):
       num_classes = tf.shape(y_pred)[-1]
       y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)

       epsilon = tf.keras.backend.epsilon()
       y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

       pt = tf.reduce_sum(y_true_onehot * y_pred, axis=-1)
       modulating_factor = tf.pow(1. - pt, gamma)
       ce = -tf.math.log(pt)

       if isinstance(alpha, (float, int)):
           alpha_factor = alpha
       else:
           alpha_factor = tf.reduce_sum(y_true_onehot * alpha, axis=-1)

       loss = alpha_factor * modulating_factor * ce
       return tf.reduce_mean(loss)
    ```

O modelo é compilado com:

    ```text
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=focal_loss_multiclass,
        metrics=['accuracy']
    )

    ```
