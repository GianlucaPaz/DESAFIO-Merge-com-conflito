# 🧠 Rede Neural - Classificador de Lixo

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


### 1️⃣ Preparação do dataset

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


#### 🔧 Padronização opcional do tamanho das imagens

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

### 2️⃣ Treinamento da rede neural (trainer_final_version.py)

#### 📥 Carregamento do dataset

O script separa automaticamente treino e validação a partir da pasta images/train:

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

#### 🎛 Aumento de dados (data augmentation)

Para melhorar a generalização, o modelo aplica várias transformações aleatórias apenas no treino:

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

#### 🧩 Arquitetura da CNN

A rede é uma CNN customizada, com 5 blocos convolucionais e pooling global:

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

#### 🎯 Função de perda: Focal Loss multiclasse

Em vez da entropia cruzada padrão, o projeto usa Focal Loss, mais robusta em cenários com classes desbalanceadas:

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

#### ⏱ Callbacks e treinamento em duas fases

O treinamento é dividido em duas fases, ambas com Early Stopping e ajuste dinâmico da taxa de aprendizado:

- EPOCHS_INITIAL = 70 – treino principal
- EPOCHS_FINE_TUNE = 35 – ajuste fino com LR reduzida

Callbacks principais:

```text
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

early_stop_initial = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

Após a primeira fase, o código mantém a learning rate atual, recompila o modelo e executa o segundo treinamento com callbacks mais agressivos.

Ao final:

```text
model.save('trash_classifier_model_finetuned.keras')
```

### 3️⃣ Avaliação do modelo (evaluate.py)

O script evaluate.py carrega:

- O modelo salvo (trash_classifier_model_finetuned.keras);
- O conjunto de teste em ./images/test/.

```text
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 24
TEST_DIR = './images/test/'

model = tf.keras.models.load_model('trash_classifier_model_finetuned.keras', compile=False)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)
class_names = test_ds.class_names
```

Ele calcula:

- Acurácia e log loss por classe
- Acurácia, precisão, recall e F1-score globais
- Matriz de confusão (visualizada via Seaborn)

Trecho principal:

```text
overall_acc = accuracy_score(y_true, y_pred)
overall_prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
overall_rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
overall_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
```

### 4️⃣ Conversão para TensorFlow Lite (tflite_converter.py)

Por fim, o modelo é convertido para um `.tflite` otimizado, que é o formato usado no app Android:

```text
import tensorflow as tf

model = tf.keras.models.load_model('trash_classifier_model_finetuned.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open('trash_classifier_model_optimized.tflite', 'wb') as f:
    f.write(tflite_model)
```

tf.lite.Optimize.DEFAULT ativa otimizações padrão do TensorFlow Lite (como quantização de pesos), reduzindo o tamanho do modelo e ajudando no desempenho em dispositivos móveis.

---

## ▶️ Como reproduzir o experimento localmente

### 1. Criar e ativar ambiente virtual (opcional, mas recomendado)

```text
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 2. Instalar dependências

```text
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow
```

(ou via requirements.txt, se criado)

### 3. Organizar o dataset

- Colocar as imagens em `images/train/<nome_da_classe>/...`
- Colocar o conjunto de teste em `images/test/<nome_da_classe>/...`
- Os nomes das pastas de `train/` e `test/` devem ser idênticos.

### 4. (Opcional) Padronizar tamanho das imagens

```text
python resize_images.py
```

### 5. Treinar o modelo

```text
python trainer_final_version.py
```

Ao final, será gerado o arquivo:

```text
trash_classifier_model_finetuned.keras
```

### 6. Avaliar em conjunto de teste

```text
python evaluate.py
```

O script imprime métricas no console e abre a matriz de confusão em uma janela gráfica.

### 7. Gerar modelo TFLite

```text
python tflite_converter.py
```

Saída esperada:

```text
trash_classifier_model_optimized.tflite
```

Este é o arquivo que será usado pelo aplicativo Android (RecycleApp) via Interpreter do TensorFlow Lite.

---

## 🔗 Integração com o RecycleApp

- O arquivo trash_classifier_model_optimized.tflite é copiado para a pasta assets/ do app Android.
- No app, uma classe utilitária (TrashClassifier.kt) faz:
1. Carregamento da imagem a partir de uma URI;
2. Redimensionamento para 256×256;
3. Conversão para ByteBuffer float32;
4. Execução do modelo TFLite;
5. Mapeamento do índice de classe para o material exibido na interface (Vidro, Papel, Plástico, Metal ou Indefinido).

---

## 👥 Equipe

Projeto de rede neural desenvolvido como parte do TCC do curso de Ciência da Computação – Universidade Veiga de Almeida, integrado ao aplicativo móvel RecycleApp.

- Responsáveis pelo desenvolvimento do modelo de IA
  - Davi Millan Alves
  - Gabriel Mesquita Gusmão
