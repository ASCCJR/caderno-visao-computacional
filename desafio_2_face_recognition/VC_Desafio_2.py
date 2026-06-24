# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Desafio 2 da Trilha de Visão Computacional — Reconhecimento de Faces
#
# > **Resumo:** Sistema de reconhecimento facial em **3 etapas** (estilo controle de acesso): (1) um
# > **detector SSD** acha a face na imagem; (2) o **OpenFace** transforma a face num vetor de 128
# > dimensões (*embedding*); (3) um **SVM** classifica esse vetor no nome da pessoa. Treinamos com um
# > pequeno dataset e validamos numa imagem de teste.
#
# > 📝 **Por que embeddings?** Em vez de comparar pixels, o OpenFace mapeia cada rosto para um ponto
# > num espaço onde faces da mesma pessoa ficam próximas. Aí um classificador simples (SVM) já separa
# > bem as identidades — e dá pra cadastrar pessoas novas só re-treinando o SVM, sem mexer nas redes.
#
# Nesse desafio você irá implementar uma solução simples para reconhecimento de faces, semelhante ao que soluções de acessos a áreas restritas fazem hoje em dia.

# %% [markdown]
# ### Instalação e Importação da Biblioteca

# %%
# !pip install opencv-python matplotlib imutils tqdm ipywidgets scikit-learn

# %%
import cv2
import matplotlib.pyplot as plt
import os
import imutils.paths as paths
import imutils
import pickle
import numpy as np
from tqdm.notebook import tqdm
import sklearn

# %% [markdown]
# ### Processar a Base de Dados
#
# Iremos processar as imagens da base de dados.

# %%
# Carregar os modelos.
protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detection_model","res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# Captura os paths do dataset
imagePaths = list(paths.list_images("dataset"))

# Lista de caracteristicas e nomes
knownEmbeddings = []
knownNames = []


# loop sobre as imagens do dataset
for imagePath in tqdm(imagePaths):
    name = imagePath.split(os.path.sep)[-2] # Captura nomes dos dataset
    image = cv2.imread(imagePath) # Le imagem
    image = imutils.resize(image, width=600) # Redimensiona imagem
    (h, w) = image.shape[:2] # Pega dimensões da imagem

    # OpenCV pre processamento para modelos de deep learning
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob) # Define entrada do modelo de deep learning
    detections = detector.forward() # Processa a imagem
	
    # Valida que pelo menos 1 face foi detectada na iamgem
    if len(detections) > 0:
        # Assumindo apenas 1 face por imagem, pegamos a maior bounding box
        # E validamos que aquela detecção está dentro do limite de 50% de confiança
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            # Calcular as coordenadas da bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Extrair a imagem da face
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            # Valida que a imagem da face é grande o suficiente
            if fW < 20 or fH < 20:
                continue
            

            # OpenCV pre processamento para imagem da face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob) # Define como entrada do modelo de extração de características
            vec = embedder.forward() # Processa a imagem

            # Adiciona as listas o nome e as características
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())

# Salva arquivo.
data = {"embeddings": knownEmbeddings, "names": knownNames}
os.makedirs("output", exist_ok=True)
f = open("output/embeddings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()

print(f"Imagens no dataset: {len(imagePaths)}")
print(f"Embeddings extraídos: {len(knownEmbeddings)}")
print(f"Pessoas (classes): {sorted(set(knownNames))}")

# %% [markdown]
# ### Treinar Modelo de Reconhecimento de Face
#
# Agora vamos treinar um modelo de aprendizado de máquina para reconhecer as faces.

# %%
# Carrega os dados de características
data = pickle.loads(open("output/embeddings.pickle", "rb").read())

# Cria o pre processamento e define os labels como os nomes
le = sklearn.preprocessing.LabelEncoder()
labels = le.fit_transform(data["names"])

# Cria um modelo de SVM para classificar usando as características
recognizer = sklearn.svm.SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)


# Salva os modelos de classificação (reconhecimento) e de labels
f = open("output/recognizer.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()
f = open("output/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()

print(f"SVM treinado em {len(labels)} embeddings. Classes reconhecíveis: {list(le.classes_)}")

# %% [markdown]
# ### Validar Modelo
#
#

# %%
########################
## Processar o Modelo ##
########################

# Define a image de teste
image_to_test_path = "test/modelo_2_1.jpeg"

# Carrega os modelos
protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detection_model",
    "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

# Carrega a imagem
image = cv2.imread(image_to_test_path)
image = imutils.resize(image, width=600) # Redimensiona a imagem
(h, w) = image.shape[:2] # Salva as dimensões
# Pre processa por OpenCV para modelo de deep learning (mesmo blob usado no dataset)
imageBlob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)), 1.0, (300, 300),
    (104.0, 177.0, 123.0), swapRB=False, crop=False)
# Detecta faces na imagem de input
detector.setInput(imageBlob)
detections = detector.forward()


# loop sobre as detecções
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2] # Valida a confiaça da detecção que estpa sendo iterada
    # Valida que a detecção tem uma confiança superior a 50%
    if confidence > 0.5:
        # Calcula coordenadas da face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # Extrai imagem da face
        face = image[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]
        # Garante o tamanho da imagem da face não é pequeno demais
        if fW < 20 or fH < 20:
            continue
        
        # OpenCV pre processamento para o modelo de extração de características
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob) # Define o input do modelo de extração de características
        vec = embedder.forward() # Processa a imagem (gera o embedding de 128 dimensões)
        
        # Realiza a classificação com o modelo de reconhecimento 
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds) # Pega o indice da maior probabilidade
        proba = preds[j] # Salva o valor da maior probabilidade
        name = le.classes_[j] # Recupera o nome da maior probabilidade

        print(f"Reconhecido: {name} ({proba*100:.2f}%) na caixa ({startX},{startY})-({endX},{endY})")

        # Desenha um retângulo com o texto do nome e da probabilidade na face reconhecida
        text = "{}: {:.2f}%".format(name, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# mostra a imagem (convertendo BGR -> RGB para o matplotlib)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# %% [markdown]
# ### Referência
#
# - https://pyimagesearch.com/2018/09/24/opencv-face-recognition/
# - https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/
