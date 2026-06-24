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
# # M3A5 - Detecção de Objetos
#
# > **Resumo:** Detecção responde **o quê** e **onde** — cada objeto ganha uma classe, uma *bounding box* e uma confiança. Usamos a **YOLOv11** (pré-treinada no COCO) numa foto de rua e comparamos os tamanhos nano vs small, inclusive testando como a rede reage a um personagem fora do domínio (o Sonic).
#
# > 📝 **Classificação vs Detecção:** classificar diz "tem um gato na imagem"; detectar diz "tem um gato *aqui*, neste retângulo, com 92% de confiança".

# %% [markdown]
# Na prática de hoje vamos utilizar um modelo pré-treinado para detecção de objetos. Para a prática de hoje iremos utilizar a biblioteca  [Ultralytics](https://docs.ultralytics.com/).
#
# Esse notebook está estruturado da seguinte forma.
#
# - Introdução
# - Detecção de Objetos
# - Próximos passos
# - ✅ Atividades Complementares (resolvidas)

# %% [markdown]
# ## Introdução
#
# **Detecção de objetos** é a tarefa de identificar **o quê** existe em uma imagem e **onde** cada objeto está.
# Diferente da classificação (que diz apenas "esta imagem contém um gato"), a detecção retorna:
#
# - **Classe** do objeto (pessoa, carro, ônibus, etc.)
# - **Bounding box** (caixa delimitadora) com coordenadas (x, y, largura, altura)
# - **Confiança** (probabilidade de que a detecção está correta)
#
# O modelo **YOLO** (You Only Look Once) é um dos detectores mais populares por ser
# extremamente rápido — processa a imagem inteira em uma única passada pela rede.

# %% [markdown]
# Instalação para os que ainda não possuem a biblioteca instalada.

# %%
# !pip install ultralytics

# %% [markdown]
# Importar as bibliotecas e Ler Imagens do Disco

# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
import ultralytics


# %%
# TODO Atualize o path da imagem.
image = cv2.imread("street.jpeg")

# Verificar se a imagem foi carregada corretamente.
if image is None:
    print("ERRO: Imagem 'street.jpeg' não encontrada!")
    print("Faça upload do arquivo street.jpeg para o mesmo diretório do notebook.")
else:
    print(f"Imagem carregada com sucesso!")
    print(f"  Shape: {image.shape} (altura, largura, canais)")
    print(f"  Tipo: {image.dtype}")
    print(f"  Resolução: {image.shape[1]}x{image.shape[0]} pixels")

# %%
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# %% [markdown]
# ## Detecção de Objetos
#
# Iremos carregar o modelo pré treinado e visualizar seus resultados.
#
# A **YOLOv11n** ("n" de nano) é a versão mais leve da família YOLO v11.
# Ela foi treinada no dataset **COCO** (Common Objects in Context).
#
# Como o professor apresentou em aula, existem três principais bases de dados utilizadas como benchmark e treinamento para modelos de detecção:
#
# 1. **COCO** (Common Objects in Context): Uma das maiores e mais populares bases, contendo milhares de imagens de cenas cotidianas complexas com 80 categorias de objetos cotidianos detalhados em bounding boxes.
# 2. **Pascal VOC**: Base clássica de visão computacional que serviu como grande referência histórica nos primeiros anos de detecção profunda, ideal para estudos teóricos e tarefas mais controladas.
# 3. **KITTI**: Base especificamente voltada para o desenvolvimento de veículos autônomos, contendo imagens urbanas com bboxes em tempo real para carros, pedestres, ciclistas, etc.


# %%
# Load a pretrained YOLO11n model
model = ultralytics.YOLO("yolo11n.pt")

# Roda a inferência do modelo YOLO.
results = model(image)

for i, result in enumerate(results):
    # Salva os resultados em uma imagem no disco.
    result.save(filename=f"results_{i}.jpg")

# Carrega a imagem que acabou de ser salva para visualizarmos.
results_image = cv2.imread("results_0.jpg")
plt.imshow(cv2.cvtColor(results_image, cv2.COLOR_BGR2RGB))

# %% [markdown]
# ### Análise dos Resultados (textual)
#
# Vamos extrair os dados da detecção em formato de texto para analisar quantitativamente.

# %%
# Extrair detecções em formato textual.
for result in results:
    boxes = result.boxes
    print(f"Total de objetos detectados: {len(boxes)}")
    print(f"Classes disponíveis no modelo: {len(result.names)} classes")
    print()
    
    # Contar objetos por classe.
    from collections import Counter
    class_counts = Counter()
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        class_counts[cls_name] += 1
    
    print("--- Contagem por classe ---")
    for cls_name, count in class_counts.most_common():
        print(f"  {cls_name}: {count}")
    print()
    
    # Tabela detalhada de cada detecção.
    print("--- Detalhes de cada detecção ---")
    print(f"{'#':<4} {'Classe':<12} {'Confiança':<12} {'x1':<8} {'y1':<8} {'x2':<8} {'y2':<8}")
    print("-" * 60)
    for idx, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"{idx:<4} {cls_name:<12} {conf:<12.4f} {x1:<8.1f} {y1:<8.1f} {x2:<8.1f} {y2:<8.1f}")

# %% [markdown]
# E basta isso para detectarmos objetos que a [YOLO](https://arxiv.org/pdf/1506.02640) foi treinada para detectar.

# %% [markdown]
# ## Próximos Passos e Referências

# %% [markdown]
# Nas próximas práticas iremos ver outras tarefas clássicas de Visão Computacional.
#
# Uma lista não exaustiva de referências segue:
#
# - https://docs.ultralytics.com/
# - https://pytorch.org/
# - https://docs.pytorch.org/vision/main/models.html
# - https://opencv.org/
# - https://learnopencv.com/blogs/
# - https://pyimagesearch.com/

# %% [markdown]
#
# %% [markdown]
# ## Atividades Complementares
#
# Nesta seção, vamos explorar a CLI (linha de comando) da ultralytics e testar o comportamento dos modelos YOLOv11n e YOLOv11s em uma imagem fora do domínio comum (imagem de anime/desenho), utilizando a imagem do **Sonic**.
#
# Isso ajuda a entender como a rede generaliza para representações artísticas e como modelos maiores se comportam em comparação com modelos menores.

# %% [markdown]
# ### 1. Exploração do YOLO via Linha de Comando (CLI)
#
# A biblioteca Ultralytics possui uma interface de linha de comando muito poderosa. Podemos usá-la diretamente no terminal ou em células do Jupyter precedidas por `!`.
#
# Exemplo de comando CLI para realizar predição:
# ```bash
# yolo predict model=yolo11n.pt source="street.jpeg" conf=0.25 device=cpu
# ```
#
# Vamos executar este comando via CLI na imagem do tráfego:

# %%
# Executando a predição via CLI da ultralytics
# Nota: O argumento 'save=True' salva o resultado na pasta 'runs/detect/predict'
# !yolo predict model=yolo11n.pt source="street.jpeg" conf=0.25 save=True

# %% [markdown]
# ### 2. Detecção com Modelos Diferentes (YOLOv11n vs YOLOv11s) na Imagem do Sonic
#
# Vamos carregar a imagem do Sonic (`sonic.jpg`) que está na pasta `img` da raiz do repositório.

# %%
import os

# Verificar onde a imagem está localizada
sonic_path = "/content/sonic.jpg"
if not os.path.exists(sonic_path):
    sonic_path = "sonic.jpg"
if not os.path.exists(sonic_path):
    sonic_path = "../../img/sonic.jpg"
if not os.path.exists(sonic_path):
    sonic_path = "img/sonic.jpg"

if not os.path.exists(sonic_path):
    print(f"ERRO: Imagem do Sonic não encontrada em '{sonic_path}'!")
    print("Certifique-se de que a imagem 'sonic.jpg' está na pasta 'img' na raiz do projeto.")
else:
    print(f"Imagem do Sonic encontrada com sucesso em: {sonic_path}")
    sonic_img = cv2.imread(sonic_path)
    print(f"  Resolução: {sonic_img.shape[1]}x{sonic_img.shape[0]} pixels")

# %% [markdown]
# Agora, vamos carregar e rodar a inferência com dois tamanhos de modelo:
# - **YOLOv11n (Nano):** ~2.6M de parâmetros. Mais rápido e leve.
# - **YOLOv11s (Small):** ~9.4M de parâmetros. Mais preciso, porém um pouco mais pesado.

# %%
# Carregar os modelos
model_nano = ultralytics.YOLO("yolo11n.pt")
model_small = ultralytics.YOLO("yolo11s.pt")  # Fará o download automaticamente

# Executar predições
results_nano = model_nano(sonic_img)
results_small = model_small(sonic_img)

# %% [markdown]
# #### Resultados Textuais - YOLOv11n (Nano)

# %%
print("=== DETECÇÕES COM YOLOv11n (Nano) ===")
for result in results_nano:
    boxes = result.boxes
    print(f"Total de objetos detectados: {len(boxes)}")
    for idx, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        conf = float(box.conf[0])
        print(f"  Detecção {idx+1}: {cls_name} (Confiança: {conf:.4f})")
    
    # Salvar resultado visual
    result.save(filename="sonic_detected_nano.jpg")

# %% [markdown]
# #### Resultados Textuais - YOLOv11s (Small)

# %%
print("=== DETECÇÕES COM YOLOv11s (Small) ===")
for result in results_small:
    boxes = result.boxes
    print(f"Total de objetos detectados: {len(boxes)}")
    for idx, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        conf = float(box.conf[0])
        print(f"  Detecção {idx+1}: {cls_name} (Confiança: {conf:.4f})")
        
    # Salvar resultado visual
    result.save(filename="sonic_detected_small.jpg")

# %% [markdown]
# #### Visualização Lado a Lado
#
# Vamos plotar os resultados das detecções visuais geradas pelos dois modelos.

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 10))

if os.path.exists("sonic_detected_nano.jpg"):
    nano_visual = cv2.imread("sonic_detected_nano.jpg")
    axes[0].imshow(cv2.cvtColor(nano_visual, cv2.COLOR_BGR2RGB))
    axes[0].set_title("YOLOv11n (Nano) no Sonic")
else:
    axes[0].text(0.5, 0.5, "Imagem nano não gerada", ha="center")

if os.path.exists("sonic_detected_small.jpg"):
    small_visual = cv2.imread("sonic_detected_small.jpg")
    axes[1].imshow(cv2.cvtColor(small_visual, cv2.COLOR_BGR2RGB))
    axes[1].set_title("YOLOv11s (Small) no Sonic")
else:
    axes[1].text(0.5, 0.5, "Imagem small não gerada", ha="center")

plt.show()

# %% [markdown]
# ### Discussão Educativa
#
# A imagem do Sonic representa um desafio para modelos treinados no dataset COCO (composto por fotos reais):
# 1. **Diferença de Domínio (Domain Shift):** O Sonic é um personagem fictício (desenho/3D), e a textura, cor azul brilhante e traços corporais dele diferem muito de animais reais.
# 2. **Como a YOLO interpreta:** neste teste, a **YOLOv11n** detectou o Sonic como **"teddy bear"** (~43% de confiança). Faz sentido — sem nenhuma classe de "personagem/desenho" no COCO, a rede encaixa a forma arredondada e colorida na classe real mais próxima.
# 3. **Comparação dos Modelos:** a **YOLOv11s** (maior) detectou **2 "baseball bats" + 1 "teddy bear"** — provavelmente interpretando os nunchakus como tacos de baseball. Ou seja, o modelo maior enxerga mais objetos, mas continua "forçando" o personagem nas 80 classes do COCO, o que reforça o problema de *domain shift*.

