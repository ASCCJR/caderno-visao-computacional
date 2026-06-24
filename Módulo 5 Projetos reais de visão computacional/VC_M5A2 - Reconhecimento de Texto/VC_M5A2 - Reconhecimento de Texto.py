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
# # M5A2 - Reconhecimento de Texto
#
# > **Resumo:** Projeto de **OCR** (reconhecimento óptico de caracteres) com a biblioteca **EasyOCR**: lemos o texto de imagens, extraímos as *bounding boxes* de cada palavra (com confiança) e ainda testamos se há algum texto legível na imagem do Sonic.

# %% [markdown]
# Na prática de hoje vamos utilizar um modelo na tarefa de reconhecimento de texto (OCR — Optical Character Recognition).
#
# O reconhecimento óptico de caracteres é um problema clássico de visão computacional. Historicamente, ele era resolvido com técnicas de extração manual de características e casamento de templates (template matching). Hoje, a abordagem moderna combina **Redes Convolucionais (CNNs)** para extração de características visuais e **Redes Neurais Recorrentes (RNNs)** para processar a sequência temporal dos caracteres e palavras.
#
# Um exemplo clássico do impacto do OCR no mundo real foi o projeto **SVHN (Street View House Numbers)** do Google. Usando a enorme base de imagens coletadas pelo Google Street View, o Google treinou modelos para ler de forma automática e precisa os números de portas de residências. Em muitas cidades, a numeração das ruas é confusa e pouco padronizada; a identificação desses números permitiu mapear os endereços exatos no Google Maps, melhorando drasticamente a precisão da navegação, sistemas de entrega e logística.
#
# Esse notebook está estruturado da seguinte forma.
#
# - Introdução
# - Utilizar Modelo
# - Próximos passos
# - Atividades Complementares

# %% [markdown]
# ## Introdução

# %% [markdown]
# Instalação para os que ainda não possuem a biblioteca instalada.

# %%
# !pip install easyocr

# %% [markdown]
# ## Importações e Imagens de Teste
#
# Importar as bibliotecas e visualizar as imagens.

# %%
import easyocr
from PIL import Image
import matplotlib.pyplot as plt

# %%
eng_image = Image.open("eng.jpeg")
num_image = Image.open("num.jpeg")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))
axes[0].imshow(eng_image)
axes[1].imshow(num_image)


# %% [markdown]
# ## Carregar Modelo e Utilizar Modelo

# %%
reader = easyocr.Reader(["en"]) # Carrega o modelo de inglês para a memória.

# %%
plt.imshow(eng_image)
result = reader.readtext("eng.jpeg", detail=0)
print(result)

# %%
plt.imshow(num_image)
result = reader.readtext("num.jpeg", detail=0)
print(result)

# %% [markdown]
# ## Próximos Passos e Referências

# %% [markdown]
# Nas próximas práticas vamos continuar trabalhando com problemas reais que envolvem Visão Computacional.
#
# Uma lista não exaustiva de referências segue:
#
# - https://github.com/JaidedAI/EasyOCR
# - https://www.jaided.ai/easyocr/
# - https://huggingface.co
# - https://pytorch.org/
# - https://docs.pytorch.org/vision/main/models.html
# - https://opencv.org/
# - https://learnopencv.com/blogs/
# - https://pyimagesearch.com/

# %% [markdown]
# ## Atividades Complementares (Opcional)

# %% [markdown]
# - [ ] Tente alterar as imagens utilizadas nos testes (outras línguas, placas ou embalagens).
# - [x] **Atividade Opcional: Extração de Bounding Boxes com `detail=1`**
#   Vamos configurar o EasyOCR para retornar caixas delimitadoras e desenhá-las sobre as imagens usando a biblioteca PIL (`ImageDraw`).

# %%
from PIL import ImageDraw

def plot_ocr_results(image_path):
    # Executar OCR com detail=1 para obter coordenadas das caixas
    results_detail = reader.readtext(image_path, detail=1)
    
    # Abrir imagem original e desenhar caixas
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    print(f"Texto detectado em {image_path}:")
    for bbox, text, confidence in results_detail:
        print(f"- '{text}' (Confiança: {confidence:.3f})")
        # bbox é uma lista de 4 pontos: [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
        # Achatamos para passar para polygon do PIL: [x0,y0,x1,y1,x2,y2,x3,y3]
        points = [coord for pt in bbox for coord in pt]
        draw.polygon(points, outline="red", width=3)
        
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Testando a função de desenho nas duas imagens originais
plot_ocr_results("eng.jpeg")
plot_ocr_results("num.jpeg")

# %% [markdown]
# - [x] **Atividade Opcional: Teste de OCR no Team Sonic**
#   Vamos buscar se existe algum texto legível na imagem do Sonic (`img/sonic.jpg`) e destacar os locais encontrados.

# %%
import os

# Caminho da imagem do Sonic
sonic_path = "/content/sonic.jpg"
if not os.path.exists(sonic_path):
    sonic_path = "sonic.jpg"
if not os.path.exists(sonic_path):
    sonic_path = "../../img/sonic.jpg"
if not os.path.exists(sonic_path):
    sonic_path = "img/sonic.jpg"

if not os.path.exists(sonic_path):
    print("ERRO: Imagem do Sonic não encontrada!")
else:
    print(f"Imagem do Sonic encontrada em: {sonic_path}")
    plot_ocr_results(sonic_path)
