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
# # M1A7 - Fundamentos de Filtros Espaciais e Convoluções
#
# > **Resumo:** Convoluções são a operação mais importante em visão computacional — são a base de CNNs, detecção de bordas, blur, sharpening, etc. Aqui aprendemos a aplicar kernels com `cv2.filter2D`. Imagem de teste: Amy 🌸

# %% [markdown]
# **Estrutura do notebook:**
#
# - Introdução
# - Convoluções
# - Próximos passos
# - ✅ Atividades Complementares (resolvidas)

# %% [markdown]
# ## Introdução

# %%
# !pip install opencv-python

# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2

# %%
image = cv2.imread("amy.jpg")
print(f"Shape: {image.shape}")

# %%
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Amy original")
plt.axis("off")
plt.show()

# %% [markdown]
# ## Convoluções
#
# Uma convolução é basicamente "deslizar um kernel (matrizinha pequena) pela imagem inteira", multiplicando e somando valores em cada posição. O resultado depende dos valores do kernel.

# %% [markdown]
# ### Definindo kernels
#
# Cada kernel produz um efeito diferente:

# %%
# Sobel — detecta bordas na direção X
x_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# Sobel — detecta bordas na direção Y
y_sobel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Sharpening — realça detalhes
sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

print(f"Sobel X:\n{x_sobel}\n")
print(f"Sobel Y:\n{y_sobel}\n")
print(f"Sharpening:\n{sharp}")

# %% [markdown]
# ### Aplicando os kernels
#
# `cv2.filter2D(imagem, ddepth, kernel)` faz a convolução. O `ddepth` é a profundidade de bits do resultado.
#
# > ⚠️ **Cuidado com Sobel em `uint8`:** o Sobel produz valores **negativos** (a borda tem gradiente nos dois sentidos). Se a imagem for `uint8` e usarmos `ddepth=-1` ("manter o tipo de entrada"), todo valor negativo é cortado pra 0 e metade das bordas some. Por isso convertemos pra `float32` antes e depois combinamos as direções X e Y pela **magnitude do gradiente** $\sqrt{g_x^2 + g_y^2}$.

# %%
# Converter pra cinza primeiro (bordas ficam mais claras de ver)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap="gray")
plt.title("Amy em escala de cinza")
plt.axis("off")
plt.show()

# %%
# Detecção de bordas com Sobel
gray_f = gray.astype(np.float32)              # float pra preservar gradientes negativos
grad_x = cv2.filter2D(gray_f, -1, x_sobel)    # bordas verticais
grad_y = cv2.filter2D(gray_f, -1, y_sobel)    # bordas horizontais
image_sobel = cv2.magnitude(grad_x, grad_y)   # combina X e Y -> intensidade da borda
image_sobel = np.clip(image_sobel, 0, 255).astype(np.uint8)
plt.imshow(image_sobel, cmap="gray")
plt.title("Bordas detectadas (Sobel X + Y)")
plt.axis("off")
plt.show()

# %%
# Sharpening na imagem colorida
image_sharp = cv2.filter2D(image, -1, sharp)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")
axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(image_sharp, cv2.COLOR_BGR2RGB))
axes[1].set_title("Com sharpening")
axes[1].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Próximos Passos e Referências

# %% [markdown]
# A partir daqui vamos avançar pra técnicas mais sofisticadas de processamento de imagens e entrar no mundo de machine learning aplicado a visão.
#
# **Referências úteis:**
#
# - https://opencv.org/
# - https://learnopencv.com/blogs/
# - https://pyimagesearch.com/

# %% [markdown]
# ## ✅ Atividades Complementares

# %% [markdown]
# ### 1. Ler outra imagem e aplicar os filtros

# %%
knuckles = cv2.imread("knuckles.jpg")
knuckles_gray = cv2.cvtColor(knuckles, cv2.COLOR_BGR2GRAY)

# Aplicar Sobel (mesma técnica: float + magnitude)
knuckles_gray_f = knuckles_gray.astype(np.float32)
k_grad_x = cv2.filter2D(knuckles_gray_f, -1, x_sobel)
k_grad_y = cv2.filter2D(knuckles_gray_f, -1, y_sobel)
k_sobel = cv2.magnitude(k_grad_x, k_grad_y)
k_sobel = np.clip(k_sobel, 0, 255).astype(np.uint8)

# Aplicar sharpening
k_sharp = cv2.filter2D(knuckles, -1, sharp)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(knuckles, cv2.COLOR_BGR2RGB))
axes[0].set_title("Knuckles original")
axes[0].axis("off")
axes[1].imshow(k_sobel, cmap="gray")
axes[1].set_title("Bordas (Sobel)")
axes[1].axis("off")
axes[2].imshow(cv2.cvtColor(k_sharp, cv2.COLOR_BGR2RGB))
axes[2].set_title("Sharpening")
axes[2].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2. Experimentar com diferentes kernels
#
# Vamos testar vários tipos de kernels e comparar os efeitos:

# %%
# --- Blur (desfoque) ---
# Kernel de média: todos os valores iguais, soma = 1
blur_3x3 = np.ones((3, 3), dtype=np.float32) / 9
blur_5x5 = np.ones((5, 5), dtype=np.float32) / 25
blur_9x9 = np.ones((9, 9), dtype=np.float32) / 81

img_blur3 = cv2.filter2D(image, -1, blur_3x3)
img_blur5 = cv2.filter2D(image, -1, blur_5x5)
img_blur9 = cv2.filter2D(image, -1, blur_9x9)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")
axes[1].imshow(cv2.cvtColor(img_blur3, cv2.COLOR_BGR2RGB))
axes[1].set_title("Blur 3x3")
axes[2].imshow(cv2.cvtColor(img_blur5, cv2.COLOR_BGR2RGB))
axes[2].set_title("Blur 5x5")
axes[3].imshow(cv2.cvtColor(img_blur9, cv2.COLOR_BGR2RGB))
axes[3].set_title("Blur 9x9")
for ax in axes:
    ax.axis("off")
plt.suptitle("Efeito do tamanho do kernel de blur", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# > 📝 Quanto maior o kernel de blur, mais desfocada fica a imagem. Isso porque a média é calculada sobre mais pixels vizinhos.

# %%
# --- Emboss (relevo) ---
emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

img_emboss = cv2.filter2D(image, -1, emboss)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")
axes[1].imshow(cv2.cvtColor(img_emboss, cv2.COLOR_BGR2RGB))
axes[1].set_title("Emboss (relevo)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %%
# --- Edge detection: Laplaciano ---
laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

img_laplacian = cv2.filter2D(gray, -1, laplacian)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Escala de cinza")
axes[1].imshow(img_laplacian, cmap="gray")
axes[1].set_title("Laplaciano (detecção de bordas)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %%
# --- Comparação final: todos os efeitos lado a lado ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

effects = [
    (cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "Original"),
    (cv2.cvtColor(img_blur5, cv2.COLOR_BGR2RGB), "Blur 5x5"),
    (cv2.cvtColor(image_sharp, cv2.COLOR_BGR2RGB), "Sharpening"),
    (image_sobel, "Sobel (bordas)"),
    (cv2.cvtColor(img_emboss, cv2.COLOR_BGR2RGB), "Emboss (relevo)"),
    (img_laplacian, "Laplaciano (bordas)")
]

for ax, (img, title) in zip(axes.flat, effects):
    if len(img.shape) == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)
    ax.set_title(title, fontsize=12)
    ax.axis("off")

plt.suptitle("Amy — Galeria de efeitos com convoluções", fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
# > 📝 **Conexão com Deep Learning:** Em CNNs (Redes Neurais Convolucionais), os kernels são **aprendidos automaticamente** durante o treinamento. A rede descobre sozinha quais filtros são úteis pra cada tarefa — detecção de objetos, classificação, segmentação, etc. Aqui fizemos "na mão" pra entender o conceito!
