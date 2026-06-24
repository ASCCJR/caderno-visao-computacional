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
# # M1A5 - Operações Básicas em Imagens
#
# > **Resumo:** Transformações geométricas (crop, translação, rotação) e funções de desenho do OpenCV. Imagem de teste: Tails 🦊

# %% [markdown]
# **Estrutura do notebook:**
#
# - Introdução
# - Transformações de imagens (recorte, translação e rotação)
# - Funções utilitárias de desenhos
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
image = cv2.imread("tails.jpg")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Tails original")
plt.axis("off")
plt.show()

# %% [markdown]
# ## Transformações de Imagens

# %% [markdown]
# Transformações geométricas são fundamentais — usadas pra data augmentation, alinhamento, e pré-processamento.

# %% [markdown]
# ### Crop (Recorte)
#
# Fatiar o array NumPy diretamente — `image[y1:y2, x1:x2]`:

# %%
crop = image[0:500, 0:500]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")
axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
axes[1].set_title("Crop [0:500, 0:500]")
axes[1].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Translação
#
# Mover a imagem usando uma matriz de transformação afim 2×3:

# %%
# Dimensões da imagem original.
height, width = image.shape[:2]

# Matriz de translação — deslocar 200px direita, 100px baixo
translation_x, translation_y = 200, 100
translation_matrix = np.float32([
    [1, 0, translation_x],
    [0, 1, translation_y]
])

# Aplicando a transformação
translated = cv2.warpAffine(image, translation_matrix, (width, height))

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")
axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(translated, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Transladado ({translation_x}px, {translation_y}px)")
axes[1].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# > 📝 A matriz `[[1, 0, tx], [0, 1, ty]]` é a forma matricial de "somar (tx, ty) a cada coordenada". Elegante!

# %% [markdown]
# ### Rotação
#
# `cv2.getRotationMatrix2D` monta a matriz de rotação pra gente:

# %%
height, width = image.shape[:2]
center = (width // 2, height // 2)
angle = 45
scale = 1.0

rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
rotated = cv2.warpAffine(image, rotation_matrix, (width, height))

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")
axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Rotacionado {angle}°")
axes[1].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Funções utilitárias de Desenhos

# %% [markdown]
# OpenCV tem funções pra desenhar formas diretamente nos arrays. Útil pra marcar regiões de interesse, bounding boxes, etc:

# %%
image_draw = image.copy()  # Sempre copiar antes de desenhar!
# Enquadrando o rosto do Tails (centro-direita da imagem 1080x1080)
image_draw = cv2.rectangle(image_draw, (400, 150), (950, 750), (0, 0, 255), 5)
plt.imshow(cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB))
plt.title("Bounding box no rosto do Tails")
plt.axis("off")
plt.show()

# %% [markdown]
# ## Próximos Passos e Referências

# %% [markdown]
# Nas próximas aulas vamos entrar em filtros e convoluções — a base de detecção de bordas e features.
#
# **Referências úteis:**
#
# - https://opencv.org/blog/image-rotation-and-translation-using-opencv/
# - https://opencv.org/
# - https://learnopencv.com/blogs/
# - https://pyimagesearch.com/

# %% [markdown]
# ## ✅ Atividades Complementares

# %% [markdown]
# ### 1. Ler outra imagem e aplicar todas as transformações

# %%
sonic = cv2.imread("sonic.jpg")
sonic_rgb = cv2.cvtColor(sonic, cv2.COLOR_BGR2RGB)

h, w = sonic.shape[:2]
print(f"Sonic — Shape: {sonic.shape}")

# Crop
crop_s = sonic_rgb[50:400, 100:500]

# Translação
t_matrix = np.float32([[1, 0, 150], [0, 1, 80]])
translated_s = cv2.warpAffine(sonic, t_matrix, (w, h))

# Rotação
r_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), 30, 1.0)
rotated_s = cv2.warpAffine(sonic, r_matrix, (w, h))

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(sonic_rgb)
axes[0, 0].set_title("Sonic original")
axes[0, 1].imshow(crop_s)
axes[0, 1].set_title("Crop")
axes[1, 0].imshow(cv2.cvtColor(translated_s, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Transladado")
axes[1, 1].imshow(cv2.cvtColor(rotated_s, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Rotacionado 30°")

for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2. Outras transformações úteis
#
# **Flip** (espelhar), **resize com diferentes interpolações**, e **transformação de perspectiva**:

# %%
# --- Flip (espelhar) ---
flip_h = cv2.flip(sonic, 1)   # 1 = horizontal
flip_v = cv2.flip(sonic, 0)   # 0 = vertical
flip_hv = cv2.flip(sonic, -1) # -1 = ambos

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
titles = ["Original", "Flip horizontal", "Flip vertical", "Flip ambos"]
images = [sonic, flip_h, flip_v, flip_hv]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()

# %%
# --- Resize com diferentes interpolações ---
small = cv2.resize(sonic_rgb, (100, 100))  # Reduzir bastante

# Agora ampliar de volta com diferentes métodos
nearest = cv2.resize(small, (400, 400), interpolation=cv2.INTER_NEAREST)
linear = cv2.resize(small, (400, 400), interpolation=cv2.INTER_LINEAR)
cubic = cv2.resize(small, (400, 400), interpolation=cv2.INTER_CUBIC)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(nearest)
axes[0].set_title("NEAREST (pixelado)")
axes[1].imshow(linear)
axes[1].set_title("LINEAR (suave)")
axes[2].imshow(cubic)
axes[2].set_title("CUBIC (mais suave)")

for ax in axes:
    ax.axis("off")
plt.suptitle("Comparação de interpolações ao ampliar", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# > 📝 **Dica:** `INTER_NEAREST` é rápido mas pixelado. `INTER_LINEAR` é o padrão. `INTER_CUBIC` é mais lento mas mais suave — bom pra ampliações.

# %%
# --- Transformação de Perspectiva ---
h, w = sonic.shape[:2]

# Pontos de origem (cantos da imagem)
pts_orig = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

# Pontos de destino (simulando uma perspectiva)
pts_dest = np.float32([[50, 50], [w - 80, 30], [30, h - 20], [w - 50, h - 50]])

# Calcula a matriz de perspectiva
M = cv2.getPerspectiveTransform(pts_orig, pts_dest)
perspective = cv2.warpPerspective(sonic, M, (w, h))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(sonic, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")
axes[1].imshow(cv2.cvtColor(perspective, cv2.COLOR_BGR2RGB))
axes[1].set_title("Transformação de perspectiva")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3. Desenhar outros formatos e textos

# %%
canvas = sonic.copy()
canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

# Retângulo
cv2.rectangle(canvas, (50, 50), (300, 200), (0, 255, 0), 3)

# Círculo
cv2.circle(canvas, (400, 150), 80, (255, 0, 0), 3)

# Linha
cv2.line(canvas, (0, 0), (canvas.shape[1], canvas.shape[0]), (0, 0, 255), 2)

# Elipse
cv2.ellipse(canvas, (300, 350), (100, 50), 45, 0, 360, (255, 255, 0), 2)

# Texto
cv2.putText(canvas, "Sonic the Hedgehog!", (30, canvas.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
plt.title("Sonic com desenhos OpenCV")
plt.axis("off")
plt.show()

# %% [markdown]
# > 📝 **Lembrete:** `cv2.rectangle`, `cv2.circle` etc. **modificam a imagem in-place**! Sempre faça `.copy()` antes se quiser preservar o original.
