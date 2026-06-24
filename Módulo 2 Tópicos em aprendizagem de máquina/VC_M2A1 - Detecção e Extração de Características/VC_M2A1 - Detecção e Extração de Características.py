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
# # M2A1 - Detecção e Extração de Características
#
# > **Resumo:** Aqui começamos o Módulo 2! O objetivo é extrair "impressões digitais" de uma imagem — pontos de interesse (keypoints) e seus descritores — usando SIFT e ORB. Isso é a base pra detecção de objetos.
#
# > 📝 **Conceito-chave:** Características precisam ser **invariantes** — não podem mudar muito se a iluminação variar, o objeto girar, ou estiver mais longe/perto.

# %% [markdown]
# **Estrutura do notebook:**
#
# - Setup (upload de imagens pro Colab)
# - Extração de Características (SIFT e ORB)
# - Próximos passos
# - ✅ Atividades Complementares (resolvidas)

# %% [markdown]
# ## Importações e Leitura da Imagem

# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2

# %%
image = cv2.imread("/content/sonic.jpg")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Sonic — nossa imagem de teste")
plt.axis("off")
plt.show()
print(f"Shape: {image.shape}")

# %% [markdown]
# ## Extração de Características

# %% [markdown]
# ### SIFT (Scale-Invariant Feature Transform)
#
# O SIFT é um dos algoritmos mais famosos da área. Ele detecta keypoints e calcula descritores que são **invariantes à escala** — ou seja, reconhece o mesmo ponto mesmo se a imagem for ampliada ou reduzida.
#
# - **Keypoints:** pontos de interesse na imagem (cantos, bordas marcantes)
# - **Descritores:** vetores que descrevem a região ao redor de cada keypoint
#
# > 📝 Curiosidade: o SIFT foi patenteado por anos, mas a patente expirou e agora está livre no OpenCV!

# %%
# Criando o detector SIFT
sift = cv2.SIFT_create()

# Detectando keypoints e calculando descritores
keypoints_sift, descriptors_sift = sift.detectAndCompute(image, None)

# Visualizando os keypoints (os círculos mostram escala e orientação)
image_kp_sift = cv2.drawKeypoints(image, keypoints_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(image_kp_sift, cv2.COLOR_BGR2RGB))
plt.title(f"SIFT — {len(keypoints_sift)} keypoints detectados")
plt.axis("off")
plt.show()
print(f"SIFT: {len(keypoints_sift)} keypoints detectados no Sonic.")


# %% [markdown]
# ### ORB (Oriented FAST and Rotated BRIEF)
#
# O ORB é uma alternativa **open-source e mais rápida** ao SIFT. Ótimo pra aplicações em tempo real (celular, câmeras embarcadas).

# %%
# Criando o detector ORB (limitando a 30 keypoints pra visualização)
orb = cv2.ORB_create(30)

# Detectando keypoints e descritores
keypoints_orb, descriptors_orb = orb.detectAndCompute(image, None)

# Visualizando
image_kp_orb = cv2.drawKeypoints(image, keypoints_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(image_kp_orb, cv2.COLOR_BGR2RGB))
plt.title(f"ORB — {len(keypoints_orb)} keypoints detectados")
plt.axis("off")
plt.show()
print(f"ORB (limitado a 30): {len(keypoints_orb)} keypoints detectados no Sonic.")


# %% [markdown]
# > 📝 **Comparação visual:** Repare como o SIFT detecta MUITO mais keypoints que o ORB (limitamos o ORB a 30). Os círculos do SIFT variam mais de tamanho porque ele trabalha em múltiplas escalas.

# %% [markdown]
# ## Próximos Passos e Referências

# %% [markdown]
# Na próxima aula vamos usar essas características pra **encontrar correspondências** entre imagens diferentes — a base da detecção de objetos!
#
# **Referências:**
#
# - [Artigo original do SIFT](https://www.cs.ubc.ca/~lsigal/425_2024W1/ijcv04.pdf)
# - [Artigo do ORB](https://par.cse.nsysu.edu.tw/resource/paper/2016/161129/ORB-an%20efficient%20alternative%20to%20SIFT%20or%20SURF.pdf)
# - https://opencv.org/
# - https://learnopencv.com/blogs/

# %% [markdown]
# ## ✅ Atividades Complementares

# %% [markdown]
# ### 1. Comparar SIFT vs ORB em outra imagem
#
# > ⚠️ **Importante:** Lembre-se de fazer o upload do arquivo `knuckles.jpg` para a pasta `/content/` no Colab para que esta célula funcione corretamente!

# %%
image2 = cv2.imread("/content/knuckles.jpg")

image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# SIFT
kp_sift2, desc_sift2 = sift.detectAndCompute(image2, None)
img_sift2 = cv2.drawKeypoints(image2, kp_sift2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# ORB (sem limite agora)
orb_full = cv2.ORB_create()
kp_orb2, desc_orb2 = orb_full.detectAndCompute(image2, None)
img_orb2 = cv2.drawKeypoints(image2, kp_orb2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(cv2.cvtColor(img_sift2, cv2.COLOR_BGR2RGB))
axes[0].set_title(f"SIFT — {len(kp_sift2)} keypoints")
axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(img_orb2, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"ORB — {len(kp_orb2)} keypoints")
axes[1].axis("off")
plt.suptitle("Knuckles — SIFT vs ORB", fontsize=14)
plt.tight_layout()
plt.show()

print(f"SIFT no Knuckles: {len(kp_sift2)} keypoints detectados.")
print(f"ORB no Knuckles (sem limite): {len(kp_orb2)} keypoints detectados.")


# %% [markdown]
# ### 2. Ajustar parâmetros do SIFT
#
# O SIFT tem parâmetros que controlam a sensibilidade da detecção:

# %%
# Variando nfeatures (quantidade máxima de keypoints)
configs = [
    ("SIFT padrão", cv2.SIFT_create()),
    ("SIFT max 50", cv2.SIFT_create(nfeatures=50)),
    ("SIFT max 500", cv2.SIFT_create(nfeatures=500)),
    ("SIFT contrastThreshold baixo", cv2.SIFT_create(contrastThreshold=0.01)),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (name, detector) in zip(axes, configs):
    kps, _ = detector.detectAndCompute(image, None)
    img_kp = cv2.drawKeypoints(image, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ax.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{name}\n({len(kps)} kps)")
    ax.axis("off")
plt.suptitle("Impacto dos parâmetros do SIFT", fontsize=14)
plt.tight_layout()
plt.show()

print("Impacto das configurações SIFT no Sonic:")
for name, detector in configs:
    kps, _ = detector.detectAndCompute(image, None)
    print(f"  - {name}: {len(kps)} keypoints detectados.")


# %% [markdown]
# > 📝 **Observação:** `contrastThreshold` baixo = mais keypoints (inclusive em regiões de baixo contraste). `nfeatures` limita a quantidade total.

# %% [markdown]
# ### 3. Outros detectores do OpenCV: AKAZE e BRISK

# %%
# AKAZE — similar ao SIFT mas mais rápido
akaze = cv2.AKAZE_create()
kp_akaze, desc_akaze = akaze.detectAndCompute(image, None)
img_akaze = cv2.drawKeypoints(image, kp_akaze, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# BRISK — Binary Robust Invariant Scalable Keypoints
brisk = cv2.BRISK_create()
kp_brisk, desc_brisk = brisk.detectAndCompute(image, None)
img_brisk = cv2.drawKeypoints(image, kp_brisk, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
detectors = [
    (image_kp_sift, f"SIFT ({len(keypoints_sift)})"),
    (image_kp_orb, f"ORB ({len(keypoints_orb)})"),
    (img_akaze, f"AKAZE ({len(kp_akaze)})"),
    (img_brisk, f"BRISK ({len(kp_brisk)})"),
]
for ax, (img, title) in zip(axes.flat, detectors):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=12)
    ax.axis("off")
plt.suptitle("Sonic — Comparação de 4 detectores", fontsize=16)
plt.tight_layout()
plt.show()

print("Resumo final de detectores no Sonic:")
print(f"  - SIFT: {len(keypoints_sift)} keypoints")
print(f"  - ORB: {len(keypoints_orb)} keypoints")
print(f"  - AKAZE: {len(kp_akaze)} keypoints")
print(f"  - BRISK: {len(kp_brisk)} keypoints")


# %% [markdown]
# > 📝 **Resumo dos detectores:**
# > - **SIFT:** mais keypoints, invariante à escala, descritores float (128D). Mais lento.
# > - **ORB:** rápido, descritores binários (32 bytes). Bom pra tempo real.
# > - **AKAZE:** meio-termo entre SIFT e ORB. Descritores binários.
# > - **BRISK:** similar ao ORB, bom pra grandes datasets.
