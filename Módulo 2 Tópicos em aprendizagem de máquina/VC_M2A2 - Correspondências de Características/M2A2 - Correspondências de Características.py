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
# # M2A2 - Correspondências de Características
#
# > **Resumo:** Agora que sabemos extrair keypoints e descritores, o próximo passo é **comparar** essas características entre imagens diferentes pra encontrar correspondências. Isso é a base de panorâmicas, realidade aumentada, e detecção de objetos!

# %% [markdown]
# **Estrutura do notebook:**
#
# - Setup (upload de imagens pro Colab)
# - Extração de Características
# - Correspondências de Características (BFMatcher)
# - Próximos passos
# - ✅ Atividades Complementares (resolvidas)

# %% [markdown]
# ## Importações e Leitura das Imagens

# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2

# %%
image_1 = cv2.imread("/content/1.jpeg")
image_2 = cv2.imread("/content/2.jpeg")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
axes[0].imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
axes[0].set_title("Imagem 1 (frontal)")
axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
axes[1].set_title("Imagem 2 (lateral)")
axes[1].axis("off")
plt.suptitle("Mesma cena, ângulos diferentes", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Extração de Características
#
# Primeiro, extraímos keypoints e descritores de **ambas** as imagens usando ORB:

# %%
# Criando o detector ORB
orb = cv2.ORB_create()

# Detectando keypoints em ambas as imagens
kp1, des1 = orb.detectAndCompute(image_1, None)
kp2, des2 = orb.detectAndCompute(image_2, None)

# Visualizando keypoints
image_1_kp = cv2.drawKeypoints(image_1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
image_2_kp = cv2.drawKeypoints(image_2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
axes[0].imshow(cv2.cvtColor(image_1_kp, cv2.COLOR_BGR2RGB))
axes[0].set_title(f"Imagem 1 — {len(kp1)} keypoints")
axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(image_2_kp, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Imagem 2 — {len(kp2)} keypoints")
axes[1].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Correspondências de Características
#
# Agora usamos um **matcher** pra comparar os descritores de ambas as imagens e encontrar pares correspondentes.
#
# O `BFMatcher` (Brute Force Matcher) compara **todos** os descritores de uma imagem contra **todos** da outra — é o mais simples mas funciona bem.

# %%
# Criando o matcher de força bruta
# NORM_HAMMING é a norma correta pra descritores binários (ORB, BRISK)
# crossCheck=True garante correspondência mútua (A↔B, não só A→B)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Realizando o matching entre descritores
matches = bf.match(des1, des2)

# Ordenando por distância (menor = melhor correspondência)
matches = sorted(matches, key=lambda x: x.distance)

print(f"Total de matches encontrados: {len(matches)}")
print(f"Distância do melhor match: {matches[0].distance:.1f}")
print(f"Distância do pior match: {matches[-1].distance:.1f}")

# %%
# Visualizar os 5 melhores matches
num = 5
image_matches = cv2.drawMatches(
    cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB), kp1,
    cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB), kp2,
    matches[:num], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(16, 8))
plt.imshow(image_matches)
plt.title(f"Top {num} correspondências (ORB + BFMatcher)")
plt.axis("off")
plt.show()

# %% [markdown]
# > 📝 **Como ler o resultado:** Cada linha conecta um keypoint da imagem 1 ao seu par correspondente na imagem 2. Quanto menor a distância, mais parecidos são os descritores — ou seja, mais confiante é a correspondência.
#
# > O matcher pode errar! Nem toda correspondência é correta. Por isso, ordenamos por distância e pegamos os melhores.

# %% [markdown]
# ## Próximos Passos e Referências

# %% [markdown]
# A partir daqui, o curso avança pra classificadores supervisionados — ou seja, usar essas características pra treinar modelos de ML!
#
# **Referências:**
#
# - https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
# - https://opencv.org/
# - https://learnopencv.com/blogs/

# %% [markdown]
# ## ✅ Atividades Complementares

# %% [markdown]
# ### 1. Usar SIFT ao invés de ORB
#
# O SIFT usa descritores float, então a norma do matcher muda pra `NORM_L2`:

# %%
# SIFT + BFMatcher com NORM_L2
sift = cv2.SIFT_create()
kp1_sift, des1_sift = sift.detectAndCompute(image_1, None)
kp2_sift, des2_sift = sift.detectAndCompute(image_2, None)

bf_l2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches_sift = bf_l2.match(des1_sift, des2_sift)
matches_sift = sorted(matches_sift, key=lambda x: x.distance)

print(f"SIFT matches: {len(matches_sift)}")
orb_top10_dist = [m.distance for m in matches[:10]]
sift_top10_dist = [m.distance for m in matches_sift[:10]]
print(f"Estatísticas de distância dos Top 10 matches:")
print(f"  - ORB: Média={np.mean(orb_top10_dist):.2f}, Mín={np.min(orb_top10_dist):.2f}, Máx={np.max(orb_top10_dist):.2f}")
print(f"  - SIFT: Média={np.mean(sift_top10_dist):.2f}, Mín={np.min(sift_top10_dist):.2f}, Máx={np.max(sift_top10_dist):.2f}")


# Comparação visual: ORB vs SIFT
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# ORB
img_orb = cv2.drawMatches(
    cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB), kp1,
    cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB), kp2,
    matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
axes[0].imshow(img_orb)
axes[0].set_title(f"ORB — top 10 matches (total: {len(matches)})")
axes[0].axis("off")

# SIFT
img_sift = cv2.drawMatches(
    cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB), kp1_sift,
    cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB), kp2_sift,
    matches_sift[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
axes[1].imshow(img_sift)
axes[1].set_title(f"SIFT — top 10 matches (total: {len(matches_sift)})")
axes[1].axis("off")

plt.suptitle("ORB vs SIFT — Correspondências", fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
# > 📝 **ORB vs SIFT pra matching:** SIFT encontrou mais matches (207 vs 124). Repare também na distância: a média do ORB foi ~31.1 (distância Hamming, binária, menor escala) e a do SIFT foi ~65.1 (distância L2/Euclidiana, float, maior escala). Escalas diferentes de distância explicam essa diferença!


# %% [markdown]
# ### 2. Variar a quantidade de matches visualizados

# %%
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
for ax, n in zip(axes.flat, [3, 10, 25, 50]):
    n_real = min(n, len(matches_sift))
    img_m = cv2.drawMatches(
        cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB), kp1_sift,
        cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB), kp2_sift,
        matches_sift[:n_real], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    ax.imshow(img_m)
    ax.set_title(f"Top {n_real} matches (SIFT)")
    ax.axis("off")
plt.suptitle("Mais matches = mais ruído (matches ruins aparecem)", fontsize=14)
plt.tight_layout()
plt.show()

print("Qualidade dos matches variando N:")
for n in [3, 10, 25, 50]:
    n_real = min(n, len(matches_sift))
    dist_mean = np.mean([m.distance for m in matches_sift[:n_real]])
    print(f"  - Top {n_real} matches: Distância média = {dist_mean:.2f}")

# %% [markdown]
# > 📝 **Impacto de N:** Como previsto, a distância média aumenta conforme pegamos mais matches (de 40.3 com N=3 para 132.6 com N=50). Isso acontece porque começamos a incluir os piores matches, que contêm muito mais ruído visual e erros de correspondência!

# %% [markdown]
# ### 3. KNN Matcher com Ratio Test de Lowe
#
#
# O ratio test é uma técnica clássica pra filtrar matches ruins: se os 2 matches mais próximos têm distâncias parecidas, o match é ambíguo e deve ser descartado.

# %%
# KNN matcher (retorna os 2 vizinhos mais próximos)
bf_knn = cv2.BFMatcher(cv2.NORM_L2)
matches_knn = bf_knn.knnMatch(des1_sift, des2_sift, k=2)

# Ratio test de Lowe (threshold = 0.75)
good_matches = []
for m, n in matches_knn:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"Total KNN matches: {len(matches_knn)}")
print(f"Após ratio test (0.75): {len(good_matches)} bons matches")
ratio_retained = len(good_matches) / len(matches_knn) * 100
print(f"Porcentagem de matches mantidos pelo Ratio Test (0.75): {ratio_retained:.1f}%")


# Visualizar
img_good = cv2.drawMatches(
    cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB), kp1_sift,
    cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB), kp2_sift,
    good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
plt.figure(figsize=(16, 8))
plt.imshow(img_good)
plt.title(f"KNN + Ratio Test de Lowe — {len(good_matches)} matches filtrados")
plt.axis("off")
plt.show()

# %% [markdown]
# > 📝 **Ratio test de Lowe:** De 930 matches possíveis, o teste de Lowe (0.75) reteve apenas **68 (7.3%)** correspondências confiáveis. Isso significa que 92.7% das correspondências eram ambíguas ou ruins e foram filtradas! Isso mostra o poder dessa técnica em limpar o ruído do matching.

