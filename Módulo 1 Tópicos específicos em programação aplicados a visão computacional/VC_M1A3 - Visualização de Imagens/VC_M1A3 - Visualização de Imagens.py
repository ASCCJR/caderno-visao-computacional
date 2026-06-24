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
# # M1A3 - Visualização de Imagens
#
# > **Resumo:** Primeiro contato com imagens de verdade! Aprendemos a ler imagens do disco e visualizar usando matplotlib. Aqui uso o Sonic como imagem de teste 🦔

# %% [markdown]
# **Estrutura do notebook:**
#
# - Instalação e importação
# - Ler imagens do disco
# - Visualizar imagens
# - Próximos passos
# - ✅ Atividades Complementares (resolvidas)

# %% [markdown]
# ## Instalação e Importação da Biblioteca

# %%
# !pip install matplotlib

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Ler Imagens do Disco

# %% [markdown]
# A função `plt.imread` lê uma imagem e retorna um **array NumPy** — é aqui que tudo se conecta com a aula anterior!

# %%
image = plt.imread("sonic.jpg")

# %% [markdown]
# Vamos verificar o tipo e espiar os valores de um pixel:

# %%
print(f"Tipo: {type(image)}")
print(f"Shape: {image.shape}")
print(f"Dtype: {image.dtype}")

# %%
# Valor RGB do pixel no canto superior esquerdo
image[0,0]

# %% [markdown]
# ## Visualizar Imagens

# %% [markdown]
# Com a imagem carregada em memória (como array NumPy), usamos `plt.imshow` pra visualizar:

# %%
plt.imshow(image)
plt.title("Sonic")
plt.axis("off")
plt.show()

# %% [markdown]
# ## Próximos Passos e Referências

# %% [markdown]
# Nas próximas aulas vamos explorar OpenCV, que tem muito mais ferramentas pra manipular imagens.
#
# **Referências úteis:**
#
# - https://matplotlib.org/stable/
# - https://matplotlib.org/stable/tutorials/images.html

# %% [markdown]
# ## ✅ Atividades Complementares

# %% [markdown]
# ### 1. Ler outra imagem e rodar todos os comandos
#
# Vamos usar o Tails como segunda imagem:

# %%
image_tails = plt.imread("tails.jpg")

print(f"Tipo: {type(image_tails)}")
print(f"Shape: {image_tails.shape}")
print(f"Dtype: {image_tails.dtype}")
print(f"Pixel [0,0]: {image_tails[0,0]}")

# %%
plt.imshow(image_tails)
plt.title("Tails")
plt.axis("off")
plt.show()

# %% [markdown]
# ### 2. Alterar o espaço de cores
#
# Imagens são armazenadas em RGB (Red, Green, Blue), mas podemos visualizar canais individuais ou converter pra escala de cinza:

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Imagem original
axes[0].imshow(image)
axes[0].set_title("Original (RGB)")
axes[0].axis("off")

# Canal Vermelho (R)
axes[1].imshow(image[:, :, 0], cmap="Reds")
axes[1].set_title("Canal R")
axes[1].axis("off")

# Canal Verde (G)
axes[2].imshow(image[:, :, 1], cmap="Greens")
axes[2].set_title("Canal G")
axes[2].axis("off")

# Canal Azul (B)
axes[3].imshow(image[:, :, 2], cmap="Blues")
axes[3].set_title("Canal B")
axes[3].axis("off")

plt.suptitle("Sonic — Canais de cor separados", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Converter pra escala de cinza (média ponderada dos canais)
gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(image)
axes[0].set_title("Original")
axes[0].axis("off")
axes[1].imshow(gray, cmap="gray")
axes[1].set_title("Escala de cinza")
axes[1].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# > 📝 **Nota:** Os pesos `[0.2989, 0.5870, 0.1140]` não são arbitrários — seguem a fórmula de luminância (ITU-R BT.601) que leva em conta como o olho humano percebe cada cor. Verde tem mais peso porque somos mais sensíveis a ele!

# %% [markdown]
# ### 3. Criar e visualizar imagens a partir de arrays NumPy
#
# Não precisamos ler do disco — podemos criar imagens "na mão" com NumPy:

# %%
# Gradiente horizontal (preto → branco)
gradient_h = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))

# Gradiente vertical (preto → branco)
gradient_v = np.tile(np.linspace(0, 255, 256, dtype=np.uint8).reshape(-1, 1), (1, 256))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(gradient_h, cmap="gray")
axes[0].set_title("Gradiente horizontal")
axes[0].axis("off")
axes[1].imshow(gradient_v, cmap="gray")
axes[1].set_title("Gradiente vertical")
axes[1].axis("off")
plt.tight_layout()
plt.show()

# %%
# Criar uma imagem RGB colorida — quadrado com 4 cores
rgb_img = np.zeros((200, 200, 3), dtype=np.uint8)
rgb_img[0:100, 0:100] = [255, 0, 0]      # Vermelho (canto superior esquerdo)
rgb_img[0:100, 100:200] = [0, 0, 255]     # Azul (canto superior direito)
rgb_img[100:200, 0:100] = [255, 255, 0]   # Amarelo (canto inferior esquerdo)
rgb_img[100:200, 100:200] = [0, 255, 0]   # Verde (canto inferior direito)

plt.imshow(rgb_img)
plt.title("Imagem criada com NumPy")
plt.axis("off")
plt.show()

# %%
# Padrão xadrez
xadrez = np.zeros((200, 200), dtype=np.uint8)
tamanho_casa = 25
for i in range(0, 200, tamanho_casa):
    for j in range(0, 200, tamanho_casa):
        if (i // tamanho_casa + j // tamanho_casa) % 2 == 0:
            xadrez[i:i+tamanho_casa, j:j+tamanho_casa] = 255

plt.imshow(xadrez, cmap="gray")
plt.title("Tabuleiro de xadrez — puro NumPy")
plt.axis("off")
plt.show()

# %% [markdown]
# > 📝 **Conexão com visão computacional:** Criar imagens com NumPy é útil pra entender como os pixels funcionam, e também pra criar máscaras, kernels e padrões de teste mais adiante no curso.
