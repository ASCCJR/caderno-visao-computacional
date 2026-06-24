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
# # M1A2 - Manipulação de arrays com NumPy
#
# > **Resumo:** Aqui aprendo a base de tudo em visão computacional — manipular vetores e matrizes com NumPy. Imagens são matrizes, então dominar isso é essencial.

# %% [markdown]
# **Estrutura do notebook:**
#
# - Instalação e importação
# - Operações básicas com vetores
# - Operações básicas com matrizes
# - Próximos passos
# - ✅ Atividades Complementares (resolvidas)

# %% [markdown]
# ## Instalação e Importação da Biblioteca

# %%
# !pip install numpy

# %%
import numpy as np

# %% [markdown]
# ## Operações Básicas com Vetores

# %% [markdown]
# Vetores são a estrutura mais simples do NumPy. Aqui vamos criar dois vetores e brincar com operações.
#
# Usamos $\vec{u} = (1, 3)$ e $\vec{v} = (3, 1)$ como na figura:
#
# ![vetores u e v](vetores.png)

# %%
# Criando vetores.
u = np.array([1, 3])
v = np.array([3, 1])

# %% [markdown]
# **Soma e subtração** — funcionam elemento a elemento, como esperado da álgebra linear:

# %%
y = u - v
w = u + v

print(f"Vetor y: {y}\nVetor w: {w}")

# %% [markdown] vscode={"languageId": "plaintext"}
# Visualizando no plano cartesiano:
#
# ![subtração e adição com u e v](sub-add-vecs.png)

# %% [markdown]
# **Multiplicação e divisão** — também elemento a elemento.
#
# Dá pra usar isso pra escalar vetores (tipo $\alpha \cdot \vec{u}$):

# %%
a = u / v
b = u * v

print(f"Vetor a: {a}\nVetor b: {b}")

alpha = 2
escalar = alpha * u
print(f"u multiplicado pelo escalar alpha: {escalar}")

# %% [markdown]
# **Produto interno e vetorial** — as operações geométricas clássicas:

# %%
interno = np.dot(u, v)


# Para o produto vetorial iremos utilizar vetores tridimensionais.
u3 = np.array([1, 3, 1])
v3 = np.array([3, 1, 1])
vetorial = np.cross(u3, v3)

print(f"Resultado produto interno: {interno}\nResultado produto vetorial: {vetorial}")

# %% [markdown]
# ## Operações Básicas com Matrizes

# %% [markdown]
# Matrizes = vetores bidimensionais no NumPy. Imagens são exatamente isso!
#
# Vamos criar uma matriz identidade e uma genérica:

# %%
I = np.array([[1, 0],[0, 1]])
A = np.array([[2, 1],[1, 2]])

# %% [markdown] vscode={"languageId": "plaintext"}
# As operações funcionam igual vetores — elemento a elemento ou usando funções específicas:

# %%
B = A + I

print(f"Matriz A:\n{A}\n\nMatriz Identidade:\n{I}\n\nMatriz B = A + I:\n{B}")

# %%
B = A - I

print(f"Matriz A:\n{A}\n\nMatriz Identidade:\n{I}\n\nMatriz B = A - I:\n{B}")

# %% [markdown]
# **Broadcasting** — NumPy é esperto, não precisa nem ter as mesmas dimensões:

# %%
C = A + u

print(f"Matriz A:\n{A}\nVetor u:\n{u}\n\nMatriz C = A + u:\n{C}")

# %% [markdown]
# **Multiplicação de matrizes** — aqui tem que respeitar as regras de dimensão (colunas de A = linhas de B):

# %%
D = np.matmul(A, I)

print(f"Matriz A:\n{A}\n\nMatriz Identidade:\n{I}\n\nMatriz D = A x I:\n{D}")

# %% [markdown]
# ## Próximos Passos e Referências

# %% [markdown]
# Nas próximas aulas vamos ver como essa base NumPy se conecta diretamente com imagens e visão computacional.
#
# **Referências úteis:**
#
# - https://numpy.org/devdocs/user/absolute_beginners.html
# - https://www.geogebra.org/
# - https://numpy.org/devdocs/user/whatisnumpy.html
# - https://numpy.org/devdocs/user/quickstart.html

# %% [markdown]
# ## ✅ Atividades Complementares

# %% [markdown]
# ### 1. Formas alternativas de instanciar matrizes e vetores especiais
#
# NumPy tem funções prontas pra criar matrizes comuns sem precisar digitar tudo na mão:

# %%
# Matriz identidade — np.eye(n) cria uma identidade n×n
I_auto = np.eye(3)
print(f"Identidade 3x3 com np.eye:\n{I_auto}\n")

# Vetor/matriz de zeros
zeros_vec = np.zeros(5)
zeros_mat = np.zeros((3, 3))
print(f"Vetor de zeros: {zeros_vec}")
print(f"Matriz de zeros:\n{zeros_mat}\n")

# Vetor/matriz de uns
ones_vec = np.ones(4)
ones_mat = np.ones((2, 3))
print(f"Vetor de uns: {ones_vec}")
print(f"Matriz de uns:\n{ones_mat}\n")

# Matriz preenchida com valor específico
full_mat = np.full((2, 2), 7)
print(f"Matriz 2x2 preenchida com 7:\n{full_mat}\n")

# Sequências úteis
arange = np.arange(0, 10, 2)  # como range() mas retorna array
linspace = np.linspace(0, 1, 5)  # 5 pontos igualmente espaçados entre 0 e 1
print(f"np.arange(0, 10, 2): {arange}")
print(f"np.linspace(0, 1, 5): {linspace}")

# %% [markdown]
# ### 2. Formas alternativas de realizar operações (dot, matmul)
#
# Além de `np.dot()` e `np.matmul()`, existem outras sintaxes:

# %%
# Recriar as matrizes
A = np.array([[2, 1], [1, 2]])
I = np.array([[1, 0], [0, 1]])

# --- Produto interno (vetores) ---
u = np.array([1, 3])
v = np.array([3, 1])

# Forma 1: np.dot()
dot1 = np.dot(u, v)

# Forma 2: método do objeto
dot2 = u.dot(v)

# Forma 3: operador @ (Python 3.5+) — meu favorito, mais limpo
dot3 = u @ v

print(f"np.dot(u, v) = {dot1}")
print(f"u.dot(v)     = {dot2}")
print(f"u @ v        = {dot3}")
print(f"Todos iguais? {dot1 == dot2 == dot3}\n")

# --- Multiplicação de matrizes ---
# Forma 1: np.matmul()
mat1 = np.matmul(A, I)

# Forma 2: operador @
mat2 = A @ I

# Forma 3: método .dot() (funciona pra matrizes também)
mat3 = A.dot(I)

print(f"np.matmul(A, I):\n{mat1}\n")
print(f"A @ I:\n{mat2}\n")
print(f"A.dot(I):\n{mat3}\n")
print(f"Todos iguais? {np.array_equal(mat1, mat2) and np.array_equal(mat2, mat3)}")

# %% [markdown]
# > 📝 **Dica de estudo:** O operador `@` é o mais pythonico e legível. Uso ele sempre que possível.

# %% [markdown]
# ### 3. Arrays multi-dimensionais (3D)
#
# Em visão computacional, imagens coloridas são arrays 3D: (altura, largura, canais RGB).
# Vamos explorar essa estrutura:

# %%
# Criar um array 3D — imagine como uma "pilha" de matrizes
arr_3d = np.array([
    [[1, 2, 3],
     [4, 5, 6]],
    
    [[7, 8, 9],
     [10, 11, 12]]
])

print(f"Array 3D:\n{arr_3d}")
print(f"\nShape: {arr_3d.shape}")  # (2, 2, 3) → 2 "camadas", 2 linhas, 3 colunas
print(f"Dimensões (ndim): {arr_3d.ndim}")
print(f"Total de elementos: {arr_3d.size}")

# %%
# Acessar elementos específicos
print(f"Camada 0:\n{arr_3d[0]}\n")
print(f"Camada 1, Linha 0: {arr_3d[1, 0]}")
print(f"Elemento [1, 0, 2]: {arr_3d[1, 0, 2]}")

# %%
# Simular uma "mini-imagem" RGB de 3x3 pixels
mini_img = np.zeros((3, 3, 3), dtype=np.uint8)

# Pixel vermelho no canto superior esquerdo
mini_img[0, 0] = [255, 0, 0]

# Pixel verde no centro
mini_img[1, 1] = [0, 255, 0]

# Pixel azul no canto inferior direito
mini_img[2, 2] = [0, 0, 255]

print(f"Mini imagem RGB (3x3):\n{mini_img}")
print(f"\nShape: {mini_img.shape} → (altura, largura, canais RGB)")

# %%
# Operações em 3D funcionam igual!
arr_a = np.ones((2, 3, 3))
arr_b = np.full((2, 3, 3), 2)

soma_3d = arr_a + arr_b
mult_3d = arr_a * arr_b

print(f"Soma 3D (todos viram 3.0):\n{soma_3d}\n")
print(f"Multiplicação 3D (todos viram 2.0):\n{mult_3d}")

# %% [markdown]
# > 📝 **Conexão com visão computacional:** Quando a gente carrega uma imagem colorida com OpenCV ou matplotlib, ela vira exatamente um array 3D com shape `(altura, largura, 3)`. Cada pixel tem 3 valores: R, G, B. Então tudo que aprendemos aqui se aplica diretamente!
