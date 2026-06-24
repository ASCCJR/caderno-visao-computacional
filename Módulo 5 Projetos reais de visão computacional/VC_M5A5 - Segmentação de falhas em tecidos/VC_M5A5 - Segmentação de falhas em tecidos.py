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
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="95176816"
# # M5A5 - Segmentação de Falhas em Tecidos
#
# > **Resumo:** Como datasets de falhas industriais reais são raros (segredo de fábrica), usamos uma **proxy task**: classificação de texturas com o dataset **DTD** (47 classes). Refinamos uma **ResNet-18** e, no fim, classificamos a "textura" do pelo do Sonic e do Tails.
#
# > 📝 **Proxy task:** quando faltam dados do problema real, treina-se numa tarefa parecida e disponível (aqui, texturas genéricas) cujas features transferem para o problema-alvo.

# %% [markdown] id="9da2d543"
# > ⚠️ **GPU recomendada** (Ambiente de execução → Alterar o tipo → GPU T4): são 100 épocas de treino da ResNet-18.

# %% [markdown] id="a2ac4c18"
# Na prática de hoje vamos trabalhar com o problema de detecção e classificação de padrões em tecidos e texturas.
#
# Em cenários industriais reais, a **segmentação de falhas em tecidos** (como rasgos, manchas e irregularidades) é vital para a automação de controle de qualidade na indústria têxtil. No entanto, coletar e anotar bases de dados públicas de falhas industriais reais é um grande desafio devido a segredos de negócio e direitos de privacidade das fábricas.
#
# Diante da escassez desses datasets industriais, os desenvolvedores de visão computacional utilizam comumente tarefas substitutas (**proxy tasks**), como a classificação de texturas. Para isso, usaremos o dataset **DTD (Describable Textures Dataset)** do PyTorch. Em vez de segmentação pixel a pixel, refinaremos uma rede convolucional **ResNet-18** para classificar diferentes tipos de texturas visuais complexas.
#
# Esse notebook está estruturado da seguinte forma.
#
# - Introdução
# - Carregar Base de Dados
# - Refinar Modelo
# - Validar Modelo
# - Próximos passos
# - Atividades Complementares

# %% [markdown] id="9a1440d4"
# ## Introdução

# %% [markdown] id="9090a15c"
# Instalação para os que ainda não possuem a biblioteca instalada.

# %% colab={"base_uri": "https://localhost:8080/"} id="2561b9db" outputId="f27e81cd-31c0-4ecf-c6dd-f0f8e01dd55d"
# !pip install torch torchvision tqdm ipywidgets

# %% [markdown] id="1ffc5e8b"
# Importar as bibliotecas

# %% id="1f7026a0"
import torch
import torchvision
from tqdm.notebook import tqdm


# %% [markdown] id="69532420"
# ## Carregar Base de Dados
#
# A primeira tarefa para refinar um modelo é criar a base de dados.
#
# Referência: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.DTD.html

# %% colab={"base_uri": "https://localhost:8080/"} id="fd318926" outputId="d23739cc-ae8d-4645-8a2a-a741bd7ec987"
# Define transforms: Resize and normalize images as expected by most pre-trained models (e.g., ImageNet models use 224x224 input).
# Data augmentation is also crucial for better performance.
image_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224), # Common input size for ImageNet models
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
# Set download=True to automatically fetch the dataset if not already present
train_dataset = torchvision.datasets.DTD(root="./data", split="train", download=True, transform=image_transforms)
val_dataset = torchvision.datasets.DTD(root="./data", split="val", download=True, transform=image_transforms)
test_dataset = torchvision.datasets.DTD(root="./data", split="test", download=True, transform=image_transforms)

# Define DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
_ = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# %% [markdown] id="ee1b1b55" vscode={"languageId": "bat"}
# ## Refinar Modelo
#
# Na prática de hoje iremos refinar o modelo **ResNets** disponível no torchvision.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["3068754d69a7461992eb1b05a6266224", "75a47e273b254c699e1ab794989999d4", "7cf849b667dc4d4baebad5646597aaaf", "fe55dfcee9d04a6a9bedc6d1470db754", "db7f7f983f0143aa859b9d1da8639092", "a8b8e831ca3a4c78a79a7a5daf2ad60a", "28a3171965054b6da3885457c0928182", "c00a3c47774d4dd9a752f3855efd1a38", "aa003e080515484298e4551a4df3eccd", "4623f54f361f45f08e7b9f340b0614c2", "e8d57cbe277a420b8e4ac24742cc132b"]} id="7583287c" outputId="88a7665b-db39-4db3-f7e9-3de4cb6b238d"
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

num_classes = 47

# Get the number of input features for the final layer
in_features = model.fc.in_features

# Replace the existing fully connected layer with a new one
model.fc = torch.nn.Linear(in_features, num_classes)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Treinamento do modelo
model.train()
epochs = 100 # Alterar para treinar mais epocas.
for epoch in tqdm(range(epochs)):
    iteration = 0
    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print(f"Total loss: {loss.item()}")
        iteration += 1



# %% [markdown] id="3f888a2b"
# ## Validar Modelo
#
# Agora vamos testar o nosso modelo.

# %% colab={"base_uri": "https://localhost:8080/"} id="264fd4ec" outputId="e8a5e0c1-0247-4264-b412-ba0a6f71d932"
num_correct = 0
num_samples = 0

# Set model to evaluation mode
model.eval()

# Disable gradient calculation for inference
with torch.no_grad():
    for images, labels in test_loader:
            x = images.to(device)
            y = labels.to(device)

            scores = model(x)

            # Get the index (class) with the highest score
            _, predictions = scores.max(1)

            # Count correct predictions
            num_correct += (predictions == y).sum().item()
            # Count total samples
            num_samples += predictions.size(0)

    # Calculate and print accuracy
    accuracy = float(num_correct) / float(num_samples) * 100
    print(f'Got {num_correct} / {num_samples} with accuracy {accuracy:.2f}%')


# %% [markdown] id="a92154ac"
# ## Próximos Passos e Referências

# %% [markdown] id="0a33dd1c"
# Nas próximas práticas vamos continuar trabalhando com problemas reais que envolvem Visão Computacional.
#
# Uma lista não exaustiva de referências segue:
#
# - https://docs.pytorch.org/vision/main/generated/torchvision.datasets.DTD.html
# - https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
# - https://huggingface.co/datasets
# - https://pytorch.org/
# - https://docs.pytorch.org/vision/main/models.html
# - https://opencv.org/
# - https://learnopencv.com/blogs/
# - https://pyimagesearch.com/

# %% [markdown] id="02185967"
# ## Atividades Complementares (Opcional)

# %% [markdown] id="ce664141"
# - [x] **Classificação de Texturas do Team Sonic** — resolvida logo abaixo (leve: só inferência no modelo já treinado).
# - [x] **Tempo de treino vs acurácia** — resolvida na seção opcional/pesada no fim (treina uma versão curta e compara).
# - [ ] **Sugestão livre:** altere o batch/learning rate e veja como muda a convergência.
#
#   Para exercitar o classificador de texturas ResNet-18 treinado no Describable Textures Dataset (DTD), vamos submeter texturas reais extraídas do Team Sonic!
#   Recortaremos a região central dos pelos do Sonic (`img/sonic.jpg`) e do Tails (`img/tails.jpg`), aplicaremos o mesmo pré-processamento de normalização e verificaremos qual das 47 classes de textura do dataset (como *furry*, *striped*, *dotted*, *stratified*) a rede associa a cada personagem.

# %% colab={"base_uri": "https://localhost:8080/", "height": 759} id="3728225e" outputId="90d9fd8f-ddd6-41bd-9fc0-1565c0febb99"
import matplotlib.pyplot as plt
import os
from PIL import Image

def classify_sonic_texture(image_name, path_to_image):
    if not os.path.exists(path_to_image):
        print(f"Erro: {image_name} não encontrado em {path_to_image}")
        return

    # Carregar imagem original
    img = Image.open(path_to_image).convert("RGB")

    # Recortar patch central de textura (300x300)
    w, h = img.size
    crop_size = min(w, h, 300)
    left = (w - crop_size)/2
    top = (h - crop_size)/2
    right = (w + crop_size)/2
    bottom = (h + crop_size)/2
    img_cropped = img.crop((left, top, right, bottom))

    # Aplicar as mesmas transformações de validação
    input_tensor = image_transforms(img_cropped).unsqueeze(0).to(device)

    # Executar inferência
    model.eval()
    with torch.no_grad():
        scores = model(input_tensor)
        _, pred_idx = scores.max(1)
        pred_class = train_dataset.classes[pred_idx.item()]

    # Exibir resultados
    plt.figure(figsize=(4, 4))
    plt.imshow(img_cropped)
    plt.axis('off')
    plt.title(f"Textura do {image_name}\nPredição DTD: '{pred_class}'")
    plt.show()

# Buscar caminhos das imagens dos personagens
sonic_path = "/content/sonic.jpg"
if not os.path.exists(sonic_path): sonic_path = "sonic.jpg"
if not os.path.exists(sonic_path): sonic_path = "../../img/sonic.jpg"
if not os.path.exists(sonic_path): sonic_path = "img/sonic.jpg"

tails_path = "/content/tails.jpg"
if not os.path.exists(tails_path): tails_path = "tails.jpg"
if not os.path.exists(tails_path): tails_path = "../../img/tails.jpg"
if not os.path.exists(tails_path): tails_path = "img/tails.jpg"

classify_sonic_texture("Sonic", sonic_path)
classify_sonic_texture("Tails", tails_path)

# %% [markdown] id="32322da2"
# > 📝 **O que esperar:** o classificador foi treinado só nas texturas do DTD, então ele "força" o pelo dos personagens em alguma das 47 classes (ex.: *furry*, *fuzzy*, *gauzy*, *smeared*...). A predição exata **varia a cada treino** (não fixamos seed) — o interessante é notar que a rede tenta descrever a *textura*, não o personagem. As predições aparecem nos títulos das figuras acima.

# %% [markdown] id="b02d01e9"
# ---
# # ⏸️ Ponto de parada

# %% [markdown] id="40ed6d61"
# ## Atividade (opcional/pesada): Tempo de Treino vs Acurácia
#
# A pergunta da atividade é: *treinar por menos tempo mantém a performance?* Vamos treinar uma
# ResNet-18 por apenas **5 épocas** e comparar a acurácia no teste com o modelo completo (100 épocas).

# %% colab={"base_uri": "https://localhost:8080/"} id="2ec84fd3" outputId="546b23a5-9219-445b-a3fb-f9591a47f32c"
# Treina uma ResNet18 "rápida" (5 épocas) e compara com o modelo completo (100 épocas, acima)
model_rapido = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model_rapido.fc = torch.nn.Linear(model_rapido.fc.in_features, num_classes)
model_rapido = model_rapido.to(device)
opt_rapido = torch.optim.SGD(model_rapido.parameters(), lr=0.001, momentum=0.9)

model_rapido.train()
for epoch in range(5):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        loss_r = criterion(model_rapido(images), labels)
        opt_rapido.zero_grad()
        loss_r.backward()
        opt_rapido.step()

# Avalia no conjunto de teste
model_rapido.eval()
num_correct_r = num_samples_r = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        _, preds = model_rapido(images).max(1)
        num_correct_r += (preds == labels).sum().item()
        num_samples_r += preds.size(0)
acc_rapido = num_correct_r / num_samples_r * 100

print(f"ResNet18 -   5 épocas:  {acc_rapido:.2f}% no teste")
print(f"ResNet18 - 100 épocas:  {accuracy:.2f}% no teste (treino completo lá em cima)")
print()
print("DTD é um dataset difícil (47 classes de textura). Mais épocas ajudam, mas com custo de tempo —")
print("o ganho costuma diminuir conforme o treino avança (retornos decrescentes).")
