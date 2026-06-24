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
# # M5A4 - Sistema de Vigilância
#
# > **Resumo:** Projeto de **detecção de anomalias em vídeo** (acidentes, roubos, brigas...). Diferente de imagens estáticas, vídeo exige modelagem **temporal** — então usamos um **Swin Transformer 3D**, que aplica atenção no espaço *e* no tempo (tensores `(C, T, H, W)`). Refinamos o modelo para 13 classes de anomalia.

# %% [markdown]
# > ⚠️ **GPU obrigatória + notebook pesado.** Treina um Transformer 3D de vídeo (Swin3D) por 10 épocas e baixa um dataset de vídeos grande; depende de `ffmpeg`/`torchcodec`. É de longe o notebook mais demorado do curso — rode com GPU e paciência.

# %% [markdown]
# Na prática de hoje vamos refinar um modelo para a tarefa de sistema de vigilância inteligente (detecção de anomalias em vídeo).
#
# Sistemas de vigilância automatizados monitoram continuamente grandes volumes de feeds de vídeo para identificar comportamentos anômalos (como acidentes de trânsito, assaltos, brigas ou invasões). Em vez de processar imagens estáticas independentes, a análise de vídeo exige **modelagem temporal**, pois a anomalia geralmente é definida pela dinâmica dos movimentos ao longo do tempo (a terceira dimensão).
#
# Para isso, utilizaremos um modelo de vídeo tridimensional, o **Swin Transformer 3D (Swin3D)**. Modelos 3D tratam os dados como tensores de forma `(C, T, H, W)` (onde `T` é o tempo/número de frames). O Swin3D aplica janelas de atenção móveis tanto no espaço bidimensional quanto no tempo para capturar dependências de movimento espaço-temporais complexas.
#
# Esse notebook está estruturado da seguinte forma.
#
# - Introdução
# - Carregar Base de Dados
# - Refinar Modelo
# - Validar Modelo
# - Próximos passos
# - Atividades Complementares

# %% [markdown]
# ## Introdução

# %% [markdown]
# Também será necessária a instalação do `ffmpeg` mais detalhes nesse [link](https://www.ffmpeg.org/download.html).
#
# Para usuários linux e possivelmente em ambientes wsl.
#
# ```console
# sudo apt update; sudo apt install ffmpeg -y
# ```
#

# %% [markdown]
# Instalação para os que ainda não possuem a biblioteca instalada.

# %%
# !pip install torch torchvision datasets tqdm ipywidgets torchcodec torchvideo ffmpeg-python av

# %% [markdown]
# Importar as bibliotecas

# %%
import datasets
import torch
import torchvision
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torchcodec

# %% [markdown]
# ## Carregar Base de Dados
#
# A primeira tarefa para refinar um modelo é criar a base de dados.
#
# Referência: https://huggingface.co/datasets/ertiaM/Anomaly_Detection_in_Surveillance_Videos

# %%
# Baixando dataset.
# Esse dataset possui apenas split de treino.
dataset = datasets.load_dataset("ertiaM/Anomaly_Detection_in_Surveillance_Videos", split="train")

# Split dataset.
split_ds = dataset.train_test_split(test_size=0.1, seed=42)

# Pegar splits de dados.
train_dataset = split_ds["train"]
test_dataset = split_ds["test"]


# Transformações dos dados para treino.
def transforms(examples):
    # Pré processamento do modelo.
    preprocess = torchvision.models.video.Swin3D_T_Weights.KINETICS400_V1.transforms()
    #Transformações das imagens.
    transforms = torchvision.transforms.Compose([
        preprocess, # Pré processamento do modelo
    ])

    videos = []
    targets = []
    for video, label in zip(examples["video"], examples["label"]):
        # Cria os clips.
        num_clips = 5
        clips =  torchcodec.samplers.clips_at_random_indices(video,
                                                             num_clips=num_clips,
                                                             num_frames_per_clip=4,
                                                             num_indices_between_frames=3)

        # Processando os clips e organizando os labels.
        clips = [transforms(clip.data) for clip in clips]
        labels = torch.tensor(label).repeat(num_clips)
        
        videos.append(torch.stack(clips))
        targets.append(labels)
    videos = [clip for video in videos for clip in video]
    targets = [label for target in targets for label in target]
    return {"video": videos, "target": targets}

train_dataset =  train_dataset.with_transform(transforms)
# test_dataset = test_dataset.with_transform(test_transforms)

def collate_fn(batch):
    # Organiza o retorno do dataloader.
    videos = torch.stack([item["video"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    return videos, targets

# Criando dataloaders.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# %% [markdown]
# ## Refinar Modelo
#
# Na prática de hoje iremos refinar o modelo **Swin Transformer 3D** disponível no torchvision.

# %%
# Definindo dispositivo.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carregando modelo.
model = torchvision.models.video.swin3d_t(weights=torchvision.models.video.Swin3D_T_Weights.KINETICS400_V1)

num_classes = 13 # número de classes no nosso dataset.
in_features = model.head.in_features # número de features da última camada.
model.head = torch.nn.Linear(in_features, num_classes) # substitui a última camada do modelo.

model.to(device) # Colocando modelo no dispositivo.

# Definindo o otimizador.
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

# Define a função de perda.
loss = torch.nn.CrossEntropyLoss()

# Treinamento do modelo
model.train()
epochs = 10 # Alterar para treinar mais epocas.
for epoch in range(epochs):
    iteration = 0
    for videos, labels in tqdm(train_loader):
        # Muda o dispostivo de processamento dos dados.
        videos = videos.to(device)
        labels = labels.to(device)

        # Processa os dados.
        output = model(videos)

        # Computa loss.
        loss_value = loss(output, labels)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        
        if iteration % 100 == 0:
            print(f"Total loss: {loss_value.item()}")
        iteration += 1

# %% [markdown]
# ## Validar Modelo
#
# Agora vamos testar o nosso modelo.

# %%
# Tamanho do dataset de treino.
len_test_data = len(test_dataset)

# Sortear amostra do dataset de teste.
idx = torch.randint(len_test_data, (1,))

# Pré processamento para o modelo.
preprocess = torchvision.models.video.Swin3D_T_Weights.KINETICS400_V1.transforms()

# Modelo em modo de avaliação.
model.eval()
with torch.no_grad():
    video = test_dataset[idx]["video"][0]

    clips =  torchcodec.samplers.clips_at_random_indices(video,
                                                        num_clips=5,
                                                        num_frames_per_clip=4,
                                                        num_indices_between_frames=3)
    
    x = [preprocess(clip.data) for clip in clips]
    images = torch.stack([clip.data for clip in clips])
    # Processa os clips e coloca no dispositivo. 
    x = torch.stack(x).to(device)
    predictions = model(x)

probs = torch.argmax(torch.softmax(predictions, dim=1), dim=1)
inverse_label_map = {7: "RoadAccidents",
                     8: "Robbery",
                     4: "Burglary",
                     11: "Stealing",
                     0: "Abuse",
                     1: "Arrest",
                     2: "Arson",
                     3: "Assault",
                     5: "Explosion",
                     6: "Fighting",
                     9: "Shooting",
                     10: "Shoplifting",
                     12: "Vandalism"}

print(f"Predição: {inverse_label_map[int(probs[0].cpu())]}")

fig, ax = plt.subplots(5, 1,  figsize=(20, 15)) # Cria uma figura e um eixo
for i in range(5):
    # Converte de (T, H, W, C) para (T, C, H, W) se o canal estiver na última dimensão, para evitar erro no make_grid
    grid_img = images[i]
    if grid_img.shape[-1] in (1, 3, 4):
        grid_img = grid_img.permute(0, 3, 1, 2)
    ax[i].imshow(torchvision.transforms.functional.to_pil_image(torchvision.utils.make_grid(grid_img)))
    ax[i].set_title(f"Predição: {inverse_label_map[int(probs[i].cpu())]}") # Coloca título a classe.

plt.show() # Exibe o gráfico final (bloqueia)

# %% [markdown]
# ## Próximos Passos e Referências

# %% [markdown]
# Nas próximas práticas vamos continuar trabalhando com problemas reais que envolvem Visão Computacional.
#
# Uma lista não exaustiva de referências segue:
#
# - https://huggingface.co/datasets/ertiaM/Anomaly_Detection_in_Surveillance_Videos
# - https://huggingface.co/datasets
# - https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.swin3d_t.html#torchvision.models.video.swin3d_t
# - https://pytorch.org/
# - https://docs.pytorch.org/vision/main/models.html
# - https://opencv.org/
# - https://learnopencv.com/blogs/
# - https://pyimagesearch.com/

# %% [markdown]
# ## Atividades Complementares (Opcional)

# %% [markdown]
# - [x] **Ajuste de Hiperparâmetros Space-Time — exercício guiado (NÃO roda automático)**
#
#   > ⚠️ **Por que este não é resolvido como os outros notebooks?** O Swin3D de vídeo é pesado
#   > demais para um *segundo* treino no Colab — re-treinar aqui poderia levar horas. Por isso
#   > deixamos como **exercício guiado**, com o passo a passo pronto para quem quiser experimentar
#   > manualmente (basta editar as células de `transforms`/treino lá em cima):
#   1. Altere `num_clips` para 8 e `num_frames_per_clip` para 8, para o modelo capturar mais informação temporal.
#   2. Use uma taxa de aprendizado menor (ex.: `lr=0.0001`) ou o otimizador `AdamW`, para ver se a perda decai de forma mais estável.
#   3. Treine por mais épocas e plote a curva de convergência para documentar o comportamento.
# - [ ] **Sugestão livre:** ajuste batch / frames por clip / learning rate e observe o efeito.
