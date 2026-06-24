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
# # M5A3 - Identificação de Elementos Visuais em UI de Aplicativos
#
# > **Resumo:** Projeto de **detecção de objetos** aplicado a interfaces: refinamos um **Faster R-CNN** (backbone ResNet-50) para localizar componentes de UI (rectangle, text, group, image) em telas de apps móveis. Útil pra testes automatizados e acessibilidade. No fim, testamos o modelo fora-de-distribuição (o Sonic).

# %% [markdown] id="83cf97a1"
# > ⚠️ **GPU recomendada** (Ambiente de execução → Alterar o tipo → GPU T4): treinar o Faster R-CNN em CPU é muito lento.

# %% [markdown] id="a2ac4c18"
# Na prática de hoje vamos refinar um modelo para a tarefa de identificação de elementos visuais em interfaces gráficas de aplicativos móveis.
#
# Em projetos reais de desenvolvimento de software, a identificação automática de componentes de UI (como botões, ícones, campos de texto e contêineres) é crucial para automatizar testes funcionais (garantindo que o layout permaneça correto em diferentes dispositivos e modos claro/escuro) e para criar ferramentas de acessibilidade inteligentes para usuários portadores de deficiência visual.
#
# Para essa tarefa, usaremos uma abordagem de **Detecção de Objetos** baseada em **Faster R-CNN** (com backbone ResNet-50). Em problemas de UI, também podem ser aplicados modelos de segmentação (como a UNet) para obter recortes precisos de botões, ou OCR (como Tesseract ou EasyOCR) para ler os rótulos internos das caixas detectadas.
#
# Esse notebook está estruturado da seguinte forma.
#
# - Introdução
# - Carregar Base de Dados
# - Refinar Modelo
# - Validar o modelo
# - Próximos passos
# - Atividades Complementares

# %% [markdown] id="9a1440d4"
# ## Introdução

# %% [markdown] id="9090a15c"
# Instalação para os que ainda não possuem a biblioteca instalada.

# %% colab={"base_uri": "https://localhost:8080/"} id="2561b9db" outputId="ff75db8c-c671-4073-e4ea-b22aef39a3ab"
# !pip install torch torchvision datasets tqdm ipywidgets

# %% [markdown] id="1ffc5e8b"
# Importar as bibliotecas

# %% id="1f7026a0"
import datasets
import torch
import torchvision
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# %% [markdown] id="69532420"
# ## Carregar Base de Dados
#
# A primeira tarefa para refinar um modelo é criar a base de dados.
#
# Referência: https://huggingface.co/datasets/mrtoy/mobile-ui-design

# %% colab={"base_uri": "https://localhost:8080/", "height": 316, "referenced_widgets": ["d2f3332f21a04351b66e111073760c93", "230cb25a6f2940668b558e5126ebe040", "fc3a71ea4ff14ed5b65d84c709a97455", "5c43bc3804fb451baa4b879d7b8f9754", "50fd354c389f4be98a557bde92f7bc61", "52c9c6afe692417cb59d1913c4d7952e", "d6f325400f864964ae6e59eb75e51e02", "a795f535e7504129852e9425fc112968", "8703dc3b84f442559afd6efe0ef8d063", "0c7d621f705540daa0f1b599737daca9", "f02966acbb04417fa56fae137d61237e", "0c8b8479fc714259b54df7227a414198", "aa8ea49c315049d8b0f42f6ae180c77f", "19a273410e7c4ab2b080a17eeb3e86d0", "3031d3127503430dbb355d435c4ffbf8", "f0e6e8c947ae4a6287899201bce388b6", "25dc5a590431498fa1e4e787ff853531", "129121cb9d2a4363ad0f297ee89c9460", "dcb2fa08bfca4527babdb6e84a9bb69d", "6487ebfbc21c41c19d00e59d22505176", "3110b9d1a0374e5da25b926b41a42a53", "96cb1c5ca50e4b4b92d9544b24bc1d65", "a83d8ea5ebcd4aa0bf7a9f53dc825327", "2b30ce38555540a28cbc9f944ff46355", "45d0325f55c24759b2063ad026654b90", "6cb272a32e0b47b68e131a0124c9819c", "79a7bb07448b4405b9b8e8072983b23b", "ecd93a1ae3754fe8904aa5560f31a9d6", "71768adc158d493aab4bd32dfae57917", "e8c6a079621444b48b2335c1fbf8c676", "8d00579b9f7e4872a59d91e498cfb380", "d822b5e42d5d41fe9cabc3090b4f058a", "8d148fa6d26640d6bdfeba82b30ba14a", "1d9b2c38dcd04510b36ca223f2745778", "af3974489e3b4271bdf2fa5e1aef980c", "168cc790b3534ad3b1b32ffde847061c", "69855451227649df94d9729ef8f8eb96", "1a2dec5319764e9482339fd50230f089", "1fd0956977444d839b948d67c3836bed", "429f574e715249158f3bb7e2e2b2aaf9", "2482dff782884fd58fa3f53e74b48702", "89af84331245418ba09715338a3363fb", "ef56115b488f4ca992fbd5e136cf8397", "270f1c9c21844c67835acb5d1556c48a", "c850154538624553b31024967440e3fd", "bff586144b4249a69ea792474867f6c4", "79f01234e2a9477db9a2d4d4f15230ee", "673e748513e341599015db0ce46d9e15", "f102cdde5c704bdb9cc268242a68391b", "644f455668be4aa9a0b5c00166ecd0b4", "74c25e71128f414eae7ec2cce2b50090", "b737cf7fc57c445c978c6a632f0cd4c4", "a8ad7edde5b64de4a90b6c759717456d", "7c783f09f49c4ed9af6f8e171bd04d70", "a77e3431c43844f6b04d419eaafca750"]} id="fd318926" outputId="c7799887-7245-452b-a243-762304c37359"
# Baixando dataset.
# Esse dataset possui apenas split de treino.
dataset = datasets.load_dataset("mrtoy/mobile-ui-design", split="train")

# Split dataset.
split_ds = dataset.train_test_split(test_size=0.1, seed=42)

# Pegar splits de dados.
train_dataset = split_ds["train"]
test_dataset = split_ds["test"]

# Transformações dos dados.
def transforms(examples):
    # Transformações das imagens.
    transform_func = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), # Transforma em tensor.
        torchvision.transforms.Resize((640, 480)), # Faz resize das imagens.
    ])

    images = []
    targets = []

    # Mapeamento de classes deslocado em +1 (pois o Faster R-CNN do PyTorch reserva a classe 0 para o background)
    label_map = {"rectangle": 1, "text": 2, "group": 3, "image": 4}

    target_h, target_w = 640, 480  # mesmo tamanho usado no Resize acima
    batch_size = len(examples["image"])
    for i in range(batch_size):
        img = examples["image"][i]
        # Tamanho original (PIL retorna largura, altura) ANTES do resize, para escalar as bboxes
        orig_w, orig_h = img.size
        sx, sy = target_w / orig_w, target_h / orig_h
        img_tensor = transform_func(img)
        # Garantir que imagens RGBA (4 canais) sejam convertidas para RGB (3 canais)
        if img_tensor.shape[0] == 4:
            img_tensor = img_tensor[:3, ...]
        images.append(img_tensor)

        objs = examples["objects"][i]
        len_objs = len(objs["category"])

        if len_objs > 0:
            boxes_list = []
            labels_list = []
            for j in range(len_objs):
                x, y, w, h = objs["bbox"][j]
                # IMPORTANTE: escalar a bbox junto com o resize da imagem, senão as caixas
                # ficam desalinhadas (o modelo treinaria com rótulos errados).
                bbox_scaled = [x * sx, y * sy, w * sx, h * sy]
                # Converte o formato da bbox de xywh para xyxy
                bbox_xyxy = torchvision.ops.box_convert(torch.tensor(bbox_scaled, dtype=torch.float32), in_fmt="xywh", out_fmt="xyxy")
                boxes_list.append(bbox_xyxy)

                cat = objs["category"][j]
                labels_list.append(label_map[cat])

            boxes_tensor = torch.stack(boxes_list)
            labels_tensor = torch.tensor(labels_list, dtype=torch.int64)
        else:
            # Caso a imagem não possua objetos anotados
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        targets.append({"boxes": boxes_tensor, "labels": labels_tensor})

    # Retorna dicionário compatível com o dataloader, onde cada chave possui listas com o mesmo tamanho do lote
    return {"image": images, "target": targets}

train_dataset =  train_dataset.with_transform(transforms)
test_dataset = test_dataset.with_transform(transforms)

def collate_fn(batch):
    # Organiza o retorno do dataloader.
    images = [item["image"] for item in batch]
    targets = [item["target"] for item in batch]
    return images, targets

# Criando dataloaders.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# %% [markdown] id="60aa9279"
# ## Refinar Modelo
#
# Na prática de hoje iremos refinar o modelo **Faster R-CNN** disponível no torchvision.

# %% colab={"base_uri": "https://localhost:8080/", "height": 396, "referenced_widgets": ["609a40580f3b42c5ab4db0ed425d1c1b", "a58bc1bcc0e641438132aeaccfc579b3", "e7e281d77fff48d0aa74535f11bcd126", "c3ce81e388ba4f5bb577277c2b11a9b2", "fc1303771a9b4ebab4fd1f533eb22dda", "9ebaef8eeeed4ffab340b11419dd0309", "972a0953cbee4b3dab96de4022f8e2a0", "005568186e9c4157b1776873a8ff7270", "af722c5fe58d4cbd8f74d3f08981bf3e", "dba1d96ef24a432a904c4bef161750c1", "b9717a2017504570bf440c4732a4ed0a"]} id="ee1b1b55" outputId="6448425b-10e5-42af-f872-bd9aa25b83fd"
# Definindo dispositivo.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carregando modelo.
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

num_classes = 5 # número de classes no nosso dataset + 1 para a classe background.
in_features = model.roi_heads.box_predictor.cls_score.in_features # número de features da última camada.
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes) # substitui a última camada do modelo.

model.to(device) # Colocando modelo no dispositivo.

# Definindo o otimizador.
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

# Treinamento do modelo
model.train()
epochs = 1 # Alterar para treinar mais epocas.
for epoch in range(epochs):
    iteration = 0
    for images, targets in tqdm(train_loader):
        # Muda o dispostivo de processamento dos dados.
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Modelo retorna um dicionário de losses durante o treinamento.
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print(f"Total loss: {losses.item()}")
        iteration += 1

# %% [markdown] id="a92154ac"
# ## Validar Modelo
#
# Agora vamos testar o nosso modelo.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="39c2f25c" outputId="72c970b9-a7db-4993-e3e8-f861a3fcdfca"
# Tamanho do dataset de teste.
len_test_data = len(test_dataset)

# Sortear amostra do dataset de teste.
idx = torch.randint(len_test_data, (1,))

# Modelo em modo de avaliação
model.eval()
with torch.no_grad():
    x = test_dataset[idx]["image"][0]
    # convert RGBA -> RGB and move to device
    x = x[:3, ...].to(device)
    predictions = model([x, ])
    pred = predictions[0]

# Mapeamento inverso das classes (respeitando o deslocamento de +1)
inverse_label_map = {1: "rectangle", 2: "text", 3: "group", 4: "image", 0: "background"}

# Filtro de confiança: manter apenas as caixas delimitadoras cuja confiança é maior que 0.5
keep = pred["scores"] > 0.5
pred_boxes = pred["boxes"][keep].long()
pred_labels = [f"{inverse_label_map.get(int(label.cpu()), 'bg')}: {score:.3f}" for label, score in zip(pred["labels"][keep], pred["scores"][keep])]

image = x
image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...]

if len(pred_boxes) > 0:
    output_image = torchvision.utils.draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")
else:
    output_image = image

plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0).cpu())

# %% [markdown] id="57ee5eb5"
# ## Próximos Passos e Referências

# %% [markdown] id="0a33dd1c"
# Nas próximas práticas vamos continuar trabalhando com problemas reais que envolvem Visão Computacional.
#
# Uma lista não exaustiva de referências segue:
#
# - https://docs.pytorch.org/vision/master/models/faster_rcnn.html
# - https://huggingface.co/datasets/mrtoy/mobile-ui-design
# - https://huggingface.co/datasets
# - https://pytorch.org/
# - https://docs.pytorch.org/vision/main/models.html
# - https://opencv.org/
# - https://learnopencv.com/blogs/
# - https://pyimagesearch.com/

# %% [markdown] id="02185967"
# ## Atividades Complementares (Opcional)

# %% [markdown] id="ce664141"
# - [x] **Teste de Robustez com Team Sonic (Out-of-Distribution)** — resolvida logo abaixo (leve: só inferência no modelo já treinado).
# - [x] **Efeito do learning rate** — resolvida na seção opcional/pesada no fim (re-treina rápido com 2 LRs).
# - [ ] **Sugestão livre:** treine por mais épocas e veja se as detecções melhoram.
#
# Vamos submeter uma imagem do Sonic (`img/sonic.jpg`) ao modelo Faster R-CNN treinado para verificar se ele detecta erroneamente algum componente visual de interface no corpo do personagem.

# %% colab={"base_uri": "https://localhost:8080/", "height": 864} id="cd0f8a59" outputId="593da523-f812-4f4f-9148-87727501a887"
import os

# Caminho para a imagem do Sonic
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

    # Carregar imagem do Sonic, converter para RGB e aplicar transformações
    from PIL import Image
    sonic_img = Image.open(sonic_path).convert("RGB")
    transform_func = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((640, 480)),
    ])
    sonic_tensor = transform_func(sonic_img).to(device)

    model.eval()
    with torch.no_grad():
        sonic_preds = model([sonic_tensor])[0]

    # Usaremos um limiar de confiança mais baixo (0.3) para esta exploração lúdica
    sonic_keep = sonic_preds["scores"] > 0.3
    sonic_boxes = sonic_preds["boxes"][sonic_keep].long()
    sonic_labels = [f"{inverse_label_map.get(int(lbl.cpu()), 'bg')}: {score:.3f}"
                    for lbl, score in zip(sonic_preds["labels"][sonic_keep], sonic_preds["scores"][sonic_keep])]

    # Formatar imagem para draw_bounding_boxes
    sonic_disp = (255.0 * (sonic_tensor - sonic_tensor.min()) / (sonic_tensor.max() - sonic_tensor.min())).to(torch.uint8)

    if len(sonic_boxes) > 0:
        sonic_out = torchvision.utils.draw_bounding_boxes(sonic_disp, sonic_boxes, sonic_labels, colors="blue")
        print(f"O modelo detectou {len(sonic_boxes)} possíveis elementos de UI no Sonic!")
    else:
        sonic_out = sonic_disp
        print("Nenhum elemento de UI foi detectado no Sonic com confiança > 0.3.")

    plt.figure(figsize=(10, 10))
    plt.imshow(sonic_out.permute(1, 2, 0).cpu())
    plt.axis("off")
    plt.title("UI Detection on Sonic (Out-of-Distribution)")
    plt.show()

# %% [markdown] id="ee8a7f57"
# ---
# # ⏸️ Ponto de parada

# %% [markdown] id="e2bc2e99"
# ## Atividade (opcional/pesada): Efeito do Learning Rate
#
# Treinamos rapidamente (apenas alguns batches) com dois *learning rates* diferentes e comparamos a
# **perda média**. Assim dá pra sentir como o LR afeta a estabilidade do treino, sem pagar um treino
# completo.

# %% colab={"base_uri": "https://localhost:8080/"} id="939eb282" outputId="6f68ccca-093b-4bc5-8f09-552e7ea7823a"
import itertools

def treino_curto(lr, n_batches=50):
    """Treina um Faster R-CNN do zero por poucos batches e retorna a perda média."""
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        m.roi_heads.box_predictor.cls_score.in_features, num_classes)
    m.to(device)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    losses = []
    for images, targets in itertools.islice(train_loader, n_batches):
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss = sum(l for l in m(images, targets).values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)

print("Comparando learning rates (50 batches cada)...")
loss_baixo = treino_curto(0.001)
loss_alto = treino_curto(0.005)
print(f"  lr=0.001  ->  perda média: {loss_baixo:.4f}")
print(f"  lr=0.005  ->  perda média: {loss_alto:.4f}")
print()
print("Intuição: LR muito alto pode tornar o treino instável (perda alta/oscilando);")
print("LR muito baixo aprende devagar. O 'ponto certo' depende do problema.")
