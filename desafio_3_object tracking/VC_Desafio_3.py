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
# ## Desafio 3 da Trilha de Visão Computacional — Rastreamento de Objetos
#
# > **Resumo:** Usamos o modo **`track`** da YOLO para não só *detectar* objetos em cada frame, mas
# > seguir cada um com um **ID persistente** ao longo do vídeo (object tracking). No fim, salvamos um
# > vídeo com as caixas e os IDs desenhados.
#
# > 📝 **Detecção vs Tracking:** a detecção é quadro-a-quadro e "esquece" tudo entre frames. O
# > tracking liga as detecções no tempo (com algoritmos tipo ByteTrack/BoT-SORT), mantendo o mesmo
# > ID para o mesmo objeto — é o que permite *contar* objetos e analisar trajetórias.

# %% [markdown]
# > ⚠️ **GPU recomendada** (Ambiente de execução → GPU T4): processar todos os frames do vídeo é
# > bem mais rápido em GPU, embora a YOLO nano também rode em CPU.

# %% [markdown]
# Nesse desafio você irá implementar uma solução simples para realizar o rastreamento de objetos.
#
# Para isso você deverá seguir os pasos desse documento atentamente.

# %% [markdown]
# ### Instalação e Importação da Biblioteca

# %%
# !pip install ultralytics opencv-python torch

# %%
import ultralytics
import cv2
import torch

# %% [markdown]
# ### Processar dados

# %%
# Baixa o modelo
model = ultralytics.YOLO("yolo11n.pt")  # modelo nano da YOLO (treinado no COCO)

# Joga o modelo para o dispositivo de computação
device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU se disponível, senão CPU
model = model.to(device)  # move o modelo para o dispositivo
print(f"Rodando em: {device}")

# processa o vídeo no modo TRACK: detecta + dá um ID persistente a cada objeto entre frames
results = model.track(source="video.mov", conf=0.1, iou=0.7)

# %% [markdown]
# ### Salvar Vídeo

# %%
# Resumo do rastreamento: quantos objetos distintos foram seguidos ao longo do vídeo
all_ids = set()
for r in results:
    if r.boxes is not None and r.boxes.id is not None:
        all_ids.update(r.boxes.id.int().tolist())

print(f"Total de frames processados: {len(results)}")
print(f"Objetos distintos rastreados (IDs únicos): {len(all_ids)}")
print(f"IDs: {sorted(all_ids)}")
print()
print("Amostra dos primeiros frames (nº de objetos rastreados por frame):")
for idx, r in enumerate(results[:10]):
    n = 0 if (r.boxes is None or r.boxes.id is None) else len(r.boxes.id)
    print(f"  Frame {idx:03d}: {n} objetos")

# %%
# Cria o escritor do vídeo de output.
# IMPORTANTE: o tamanho do writer precisa bater com o tamanho dos frames anotados (r.plot()),
# senão o OpenCV salva um vídeo vazio/corrompido. Por isso pegamos o tamanho real do 1º frame.
annotated = results[0].plot()
height, width = annotated.shape[:2]
writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), 20, (width, height))

# Escreve cada frame anotado (caixas + IDs de tracking) no vídeo
for r in results:
    writer.write(r.plot())

writer.release()  # IMPORTANTE: fecha o arquivo — sem isso o vídeo fica incompleto/inválido
print(f"Vídeo salvo: output.avi  ({width}x{height}, {len(results)} frames)")

# %% [markdown]
# ### Referência
#
# - https://docs.ultralytics.com/modes/track/
