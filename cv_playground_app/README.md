---
title: CV Playground
emoji: 🎮
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.19.0
python_version: '3.13'
app_file: app.py
pinned: false
license: mit
short_description: Playground interativo de Visão Computacional
---

# 🎮 CV Playground

App interativo de Visão Computacional. Sobe uma imagem, escolhe a técnica e vê o resultado na hora:

- **Detecção de objetos** (YOLO11) — Módulos 3/5, Desafio 3
- **Estimativa de pose** (YOLO11-pose) — Desafio 4
- **Borrar rostos / privacidade (LGPD)** — Desafio 1
- **CLIP zero-shot** (imagem ↔ texto) — Módulo 4

Parte de um caderno de estudos da trilha de Visão Computacional.

## Como rodar localmente

```bash
pip install -r requirements.txt
python app.py
```

Abra o endereço local que o Gradio imprimir (ex.: http://127.0.0.1:7860).

## Deploy grátis no Hugging Face Spaces

1. Crie um Space novo em https://huggingface.co/new-space → SDK: **Gradio**.
2. Suba os arquivos `app.py`, `requirements.txt` e este `README.md`.
3. O Space instala as dependências e publica a URL pública sozinho.

> Os pesos do YOLO baixam automaticamente; o CLIP é baixado do 🤗 Hub no primeiro uso.
> Roda em CPU (gratuito) — tarefas de imagem respondem em poucos segundos.
