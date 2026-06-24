"""
CV Playground — app interativo de Visão Computacional.

Demonstra, de forma "tocável", várias técnicas estudadas na trilha de Visão Computacional:
- Detecção de objetos (YOLO)            -> Módulos 3/5, Desafio 3
- Estimativa de pose (YOLO-pose)        -> Desafio 4
- Borrar rostos / privacidade (LGPD)    -> Desafio 1
- CLIP zero-shot (multimodal)           -> Módulo 4

Sobe uma imagem, escolhe a técnica e vê o resultado na hora.
Roda em CPU (Hugging Face Spaces / local).
"""

import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPModel, CLIPProcessor

# ----------------------------------------------------------------------
# Carregamento dos modelos (uma vez, no startup)
# ----------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

yolo_det = YOLO("yolo11n.pt")        # detecção (COCO)
yolo_pose = YOLO("yolo11n-pose.pt")  # pose (keypoints)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# ----------------------------------------------------------------------
# Funções de cada técnica  (a entrada do Gradio é numpy RGB)
# ----------------------------------------------------------------------
def _rgb_to_bgr(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def detectar(img_rgb):
    res = yolo_det(_rgb_to_bgr(img_rgb), verbose=False)[0]
    anotada_bgr = res.plot()  # plot() devolve BGR
    n = 0 if res.boxes is None else len(res.boxes)
    return cv2.cvtColor(anotada_bgr, cv2.COLOR_BGR2RGB), f"{n} objeto(s) detectado(s)."


def estimar_pose(img_rgb):
    res = yolo_pose(_rgb_to_bgr(img_rgb), verbose=False)[0]
    anotada_bgr = res.plot()
    n = 0 if res.keypoints is None else len(res.keypoints)
    return cv2.cvtColor(anotada_bgr, cv2.COLOR_BGR2RGB), f"{n} pessoa(s) com pose estimada."


def borrar_rostos(img_rgb):
    bgr = _rgb_to_bgr(img_rgb).copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi = bgr[y:y + h, x:x + w]
        bgr[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (45, 45), 0)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), f"{len(faces)} rosto(s) borrado(s) (privacidade/LGPD)."


def clip_zero_shot(img_rgb, rotulos_txt):
    rotulos = [r.strip() for r in rotulos_txt.split(",") if r.strip()]
    if not rotulos:
        rotulos = ["a photo", "a person", "an animal"]
    pil = Image.fromarray(img_rgb)
    inputs = clip_proc(text=rotulos, images=pil, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        out = clip_model(**inputs)
        probs = out.logits_per_image.softmax(dim=1)[0].cpu().numpy()
    return {rot: float(p) for rot, p in zip(rotulos, probs)}


# ----------------------------------------------------------------------
# Dispatcher chamado pelo botão
# ----------------------------------------------------------------------
def rodar(tecnica, img, rotulos_txt):
    if img is None:
        return None, {}, "Suba uma imagem primeiro."
    if tecnica == "Detecção de objetos":
        out, msg = detectar(img)
        return out, {}, msg
    if tecnica == "Estimativa de pose":
        out, msg = estimar_pose(img)
        return out, {}, msg
    if tecnica == "Borrar rostos (privacidade)":
        out, msg = borrar_rostos(img)
        return out, {}, msg
    if tecnica == "CLIP (zero-shot)":
        probs = clip_zero_shot(img, rotulos_txt)
        melhor = max(probs, key=probs.get)
        return img, probs, f"CLIP: melhor correspondência = '{melhor}'."
    return img, {}, ""


# ----------------------------------------------------------------------
# Interface Gradio
# ----------------------------------------------------------------------
DESCRICAO = """
# 🎮 CV Playground — Visão Computacional Interativa

Sobe uma imagem, escolhe a técnica e vê o resultado na hora. Cada técnica vem de um tema da
**trilha de Visão Computacional**:

| Técnica | De onde vem |
| :-- | :-- |
| **Detecção de objetos** (YOLO) | Módulos 3 e 5 · Desafio 3 |
| **Estimativa de pose** (YOLO-pose) | Desafio 4 |
| **Borrar rostos** (privacidade/LGPD) | Desafio 1 |
| **CLIP zero-shot** (imagem ↔ texto) | Módulo 4 |

> 💡 No **CLIP zero-shot**, escreva rótulos em inglês separados por vírgula (ex.: `a cat, a dog,
> a person`) e o modelo diz qual combina mais com a imagem — sem ter sido treinado nessas classes.
"""

with gr.Blocks(title="CV Playground", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRICAO)
    with gr.Row():
        with gr.Column():
            tecnica = gr.Radio(
                ["Detecção de objetos", "Estimativa de pose",
                 "Borrar rostos (privacidade)", "CLIP (zero-shot)"],
                value="Detecção de objetos",
                label="Técnica",
            )
            img_in = gr.Image(type="numpy", label="Imagem de entrada")
            rotulos_in = gr.Textbox(
                value="a cat, a dog, a person, a car",
                label="Rótulos para o CLIP (separados por vírgula, em inglês)",
            )
            btn = gr.Button("▶ Rodar", variant="primary")
        with gr.Column():
            img_out = gr.Image(label="Resultado")
            label_out = gr.Label(label="CLIP — probabilidades", num_top_classes=5)
            msg_out = gr.Textbox(label="Resumo", interactive=False)

    btn.click(rodar, [tecnica, img_in, rotulos_in], [img_out, label_out, msg_out])

    gr.Markdown(
        "Feito como parte de um caderno de estudos de Visão Computacional. "
        "Modelos: YOLO11 (Ultralytics) e CLIP (OpenAI via 🤗 Transformers)."
    )

if __name__ == "__main__":
    demo.launch()
