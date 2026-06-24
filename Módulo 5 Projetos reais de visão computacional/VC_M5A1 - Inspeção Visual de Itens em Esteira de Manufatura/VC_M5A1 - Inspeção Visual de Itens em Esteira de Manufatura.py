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
# # M5A1 - Inspeção Visual de Itens em Esteira de Manufatura
#
# > **Resumo:** Primeiro projeto real do módulo! Treinamos (refinamos) um detector **YOLOv11** para inspeção visual numa esteira de manufatura: baixamos um dataset do Hugging Face, treinamos a YOLO, rodamos a detecção num **vídeo** da esteira e testamos a robustez do modelo fora-de-distribuição (o Sonic).

# %% [markdown] id="a687789c"
# > ⚠️ **GPU recomendada** (Ambiente de execução → Alterar o tipo → GPU T4): o `model.train(...)` da YOLO roda muito mais rápido em GPU.

# %% [markdown] id="a2ac4c18"
# Na prática de hoje vamos refinar um modelo para a tarefa de inspeção visual de itens em uma esteira de manufatura.
#
# Na indústria de manufatura, a inspeção visual manual é suscetível a erros devido à fadiga humana e à alta velocidade das linhas de produção. A automatização com visão computacional é ideal nesses cenários. No entanto, um desafio clássico é que defeitos reais de fabricação são raros, resultando em bases de dados extremamente desbalanceadas.
#
# Para contornar esse problema, técnicas de **Data Augmentation** e geração de dados sintéticos por meio de **Modelos Generativos** (como GANs ou Modelos de Difusão) são amplamente utilizadas para enriquecer a base com exemplos realistas de peças defeituosas. Além disso, modelos de detecção baseados em **CNNs** (como a família YOLO) são empregados para classificar e, ao mesmo tempo, localizar as anomalias nas peças.
#
# Esse notebook está estruturado da seguinte forma.
#
# - Introdução
# - Carregar Base de Dados
# - Refinar Modelo
# - Próximos passos
# - Atividades Complementares

# %% [markdown] id="9a1440d4"
# ## Introdução

# %% [markdown] id="9090a15c"
# Instalação para os que ainda não possuem a biblioteca instalada.

# %% colab={"base_uri": "https://localhost:8080/"} id="2561b9db" outputId="7a16189f-9140-4a21-a274-39e9d1370b19"
# !pip install torch torchvision huggingface_hub ultralytics

# %% [markdown] id="1ffc5e8b"
# Importar as bibliotecas

# %% colab={"base_uri": "https://localhost:8080/"} id="1f7026a0" outputId="1305f126-5e91-4a15-c9eb-c16fd5a51027"
import yaml
import os

from ultralytics import YOLO
from huggingface_hub import snapshot_download
from pathlib import Path
import shutil
from IPython.display import Video

# %% [markdown] id="69532420"
# ## Carregar Base de Dados
#
# A primeira tarefa para refinar um modelo é criar a base de dados.
#
# Referência: https://huggingface.co/datasets/johnatanvq/fruits-dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 261, "referenced_widgets": ["b9177fa127a344fc81d1a44b158f0d5c", "5d7fa437ba9f440ebe708fb8709c0606", "a00df7d2f59546baaddef09f3b05cff8", "9079243e1e69417bb0212dc4516707bb", "f7075c09160b44e080def3b51e9b5be0", "d952d90b75d0482993097feeaf4b682d", "8bf0df1dac4749b0928cfa8ea8effd24", "b7f1fa2455f849aa8371965b75c73537", "fd65f87743114f61aa410975c844fc88", "b2200852b93f4ed7b16a6615ec30984a", "936b1d09cf024792946e0640511c305e", "db7ec688616d403ca4cee2b3039112b5", "26f58eb67c784eebaecef13a77c6e837", "d3cbe8bc8d2449a5b65b219f9ae93d43", "24a44d4a70f34031ae0a57aaa617d2b1", "eb1762b1a17248248f01fee1177ce420", "bbb7c9f8704d420b8e45ffa4f44a35c8", "f24eb2897c624097a2ae97766d59a205", "689894da4c264e3aa6991aaecc470c10", "2e104bae55f741a281bdc12f29aebbca", "47558274a79c40a48a3045703cf909e9", "756ed448bac145d489ff42736f87296a"]} id="fd318926" outputId="428db8ee-a4e4-443c-9062-f24a00fe5a93"
# 1) Baixar somente o subdiretório fruitsData/
local_repo_dir = snapshot_download(
    repo_id="johnatanvq/fruits-dataset",
    repo_type="dataset",
    allow_patterns=["fruitsData/**"],  # baixa só essa pasta
)

print("Arquivos baixados em:", local_repo_dir)

# 2) Mover/copiar para uma pasta final estilo ImageFolder (se quiser customizar o caminho)
src = Path(local_repo_dir) / "fruitsData"
dst = Path("data/fruits")  # pasta final onde você quer o ImageFolder

# Copiando arquivos para dst.
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)

print("ImageFolder pronto em:", dst)


# %% colab={"base_uri": "https://localhost:8080/"} id="d35223e4" outputId="34a7e051-b5d7-4f38-f93e-3a5c45f28f7c"
def create_data_yaml(path_to_classes_txt, path_to_data_yaml):
  # Lê o arquivos "classes.txt".
  if not os.path.exists(path_to_classes_txt):
    print(f'classes.txt file not found! Please create a classes.txt labelmap and move it to {path_to_classes_txt}')
    return
  with open(path_to_classes_txt, 'r') as f:
    classes = []
    for line in f.readlines():
      if len(line.strip()) == 0: continue
      classes.append(line.strip())
  number_of_classes = len(classes)

  # Cria o dicionário a ser salvo.
  data = {
      'path': 'data/fruits',
      'train': 'images',
      'val': 'images',
      'nc': number_of_classes,
      'names': classes
  }

  # Escreve o arquivo YAML.
  with open(path_to_data_yaml, 'w') as f:
    yaml.dump(data, f, sort_keys=False)
  print(f'Created config file at {path_to_data_yaml}')

  return

# Chama a função.
create_data_yaml("data/fruits/classes.txt", "yolo_train.yaml")

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="6277fa8c" outputId="25713af8-c886-4df2-d57f-303334130589"
# Carrega o modelo pré-treinado.
model = YOLO("yolo11n.pt")

# Treina o modelo utilizando as informações do arquivo YAML.
# Definimos também a quantidade de épocas, o batch, e o tamanho das imagens.
results = model.train(data="./yolo_train.yaml", project="praticas/modulo_5/aula_1", epochs=10, batch=2, imgsz=480)

# %% colab={"base_uri": "https://localhost:8080/", "height": 171, "resources": {"http://localhost:8080/video.mov": {"data": "", "headers": [["content-length", "0"]], "ok": false, "status": 404, "status_text": ""}}} id="707bf398" outputId="ac3cb44e-42c9-4443-d8d7-63bcf6ca6aec"
# Visualizar o vídeo original.
Video("video.mov")


# %% colab={"base_uri": "https://localhost:8080/"} id="60aa9279" outputId="641a32d1-ae68-4e89-f994-4c099508c1fd"
# Rodar predição do modelo no vídeo e salvar os resultados de forma determinística
# Usando name="predict" e exist_ok=True para evitar a criação de múltiplas pastas (predict2, predict3, etc.)
pred_video_results = model.predict("video.mov", save=True, project="praticas/modulo_5/aula_1", name="predict", exist_ok=True)

# %% colab={"base_uri": "https://localhost:8080/"} id="ee1b1b55" outputId="c8abceec-f7f1-4c82-e1af-ba45815c102d"
# Exibir o vídeo com as predições do YOLO (resolvendo o caminho absoluto de forma dinâmica)
import glob, os
# A ultralytics costuma salvar sob 'runs/detect/...'; pegamos o diretório real do resultado.
save_dir = str(pred_video_results[0].save_dir)
predict_videos = glob.glob(os.path.join(save_dir, "video.*"))
if not predict_videos:  # fallback: busca recursiva caso a estrutura mude
    predict_videos = glob.glob("**/predict/video.*", recursive=True)
if predict_videos:
    video_to_show = predict_videos[0]
    print(f"Exibindo vídeo resultante: {video_to_show}")
    Video(video_to_show)
else:
    print("Nenhum vídeo resultante de predição foi encontrado em 'praticas/modulo_5/aula_1/predict/'")

# %% [markdown] id="a92154ac"
# ## Próximos Passos e Referências

# %% [markdown] id="0a33dd1c"
# Nas próximas práticas vamos continuar trabalhando com problemas reais que envolvem Visão Computacional.
#
# Uma lista não exaustiva de referências segue:
#
# - https://docs.ultralytics.com/modes/train/
# - https://docs.ultralytics.com/modes/predict/
# - https://huggingface.co/datasets/johnatanvq/fruits-dataset
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
# - [x] **Comparação de hiperparâmetros** — resolvida na seção opcional/pesada no fim (re-treina a YOLO).
# - [ ] **Sugestão livre:** troque o dataset de frutas por outro do Hugging Face e veja se o pipeline continua funcionando.
#
# Vamos rodar o modelo treinado em uma imagem do Sonic para ver se ele faz falsas detecções ou ignora a imagem corretamente, já que o Sonic não faz parte do dataset de frutas!

# %% colab={"base_uri": "https://localhost:8080/", "height": 498} id="01469ae0" outputId="abf7ecb9-260b-4e54-c95f-632d12d4d891"
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
    # Rodar predição do YOLO na imagem do Sonic
    # Usando predict_sonic e exist_ok=True para garantir um caminho previsível
    sonic_results = model.predict(sonic_path, save=True, project="praticas/modulo_5/aula_1", name="predict_sonic", exist_ok=True)

    # Exibir a imagem resultante com as detecções do YOLO (usando o save_dir real da ultralytics,
    # que costuma ficar sob 'runs/detect/...')
    predicted_sonic_path = os.path.join(str(sonic_results[0].save_dir), "sonic.jpg")
    if os.path.exists(predicted_sonic_path):
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        img = mpimg.imread(predicted_sonic_path)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Detecção YOLO no Sonic (Out-of-Distribution)")
        plt.show()
    else:
        print("Não foi possível carregar a imagem predita do Sonic.")

# %% [markdown] id="HS01MePrr9aV"
# a identificação de carrot kkkk

# %% [markdown] id="40f380af"
# ---
# # ⏸️ Ponto de parada

# %% [markdown] id="ad2b58a6"
# ## Atividade (opcional/pesada): Comparação de Hiperparâmetros
#
# Treinamos uma **segunda** YOLO com uma configuração mais "enxuta" (menos épocas, imagem menor,
# batch maior) e comparamos a qualidade de detecção (**mAP**) com o treino original. Assim dá pra
# sentir o trade-off entre tempo de treino e desempenho.

# %% colab={"base_uri": "https://localhost:8080/"} id="9d0097b5" outputId="9ad7681e-3c29-4f8e-f947-650c720131d4"
# Segundo treino com hiperparâmetros diferentes (config enxuta: mais rápida)
model_v2 = YOLO("yolo11n.pt")
results_v2 = model_v2.train(data="./yolo_train.yaml", project="praticas/modulo_5/aula_1",
                            name="train_v2", exist_ok=True, epochs=5, batch=4, imgsz=320)

# Extrai as métricas de cada treino de forma robusta (mAP50 e mAP50-95)
def get_map(res):
    try:
        return res.box.map50, res.box.map
    except Exception:
        return None, None

m50_v1, m_v1 = get_map(results)
m50_v2, m_v2 = get_map(results_v2)

def fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "n/d"

print("=== Comparação de configurações da YOLO ===")
print(f"{'Config':<30} {'mAP50':<10} {'mAP50-95':<10}")
print(f"{'Original (ep=10, bs=2, 480px)':<30} {fmt(m50_v1):<10} {fmt(m_v1):<10}")
print(f"{'Enxuta   (ep=5,  bs=4, 320px)':<30} {fmt(m50_v2):<10} {fmt(m_v2):<10}")
print()
print("Esperado: a config enxuta treina bem mais rápido, geralmente com mAP um pouco menor —")
print("o clássico trade-off entre custo de treino e acurácia.")
