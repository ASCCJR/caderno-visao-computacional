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
# # M6A3 - Sistemas de Monitoramento de Experimentos
#
# > **Resumo:** Última aula da trilha! Em **MLOps**, rastrear experimentos é essencial para reproduzir, comparar e auditar treinos. Aqui re-treinamos a GAN do M4A2, mas agora **instrumentada com o TensorBoard**: logamos as perdas (por batch e por época), as imagens geradas ao longo do treino, o grafo dos modelos e os hiperparâmetros.
#
# > 📝 **Por que tracking?** Depois de dezenas de experimentos com hiperparâmetros diferentes, é fácil esquecer qual configuração deu o melhor resultado. Ferramentas como TensorBoard, MLflow e W&B organizam isso (parâmetros, métricas, artefatos) e tornam o trabalho reproduzível.

# %% [markdown]
# > ⚠️ **Use GPU!** Treina a GAN por 200 épocas no MNIST (a mesma do M4A2). No Colab: **Ambiente de execução → Alterar o tipo → GPU (T4)**. Em CPU levaria horas.

# %% [markdown]
# Na prática de hoje vamos monitorar um experimento utilizando o [TensorBoard](https://www.tensorflow.org/tensorboard?hl=pt-br).
#
# No ciclo de vida do desenvolvimento de aprendizado de máquina (**MLOps**), o **monitoramento de experimentos** (experiment tracking) é essencial. Ele garante a reprodutibilidade dos modelos, permite a governança e auditoria dos dados, facilita a comparação de curvas de aprendizado e auxilia na detecção precoce de anomalias de treino, como desvanecimento de gradientes ou colapso de modo em redes generativas adversariais (GANs).
#
# Embora o **TensorBoard** seja a ferramenta integrada mais comum para monitoramento no ecossistema do TensorFlow e PyTorch, a indústria também utiliza outras plataformas robustas, como:
# - **MLflow**: Uma plataforma de código aberto para gerenciar o ciclo de vida completo de ML, incluindo rastreamento de parâmetros, métricas e artefatos (modelos salvos).
# - **Weights & Biases (W&B)**: Uma ferramenta comercial popular com foco em colaboração e visualizações avançadas.
# - **Comet ML**: Outro serviço de nuvem robusto para rastreamento automatizado de dados e códigos de treino.
#
# Esse notebook está estruturado da seguinte forma.
#
# - Introdução
# - Rodar experimento simples
# - Acompanhar logs
# - Próximos passos
# - ✅ Atividades Complementares (resolvidas)

# %% [markdown]
# ## Introdução

# %% [markdown]
# Instalação para os que ainda não possuem a biblioteca instalada.

# %%
# !pip install torch torchvision tensorboard

# %% [markdown]
# Importar as bibliotecas

# %%
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


# %% [markdown]
# ## Rodar Experimentos Simples
#
# Para isso iremos reproduzir o experimento da aula M4A2 sobre GANs.

# %%
####################
## Carregar Dados ##
####################

batch_size = 100

# MNIST Dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5), std=(0.5))])

train_dataset = torchvision.datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


########################
## Criando os modelos ##
########################


# Gerador.
class Generator(torch.nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = torch.nn.Linear(g_input_dim, 256)
        self.fc2 = torch.nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = torch.nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = torch.nn.Linear(self.fc3.out_features, g_output_dim)
    
    # método forward. 
    def forward(self, x): 
        x = torch.nn.functional.leaky_relu(self.fc1(x), 0.2)
        x = torch.nn.functional.leaky_relu(self.fc2(x), 0.2)
        x = torch.nn.functional.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

# Discrimador.
class Discriminator(torch.nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(d_input_dim, 1024)
        self.fc2 = torch.nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = torch.nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = torch.nn.Linear(self.fc3.out_features, 1)
    
    # método forward. 
    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x), 0.2)
        x = torch.nn.functional.dropout(x, 0.3)
        x = torch.nn.functional.leaky_relu(self.fc2(x), 0.2)
        x = torch.nn.functional.dropout(x, 0.3)
        x = torch.nn.functional.leaky_relu(self.fc3(x), 0.2)
        x = torch.nn.functional.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))
    

#################
## Treinamento ##
#################

# Instanciar o logger do Tensorboard.
# Pode alterar o path.
writer =  SummaryWriter("logs/gan_2")

# Instanciar as redes.
z_dim = 100
mnist_dim = train_dataset.data.size(1) * train_dataset.data.size(2)  # 28*28 = 784

device = "cuda" if torch.cuda.is_available() else "cpu"
G = Generator(g_input_dim = z_dim, g_output_dim = mnist_dim).to(device)
D = Discriminator(mnist_dim).to(device)

# Função de perda.
criterion = torch.nn.BCELoss() 

# Otimizador.
lr = 0.0002 
G_optimizer = torch.optim.Adam(G.parameters(), lr = lr)
D_optimizer = torch.optim.Adam(D.parameters(), lr = lr)

def D_train(x):
    #=======================Treino do discriminador=======================#
    D.zero_grad()

    # Treina discriminador em dados reais.
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(batch_size, 1)
    x_real, y_real = torch.autograd.Variable(x_real.to(device)), torch.autograd.Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # Treina discriminador em dados falsos.
    z = torch.autograd.Variable(torch.randn(batch_size, z_dim).to(device))
    x_fake, y_fake = G(z), torch.autograd.Variable(torch.zeros(batch_size, 1).to(device))

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # Backpropagation e otimização dos parâmetros do discriminador.
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()

def G_train(x):
    #=======================Treino do gerador=======================#
    G.zero_grad()

    z = torch.autograd.Variable(torch.randn(batch_size, z_dim).to(device))
    y = torch.autograd.Variable(torch.ones(batch_size, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # Backpropagation e otimização dos parâmetros do gerador.
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()

# Laço de treino.
n_epoch = 200

# Registrar os grafos do Gerador e Discriminador antes do treino (Atividade Complementar)
try:
    dummy_input_g = torch.randn(batch_size, z_dim).to(device)
    writer.add_graph(G, dummy_input_g)
    
    # Para o discriminador, a entrada é uma imagem achatada do MNIST (dim=784)
    dummy_input_d = torch.randn(batch_size, mnist_dim).to(device)
    writer.add_graph(D, dummy_input_d)
    print("Grafos dos modelos registrados no TensorBoard com sucesso!")
except Exception as e:
    print(f"Aviso ao registrar grafos no TensorBoard: {e}")

for epoch in range(1, n_epoch+1):           
    D_losses, G_losses = [], []
    for batch_idx, (x, _) in enumerate(train_loader):
        D_loss_val = D_train(x)
        G_loss_val = G_train(x)
        D_losses.append(D_loss_val)
        G_losses.append(G_loss_val)
        
        # Calcular o global step acumulador contínuo ao longo das épocas (resolvendo o bug do batch_idx sobreposto)
        global_step = (epoch - 1) * len(train_loader) + batch_idx
        writer.add_scalar("Loss/Treino_Discriminator_step", D_loss_val, global_step)
        writer.add_scalar("Loss/Treino_Generator_step", G_loss_val, global_step)

    mean_d_loss = torch.mean(torch.FloatTensor(D_losses)).item()
    mean_g_loss = torch.mean(torch.FloatTensor(G_losses)).item()

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epoch, mean_d_loss, mean_g_loss))
    writer.add_scalar("Loss/Treino_Discriminator_epoch", mean_d_loss, epoch)
    writer.add_scalar("Loss/Treino_Generator_epoch", mean_g_loss, epoch)
    
    # Gerando imagens para os logs.
    with torch.no_grad():
        test_z = torch.autograd.Variable(torch.randn(batch_size, z_dim).to(device))
        generated = G(test_z)

    generated = generated.view(generated.size(0), 1, 28, 28)
    grid = torchvision.utils.make_grid(generated.cpu(), 20)
    writer.add_image("Imagens Geradas", grid, epoch)

# Registrar hiperparâmetros e as perdas finais obtidas (Atividade Complementar)
try:
    writer.add_hparams(
        {"lr": lr, "batch_size": batch_size, "epochs": n_epoch},
        {"hparam/loss_discriminator": mean_d_loss, "hparam/loss_generator": mean_g_loss}
    )
    print("Hiperparâmetros registrados no TensorBoard com sucesso!")
except Exception as e:
    print(f"Aviso ao registrar hiperparâmetros no TensorBoard: {e}")

writer.flush()
writer.close()

# %% [markdown]
# ## Acompanhar Logs
#
# Para visualizar os experimentos monitorados localmente no seu terminal, execute:
#
# ```bash
# tensorboard --logdir logs
# ```
#
# Se você estiver executando no **Google Colab**, você pode carregar a extensão mágica e visualizar o TensorBoard diretamente dentro do seu notebook com os comandos abaixo:

# %%
# %load_ext tensorboard
# %tensorboard --logdir logs

# %% [markdown]
# ## Próximos Passos e Referências

# %% [markdown]
# E essa é a última prática da nossa trilha de Visão Computacional, espero que tenho aproveitado.
#
# Uma lista não exaustiva de referências segue:
#
# - https://www.tensorflow.org/tensorboard?hl=pt-br
# - https://docs.pytorch.org/docs/stable/tensorboard.html
# - https://github.com/lyeoni/pytorch-mnist-GAN
# - https://pytorch.org/
# - https://docs.pytorch.org/vision/main/models.html
# - https://opencv.org/
# - https://learnopencv.com/blogs/
# - https://pyimagesearch.com/

# %% [markdown]
# ## Atividades Complementares (Opcional)

# %% [markdown]
# - [x] **Explore outras funções de logs possíveis no tensorboard e veja os logs.**
#   Implementamos com sucesso os logs do grafo estrutural dos modelos (`add_graph`) e hiperparâmetros com métricas finais agregadas (`add_hparams`). Explore esses gráficos e tabelas de hiperparâmetros abrindo o painel do TensorBoard!
# - [x] **Existem outras soluções para controle de experimentos?**
#   Conforme discutido na introdução, alternativas maduras para MLOps na indústria incluem o **MLflow**, **Weights & Biases (W&B)** e **Comet ML**. Eles oferecem dashboards de equipe, versionamento de datasets e gerenciamento de artefatos de modelos avançados.
