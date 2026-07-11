import tensorflow as tf
from tensorflow.keras import layers, losses
import numpy as np

# 1. Dataset expandido e calibrado (Português e saudações comuns)
training_data = [
    # ---- HAM (Mensagens Seguras / Neutras - Classe 0) ----
    "hello",
    "i love you",
    "bom dia",
    "boa tarde, tudo bem?",
    "oi, tudo bem com você?",
    "precisamos alinhar a reunião de hoje mais tarde",
    "consegue me enviar o relatório por e-mail até o fim do dia?",
    "vamos almoçar juntos hoje?",
    "tudo certo para o início das aulas na segunda-feira",
    "você pode verificar esse erro no código para mim?",
    "obrigado pelo retorno, fico no aguardo",
    "olá, gostaria de tirar uma dúvida sobre o projeto",
    "seja bem vindo"
    
    # ---- SPAM (Golpes, Promoções, Links Suspeitos - Classe 1) ----
    "Ganhe dinheiro fácil trabalhando de casa! Clique aqui no link",
    "Você ganhou um sorteio exclusivo! Cadastre seu Pix agora para receber",
    "URGENTE: Sua conta bancária será bloqueada. Acesse para atualizar seus dados",
    "PROMOÇÃO IMPERDÍVEL: Compre hoje com 90% de desconto e frete grátis",
    "Acesse o link e resgate seu prêmio de 10 mil reais agora mesmo",
    "Invista apenas 50 reais e ganhe 5000 por dia de forma garantida!",
    "Parabéns! Seu número foi selecionado para uma vaga de emprego de meio período",
    "Aumente suas vendas de forma rápida e mágica com nosso robô de automação",
    "Clique aqui e mude de vida agora com o método secreto das apostas",
    "Você ganhou um carro"
]

# 0 = Seguro (Ham), 1 = Spam
labels = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,  # 12 exemplos de Ham
    1, 1, 1, 1, 1, 1, 1, 1, 1,1            # 9 exemplos de Spam
], dtype=np.int32)

# Convertendo os dados para o Tensor de strings do TF
X_train = tf.convert_to_tensor(training_data, dtype=tf.string)

# 2. Configuração do Vocabulário
max_features = 2000
sequence_length = 50

vectorize_layer = layers.TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)
vectorize_layer.adapt(X_train)

# 3. Arquitetura da Rede Neural
model = tf.keras.Sequential([
    vectorize_layer,
    layers.Embedding(max_features, 32), # Aumentado para 32 para capturar melhor o contexto das palavras
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2), # Dropout para evitar que decore apenas palavras específicas
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss=losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# 4. Treinamento (Aumentado para 100 épocas para estabilizar os pesos das palavras neutras)
model.fit(X_train, labels, epochs=200, verbose=0)

# 5. Salvar o novo modelo calibrado
model.save("spam_model.keras")
print("🔥 Novo modelo 'spam_model.keras' recalibrado com sucesso!")