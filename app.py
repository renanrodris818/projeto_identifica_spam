import streamlit as st
import tensorflow as tf

# Configuração da página
st.set_page_config(page_title="Detector de Spam IA", page_icon="🛡️")

st.title("🛡️ Detector de Mensagens Spam")
st.write("Insira o texto da mensagem abaixo para verificar se ela é segura ou potencial Spam.")

# Cache para evitar recarregar o modelo a cada clique de botão
@st.cache_resource
def load_spam_model():
    return tf.keras.models.load_model("spam_model.keras")

try:
    model = load_spam_model()
except Exception as e:
    st.error("Erro ao carregar o modelo. Certifique-se de que 'spam_model.keras' está na mesma pasta.")
    st.stop()

# Input do usuário
user_input = st.text_area("Texto da Mensagem:", placeholder="Digite ou cole a mensagem aqui...")

# Botão de ação
if st.button("Analisar Mensagem", type="primary"):
    if user_input.strip() == "":
        st.warning("Por favor, digite alguma mensagem antes de analisar.")
    else:
        # Força o input do usuário a ser um Tensor de string estruturado para o Keras 3 / optree
        input_tensor = tf.convert_to_tensor([user_input], dtype=tf.string)
        
        # Predição
        prediction = model.predict(input_tensor)[0][0]

        st.write( prediction)
        
        st.subheader("Resultado da Análise:")
        
        # Threshold padrão de 50%
        if prediction > 0.50129:

            st.success(f"✅ **Mensagem Segura (Ham)** (Confiança de Spam: {prediction * 100:.2f}%)")
            st.markdown("> **Motivo:** O texto parece legítimo e seguro para comunicação padrão.")

        else:
            st.error(f"🚨 **Alerta de Spam!** (Confiança: {prediction * 100:.2f}%)")
            st.markdown("> **Motivo:** Esta mensagem contém padrões frequentemente associados a golpes, promoções falsas ou urgência artificial.")
      