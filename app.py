import streamlit as st
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Preditor de Churn", layout="centered")

st.title("📊 Previsão de Cancelamento de Cliente (Churn)")
st.write("Insira os dados do cliente abaixo para analisar o risco de cancelamento.")

# Inputs do usuário
idade = st.number_input("Idade do Cliente:", min_value=18, max_value=100, value=30)
contrato = st.number_input("Meses de Contrato:", min_value=0, max_value=120, value=12)
gasto = st.number_input("Gasto Mensal (R$):", min_value=0.0, max_value=1000.0, value=150.0)

# Botão de previsão
if st.button("Analisar Risco"):
    try:
        # Carregar modelo carregado previamente
        model = tf.keras.models.load_model('modelo_churn.h5')
        
        # Preparar dados de entrada (idealmente aplicar o mesmo scaler do treino)
        dados_entrada = np.array([[idade, contrato, gasto]])
        
        # Predição
        predicao = model.predict(dados_entrada)[0][0]
        
        st.subheader("Resultado da Análise:")
        if predicao > 0.5:
            st.error(f"Alerta! Alta chance de Churn: {predicao:.2%}")
        else:
            st.success(f"Cliente Seguro! Chance de Churn: {predicao:.2%}")
            
    except Exception as e:
        st.warning("Nota: Certifique-se de que o arquivo 'modelo_churn.h5' está no mesmo diretório.")
        st.error(f"Erro ao carregar/processar o modelo: {e}")