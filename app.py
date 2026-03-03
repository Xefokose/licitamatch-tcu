import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import pdfplumber
import docx
import io
import os

# Configuração da Página
st.set_page_config(page_title="LicitaMatch TCU", page_icon="⚖️", layout="wide")

# Título e Estilo
st.title("⚖️ LicitaMatch TCU - Inteligência Jurisprudencial")
st.markdown("""
    **Sistema Automático de Sugestão de Acórdãos**  
    Faça o upload da sua peça (Impugnação, Recurso, Contrarrazão) e receba 
    os acórdãos do TCU semanticamente compatíveis.
""")

# Cache do Modelo de IA (Para não recarregar toda vez)
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Cache da Base de Dados
@st.cache_data
def load_database():
    # Tenta carregar a base de conhecimento do próprio repositório
    try:
        # URL direta do arquivo CSV no seu GitHub (substitua pelo seu usuário se necessário)
        # Para simplificar, vamos ler localmente se existir, ou baixar um sample
        if os.path.exists('base_tcu.csv'):
            df = pd.read_csv('base_tcu.csv')
            return df
        else:
            return None
    except:
        return None

# Função para extrair texto de PDF
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Função para extrair texto de DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Função de Busca Semântica
def search_jurisprudence(query_text, df, model, top_k=5):
    if df is None or df.empty:
        return []
    
    # Limpeza
    df = df.dropna(subset=['ementa', 'texto_decisao'])
    textos_base = (df['ementa'].astype(str) + " " + df['texto_decisao'].astype(str)).tolist()
    
    # Vetorização
    query_embedding = model.encode(query_text, convert_to_tensor=True)
    corpus_embeddings = model.encode(textos_base, convert_to_tensor=True)
    
    # Similaridade
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))
    
    resultados = []
    for score, idx in zip(top_results[0], top_results[1]):
        idx = idx.item()
        resultados.append({
            "score": f"{score.item()*100:.2f}%",
            "ementa": df.iloc[idx]['ementa'],
            "decisao": df.iloc[idx]['texto_decisao'][:300] + "...",
            "numero": df.iloc[idx].get('numero_acordao', 'N/A')
        })
    return resultados

# --- INTERFACE DO USUÁRIO ---

# Sidebar para Upload de Base (Caso queira atualizar)
with st.sidebar:
    st.header("🗄️ Gestão de Dados")
    uploaded_db = st.file_uploader("Atualizar Base de Acórdãos (CSV)", type=['csv'])
    if uploaded_db:
        df_temp = pd.read_csv(uploaded_db)
        df_temp.to_csv('base_tcu.csv', index=False)
        st.success("Base atualizada com sucesso! Recarregue a página.")
        st.cache_data.clear()

# Área Principal
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 1. Upload da Peça Jurídica")
    uploaded_file = st.file_uploader("Arraste seu PDF ou DOCX", type=['pdf', 'docx'])
    
    texto_peca = ""
    if uploaded_file:
        if uploaded_file.name.endswith('.pdf'):
            texto_peca = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith('.docx'):
            texto_peca = extract_text_from_docx(uploaded_file)
        
        st.info(f"📄 Texto extraído: {len(texto_peca)} caracteres")
        
        # Permitir edição manual caso a extração falhe
        texto_final = st.text_area("Edite o texto se necessário (Foque nos argumentos jurídicos):", value=texto_peca, height=200)

with col2:
    st.subheader("🔍 2. Resultados Sugeridos")
    if st.button("🚀 Analisar Compatibilidade"):
        if not texto_final:
            st.warning("Por favor, faça o upload de um arquivo ou digite o texto.")
        else:
            with st.spinner("A IA está lendo o TCU..."):
                model = load_model()
                df = load_database()
                
                # Se não houver base, avisa
                if df is None:
                    st.error("⚠️ Base de dados não encontrada. Use a barra lateral para subir um CSV de acórdãos.")
                else:
                    resultados = search_jurisprudence(texto_final, df, model)
                    
                    if resultados:
                        st.success("Análise Concluída!")
                        for res in resultados:
                            with st.expander(f"🏛️ Compatibilidade: {res['score']} - Acórdão {res['numero']}"):
                                st.markdown(f"**Ementa:** {res['ementa']}")
                                st.markdown(f"**Trecho da Decisão:** {res['decisao']}")
                                st.code(f"Cite como: TCU, Acórdão {res['numero']}, Rel. Min. [Nome], {res['ementa'][:50]}...")
                    else:
                        st.warning("Nenhum acórdão compatível encontrado na base atual.")

# Rodapé
st.markdown("---")
st.caption("Sistema desenvolvido para uso jurídico. Sempre verifique a vigência da jurisprudência.")
