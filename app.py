import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import pdfplumber
import docx
import io
import os
import re

# Configuração da Página
st.set_page_config(page_title="LicitaMatch TCU", page_icon="⚖️", layout="wide")

# Título e Estilo
st.title("⚖️ LicitaMatch TCU - Inteligência Jurisprudencial")
st.markdown("""
    **Sistema Automático de Sugestão de Acórdãos**  
    Faça o upload da sua peça (Impugnação, Recurso, Contrarrazão) e receba 
    os acórdãos do TCU semanticamente compatíveis.
""")

# Cache do Modelo de IA
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Cache da Base de Dados
@st.cache_data
def load_database():
    try:
        if os.path.exists('base_tcu.csv'):
            # Tenta ler com pipe primeiro (formato do seu arquivo)
            try:
                df = pd.read_csv('base_tcu.csv', sep='|', engine='python', encoding='utf-8')
            except:
                # Tenta ler com vírgula
                df = pd.read_csv('base_tcu.csv', sep=',', engine='python', encoding='utf-8')
            
            # Limpeza automática dos dados
            df = limpar_dados_tcu(df)
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Erro ao carregar base: {str(e)}")
        return None

# Função para limpar e organizar os dados do TCU
def limpar_dados_tcu(df):
    # Mapeia as colunas do seu formato específico
    coluna_map = {
        'INFORMATIVO-LC': 'id',
        'Informativo de Licitações': 'titulo',
        'Plenário': 'colegiado',
        'Acórdão': 'acordao',
        'Nos contratos': 'ementa'
    }
    
    # Renomeia colunas baseado no conteúdo
    novas_colunas = []
    for col in df.columns:
        if 'INFORMATIVO' in str(col).upper():
            novas_colunas.append('id')
        elif 'LICITA'' in str(col).upper():
            novas_colunas.append('titulo')
        elif 'PLEN' in str(col).upper() or 'CÂMARA' in str(col).upper():
            novas_colunas.append('colegiado')
        elif 'ACÓRDÃO' in str(col).upper() or 'numero' in str(col).lower():
            novas_colunas.append('numero_acordao')
        elif 'EMENTA' in str(col).upper() or 'contratos' in str(col).lower():
            novas_colunas.append('ementa')
        else:
            novas_colunas.append(f'col_{len(novas_colunas)}')
    
    df.columns = novas_colunas[:len(df.columns)]
    
    # Remove tags HTML/XML
    if 'ementa' in df.columns:
        df['ementa'] = df['ementa'].astype(str).apply(lambda x: re.sub(r'<[^>]+>', '', x))
    if 'numero_acordao' in df.columns:
        df['numero_acordao'] = df['numero_acordao'].astype(str).apply(lambda x: re.sub(r'<[^>]+>', '', x))
    
    return df

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
    
    # Identifica colunas disponíveis
    col_ementa = 'ementa' if 'ementa' in df.columns else df.columns[0]
    col_numero = 'numero_acordao' if 'numero_acordao' in df.columns else 'id'
    
    # Limpeza
    df = df.dropna(subset=[col_ementa])
    textos_base = df[col_ementa].astype(str).tolist()
    
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
            "ementa": df.iloc[idx][col_ementa][:500],
            "numero": df.iloc[idx].get(col_numero, 'N/A'),
            "colegiado": df.iloc[idx].get('colegiado', 'N/A')
        })
    return resultados

# --- INTERFACE DO USUÁRIO ---

# Sidebar
with st.sidebar:
    st.header("🗄️ Gestão de Dados")
    uploaded_db = st.file_uploader("Upload da Base TCU (CSV)", type=['csv', 'txt'])
    if uploaded_db:
        try:
            # Detecta automaticamente o delimitador
            content = uploaded_db.read().decode('utf-8')
            if '|' in content:
                df_temp = pd.read_csv(io.StringIO(content), sep='|', engine='python')
            else:
                df_temp = pd.read_csv(io.StringIO(content), sep=',', engine='python')
            
            df_temp = limpar_dados_tcu(df_temp)
            df_temp.to_csv('base_tcu.csv', index=False)
            st.success("✅ Base atualizada! Recarregue a página (F5).")
            st.cache_data.clear()
        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")
    
    st.info("📊 Seu arquivo `boletim-informativo-lc.csv` é compatível!")

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
        texto_final = st.text_area("Edite o texto se necessário:", value=texto_peca, height=200)

with col2:
    st.subheader("🔍 2. Resultados Sugeridos")
    if st.button("🚀 Analisar Compatibilidade"):
        if 'texto_final' not in locals() or not texto_final:
            st.warning("Por favor, faça o upload de um arquivo.")
        else:
            with st.spinner("🤖 A IA está analisando a jurisprudência do TCU..."):
                model = load_model()
                df = load_database()
                
                if df is None:
                    st.error("⚠️ Base não encontrada. Use a barra lateral para upload.")
                else:
                    resultados = search_jurisprudence(texto_final, df, model)
                    
                    if resultados:
                        st.success("✅ Análise Concluída!")
                        for res in resultados:
                            with st.expander(f"🏛️ {res['score']} - Acórdão {res['numero']}"):
                                st.markdown(f"**Ementa:** {res['ementa']}")
                                st.markdown(f"**Colegiado:** {res['colegiado']}")
                                st.code(f"Sugestão de citação: TCU, Acórdão {res['numero']}, {res['colegiado']}")
                    else:
                        st.warning("Nenhum acórdão compatível encontrado.")

st.markdown("---")
st.caption("Sistema desenvolvido para uso jurídico interno. Sempre verifique a vigência da jurisprudência.")
