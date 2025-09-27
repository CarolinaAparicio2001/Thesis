import streamlit as st
import os
import yaml
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import tempfile
import shutil # Para gerir o diretório temporário do FAISS store

# --- Funções adaptadas do Notebook ---

# Função para carregar a chave API (modificada para não usar ficheiro YAML diretamente no Streamlit)
def set_openai_api_key(api_key):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        return True
    return False

# Constantes (algumas podem ser configuráveis via UI)
PHASES = ["I", "II", "III"]
FAISS_STORE_BASE_DIR = "faiss_store_streamlit" # Diretório base para os FAISS stores

# Funções de carregamento e pré-processamento de dados
# Modificar load_phase_data para aceitar um DataFrame diretamente ou um caminho de ficheiro
def load_phase_data(phase, split="train", base_path="."):
    csv_path = os.path.join(base_path, f"filtered_phase_{phase}_{split}.csv")
    try:
        csv_df = pd.read_csv(csv_path)
        return csv_df
    except FileNotFoundError:
        st.warning(f"Ficheiro não encontrado: {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao ler {csv_path}: {e}")
        return pd.DataFrame()

def clean_criteria(criteria):
    if isinstance(criteria, str):
        criteria = criteria.replace("\\r\\n", " ").replace("\\n", " ").strip()
        criteria = ' '.join(criteria.split())
    return criteria

# Modificar load_all_data para usar um base_path fornecido
@st.cache_data # Cache para evitar recarregar e processar dados desnecessariamente
def load_all_data_cached(temp_data_path):
    all_dfs = []
    for phase_val in PHASES:
        for split_val in ["train", "test"]:
            df_phase_split = load_phase_data(phase_val, split_val, base_path=temp_data_path)
            if not df_phase_split.empty:
                required_cols = ["NCT Number", "Enrollment", "criteria", "label", "smiles", "icdcodes", "fused_pred", "diseases", "drugs"]
                for col in required_cols:
                    if col not in df_phase_split.columns:
                        st.warning(f"Coluna '{col}' não encontrada em phase {phase_val}, {split_val}. Será preenchida com 'N/A'.")
                        df_phase_split[col] = "N/A"
                
                if "criteria" in df_phase_split.columns:
                     df_phase_split["criteria"] = df_phase_split["criteria"].apply(clean_criteria)
                else:
                    df_phase_split["criteria"] = "N/A"

                df_phase_split["phase_meta"] = phase_val
                all_dfs.append(df_phase_split)
    
    if not all_dfs:
        st.error("Nenhum dado foi carregado. Verifique os ficheiros CSV no diretório fornecido.")
        return pd.DataFrame()
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    final_df["combined_text"] = (
        final_df["NCT Number"].astype(str) + " " +
        final_df["Enrollment"].astype(str) + " " +
        final_df["criteria"].astype(str) + " " +
        final_df["label"].astype(str) + " " +
        final_df["smiles"].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)) + " " +
        final_df["icdcodes"].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)) + " " +
        final_df["fused_pred"].astype(str)
    )
    return final_df

@st.cache_data # Cache para evitar recriar documentos desnecessariamente
def create_langchain_documents_cached(_df_main_data): # O _ antes de df_main_data é uma convenção para indicar que o input é usado para caching
    docs = []
    for index, row in _df_main_data.iterrows():
        content = f"""
        Clinical trial NCT Number: {row.get('NCT Number', 'N/A')}.
        Phase: {row.get('phase_meta', 'N/A')}.
        Enrollment: {row.get('Enrollment', 'N/A')} participants.
        Diseases: {str(row.get('diseases', 'N/A'))}.
        Drugs: {str(row.get('drugs', 'N/A'))}.
        Eligibility Criteria: {row.get('criteria', 'N/A')}.
        ICD Codes: {str(row.get('icdcodes', 'N/A'))}.
        SMILES (Molecular Information): {str(row.get('smiles', 'N/A'))}.
        Original Model Prediction Score (fused_pred): {row.get('fused_pred', 'N/A')}.
        Actual Outcome (label - 1 for success, 0 for failure): {row.get('label', 'N/A')}.
        """
        metadata = {
            "source": "csv_clinical_trials", 
            "nct_number": row.get("NCT Number", "N/A"), 
            "phase": row.get("phase_meta", "N/A"), 
            "label": row.get("label", "N/A"),
            "original_fused_pred": row.get("fused_pred", "N/A"),
            "row_index": index
        }
        docs.append(Document(page_content=content.strip(), metadata=metadata))
    return docs

# Embedding e FAISS
@st.cache_resource # Cache para o modelo de embedding
def get_embedding_function():
    eh_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    return HuggingFaceEmbeddings(model_name=eh_model_name)

def get_faiss_store_path(data_dir_name):
    return os.path.join(FAISS_STORE_BASE_DIR, f"faiss_store_{data_dir_name}")


@st.cache_resource # Cache para o vector store
def load_or_create_faiss_store(_lc_documents, _embedding_function, faiss_store_path_specific):
    if os.path.exists(os.path.join(faiss_store_path_specific, "index.faiss")) and \
       os.path.exists(os.path.join(faiss_store_path_specific, "index.pkl")):
        try:
            vector_store = FAISS.load_local(folder_path=faiss_store_path_specific, embeddings=_embedding_function, allow_dangerous_deserialization=True)
            st.success(f"Vector store FAISS carregado de {faiss_store_path_specific}")
            return vector_store
        except Exception as e:
            st.warning(f"Erro ao carregar o vector store FAISS de {faiss_store_path_specific}: {e}. A criar um novo.")
            if os.path.exists(faiss_store_path_specific):
                shutil.rmtree(faiss_store_path_specific)
    
    if not _lc_documents:
        st.error("Nenhum documento LangChain foi criado; não é possível construir o vector store.")
        return None
        
    try:
        os.makedirs(faiss_store_path_specific, exist_ok=True)
        vector_store = FAISS.from_documents(_lc_documents, _embedding_function)
        vector_store.save_local(folder_path=faiss_store_path_specific)
        st.success(f"Novo vector store FAISS criado e guardado em {faiss_store_path_specific}")
        return vector_store
    except Exception as e:
        st.error(f"ERRO CRÍTICO ao criar o vector store FAISS: {e}")
        return None

# Cadeia QA
@st.cache_resource # Cache para a cadeia QA
def get_qa_chain(_vector_store):
    if not _vector_store:
        return None
    try:
        llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo")
        retriever = _vector_store.as_retriever(search_kwargs={"k": 3})

        prompt_template_rag = """
        You are an expert assistant specializing in clinical trial analysis.
        Use the following retrieved CONTEXT from a clinical trials database to answer the QUESTION.
        If the context does not contain enough information to answer the question, explicitly state that you do not have sufficient information based on the provided documents.
        Do not attempt to invent an answer or use external knowledge.

        CONTEXT:
        {context}

        QUESTION: {question}

        Based ONLY on the provided context, answer as follows:
        1. PREDICTION OF TRIAL OUTCOME: (Indicate if the clinical trial described in the context is likely to SUCCEED or FAIL. Base this on qualitative data from the context. If the context mentions an 'Actual Outcome (label)' or 'Original Model Prediction (fused_pred)', you can use them as a reference, but form your own prediction based on the provided details.)
        2. DETAILED EXPLANATION OF THE PREDICTION: (Provide a detailed explanation for your prediction, citing the most important factors from the context, such as eligibility criteria, trial phase, diseases, drugs, sample size (enrollment), etc., that led you to your conclusion. If the 'Actual Outcome (label)' is available, compare your prediction with it.)
        3. KEY RISK AND SUCCESS FACTORS IDENTIFIED IN CONTEXT: (List the main risk factors for failure and the main factors that could contribute to the success of this specific trial, based strictly on the provided context.)

        ANALYTICAL RESPONSE:
        """
        RAG_PROMPT = PromptTemplate(
            template=prompt_template_rag, input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": RAG_PROMPT}
        )
        st.success("Cadeia RetrievalQA configurada com sucesso.")
        return qa_chain
    except Exception as e:
        st.error(f"Erro ao configurar a cadeia RetrievalQA: {e}")
        return None

# --- Interface Streamlit ---
st.set_page_config(page_title="Chatbot de Análise de Ensaios Clínicos", layout="wide")
st.title("🔬 Chatbot de Análise de Ensaios Clínicos (RAG)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "temp_data_dir" not in st.session_state:
    st.session_state.temp_data_dir = None
if "faiss_store_path_current" not in st.session_state:
    st.session_state.faiss_store_path_current = None

with st.sidebar:
    st.header("Configurações")
    
    openai_api_key = st.text_input("Chave API OpenAI", type="password", key="api_key_input")
    if openai_api_key:
        if set_openai_api_key(openai_api_key):
            st.success("Chave API OpenAI configurada.")
        else:
            st.error("Por favor, insira uma chave API OpenAI válida.")

    st.markdown("---")
    st.subheader("Fonte de Dados (Ficheiros CSV)")
    st.markdown("""
    Por favor, carregue os 6 ficheiros CSV necessários para as fases I, II, e III (train e test splits).
    Os ficheiros devem seguir o padrão de nomenclatura: `filtered_phase_{PHASE}_{SPLIT}.csv`
    (ex: `filtered_phase_I_train.csv`, `filtered_phase_I_test.csv`, etc.)
    """)

    uploaded_files = st.file_uploader(
        "Carregar ficheiros CSV (filtered_phase_X_Y.csv)", 
        type="csv", 
        accept_multiple_files=True,
        key="csv_uploader"
    )

    if st.button("Carregar Dados e Inicializar Chatbot", key="init_button"):
        if not openai_api_key:
            st.error("Por favor, insira a sua chave API OpenAI.")
        elif not uploaded_files or len(uploaded_files) < 6:
            st.error("Por favor, carregue todos os 6 ficheiros CSV necessários (Fase I, II, III - train/test).")
        else:
            with st.spinner("A processar dados e a inicializar o sistema... Isto pode demorar alguns minutos."):
                if st.session_state.temp_data_dir and os.path.exists(st.session_state.temp_data_dir):
                    shutil.rmtree(st.session_state.temp_data_dir)
                
                st.session_state.temp_data_dir = tempfile.mkdtemp(prefix="streamlit_csv_data_")
                
                expected_files_pattern = {
                    f"filtered_phase_{p}_{s}.csv": False for p in PHASES for s in ["train", "test"]
                }
                files_saved_count = 0
                for uploaded_file in uploaded_files:
                    if uploaded_file.name in expected_files_pattern:
                        file_path = os.path.join(st.session_state.temp_data_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        expected_files_pattern[uploaded_file.name] = True
                        files_saved_count += 1
                    else:
                        st.warning(f"Ficheiro '{uploaded_file.name}' ignorado pois não corresponde ao padrão de nomenclatura esperado.")

                if files_saved_count < 6:
                    missing_files = [fname for fname, found in expected_files_pattern.items() if not found]
                    st.error(f"Faltam os seguintes ficheiros CSV: {', '.join(missing_files)}. Por favor, carregue todos os ficheiros necessários.")
                    st.session_state.data_loaded = False
                    if os.path.exists(st.session_state.temp_data_dir):
                         shutil.rmtree(st.session_state.temp_data_dir)
                         st.session_state.temp_data_dir = None
                else:
                    st.info(f"Todos os {files_saved_count} ficheiros CSV foram carregados para: {st.session_state.temp_data_dir}")
                    
                    data_dir_name_for_faiss = os.path.basename(st.session_state.temp_data_dir)
                    st.session_state.faiss_store_path_current = get_faiss_store_path(data_dir_name_for_faiss)
                    st.info(f"Caminho do FAISS store: {st.session_state.faiss_store_path_current}")

                    if not os.path.exists(FAISS_STORE_BASE_DIR):
                        os.makedirs(FAISS_STORE_BASE_DIR)

                    df_main_data = load_all_data_cached(st.session_state.temp_data_dir)
                    
                    if not df_main_data.empty:
                        st.success(f"Total de {len(df_main_data)} registos carregados e processados.")
                        lc_documents = create_langchain_documents_cached(df_main_data)
                        st.success(f"Total de {len(lc_documents)} documentos LangChain criados.")

                        if lc_documents:
                            embedding_function = get_embedding_function()
                            vector_store = load_or_create_faiss_store(lc_documents, embedding_function, st.session_state.faiss_store_path_current)
                            
                            if vector_store:
                                st.session_state.qa_chain = get_qa_chain(vector_store)
                                if st.session_state.qa_chain:
                                    st.session_state.data_loaded = True
                                    st.success("Sistema inicializado e pronto para consultas!")
                                else:
                                    st.error("Falha ao inicializar a cadeia QA.")
                                    st.session_state.data_loaded = False
                            else:
                                st.error("Falha ao carregar ou criar o vector store FAISS.")
                                st.session_state.data_loaded = False
                        else:
                            st.error("Nenhum documento LangChain foi criado.")
                            st.session_state.data_loaded = False
                    else:
                        st.error("Falha ao carregar os dados dos ficheiros CSV.")
                        st.session_state.data_loaded = False
                        if os.path.exists(st.session_state.temp_data_dir):
                            shutil.rmtree(st.session_state.temp_data_dir)
                            st.session_state.temp_data_dir = None

if not st.session_state.data_loaded:
    st.info("Por favor, configure a chave API OpenAI e carregue os ficheiros de dados na barra lateral para iniciar o chatbot.")
else:
    st.success("Chatbot pronto! Faça a sua pergunta sobre os ensaios clínicos.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Qual a sua pergunta?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            if st.session_state.qa_chain:
                try:
                    with st.spinner("A pensar..."):
                        response = st.session_state.qa_chain({"query": prompt})
                    
                    result_text = response.get("result", "Não foi possível obter uma resposta.")
                    full_response += result_text
                    
                    source_docs = response.get("source_documents")
                    if source_docs:
                        full_response += "\n\n--- Documentos Fonte ---\n"
                        for i, doc in enumerate(source_docs):
                            full_response += f"\n**Fonte {i+1} (NCT: {doc.metadata.get('nct_number', 'N/A')}, Fase: {doc.metadata.get('phase', 'N/A')})**\n"
                            content_snippet = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                            full_response += f"```\n{content_snippet}\n```\n"
                    
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"Erro ao processar a pergunta: {e}")
                    full_response = "Desculpe, ocorreu um erro ao processar a sua pergunta."
                    message_placeholder.markdown(full_response)
            else:
                full_response = "A cadeia QA não está inicializada. Por favor, verifique as configurações."
                message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

with st.sidebar:
    st.markdown("---")
    st.subheader("Opções de Debug/Manutenção")
    if st.button("Limpar Cache do FAISS Store Atual"):
        if st.session_state.faiss_store_path_current and os.path.exists(st.session_state.faiss_store_path_current):
            try:
                shutil.rmtree(st.session_state.faiss_store_path_current)
                st.success(f"Cache do FAISS store '{st.session_state.faiss_store_path_current}' limpo com sucesso.")
                load_or_create_faiss_store.clear()
                get_qa_chain.clear()
                st.session_state.qa_chain = None
                st.session_state.data_loaded = False
                st.warning("Cache do FAISS limpo. Por favor, clique em 'Carregar Dados e Inicializar Chatbot' novamente.")

            except Exception as e:
                st.error(f"Erro ao limpar o cache do FAISS: {e}")
        else:
            st.info("Nenhum FAISS store atual para limpar ou o caminho não existe.")

    if st.button("Limpar Todos os Caches do Streamlit"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Todos os caches do Streamlit (dados e recursos) foram limpos. Recarregue a página.")

