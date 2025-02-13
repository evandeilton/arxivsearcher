#!/usr/bin/env python3
import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import os
from styles import apply_custom_styles

try:
    from integrated_reviewer import IntegratedReviewSystem
except ImportError as ie:
    st.error("Módulo 'integrated_reviewer' não encontrado. Verifique se o arquivo existe e está no caminho correto.")
    raise ie

def display_timeline(df: pd.DataFrame):
    """Exibe histograma de publicações por ano com estilo melhorado."""
    if "published" in df.columns:
        df = df.copy()
        df['published_date'] = pd.to_datetime(df['published'], errors='coerce')
        df_clean = df.dropna(subset=['published_date'])
        if not df_clean.empty:
            df_clean['year'] = df_clean['published_date'].dt.year
            fig = px.histogram(
                df_clean,
                x='year',
                nbins=10,
                title="Distribuição de Publicações por Ano",
                labels={'year': 'Ano de Publicação', 'count': 'Número de Artigos'},
                color_discrete_sequence=['#0096c7'],
                template='plotly_white'
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                title_font=dict(size=20, color='#023e8a'),
                showlegend=False,
                margin=dict(t=50, l=0, r=0, b=0),
                xaxis=dict(gridcolor='#f0f0f0'),
                yaxis=dict(gridcolor='#f0f0f0')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Não há dados suficientes para criar a linha do tempo.")
    else:
        st.warning("⚠️ Coluna 'published' não encontrada no conjunto de dados.")

def create_wordcloud(text: str):
    """Gera uma nuvem de palavras com estilo melhorado."""
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        colormap='viridis',
        max_words=200,
        contour_width=3,
        contour_color='steelblue'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

def initialize_session_state():
    """Inicializa o estado da sessão com valores padrão."""
    default_states = {
        'search_results': None,
        'review_text': None,
        'saved_files': [],
        'error_message': None,
        'selected_rows': {},
        'search_history': [],
        'last_update': None
    }
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

def execute_search(theme: str, max_results: int, provider: str, model: str,
                  output_lang: str, download_all: bool, output_dir: str,
                  date_range: Optional[Tuple[datetime, datetime]], sort_by: str,
                  sort_order: str, download_count: int):
    """Executa a busca de artigos."""
    try:
        system = IntegratedReviewSystem()
        results_df = system.search_papers(
            query=theme,
            max_results=max_results,
            download_count=download_count,
            download_pdfs=download_all,
            save_results=True,
            output_dir=output_dir,
            date_range=date_range,
            sort_by=sort_by,
            sort_order=sort_order
        )
        st.session_state.search_results = results_df
        st.session_state.review_text = None
        st.session_state.saved_files = []
        st.success(f"🎉 Foram encontrados {len(results_df)} artigos!")
        
    except Exception as e:
        st.session_state.error_message = str(e)

def execute_review(theme: str, provider: str, model: str,
                  output_lang: str, output_dir: str):
    """Executa a revisão de literatura."""
    try:
        system = IntegratedReviewSystem()
        df_all = st.session_state.search_results.copy()
        selected_indices = [i for i, val in st.session_state.selected_rows.items() if val]
        
        if not selected_indices:
            st.warning("⚠️ Selecione pelo menos um artigo para gerar a revisão.")
            return

        df_filtered = df_all.iloc[selected_indices]
        
        with st.spinner("🔄 Gerando revisão de literatura..."):
            review_text, saved_files = system.review_papers(
                df=df_filtered,
                theme=theme,
                provider=provider,
                model=model,
                output_lang=output_lang,
                save_results=True,
                output_dir=output_dir
            )
            
        st.session_state.review_text = review_text
        st.session_state.saved_files = saved_files
        st.success(f"✨ Revisão gerada com sucesso! Analisados {len(selected_indices)} artigos.")
        
    except Exception as e:
        st.session_state.error_message = str(e)

def main():
    apply_custom_styles()
    initialize_session_state()
    
    st.markdown("""
        <div style="text-align: center;">
            <h1>📚 Sistema Avançado de Revisão de Literatura</h1>
            <p class="subtitle">
                Automatize sua pesquisa acadêmica com Inteligência Artificial
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("<h3 style='text-align: center;'>🔧 Configurações</h3>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📝 Básico", "🤖 IA", "⚙️ Avançado"])
        
        with tab1:
            theme = st.text_input(
                "Tema da Pesquisa",
                help="Digite o tema principal da sua pesquisa"
            )
            
            max_results = st.slider(
                "Número de Artigos",
                2, 50, 10,
                help="Defina a quantidade de artigos para busca"
            )

        with tab2:
            provider = st.selectbox(
                "Provedor de IA",
                options=["anthropic", "openai", "gemini", "deepseek"],
                format_func=lambda x: {
                    "anthropic": "🌟 Anthropic",
                    "openai": "🤖 OpenAI",
                    "gemini": "🔵 Gemini",
                    "deepseek": "🎯 DeepSeek"
                }[x]
            )

            provider_model_map = {
                "anthropic": ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"],
                "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                "gemini": ["gemini-pro", "gemini-pro-vision"],
                "deepseek": ["deepseek-chat", "deepseek-coder"]
            }
            
            model = st.selectbox(
                "Modelo de IA",
                options=provider_model_map[provider]
            )
            
            output_lang = st.selectbox(
                "Idioma da Revisão",
                options=["pt-BR", "en-US", "es-ES"],
                format_func=lambda x: {
                    "pt-BR": "🇧🇷 Português (Brasil)",
                    "en-US": "🇺🇸 Inglês (EUA)",
                    "es-ES": "🇪🇸 Espanhol (Espanha)"
                }[x]
            )

        with tab3:
            with st.expander("📅 Filtros de Data", expanded=False):
                use_date_range = st.checkbox("Filtrar por Data")
                date_range = None
                if use_date_range:
                    col1, col2 = st.columns(2)
                    with col1:
                        start_date = st.date_input("Data Inicial")
                    with col2:
                        end_date = st.date_input("Data Final")
                    if start_date and end_date:
                        date_range = (start_date, end_date)

            sort_by = st.selectbox(
                "Ordenar Por",
                options=["relevance", "lastUpdatedDate", "submittedDate"],
                format_func=lambda x: {
                    "relevance": "Relevância",
                    "lastUpdatedDate": "Última Atualização",
                    "submittedDate": "Data de Submissão"
                }[x]
            )
            
            sort_order = st.selectbox(
                "Ordem",
                options=["descending", "ascending"],
                format_func=lambda x: "Decrescente" if x == "descending" else "Crescente"
            )

            with st.expander("📥 Download", expanded=False):
                download_all = st.checkbox("Baixar PDFs")
                output_dir = st.text_input(
                    "Diretório de Saída",
                    value="reviews",
                    help="Pasta onde os arquivos serão salvos"
                )
                if download_all:
                    download_count = st.slider(
                        "Quantidade",
                        1, max_results, 2
                    )
                else:
                    download_count = 0

# Botões de ação
        col1, col2 = st.columns([2, 2])
        with col1:
            search_button = st.button(
                "🔍 Pesquisar Artigos",
                help="Realizar Pesquisa",
                key="search_button"
            )
        with col2:
            clear_button = st.button(
                "🔄 Limpar Tabela",
                help="Limpar dados",
                key="clear_button"
            )

    # Área Principal
    if search_button and theme:
        with st.spinner("🔄 Realizando pesquisa..."):
            execute_search(
                theme=theme,
                max_results=max_results,
                provider=provider,
                model=model,
                output_lang=output_lang,
                download_all=download_all,
                output_dir=output_dir,
                date_range=date_range if 'date_range' in locals() else None,
                sort_by=sort_by,
                sort_order=sort_order,
                download_count=download_count if 'download_count' in locals() else 0
            )

    if clear_button:
        st.session_state.search_results = None
        st.session_state.review_text = None
        st.session_state.selected_rows = {}
        # st.experimental_rerun()

    # Exibição dos Resultados
    if st.session_state.search_results is not None:
        tabs = st.tabs(["📊 Resultados", "📈 Análises", "📝 Revisão"])
        
        with tabs[0]:
            st.markdown("### 📚 Artigos Encontrados")
            
            df = st.session_state.search_results.copy()
            if 'selected' not in df.columns:
                df['selected'] = False

            desired_columns = ["selected", "title", "pdf_url", "summary"]
            other_columns = [col for col in df.columns if col not in desired_columns]
            ordered_columns = desired_columns + other_columns
            df = df[ordered_columns]
            
            edited_df = st.data_editor(
                df,
                column_config={
                    "selected": st.column_config.CheckboxColumn(
                        "✓",
                        help="Selecionar para revisão"
                    ),
                    "title": st.column_config.TextColumn(
                        "Título",
                        width="large"
                    ),
                    "pdf_url": st.column_config.LinkColumn(
                        "PDF",
                        width="small"
                    ),
                    "summary": st.column_config.TextColumn(
                        "Resumo",
                        width="medium"
                    ),
                },
                hide_index=True,
                use_container_width=True,
                disabled=["title", "authors", "summary", "published", "pdf_url"]
            )

            st.session_state.selected_rows = {
                i: bool(edited_df.loc[i, 'selected']) for i in edited_df.index
            }

            if any(st.session_state.selected_rows.values()):
                review_button = st.button(
                    "📝 Gerar Revisão",
                    help="Iniciar Revisão de Literatura",
                    key="review_button"
                )
                if review_button:
                    execute_review(
                        theme=theme,
                        provider=provider,
                        model=model,
                        output_lang=output_lang,
                        output_dir=output_dir
                    )

        with tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                display_timeline(df)
            with col2:
                if st.session_state.review_text:
                    st.pyplot(create_wordcloud(st.session_state.review_text))
                else:
                    st.info("⏳ A nuvem de palavras será gerada após a criação da revisão.")

        with tabs[2]:
            if st.session_state.review_text:
                st.markdown(st.session_state.review_text)
                
                st.download_button(
                    "📄 Salvar",
                    st.session_state.review_text,
                    file_name=f"revisao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    help="Salvar revisão em arquivo de texto"
                )
            else:
                st.info("👆 Selecione os artigos e clique em 'Gerar Revisão' para criar sua revisão de literatura.")

if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# import streamlit as st
# import pandas as pd
# from datetime import datetime
# from pathlib import Path
# from typing import Optional, Tuple
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import plotly.express as px
# import os

# # Importa o sistema integrado de revisão (refatorado)
# try:
#     from integrated_reviewer import IntegratedReviewSystem
# except ImportError as ie:
#     st.error("Módulo 'integrated_reviewer' não encontrado. Verifique se o arquivo existe e está no caminho correto.")
#     raise ie

# # Estilo e Configurações Iniciais
# st.set_page_config(
#     page_title="Revisão de Literatura",
#     page_icon="📚",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )
# st.markdown("""
#     <style>
#     .main { padding: 0rem 1rem; }
#     .stButton>button {
#         width: 100%;
#         color: white;
#         background-color: #0096c7;
#         border: none;
#         padding: 10px 24px;
#         border-radius: 6px;
#         font-weight: 500;
#         transition: all 0.3s ease;
#     }
#     .stButton>button:hover {
#         background-color: #0077b6;
#         transform: translateY(-2px);
#         box-shadow: 0 2px 6px rgba(0,0,0,0.1);
#     }
#     .sidebar .sidebar-content { background-color: #f8f9fa; }
#     .block-container { padding-top: 2rem; }
#     h1 { color: #023e8a; font-weight: 600; }
#     h2 { color: #0077b6; font-weight: 500; }
#     h3 { color: #0096c7; font-weight: 500; }
#     .highlight { background-color: #e9ecef; padding: 1.2rem; border-radius: 8px; border-left: 4px solid #0096c7; }
#     .stSelectbox, .stTextArea { border-radius: 6px; }
#     .selected-label { color: green; font-weight: bold; }
#     </style>
#     """, unsafe_allow_html=True)

# # --------------------- Funções de Visualização --------------------- #
# def display_timeline(df: pd.DataFrame):
#     """Exibe histograma de publicações por ano."""
#     if "published" in df.columns:
#         df = df.copy()
#         df['published_date'] = pd.to_datetime(df['published'], errors='coerce')
#         df_clean = df.dropna(subset=['published_date'])
#         if not df_clean.empty:
#             df_clean['year'] = df_clean['published_date'].dt.year
#             fig = px.histogram(
#                 df_clean, 
#                 x='year', 
#                 nbins=10, 
#                 title="Distribuição de Publicações por Ano",
#                 labels={'year': 'Ano de Publicação', 'count': 'Número de Artigos'}
#             )
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.write("Não há dados suficientes para criar a linha do tempo.")
#     else:
#         st.write("Coluna 'published' não encontrada.")

# def create_wordcloud(text: str):
#     """Gera e retorna um objeto Figure com a nuvem de palavras."""
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.imshow(wordcloud, interpolation='bilinear')
#     ax.axis('off')
#     return fig

# # --------------------- Funções de Estado e Execução --------------------- #
# def initialize_session_state():
#     default_states = {
#         'search_results': None,    # DataFrame com resultados da busca
#         'review_text': None,       # Texto da revisão final
#         'saved_files': [],         # Lista de arquivos salvos
#         'error_message': None,     
#         'selected_rows': {}        # Dicionário {row_index: True/False}
#     }
#     for key, value in default_states.items():
#         if key not in st.session_state:
#             st.session_state[key] = value

# def execute_search(
#     theme,
#     max_results,
#     provider,
#     model,
#     output_lang,
#     download_all,
#     output_dir,
#     date_range,
#     sort_by,
#     sort_order,
#     download_count
# ):
#     """
#     Apenas realiza a busca de artigos no arXiv. 
#     Armazena o DataFrame em `st.session_state.search_results`.
#     """
#     try:
#         system = IntegratedReviewSystem()
#         results_df = system.search_papers(
#             query=theme,                # <--- Usamos o "theme" como query
#             max_results=max_results,
#             download_count=download_count,
#             download_pdfs=download_all,
#             save_results=True,
#             output_dir=output_dir,
#             date_range=date_range,
#             sort_by=sort_by,
#             sort_order=sort_order
#         )
#         st.session_state.search_results = results_df
#         st.session_state.review_text = None  # Zera texto de revisão anterior, se houver
#         st.session_state.saved_files = []
#         st.success(f"Foram encontrados {len(results_df)} artigos!")
#     except Exception as e:
#         st.session_state.error_message = str(e)

# def execute_review(
#     theme,
#     provider,
#     model,
#     output_lang,
#     output_dir
# ):
#     """
#     Executa a revisão de literatura apenas para os artigos selecionados.
#     Armazena o texto em `st.session_state.review_text` e arquivos em `st.session_state.saved_files`.
#     """
#     try:
#         system = IntegratedReviewSystem()

#         # Filtra o DataFrame original com base nos itens selecionados
#         df_all = st.session_state.search_results.copy()
#         # Índices selecionados
#         selected_indices = [i for i, val in st.session_state.selected_rows.items() if val]
#         if not selected_indices:
#             st.warning("Nenhum artigo selecionado.")
#             return

#         df_filtered = df_all.iloc[selected_indices]

#         review_text, saved_files = system.review_papers(
#             df=df_filtered,
#             theme=theme,
#             provider=provider,
#             model=model,
#             output_lang=output_lang,
#             save_results=True,
#             output_dir=output_dir
#         )
#         st.session_state.review_text = review_text
#         st.session_state.saved_files = saved_files
#         st.success(f"Revisão de literatura gerada com sucesso com {len(selected_indices)}!")
#     except Exception as e:
#         st.session_state.error_message = str(e)

# def main():
#     initialize_session_state()
#     st.title("📚 Revisão de Literatura")
#     st.markdown("""
#     Um sistema avançado para pesquisar artigos acadêmicos no arXiv e gerar 
#     revisões de literatura abrangentes usando IA. Otimize seu processo de pesquisa
#     com recursos poderosos de busca, análise e síntese.
#     """)

#     # Placeholder para a tabela - possibilita limpar os resultados em nova pesquisa
#     table_placeholder = st.empty()

#     # --------------------- SIDEBAR --------------------- #
#     with st.sidebar:
#         st.header("Configurações")

#         # Grupo "Pesquisa Básica" - agora apenas 1 campo (Tema da Revisão)
#         with st.expander("Pesquisa Básica", expanded=True):
#             theme = st.text_input(
#                 "Tema da Revisão",
#                 placeholder="Digite o tema que deseja pesquisar...",
#                 help="Este tema será usado para buscar artigos no arXiv e gerar a revisão."
#             )

#         with st.expander("Configurações de IA", expanded=True):
#             provider = st.selectbox(
#                 "Provedor de IA",
#                 options=["anthropic", "openai", "gemini", "deepseek"],
#                 format_func=lambda x: {
#                     "anthropic": "Anthropic",
#                     "openai": "OpenAI",
#                     "gemini": "Gemini",
#                     "deepseek": "DeepSeek"
#                 }[x]
#             )
#             # Modelos default para cada provider (pode customizar se desejar)
#             provider_model_map = {
#                 "anthropic": ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"],
#                 "openai": ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini"],
#                 "gemini": ["gemini-2.0-flash", "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash", "gemini-1.5-pro"],
#                 "deepseek": ["deepseek-chat", "deepseek-reasoner"]
#             }
#             model = st.selectbox(
#                 "Modelo",
#                 options=provider_model_map[provider],
#                 index=0
#             )
#             output_lang = st.selectbox(
#                 "Idioma de Saída",
#                 options=["pt-BR", "en-US", "es-ES"],
#                 format_func=lambda x: {
#                     "pt-BR": "Português (Brasil)",
#                     "en-US": "Inglês (EUA)",
#                     "es-ES": "Espanhol (Espanha)"
#                 }[x]
#             )

#         with st.expander("Configurações Avançadas"):
#             max_results = st.slider(
#                 "Número de Artigos",
#                 2, 50, 5,
#                 help="Número máximo de artigos a incluir na busca inicial."
#             )
#             use_date_range = st.checkbox("Filtrar por Data")
#             date_range = None
#             if use_date_range:
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     start_date = st.date_input("Data Inicial", value=None)
#                 with col2:
#                     end_date = st.date_input("Data Final", value=None)
#                 if start_date and end_date:
#                     date_range = (start_date, end_date)

#             sort_by = st.selectbox(
#                 "Ordenar Por",
#                 options=["relevance", "lastUpdatedDate", "submittedDate"],
#                 format_func=lambda x: {
#                     "relevance": "Relevância",
#                     "lastUpdatedDate": "Última Atualização",
#                     "submittedDate": "Data de Submissão"
#                 }[x]
#             )
#             sort_order = st.selectbox(
#                 "Ordem",
#                 options=["descending", "ascending"],
#                 format_func=lambda x: "Decrescente" if x == "descending" else "Crescente"
#             )

#         with st.expander("Configurações de Download"):
#             download_all = st.checkbox(
#                 "Baixar PDFs dos artigos",
#                 help="Baixar PDFs dos artigos encontrados"
#             )
#             output_dir = st.text_input(
#                 "Diretório de Saída",
#                 "reviews",
#                 help="Diretório para salvar os arquivos gerados"
#             )
#             download_count = st.slider(
#                 "Número de Artigos para Download",
#                 min_value=1,
#                 max_value=max_results,
#                 value=2,
#                 help="Define quantos PDFs serão baixados."
#             )

#         # Botão de pesquisa (APENAS busca, sem revisão ainda)
#         search_button = st.button("🔍 Pesquisar", type="primary")

#     # --------------------- EXECUÇÃO DA PESQUISA --------------------- #
#     if search_button:
#         # Limpa a tabela de resultados anteriores a cada nova pesquisa
#         table_placeholder.empty()
#         if not theme.strip():
#             st.error("Por favor, insira o tema da revisão.")
#         else:
#             with st.spinner("🔄 Pesquisando artigos no arXiv..."):
#                 execute_search(
#                     theme=theme,
#                     max_results=max_results,
#                     provider=provider,
#                     model=model,
#                     output_lang=output_lang,
#                     download_all=download_all,
#                     output_dir=output_dir,
#                     date_range=date_range,
#                     sort_by=sort_by,
#                     sort_order=sort_order,
#                     download_count=download_count
#                 )

#     # --------------------- EXIBIÇÃO DOS RESULTADOS --------------------- #
#     if st.session_state.search_results is not None:
#         if len(st.session_state.search_results) > 0:
#             st.header("📊 Resultados da Busca")
#             df = st.session_state.search_results.copy()
            
#             # Garante que a coluna de seleção exista
#             if 'selected' not in df.columns:
#                 df['selected'] = False

#             # Reorganiza as colunas conforme a ordem desejada:
#             # "Marcar" (coluna 'selected'), "Título" (coluna 'title'), 
#             # "link PDF" (coluna 'pdf_url'), "Sumário" (coluna 'summary') e demais colunas.
#             desired_columns = ["selected", "title", "pdf_url", "summary"]
#             other_columns = [col for col in df.columns if col not in desired_columns]
#             ordered_columns = desired_columns + other_columns
#             df = df[ordered_columns]
            
#             # Exibimos a tabela usando o st.data_editor com a configuração das colunas atualizada
#             edited_df = st.data_editor(
#                 df,
#                 column_config={
#                     "selected": st.column_config.CheckboxColumn("Marcar", help="Marque para incluir na revisão"),
#                     "title": st.column_config.TextColumn("Título", width="large"),
#                     "pdf_url": st.column_config.LinkColumn("link PDF"),
#                     "summary": st.column_config.TextColumn("Sumário", width="medium"),
#                     "authors": st.column_config.TextColumn("Autores", width="medium"),
#                     "published": "Data de Publicação"
#                 },
#                 hide_index=True,
#                 use_container_width=True,
#                 num_rows="fixed",
#                 disabled=["title", "authors", "summary", "published", "pdf_url"]
#             )

#             # Atualiza em session_state quais linhas estão selecionadas
#             st.session_state.selected_rows = {
#                 i: bool(edited_df.loc[i, 'selected']) for i in edited_df.index
#             }

#             # Aba de visualizações
#             with st.expander("Visualizações dos Artigos"):
#                 col_viz1, col_viz2 = st.columns(2)
#                 with col_viz1:
#                     display_timeline(df)
#                 with col_viz2:
#                     st.markdown("#### (Espaço para outra visualização futura)")

#             # Botão para iniciar revisão apenas dos selecionados
#             review_button = st.button("📝 Iniciar Revisão", type="primary")
#             if review_button:
#                 with st.spinner("🔄 Gerando revisão de literatura..."):
#                     execute_review(
#                         theme=theme,
#                         provider=provider,
#                         model=model,
#                         output_lang=output_lang,
#                         output_dir=output_dir
#                     )

#         else:
#             st.info("Nenhum artigo encontrado na busca.")

#     # --------------------- EXIBIÇÃO DA REVISÃO --------------------- #
#     if st.session_state.review_text:
#         st.header("📝 Revisão Gerada")
#         review_tabs = st.tabs(["Texto da Revisão", "Nuvem de Palavras"])
#         with review_tabs[0]:
#             st.markdown(st.session_state.review_text)
#         with review_tabs[1]:
#             wordcloud_fig = create_wordcloud(st.session_state.review_text)
#             st.pyplot(wordcloud_fig)

#     # Exibe possíveis erros
#     if st.session_state.error_message:
#         st.error(f"Erro: {st.session_state.error_message}")
#         st.session_state.error_message = None


# if __name__ == "__main__":
#     main()
