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
        st.success(f"🎉 Foram encontrados {len(results_df)} artigos. Selecione quais deseja utilizar na revisão!")
        
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
                help="Digite o tema principal da sua pesquisa",
                placeholder="Ex: COVID-19 and Vaccines"
            )
            
            max_results = st.slider(
                "Número de Artigos",
                2, 50, 10,
                help="Defina a quantidade de artigos para busca"
            )

        with tab2:
            provider = st.selectbox(
                "Provedor de IA",
                options=["anthropic", "gemini", "openai", "deepseek"],
                format_func=lambda x: {
                    "anthropic": "🌟 Anthropic",                    
                    "gemini": "🔵 Gemini",
                    "openai": "🤖 OpenAI",
                    "deepseek": "🎯 DeepSeek"
                }[x]
            )

            provider_model_map = {
                "anthropic": ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"],
                "gemini": ["gemini-2.0-flash", "gemini-1.5-flash","gemini-1.5-pro"],                
                "openai": ["gpt-4o", "gpt-4o-mini", "o1-mini"],
                "deepseek": ["deepseek-chat", "deepseek-reasoner"]
            }
            
            model = st.selectbox(
                "Modelo de IA",
                options=provider_model_map[provider]
            )
            
            output_lang = st.selectbox(
                "Idioma da Revisão",
                options=["pt-BR", "en-US", "es-ES", "fr-FR", "it-IT", "ru-RU", "ar-AE", "zh-HK"],
                format_func=lambda x: {
                    "pt-BR": "🇧🇷 Português (Brasil)",
                    "en-US": "🇺🇸 Inglês (EUA)",
                    "es-ES": "🇪🇸 Espanhol (Espanha)",
                    "fr-FR": "🇫🇷 Francês (França)",
                    "it-IT": "🇮🇹 Italiano (Itália)",
                    "ru-RU": "🇷🇺 Russo (Rússia)",
                    "ar-AE": "🇦🇪 Árabe (Emirados)",
                    "zh-HK": "🇭🇰 Chinês (Hong Kong)"
                }[x]
            )

        with tab3:
            with st.expander("📅 Filtros de Data", expanded=False):
                use_date_range = st.checkbox("Filtrar por Data")
                date_range = None
                if use_date_range:
                    col1, col2 = st.columns(2)
                    with col1:
                        start_date = st.date_input("Data Inicial", value='2010-01-01', format="DD/MM/YYYY")
                    with col2:
                        end_date = st.date_input("Data Final", format="DD/MM/YYYY")
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
    if search_button:
        if not theme.strip():
            st.error("⚠️ Por favor, digite um tema para pesquisar.")
        else:
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

            desired_columns = ["selected", "pdf_url", "title", "summary"]
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
                        width="medium"
                    ),
                    "summary": st.column_config.TextColumn(
                        "Resumo",
                        width="large"
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
            display_timeline(df)
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
