# reviewer_ui.py
#!/usr/bin/env python3
import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Callable
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from styles import (
    apply_custom_styles, 
    create_notification, 
    create_card, 
    create_data_stats,
    create_progress_indicator
)
# from mdtopdf import MarkdownPDFConverter
# from fpdf.enums import XPos, YPos
# import markdown
# from fpdf import HTMLMixin, FPDF  # adicionado HTMLMixin para renderizar HTML

try:
    from integrated_reviewer import IntegratedReviewSystem
except ImportError as ie:
    st.error("Módulo 'integrated_reviewer' não encontrado. Verifique se o arquivo existe e está no caminho correto.")
    raise ie

# Adiciona uma classe auxiliar para renderização HTML no PDF
# class MyFPDF(FPDF, HTMLMixin):
#     pass

class ReviewerUI:
    """
    Classe principal para gerenciar a interface do usuário do sistema de revisão.
    Implementa padrão Singleton para garantir única instância.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ReviewerUI, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.initialize_session_state()
        self.config = {}
        self.system = IntegratedReviewSystem()  # Inicializa o sistema uma única vez
        apply_custom_styles()

    def initialize_session_state(self) -> None:
        """Inicializa o estado da sessão com valores padrão."""
        default_states = {
            'search_results': None,
            'review_text': None,
            'saved_files': [],
            'error_message': None,
            'selected_rows': {},
            'search_history': [],
            'last_update': None,
            'theme_color': 'light',
            'language': 'pt-BR',
            'show_success_animation': False,
            'processing': False  # Novo estado para controlar processamento
        }
        
        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def execute_review(self) -> None:
        """Executa a geração da revisão de literatura com tratamento de erros aprimorado."""
        if st.session_state.processing:
            create_notification(
                "Já existe um processamento em andamento.",
                type="warning"
            )
            return

        try:
            df_all = st.session_state.search_results.copy()
            selected_indices = [i for i, val in st.session_state.selected_rows.items() if val]
            
            if not selected_indices:
                create_notification(
                    "Selecione pelo menos um artigo para gerar a revisão.",
                    type="warning"
                )
                return

            df_filtered = df_all.iloc[selected_indices]
            total_articles = len(df_filtered)
            
            st.session_state.processing = True
            progress_placeholder = st.empty()
            
            def update_progress(current: int, total: int, message: str = "") -> None:
                if not st.session_state.processing:
                    return
                progress = int((current / total) * 100)
                with progress_placeholder.container():
                    create_progress_indicator(current, total, f"Processando artigos: {message}")
                    if progress < 100:
                        create_notification(
                            f"Processando artigo {current} de {total}: {message}",
                            type="info",
                            duration=30
                        )

            # Configura o callback no sistema
            self.system.set_progress_callback(update_progress)

            with st.spinner("🔄 Gerando revisão de literatura..."):
                update_progress(0, total_articles, "Iniciando processamento")
                
                review_text, saved_files = self.system.review_papers(
                    df=df_filtered,
                    theme=self.config['theme'],
                    provider=self.config['provider'],
                    model=self.config['model'],
                    output_lang=self.config['output_lang'],
                    save_results=True,
                    output_dir=self.config['output_dir']
                )
                
                st.session_state.review_text = review_text
                st.session_state.saved_files = saved_files
                st.session_state.show_success_animation = True
                
                update_progress(total_articles, total_articles, "Processamento concluído")
                st.balloons()
                
                # Rola para a aba de revisão
                js = """
                    <script>
                        setTimeout(() => {
                            const reviewTab = document.querySelector('button[role="tab"]:nth-child(3)');
                            if (reviewTab) {
                                reviewTab.click();
                                reviewTab.scrollIntoView({ behavior: 'smooth' });
                            }
                        }, 1000);
                    </script>
                """
                st.markdown(js, unsafe_allow_html=True)
                
        except Exception as e:
            create_notification(
                f"❌ Erro ao gerar revisão: {str(e)}",
                type="error",
                duration=600
            )
            progress_placeholder.empty()
        finally:
            st.session_state.processing = False

    def display_results(self) -> None:
        """Exibe os resultados da pesquisa em uma interface moderna e interativa."""
        if st.session_state.search_results is None:
            return

        if st.session_state.show_success_animation:
            st.balloons()
            create_notification("✨ Revisão concluída com sucesso!", type="success", duration=10)
            st.session_state.show_success_animation = False

        tabs = st.tabs(["📊 Resultados", "📈 Análises", "📝 Revisão"])
        
        with tabs[0]:
            self.display_results_tab()
        
        with tabs[1]:
            self.display_analysis_tab()
        
        with tabs[2]:
            self.display_review_tab()

    def create_header(self) -> None:
        """Cria o cabeçalho da aplicação com animação e estilo moderno."""
        st.markdown("""
            <div style="text-align: center; padding: 0 0 2rem 0; animation: fadeIn 0.5s ease-out;">
                <h1 style="color: var(--primary-700); font-size: 2.5rem; margin-bottom: 1rem;">
                    📚 Revisão de Literatura Inteligente
                </h1>
                <p style="color: var(--primary-500); font-size: 1.2rem; font-weight: 500;">
                    Automatize sua pesquisa acadêmica com IA
                </p>
            </div>
        """, unsafe_allow_html=True)

    def create_sidebar_config(self) -> Dict:
        """
        Cria a configuração da barra lateral com interface moderna e validações.
        
        Returns:
            Dict: Configurações validadas selecionadas pelo usuário
        """
        with st.sidebar:
            st.markdown("""
                <h3 style="text-align: center; color: var(--primary-700); margin-bottom: 1rem;">
                    🔧 Configurações
                </h3>
            """, unsafe_allow_html=True)

            tabs = st.tabs(["📝 Básico", "🤖 IA", "⚙️ Avançado"])
            
            with tabs[0]:
                self.config['theme'] = st.text_input(
                    "Tema da Pesquisa",
                    help="Digite o tema principal da sua pesquisa",
                    placeholder="Ex: COVID-19 and Vaccines"
                ).strip()
                
                self.config['max_results'] = st.slider(
                    "Número de Artigos",
                    2, 50, 10,
                    help="Defina a quantidade de artigos para busca"
                )

            with tabs[1]:
                providers = {
                    "openrouter": "🔍 OpenRouter",
                    "anthropic": "🌟 Anthropic",
                    "gemini": "🔵 Gemini",
                    "openai": "🤖 OpenAI",
                    "deepseek": "🎯 DeepSeek"
                }
                
                self.config['provider'] = st.selectbox(
                    "Provedor de IA",
                    options=list(providers.keys()),
                    format_func=lambda x: providers[x]
                )

                provider_model_map = {
                    "openrouter": ["google/gemini-2.0-pro-exp-02-05:free","cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
                                    "cognitivecomputations/dolphin3.0-mistral-24b:free","openai/o3-mini-high","openai/o3-mini",
                                    "openai/chatgpt-4o-latest","openai/gpt-4o-mini","google/gemini-2.0-flash-001",
                                    "google/gemini-2.0-flash-thinking-exp:free","google/gemini-2.0-flash-lite-preview-02-05:free",
                                    "google/gemini-2.0-pro-exp-02-05:free", "deepseek/deepseek-r1-distill-llama-70b:free",
                                    "deepseek/deepseek-r1-distill-qwen-32b","deepseek/deepseek-r1:free","qwen/qwen-plus",
                                    "qwen/qwen-max","qwen/qwen-turbo","mistralai/codestral-2501","mistralai/mistral-small-24b-instruct-2501:free",
                                    "anthropic/claude-3.5-haiku-20241022:beta","anthropic/claude-3.5-sonnet","perplexity/sonar-reasoning",
                                    "perplexity/sonar","perplexity/llama-3.1-sonar-large-128k-online"],
                    "anthropic": ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"],
                    "gemini": ["gemini-2.0-flash-lite-preview-02-05","gemini-2.0-flash", "gemini-1.5-flash","gemini-1.5-pro"],
                    "openai": ["gpt-4o", "gpt-4o-mini", "o1-mini"],
                    "deepseek": ["deepseek-chat", "deepseek-reasoner"]
                }
                
                self.config['model'] = st.selectbox(
                    "Modelo de IA",
                    options=provider_model_map[self.config['provider']]
                )
                
                languages = {
                    "pt-BR": "🇧🇷 Português (Brasil)",
                    "en-US": "🇺🇸 Inglês (EUA)",
                    "es-ES": "🇪🇸 Espanhol (Espanha)",
                    "fr-FR": "🇫🇷 Francês (França)",
                    "it-IT": "🇮🇹 Italiano (Itália)",
                    "ru-RU": "🇷🇺 Russo (Rússia)",
                    "ar-AE": "🇦🇪 Árabe (Emirados)",
                    "zh-HK": "🇭🇰 Chinês (Hong Kong)"
                }
                
                self.config['output_lang'] = st.selectbox(
                    "Idioma da Revisão",
                    options=list(languages.keys()),
                    format_func=lambda x: languages[x]
                )

            with tabs[2]:
                with st.expander("📅 Filtros de Data", expanded=False):
                    self.config['use_date_range'] = st.checkbox("Filtrar por Data")
                    self.config['date_range'] = None
                    
                    if self.config['use_date_range']:
                        col1, col2 = st.columns(2)
                        with col1:
                            start_date = st.date_input(
                                "Data Inicial",
                                value=datetime(2010, 1, 1).date(),
                                format="DD/MM/YYYY"
                            )
                        with col2:
                            end_date = st.date_input(
                                "Data Final",
                                format="DD/MM/YYYY"
                            )
                        if start_date and end_date:
                            if start_date > end_date:
                                st.error("Data inicial deve ser anterior à data final")
                            else:
                                self.config['date_range'] = (start_date, end_date)

                sort_options = {
                    "relevance": "Relevância",
                    "lastUpdatedDate": "Última Atualização",
                    "submittedDate": "Data de Submissão"
                }
                
                self.config['sort_by'] = st.selectbox(
                    "Ordenar Por",
                    options=list(sort_options.keys()),
                    format_func=lambda x: sort_options[x]
                )
                
                self.config['sort_order'] = st.selectbox(
                    "Ordem",
                    options=["descending", "ascending"],
                    format_func=lambda x: "Decrescente" if x == "descending" else "Crescente"
                )

                with st.expander("📥 Download", expanded=False):
                    self.config['download_all'] = st.checkbox("Baixar PDFs")
                    self.config['output_dir'] = st.text_input(
                        "Diretório de Saída",
                        value="reviews",
                        help="Pasta onde os arquivos serão salvos"
                    ).strip()
                    
                    if self.config['download_all']:
                        self.config['download_count'] = st.slider("Quantidade",
                            1, self.config['max_results'], 2,
                            help="Quantidade de PDFs para download"
                        )
                    else:
                        self.config['download_count'] = 0

            # Área de Ações
            st.markdown("<hr>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🔍 Pesquisar", 
                            use_container_width=True,
                            help="Iniciar pesquisa de artigos",
                            disabled=st.session_state.processing):
                    if not self.config['theme']:
                        create_notification(
                            "Por favor, digite um tema para pesquisar.",
                            type="error",
                            duration=5000
                        )
                    else:
                        self.execute_search()
            
            with col2:
                if st.button("🔄 Limpar",
                            use_container_width=True,
                            help="Limpar resultados",
                            disabled=st.session_state.processing):
                    self.clear_results()
                    create_notification(
                        "Resultados limpos com sucesso!",
                        type="success"
                    )

            return self.config

    def execute_search(self) -> None:
        """Executa a busca de artigos com tratamento de erros aprimorado."""
        if st.session_state.processing:
            return

        try:
            st.session_state.processing = True
            with st.spinner("🔄 Realizando pesquisa..."):
                results_df = self.system.search_papers(
                    query=self.config['theme'],
                    max_results=self.config['max_results'],
                    download_count=self.config.get('download_count', 0),
                    download_pdfs=self.config.get('download_all', False),
                    save_results=True,
                    output_dir=self.config.get('output_dir', 'reviews'),
                    date_range=self.config.get('date_range'),
                    sort_by=self.config.get('sort_by'),
                    sort_order=self.config.get('sort_order')
                )
                
                st.session_state.search_results = results_df
                st.session_state.review_text = None
                st.session_state.saved_files = []
                st.session_state.selected_rows = {}
                
                create_notification(
                    f"🎉 Encontrados {len(results_df)} artigos relevantes!",
                    type="success"
                )
                
        except Exception as e:
            create_notification(
                f"❌ Erro na busca: {str(e)}",
                type="error",
                duration=5000
            )
        finally:
            st.session_state.processing = False

    def display_results_tab(self) -> None:
        """Exibe a aba de resultados com tabela interativa e opções de seleção."""
        st.markdown("""
            <h3 style="color: var(--primary-700); margin-bottom: 1rem;">
                📚 Artigos Encontrados
            </h3>
        """, unsafe_allow_html=True)

        df = st.session_state.search_results.copy()
        if 'selected' not in df.columns:
            df['selected'] = False

        # Configuração das colunas
        column_config = {
            "selected": st.column_config.CheckboxColumn(
                "✓",
                help="Selecionar para revisão",
                default=False
            ),
            "title": st.column_config.TextColumn(
                "Título",
                width="large"
            ),
            "pdf_url": st.column_config.LinkColumn(
                "PDF",
                width="medium",
                validate="url"
            ),
            "summary": st.column_config.TextColumn(
                "Resumo",
                width="large"
            )
        }

        # Reordenação das colunas
        desired_columns = ["selected", "pdf_url", "title", "summary"]
        other_columns = [col for col in df.columns if col not in desired_columns]
        df = df[desired_columns + other_columns]

        # Editor de dados aprimorado
        edited_df = st.data_editor(
            df,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
            disabled=["title", "authors", "summary", "published", "pdf_url"],
            column_order=desired_columns + other_columns,
            key="results_editor"
        )

        # Atualização das seleções
        st.session_state.selected_rows = {
            i: bool(edited_df.loc[i, 'selected']) for i in edited_df.index
        }

        # Botão de geração de revisão
        selected_count = sum(st.session_state.selected_rows.values())
        if selected_count > 0:
            st.markdown(f"""
                <div style="margin: 1rem 0;">
                    <p style="color: var(--primary-700);">
                        ✨ {selected_count} artigos selecionados para revisão
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("📝 Gerar Revisão",
                        use_container_width=True,
                        help="Iniciar geração da revisão de literatura",
                        disabled=st.session_state.processing):
                self.execute_review()

    def display_analysis_tab(self) -> None:
        """Exibe a aba de análises com visualizações interativas."""
        df = st.session_state.search_results

        if df is not None and not df.empty:
            # Estatísticas gerais
            stats = {
                "Total de Artigos": len(df),
                "Período": f"{df['published'].min()[:4]} - {df['published'].max()[:4]}",
                "Artigos Selecionados": sum(st.session_state.selected_rows.values())
            }
            create_data_stats(stats)

            # Timeline
            self.display_timeline(df)

            # Word Cloud
            if st.session_state.review_text:
                st.pyplot(self.create_wordcloud(st.session_state.review_text))
            else:
                create_card(
                    "Nuvem de Palavras",
                    "A nuvem de palavras será gerada após a criação da revisão.",
                    "⏳ Aguardando geração da revisão..."
                )

    def display_timeline(self, df: pd.DataFrame) -> None:
        """
        Cria um gráfico de linha do tempo interativo.
        
        Args:
            df (pd.DataFrame): DataFrame com os dados dos artigos
        """
        if "published" in df.columns:
            df = df.copy()
            df['published_date'] = pd.to_datetime(df['published'], errors='coerce')
            df_clean = df.dropna(subset=['published_date'])
            
            if not df_clean.empty:
                df_clean['year'] = df_clean['published_date'].dt.year
                year_counts = df_clean['year'].value_counts().sort_index()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=year_counts.index,
                    y=year_counts.values,
                    mode='lines+markers',
                    name='Publicações',
                    line=dict(color='var(--primary-500)', width=3),
                    marker=dict(
                        size=8,
                        color='var(--primary-700)',
                        symbol='circle'
                    )
                ))
                
                fig.update_layout(
                    title={
                        'text': 'Distribuição de Publicações por Ano',
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    xaxis_title='Ano de Publicação',
                    yaxis_title='Número de Artigos',
                    template='plotly_white',
                    hovermode='x unified',
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                create_notification(
                    "Não há dados suficientes para criar a linha do tempo.",
                    type="warning"
                )

    def create_wordcloud(self, text: str) -> plt.Figure:
        """
        Gera uma nuvem de palavras estilizada.
        
        Args:
            text (str): Texto para gerar a nuvem de palavras
            
        Returns:
            plt.Figure: Figura matplotlib com a nuvem de palavras
        """
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

    def display_review_tab(self) -> None:
        """Exibe a aba de revisão com o texto gerado e opções de download."""
        if st.session_state.review_text:
            st.markdown(st.session_state.review_text)
            st.download_button(
                "📥 Baixar Revisão em TXT",
                st.session_state.review_text,
                file_name=f"revisao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                help="Baixar a revisão em formato TXT",
                key="download_txt_button"
            )
        else:
            st.info("Nenhuma revisão gerada no momento.")

    def clear_results(self) -> None:
        """Limpa todos os resultados e reinicia o estado da sessão."""
        st.session_state.search_results = None
        st.session_state.review_text = None
        st.session_state.selected_rows = {}
        st.session_state.saved_files = []
        st.session_state.error_message = None
        st.session_state.processing = False
        st.session_state.show_success_animation = False

def main():
    """Função principal que inicializa e executa a interface do usuário."""
    try:
        # Inicialização da UI
        ui = ReviewerUI()
        ui.create_header()
        
        # Configuração via sidebar
        ui.create_sidebar_config()
        
        # Área principal de resultados
        if st.session_state.search_results is not None:
            ui.display_results()
            
        # Tratamento de erros globais
        if st.session_state.error_message:
            create_notification(
                st.session_state.error_message,
                type="error"
            )
            st.session_state.error_message = None
            
    except Exception as e:
        create_notification(
            f"❌ Erro inesperado: {str(e)}",
            type="error"
        )
        st.error("Ocorreu um erro inesperado. Por favor, tente novamente ou contate o suporte.")

if __name__ == "__main__":
    main()
