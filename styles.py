import streamlit as st

def apply_custom_styles():
    """Aplica estilos customizados à interface."""
    st.set_page_config(
        page_title="Sistema de Revisão de Literatura",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    /* Remove quebras de página transparentes */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Remove margens extras */
    .main > div {
        padding-top: 0rem;
    }

    /* Ajusta o header */
    header {
        background-color: transparent !important;
    }

    /* Melhoria nos botões - apenas ícones */
    .stButton>button {
        background: transparent;
        border: none;
        padding: 0.5rem;
        color: #0096c7;
        width: auto;
        min-width: 40px;
        height: 40px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background: rgba(0, 150, 199, 0.1);
        transform: translateY(0);
        box-shadow: none;
        border-radius: 50%;
    }
    
    /* Ajustes da sidebar */
    .sidebar .sidebar-content {
        padding: 1.5rem;
        background: white;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 16px;
        color: #666;
        border-radius: 0;
        border-bottom: 2px solid transparent;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #0096c7;
        border-bottom: 2px solid #0096c7;
        background: transparent;
    }

    /* Remove bordas extras */
    .stMarkdown div {
        padding: 0;
        margin: 0;
    }

    /* Título principal */
    h1 {
        padding: 1rem 0;
        margin: 0;
        color: #2c3e50;
        font-size: 2rem;
        font-weight: 600;
    }

    /* Subtítulo */
    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-top: -0.5rem;
        margin-bottom: 2rem;
    }

    /* Input de pesquisa */
    .stTextInput input {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stTextInput input:focus {
        border-color: #0096c7;
        box-shadow: 0 0 0 2px rgba(0,150,199,0.1);
    }

    /* Slider */
    .stSlider {
        padding: 1rem 0;
    }
    .stSlider .stSlider > div > div {
        height: 3px;
    }
    .stSlider .stSlider > div > div > div {
        background-color: #0096c7;
    }

    /* Esconde elementos desnecessários */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tooltips para botões */
    [data-tooltip]:before {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        padding: 4px 8px;
        background: #333;
        color: white;
        font-size: 12px;
        border-radius: 4px;
        white-space: nowrap;
        visibility: hidden;
        opacity: 0;
        transition: opacity 0.2s;
    }
    [data-tooltip]:hover:before {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)