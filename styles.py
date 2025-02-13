# styles.py
import streamlit as st

def apply_custom_styles():
    """
    Aplica estilos customizados para melhorar a UI/UX e acessibilidade.
    """
    st.markdown("""
        <style>
        /* Estilo Base */
        :root {
            --primary-color: #0066cc;
            --secondary-color: #2196F3;
            --background-color: #ffffff;
            --text-color: #333333;
            --error-color: #dc3545;
            --success-color: #28a745;
            --warning-color: #ffc107;
        }

        /* Melhorias de Contraste e Visibilidade */
        .stButton button {
            color: var(--background-color);
            background-color: var(--primary-color);
            font-weight: 600;
            min-height: 44px;
            border-radius: 8px;
            transition: all 0.3s ease;
        }

        .stButton button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }

        /* Foco e Acessibilidade */
        *:focus {
            outline: 3px solid var(--secondary-color);
            outline-offset: 2px;
        }

        /* Tamanhos Mínimos para Elementos Interativos */
        .stSelectbox, .stTextInput, .stNumberInput {
            min-height: 44px;
        }

        /* Responsividade */
        .table-wrapper {
            overflow-x: auto;
            margin: 1em 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Breadcrumbs */
        .breadcrumb {
            padding: 8px 16px;
            margin-bottom: 20px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }

        .breadcrumb-item {
            display: inline;
            color: var(--text-color);
        }

        /* Toast Notifications */
        .toast {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 16px;
            border-radius: 8px;
            background-color: var(--background-color);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            z-index: 1000;
            animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }

        /* Loading Indicators */
        .loading-spinner {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        /* Seleção de Artigos */
        .selected-row {
            background-color: rgba(33, 150, 243, 0.1);
        }

        /* Back to Top Button */
        .back-to-top {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 12px;
            border-radius: 50%;
            background-color: var(--primary-color);
            color: var(--background-color);
            cursor: pointer;
            transition: all 0.3s ease;
            z-index: 999;
        }

        /* Progress Bar */
        .progress-bar {
            width: 100%;
            height: 4px;
            background-color: #e9ecef;
            border-radius: 2px;
            margin: 10px 0;
        }

        .progress-bar-fill {
            height: 100%;
            background-color: var(--primary-color);
            border-radius: 2px;
            transition: width 0.3s ease;
        }
        </style>
    """, unsafe_allow_html=True)

def create_toast(message, type="info"):
    """
    Cria uma notificação toast.
    
    Args:
        message (str): Mensagem a ser exibida
        type (str): Tipo da notificação (info, success, error, warning)
    """
    colors = {
        "info": "#2196F3",
        "success": "#28a745",
        "error": "#dc3545",
        "warning": "#ffc107"
    }
    
    st.markdown(f"""
        <div class="toast" style="border-left: 4px solid {colors[type]}">
            {message}
        </div>
    """, unsafe_allow_html=True)

def create_breadcrumbs(items):
    """
    Cria uma navegação breadcrumb.
    
    Args:
        items (list): Lista de itens do breadcrumb
    """
    breadcrumb_html = '<div class="breadcrumb">'
    for i, item in enumerate(items):
        if i > 0:
            breadcrumb_html += ' / '
        breadcrumb_html += f'<span class="breadcrumb-item">{item}</span>'
    breadcrumb_html += '</div>'
    
    st.markdown(breadcrumb_html, unsafe_allow_html=True)

def create_progress_bar(progress):
    """
    Cria uma barra de progresso.
    
    Args:
        progress (float): Valor entre 0 e 1 representando o progresso
    """
    st.markdown(f"""
        <div class="progress-bar">
            <div class="progress-bar-fill" style="width: {progress * 100}%"></div>
        </div>
    """, unsafe_allow_html=True)