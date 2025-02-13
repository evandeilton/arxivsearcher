# styles.py
import streamlit as st
from typing import List, Dict, Optional

def apply_custom_styles():
    """
    Aplica estilos customizados modernos para melhorar a UI/UX.
    Inclui variáveis CSS, temas escuro/claro e componentes acessíveis.
    """
    st.markdown("""
        <style>
        /* Sistema de Design - Variáveis CSS */
        :root {
            /* Paleta de Cores Principal */
            --primary-50: #e3f2fd;
            --primary-100: #bbdefb;
            --primary-200: #90caf9;
            --primary-300: #64b5f6;
            --primary-400: #42a5f5;
            --primary-500: #2196f3;
            --primary-600: #1e88e5;
            --primary-700: #1976d2;
            --primary-800: #1565c0;
            --primary-900: #0d47a1;
            
            /* Cores Semânticas */
            --success: #4caf50;
            --warning: #ff9800;
            --error: #f44336;
            --info: #2196f3;
            
            /* Cores de Superfície */
            --surface-light: #ffffff;
            --surface-dark: #121212;
            --overlay: rgba(0, 0, 0, 0.5);
            
            /* Tipografia */
            --font-primary: 'Inter', -apple-system, system-ui, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
            
            /* Espaçamento */
            --spacing-xs: 4px;
            --spacing-sm: 8px;
            --spacing-md: 16px;
            --spacing-lg: 24px;
            --spacing-xl: 32px;
            
            /* Sombras */
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.12);
            --shadow-md: 0 4px 6px rgba(0,0,0,0.1);
            --shadow-lg: 0 10px 15px rgba(0,0,0,0.1);
            
            /* Border Radius */
            --radius-sm: 4px;
            --radius-md: 8px;
            --radius-lg: 16px;
            --radius-full: 9999px;
        }

        /* Reset e Estilos Base */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* Tipografia Aprimorada */
        body {
            font-family: var(--font-primary);
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }

        h1, h2, h3, h4, h5, h6 {
            font-weight: 600;
            color: var(--primary-900);
            margin-bottom: var(--spacing-md);
        }

        /* Componentes Estilizados */
        .stButton button {
            background: linear-gradient(45deg, var(--primary-600), var(--primary-500));
            color: white;
            border: none;
            padding: var(--spacing-sm) var(--spacing-md);
            border-radius: var(--radius-md);
            font-weight: 500;
            transition: transform 0.2s, box-shadow 0.2s;
            min-height: 44px;
        }

        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }

        /* Cards Modernos */
        .modern-card {
            background: var(--surface-light);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            box-shadow: var(--shadow-md);
            margin: var(--spacing-md) 0;
            border: 1px solid rgba(0,0,0,0.1);
        }

        /* Animações Suaves */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .animate-fade-in {
            animation: fadeIn 0.3s ease-out;
        }

        /* Tabelas Responsivas */
        .dataframe {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            border-radius: var(--radius-md);
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }

        .dataframe th {
            background-color: var(--primary-50);
            padding: var(--spacing-sm) var(--spacing-md);
            text-align: left;
            font-weight: 600;
        }

        .dataframe td {
            padding: var(--spacing-sm) var(--spacing-md);
            border-top: 1px solid rgba(0,0,0,0.1);
        }

        /* Inputs Aprimorados */
        .stTextInput input, .stSelectbox select {
            border-radius: var(--radius-md);
            border: 1px solid rgba(0,0,0,0.2);
            padding: var(--spacing-sm) var(--spacing-md);
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        .stTextInput input:focus, .stSelectbox select:focus {
            border-color: var(--primary-500);
            box-shadow: 0 0 0 2px var(--primary-100);
            outline: none;
        }
        </style>
    """, unsafe_allow_html=True)

def create_notification(message: str, type: str = "info", duration: int = 3000):
    """
    Cria uma notificação toast moderna e animada.
    
    Args:
        message (str): Mensagem a ser exibida
        type (str): Tipo da notificação (info, success, error, warning)
        duration (int): Duração em milissegundos
    """
    colors = {
        "info": "var(--info)",
        "success": "var(--success)",
        "error": "var(--error)",
        "warning": "var(--warning)"
    }
    
    st.markdown(f"""
        <div class="notification {type} animate-fade-in"
             style="position: fixed; top: 20px; right: 20px; 
                    background: white; padding: 16px; 
                    border-radius: var(--radius-md);
                    border-left: 4px solid {colors[type]};
                    box-shadow: var(--shadow-md);
                    z-index: 9999;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span>{message}</span>
            </div>
        </div>
        <script>
            setTimeout(() => {{
                document.querySelector('.notification').style.opacity = '0';
                setTimeout(() => {{
                    document.querySelector('.notification').remove();
                }}, 300);
            }}, {duration});
        </script>
    """, unsafe_allow_html=True)

def create_card(title: str, content: str, footer: Optional[str] = None):
    """
    Cria um card moderno e responsivo.
    
    Args:
        title (str): Título do card
        content (str): Conteúdo principal
        footer (str, optional): Rodapé do card
    """
    footer_html = f'<div style="border-top: 1px solid rgba(0,0,0,0.1); padding-top: var(--spacing-md);">{footer}</div>' if footer else ''
    
    st.markdown(f"""
        <div class="modern-card animate-fade-in">
            <h3 style="margin-bottom: var(--spacing-md);">{title}</h3>
            <div style="margin-bottom: var(--spacing-md);">{content}</div>
            {footer_html}
        </div>
    """, unsafe_allow_html=True)

def create_data_stats(data: Dict[str, any]):
    """
    Cria um painel de estatísticas com cards modernos.
    
    Args:
        data (Dict[str, any]): Dicionário com dados estatísticos
    """
    cols = st.columns(len(data))
    for i, (label, value) in enumerate(data.items()):
        with cols[i]:
            st.markdown(f"""
                <div class="modern-card" style="text-align: center;">
                    <h4 style="color: var(--primary-500);">{value}</h4>
                    <p style="color: var(--primary-900);">{label}</p>
                </div>
            """, unsafe_allow_html=True)

def create_progress_indicator(current: int, total: int, label: str = "Progresso"):
    """
    Cria um indicador de progresso moderno.
    
    Args:
        current (int): Valor atual
        total (int): Valor total
        label (str): Rótulo do progresso
    """
    progress = min(current / total, 1.0)
    st.markdown(f"""
        <div style="margin: var(--spacing-md) 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: var(--spacing-xs);">
                <span>{label}</span>
                <span>{int(progress * 100)}%</span>
            </div>
            <div style="background: var(--primary-100); border-radius: var(--radius-full); height: 8px;">
                <div style="width: {progress * 100}%; height: 100%; 
                            background: linear-gradient(90deg, var(--primary-500), var(--primary-600));
                            border-radius: var(--radius-full); transition: width 0.3s ease-in-out;">
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def create_glitter_animation():
    """
    Cria uma animação sutil de glitter quando a revisão é concluída.
    """
    st.markdown("""
        <style>
        @keyframes glitter {
            0% { transform: scale(0) rotate(0deg); opacity: 0; }
            50% { opacity: 1; }
            100% { transform: scale(1) rotate(180deg); opacity: 0; }
        }

        .glitter-wrapper {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 9999;
        }

        .glitter-particle {
            position: absolute;
            width: 10px;
            height: 10px;
            background: radial-gradient(circle, var(--primary-300), var(--primary-500));
            border-radius: 50%;
            animation: glitter 1.5s ease-out forwards;
        }
        </style>

        <div class="glitter-wrapper" id="glitter-container">
            <script>
            function createGlitter() {
                const container = document.getElementById('glitter-container');
                const particleCount = 30;
                
                for (let i = 0; i < particleCount; i++) {
                    setTimeout(() => {
                        const particle = document.createElement('div');
                        particle.className = 'glitter-particle';
                        
                        // Posição aleatória
                        particle.style.left = Math.random() * 100 + '%';
                        particle.style.top = Math.random() * 100 + '%';
                        
                        // Tamanho aleatório
                        const size = Math.random() * 8 + 4;
                        particle.style.width = size + 'px';
                        particle.style.height = size + 'px';
                        
                        container.appendChild(particle);
                        
                        // Remove a partícula após a animação
                        setTimeout(() => {
                            particle.remove();
                        }, 1500);
                    }, i * 50);
                }
                
                // Remove o container após todas as animações
                setTimeout(() => {
                    container.remove();
                }, 3000);
            }
            
            // Inicia a animação
            createGlitter();
            </script>
        </div>
    """, unsafe_allow_html=True)

def create_success_message(message: str):
    """
    Cria uma mensagem de sucesso com animação sutil.
    
    Args:
        message (str): Mensagem a ser exibida
    """
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, var(--success) 0%, var(--primary-500) 100%);
            color: white;
            padding: 1rem;
            border-radius: var(--radius-lg);
            text-align: center;
            animation: slideDown 0.5s ease-out;
            margin: 1rem 0;
            box-shadow: var(--shadow-md);
        ">
            <h3 style="margin: 0; font-weight: 500;">
                ✨ {message}
            </h3>
        </div>

        <style>
        @keyframes slideDown {{
            from {{ transform: translateY(-20px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}
        </style>
    """, unsafe_allow_html=True)
