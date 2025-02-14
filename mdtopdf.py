from fpdf import FPDF
import markdown
from datetime import datetime
from typing import Optional, Dict, Union
import logging
from pathlib import Path
import css_inline
from bs4 import BeautifulSoup
import os
import urllib.request
import zipfile
import io

class MarkdownPDFConverter:
    """
    Classe para converter textos em Markdown para PDFs elegantes e bem formatados.
    Utiliza fonte Roboto para suporte completo a caracteres Unicode.
    """
    
    DEFAULT_STYLES = {
        'font_family': 'Roboto',
        'title_size': 16,
        'subtitle_size': 12,
        'body_size': 11,
        'line_height': 1.5,
        'margin': 15,
        'colors': {
            'title': '#1a1a1a',
            'subtitle': '#4a4a4a',
            'body': '#333333'
        }
    }
    
    ROBOTO_FONTS = {
        'regular': 'Roboto-Regular.ttf',
        'bold': 'Roboto-Bold.ttf',
        'italic': 'Roboto-Italic.ttf',
        'bolditalic': 'Roboto-BoldItalic.ttf'
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o conversor com configurações opcionais.
        
        Args:
            config (Dict, optional): Dicionário com configurações personalizadas
        """
        self.config = {**self.DEFAULT_STYLES, **(config or {})}
        self.logger = logging.getLogger(__name__)
        self.fonts_dir = self._setup_fonts()
        
    def _setup_fonts(self) -> str:
        """
        Configura as fontes Roboto, baixando-as se necessário.
        
        Returns:
            str: Caminho para o diretório de fontes
        """
        # Cria diretório de fontes se não existir
        fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
        os.makedirs(fonts_dir, exist_ok=True)
        
        # URL do arquivo zip da fonte Roboto
        roboto_url = "https://github.com/googlefonts/roboto/releases/download/v2.138/roboto-unhinted.zip"
        
        # Verifica se todas as fontes necessárias já existem
        fonts_exist = all(
            os.path.exists(os.path.join(fonts_dir, font))
            for font in self.ROBOTO_FONTS.values()
        )
        
        if not fonts_exist:
            try:
                self.logger.info("Baixando fontes Roboto...")
                
                # Baixa e extrai as fontes
                response = urllib.request.urlopen(roboto_url)
                zip_data = io.BytesIO(response.read())
                
                with zipfile.ZipFile(zip_data) as zip_ref:
                    # Lista todos os arquivos no zip
                    font_files = [f for f in zip_ref.namelist() if f.endswith('.ttf')]
                    
                    # Extrai apenas as fontes necessárias
                    for font_file in font_files:
                        if any(rob_font in font_file for rob_font in self.ROBOTO_FONTS.values()):
                            zip_ref.extract(font_file, fonts_dir)
                            
                            # Move os arquivos para o diretório raiz de fontes
                            font_name = os.path.basename(font_file)
                            old_path = os.path.join(fonts_dir, font_file)
                            new_path = os.path.join(fonts_dir, font_name)
                            
                            if old_path != new_path:
                                os.rename(old_path, new_path)
                
                self.logger.info("Fontes Roboto instaladas com sucesso!")
                
            except Exception as e:
                self.logger.error(f"Erro ao baixar fontes: {str(e)}")
                raise
                
        return fonts_dir
    
    def _setup_pdf(self) -> FPDF:
        """
        Configura as propriedades básicas do PDF com suporte Unicode.
        
        Returns:
            FPDF: Instância configurada do FPDF
        """
        pdf = FPDF()
        pdf.add_page()
        
        # Adiciona as fontes Roboto
        for style, font_file in self.ROBOTO_FONTS.items():
            font_path = os.path.join(self.fonts_dir, font_file)
            if style == 'regular':
                pdf.add_font('Roboto', '', font_path, uni=True)
            elif style == 'bold':
                pdf.add_font('Roboto', 'B', font_path, uni=True)
            elif style == 'italic':
                pdf.add_font('Roboto', 'I', font_path, uni=True)
            elif style == 'bolditalic':
                pdf.add_font('Roboto', 'BI', font_path, uni=True)
                
        pdf.set_auto_page_break(auto=True, margin=self.config['margin'])
        return pdf
    
    def _add_header(self, pdf: FPDF, title: str) -> None:
        """
        Adiciona o cabeçalho ao PDF com título e metadados.
        
        Args:
            pdf (FPDF): Instância do PDF
            title (str): Título do documento
        """
        # Título principal
        pdf.set_font(self.config['font_family'], 'B', self.config['title_size'])
        pdf.set_text_color(*self._hex_to_rgb(self.config['colors']['title']))
        pdf.cell(0, 10, title, align='C', ln=True)
        
        # Subtítulo (tema se disponível)
        if 'theme' in self.config:
            pdf.set_font(self.config['font_family'], 'I', self.config['subtitle_size'])
            pdf.set_text_color(*self._hex_to_rgb(self.config['colors']['subtitle']))
            pdf.cell(0, 10, f"Tema: {self.config['theme']}", align='C', ln=True)
        
        # Data e hora
        pdf.set_font(self.config['font_family'], 'I', self.config['subtitle_size'])
        pdf.cell(0, 10, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
        pdf.ln(10)
    
    def create_pdf(self, markdown_text: str, title: str = "Documento") -> Union[bytes, None]:
        """
        Converte texto Markdown em um PDF elegante e bem formatado.
        
        Args:
            markdown_text (str): Texto em formato Markdown para conversão
            title (str, optional): Título do documento. Padrão: "Documento"
            
        Returns:
            bytes: PDF em formato de bytes
            None: Em caso de erro na conversão
            
        Raises:
            Exception: Qualquer erro durante a conversão é logado e re-levantado
        """
        try:
            pdf = self._setup_pdf()
            self._add_header(pdf, title)
            
            # Conversão do Markdown para HTML com estilos CSS inline
            html = markdown.markdown(
                markdown_text,
                extensions=['extra', 'smarty', 'tables']
            )
            
            # Processamento do HTML
            inliner = css_inline.CSSInliner()
            html = inliner.inline(html)
            
            # Configuração do corpo do texto
            pdf.set_font(self.config['font_family'], '', self.config['body_size'])
            pdf.set_text_color(*self._hex_to_rgb(self.config['colors']['body']))
            
            # Renderização do HTML processado
            soup = BeautifulSoup(html, 'html.parser')
            for element in soup.find_all(True):
                text = element.get_text().strip()
                if not text:
                    continue
                    
                if element.name in ['h1', 'h2', 'h3']:
                    pdf.set_font(self.config['font_family'], 'B', 
                               self.config['body_size'] + (4 - int(element.name[1])))
                    pdf.ln(5)
                    pdf.write(5, text)
                    pdf.ln(5)
                elif element.name == 'p':
                    pdf.set_font(self.config['font_family'], '', self.config['body_size'])
                    pdf.write(self.config['line_height'] * 5, text)
                    pdf.ln(5)
                
            return pdf.output()
            
        except Exception as e:
            self.logger.error(f"Erro na conversão do PDF: {str(e)}")
            raise
    
    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple:
        """
        Converte cor hexadecimal para RGB.
        
        Args:
            hex_color (str): Cor em formato hexadecimal (#RRGGBB)
            
        Returns:
            tuple: Valores RGB (r, g, b)
        """
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))