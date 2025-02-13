#integrated_reviewer.py
#!/usr/bin/env python3
import anthropic
import logging
import os
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
from datetime import datetime

from tenacity import retry, stop_after_attempt, wait_exponential

from google import genai

# Import do módulo arxivsearcher (deve estar instalado ou no path)
from arxivsearcher import ArxivDownloader, run_arxiv_search

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    key: str
    base_url: Optional[str] = None
    default_model: str = ""


class APIProvider(ABC):
    """Classe abstrata base para provedores de API."""
    @abstractmethod
    def generate_content(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        pass


#############################################################################
#              >>>> NÃO ALTERAR CÓDIGO DOS PROVIDERS ABAIXO <<<<            #
#############################################################################

class DeepseekProvider(APIProvider):
    """Provedor de API Deepseek."""
    def __init__(self, config: APIConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.key}",
            "Content-Type": "application/json"
        }
    def generate_content(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        response = requests.post(
            self.config.base_url,
            headers=self.headers,
            json={
                "model": self.config.default_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class AnthropicProvider(APIProvider):
    """
    Provedor de API Anthropic, atualizado para o novo modo de uso da biblioteca.
    """
    def __init__(self, config: APIConfig):
        # Inicializa o cliente Anthropic com a chave de API fornecida
        self.client = anthropic.Anthropic(api_key=config.key)
        self.config = config

    def generate_content(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        """
        Gera conteúdo usando a API da Anthropic, seguindo a interface de mensagens.
        
        Args:
            system_prompt (str): Instruções do sistema para definir comportamento, tom, etc.
            user_prompt (str): Mensagem que representa a consulta/fala do usuário.
            temperature (float): Parâmetro de amostragem que controla a aleatoriedade da saída.

        Returns:
            str: Conteúdo textual retornado pela API Anthropic.
        """
        message = self.client.messages.create(
            model=self.config.default_model,
            max_tokens=10000,
            temperature=temperature,
            # O parâmetro "system" contém as instruções de sistema
            system=system_prompt,
            # "messages" recebe uma lista de "turnos" de conversa,
            # cada um com "role" (system/user) e "content" (lista de objetos)
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }
            ]
        )
        return message.content[0].text


class OpenAIProvider(APIProvider):
    """Provedor de API OpenAI."""
    def __init__(self, config: APIConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.key}",
            "Content-Type": "application/json"
        }
    def generate_content(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
        response = requests.post(
            self.config.base_url,
            headers=self.headers,
            json={
                "model": self.config.default_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class GeminiProvider(APIProvider):
    """
    Provedor Google Generative AI (Gemini) atualizado para o novo SDK ("google-genai").
    """
    
    def __init__(self, config: APIConfig):
        """
        Inicializa o provedor Gemini.
        
        Args:
            config (APIConfig): Objeto de configuração com a chave de API e modelo padrão.
        """
        if genai is None:
            raise ImportError(
                "Google AI package não está instalado. "
                "Instale via 'pip install google-genai' e 'pip install google-generativeai' se necessário."
            )
        self.config = config
        # Cria um cliente do Google Generative AI, utilizando a chave do config
        self.client = genai.Client(api_key=self.config.key)
        
    def generate_content(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0
    ) -> str:
        """
        Gera conteúdo chamando a API do Google Generative AI (modelo Gemini).
        
        Args:
            system_prompt (str): Instruções de sistema (tonalidade, papel etc.).
            user_prompt (str): Pergunta ou comando do usuário.
            temperature (float, optional): Controla a aleatoriedade da geração. 
                                           Padrão em 0.3 para gerar textos consistentes.
                                           
        Returns:
            str: Texto retornado pela API do modelo Gemini.
        """
        config_params = genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=20000  # Ajuste conforme sua necessidade
        )
        
        response = self.client.models.generate_content(
            model=self.config.default_model,
            contents=user_prompt,
            config=config_params
        )
        
        return response.text


#############################################################################
#           >>>> CÓDIGO DO AGENTE DE REVISÃO (PODE SER ADAPTADO) <<<<       #
#############################################################################

class LiteratureReviewAgent:
    """Agente para geração de revisões de literatura usando vários provedores de IA."""
    PROVIDER_CONFIGS = {
        "deepseek": APIConfig(
            key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com/v1/chat/completions",
            default_model="deepseek-chat"
        ),
        "openai": APIConfig(
            key=os.getenv("OPENAI_API_KEY", ""),
            base_url="https://api.openai.com/v1/chat/completions",
            default_model="gpt-4o"
        ),
        "anthropic": APIConfig(
            key=os.getenv("ANTHROPIC_API_KEY", ""),
            default_model="claude-3-5-haiku-20241022"
        ),
        "gemini": APIConfig(
            key=os.getenv("GEMINI_API_KEY", ""),
            default_model="gemini-2.0-flash"
        )
    }

    PROVIDER_CLASSES = {
        "deepseek": DeepseekProvider,
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider
    }

    def __init__(self, provider: str, model: Optional[str] = None):
        config = self.PROVIDER_CONFIGS[provider]
        if not config.key:
            raise ValueError(
                f"API key not found for {provider}. "
                f"Please set the {provider.upper()}_API_KEY environment variable."
            )
        if model:
            config.default_model = model
        self.provider = self.PROVIDER_CLASSES[provider](config)

    def _create_system_prompt(self, theme: str) -> str:
        prompt = f"""
        <role>
            You are a researcher specialized in systematic literature reviews about {theme}. 
            Your task is to analyze all articles provided by the user, extracting and synthesizing 
            relevant information for the theme "{theme}".
        <role\>
        <important>
            - Use ALL content from the provided articles. Do not ignore any relevant information.
            - Read and process all provided articles carefully and extract relevant information
            - Identify connections with the theme "{theme}"
            - Extract relevant evidence, methods, and results
            - Synthesize findings while maintaining focus on the theme
            - write a comprehensive literature review with near 4000 words about the theme "{theme}"
        <important\>

        <guidelines>
            "Atention": [Summary, Methodology, Results, Discussion, Conclusion]
            "Analysis": [Theoretical frameworks, Methods used, Empirical evidence, Patterns and trends, 
                         Identified gaps, Relationship between papers and theme]
        <guidelines\>

        <output>
            - Literature Review
            - Methodology
            - Results
        <output\>
"""
        return prompt.strip()

    def process_papers(self, df, theme: str, output_lang: str) -> Tuple[str, List[str]]:
        """
        Gera o texto de revisão usando o conjunto de artigos (df) selecionados.
        
        Args:
            df (pd.DataFrame): DataFrame com colunas ["title", "authors", "summary", ...].
            theme (str): Tema da revisão.
            output_lang (str): Idioma em que a revisão será gerada (ex: "pt-BR", "en-US").
        
        Returns:
            Tuple[str, List[str]]: (texto da revisão, lista de entradas BibTeX)
        """
        try:
            for col in ["title", "authors", "summary"]:
                if col not in df.columns:
                    raise ValueError(f"Coluna '{col}' não encontrada nos resultados.")
            formatted_papers = []
            bibtex_entries = []

            for _, paper in df.iterrows():
                authors_split = paper['authors'].split(',')
                first_author = authors_split[0].split()[-1] if authors_split else "Unknown"
                publication_year = str(paper.get('published', '2024'))[:4]
                citation_key = f"{first_author}{publication_year}"
                
                paper_info = (
                    f"Title: {paper['title']}\n"
                    f"Authors: {paper['authors']}\n"
                    f"Summary: {paper['summary']}\n"
                    f"Citation: @{citation_key}\n"
                    "---"
                )
                formatted_papers.append(paper_info)
                
                bibtex = (
                    f"@article{{{citation_key},\n"
                    f"  title = {{{paper['title']}}},\n"
                    f"  author = {{{paper['authors']}}},\n"
                    f"  year = {{{publication_year}}},\n"
                    f"  journal = {{arXiv}},\n"
                    f"  url = {{{paper.get('pdf_url', '')}}}\n"
                    f"}}\n"
                )
                bibtex_entries.append(bibtex)
            
            content = "\n\n".join(formatted_papers)
            system_prompt = self._create_system_prompt(theme)
            user_prompt = f"""Create a comprehensive literature review based on the following papers. 
The review should be in {output_lang}.

Papers:
{content}

Follow the structure in the system prompt while maintaining academic rigor and a critical perspective.
"""
            review = self.provider.generate_content(system_prompt, user_prompt)
            return review, bibtex_entries
        except Exception as e:
            raise RuntimeError(f"Error processing papers: {str(e)}")


#############################################################################
#         >>>> SISTEMA INTEGRADO DE BUSCA E REVISÃO (REFATORADO) <<<<       #
#############################################################################
class IntegratedReviewSystem:
    """
    Sistema integrado para busca de artigos e geração de revisões de literatura.
    """
    def __init__(self):
        self.arxiv_downloader = ArxivDownloader()
        self.review_agent = None
        self.progress_callback = None

    def set_progress_callback(self, callback):
        """
        Define a função de callback para atualizar o progresso.
        
        Args:
            callback: Função que recebe (current, total, message)
        """
        self.progress_callback = callback

    def update_progress(self, current: int, total: int, message: str = ""):
        """
        Atualiza o progresso usando o callback se disponível.
        
        Args:
            current (int): Valor atual do progresso
            total (int): Valor total para 100%
            message (str): Mensagem descritiva opcional
        """
        if self.progress_callback:
            self.progress_callback(current, total, message)

    def search_papers(
        self,
        query: str,
        max_results: int = 10,
        download_count: int = 5,
        download_pdfs: bool = False,
        save_results: bool = True,
        output_dir: str = "reviews",
        date_range: Optional[Tuple[datetime, datetime]] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None
    ):
        """
        Realiza a busca no arXiv com feedback de progresso.
        """
        try:
            self.update_progress(0, 100, "Iniciando busca no arXiv...")
            
            logger.info(f"Searching arXiv for: {query}")
            results_df = run_arxiv_search(
                query=query,
                max_results=max_results,
                download=download_pdfs,
                save_csv=save_results,
                output_dir=output_dir,
                date_range=date_range,
                download_count=download_count
            )
            
            total_found = len(results_df)
            logger.info(f"Found {total_found} papers.")
            self.update_progress(50, 100, f"Encontrados {total_found} artigos")

            # Ordenação, se aplicável
            if sort_by and sort_order and sort_by != "relevance":
                if sort_by in results_df.columns:
                    self.update_progress(75, 100, "Ordenando resultados...")
                    ascending = (sort_order == "ascending")
                    results_df = results_df.sort_values(by=sort_by, ascending=ascending)
                else:
                    logger.warning(f"Column '{sort_by}' not found in results. Skipping sorting.")
            
            self.update_progress(100, 100, "Busca concluída")
            return results_df
            
        except Exception as e:
            logger.error(f"Erro na busca: {str(e)}")
            raise

    def review_papers(
        self,
        df,
        theme: str,
        provider: str = "anthropic",
        model: Optional[str] = None,
        output_lang: str = "en-US",
        save_results: bool = True,
        output_dir: str = "reviews"
    ):
        """
        Gera a revisão de literatura com feedback detalhado de progresso.
        """
        try:
            total_papers = len(df)
            current_step = 0
            steps_per_paper = 3  # Análise, Processamento, Síntese
            total_steps = total_papers * steps_per_paper + 2  # +2 para inicialização e finalização
            
            # Step 1: Inicialização
            self.update_progress(current_step, total_steps, "Inicializando agente de revisão...")
            self.review_agent = LiteratureReviewAgent(provider=provider, model=model)
            current_step += 1
            
            logger.info("Iniciando geração da revisão de literatura...")
            
            # Step 2: Processamento dos artigos
            formatted_papers = []
            bibtex_entries = []
            
            for idx, (_, paper) in enumerate(df.iterrows(), 1):
                # 2.1 Análise do artigo
                paper_title = paper['title'][:50] + "..." if len(paper['title']) > 50 else paper['title']
                self.update_progress(
                    current_step, 
                    total_steps,
                    f"Analisando artigo {idx}/{total_papers}: {paper_title}"
                )
                current_step += 1
                
                # 2.2 Processamento das informações
                authors_split = paper['authors'].split(',')
                first_author = authors_split[0].split()[-1] if authors_split else "Unknown"
                publication_year = str(paper.get('published', '2024'))[:4]
                citation_key = f"{first_author}{publication_year}"
                
                self.update_progress(
                    current_step,
                    total_steps,
                    f"Processando metadados do artigo {idx}/{total_papers}"
                )
                current_step += 1
                
                # 2.3 Formatação e adição aos resultados
                paper_info = (
                    f"Title: {paper['title']}\n"
                    f"Authors: {paper['authors']}\n"
                    f"Summary: {paper['summary']}\n"
                    f"Citation: @{citation_key}\n"
                    "---"
                )
                formatted_papers.append(paper_info)
                
                bibtex = (
                    f"@article{{{citation_key},\n"
                    f"  title = {{{paper['title']}}},\n"
                    f"  author = {{{paper['authors']}}},\n"
                    f"  year = {{{publication_year}}},\n"
                    f"  journal = {{arXiv}},\n"
                    f"  url = {{{paper.get('pdf_url', '')}}}\n"
                    f"}}\n"
                )
                bibtex_entries.append(bibtex)
                
                self.update_progress(
                    current_step,
                    total_steps,
                    f"Concluído artigo {idx}/{total_papers}"
                )
                current_step += 1
            
            # Step 3: Geração da revisão
            self.update_progress(current_step, total_steps, "Gerando revisão de literatura...")
            content = "\n\n".join(formatted_papers)
            system_prompt = self.review_agent._create_system_prompt(theme)
            user_prompt = f"""Create a comprehensive literature review based on the following papers. 
The review should be in {output_lang}.

Papers:
{content}

Follow the structure in the system prompt while maintaining academic rigor and a critical perspective.
"""
            review = self.review_agent.provider.generate_content(system_prompt, user_prompt)
            
            # Step 4: Salvamento dos resultados
            saved_files = []
            if save_results:
                self.update_progress(current_step, total_steps, "Salvando resultados...")
                output_dir_path = Path(output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)
                
                # Salva a revisão
                review_path = output_dir_path / f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.Rmd"
                with open(review_path, 'w', encoding='utf-8') as f:
                    f.write(review)
                saved_files.append(str(review_path))
                
                # Salva as referências
                bib_path = output_dir_path / "references.bib"
                with open(bib_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(bibtex_entries))
                saved_files.append(str(bib_path))
                
                logger.info(f"Resultados salvos em {output_dir_path}")
            
            # Finalização
            self.update_progress(total_steps, total_steps, "Revisão concluída com sucesso!")
            return review, saved_files
            
        except Exception as e:
            logger.error(f"Erro ao gerar revisão: {str(e)}")
            raise RuntimeError(f"Erro ao processar artigos: {str(e)}")

    def search_and_review(
        self,
        query: str,
        theme: str,
        max_results: int = 10,
        download_count: int = 5,
        provider: str = "anthropic",
        model: Optional[str] = None,
        output_lang: str = "en-US",
        download_pdfs: bool = False,
        save_results: bool = True,
        output_dir: str = "reviews",
        date_range: Optional[Tuple[datetime, datetime]] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> Tuple:
        """
        Fluxo completo: busca e revisão com progresso detalhado.
        """
        try:
            # Passo 1: Busca
            self.update_progress(0, 100, "Iniciando busca de artigos...")
            results_df = self.search_papers(
                query=query,
                max_results=max_results,
                download_count=download_count,
                download_pdfs=download_pdfs,
                save_results=save_results,
                output_dir=output_dir,
                date_range=date_range,
                sort_by=sort_by,
                sort_order=sort_order
            )
            
            self.update_progress(50, 100, "Busca concluída. Iniciando revisão...")
            
            # Passo 2: Revisão
            review_text, saved_files = self.review_papers(
                df=results_df,
                theme=theme,
                provider=provider,
                model=model,
                output_lang=output_lang,
                save_results=save_results,
                output_dir=output_dir
            )
            
            self.update_progress(100, 100, "Processo completo!")
            return results_df, review_text, saved_files
            
        except Exception as e:
            logger.error(f"Erro no processo completo: {str(e)}")
            raise


# Função para executar via CLI
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Integrated Literature Review System")
    parser.add_argument("--query", required=True, help="arXiv search query")
    parser.add_argument("--theme", required=True, help="Theme for literature review")
    parser.add_argument("--max_results", type=int, default=10, help="Maximum number of papers")
    parser.add_argument("--download_count", type=int, default=5,
                        help="Number of PDFs to download if --download is set.")
    parser.add_argument("--provider", choices=["deepseek", "anthropic", "openai", "gemini"],
                        default="anthropic", help="AI provider")
    parser.add_argument("--model", help="Specific model to use")
    parser.add_argument("--output_lang", default="en-US", help="Output language")
    parser.add_argument("--download_pdfs", action="store_true", help="Download PDF files")
    parser.add_argument("--output_dir", default="reviews", help="Output directory")
    parser.add_argument("--no_save", action="store_true", help="Don't save results")
    parser.add_argument("--start_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--sort_by", choices=["relevance", "lastUpdatedDate", "submittedDate"],
                        default="relevance", help="Sort the results by the selected field")
    parser.add_argument("--sort_order", choices=["ascending", "descending"],
                        default="descending", help="Sort order for the results")
    return parser.parse_args()


def main():
    import sys
    args = parse_args()
    date_range = None
    if args.start_date and args.end_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
            date_range = (start_date, end_date)
        except ValueError:
            logger.error("Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)
    system = IntegratedReviewSystem()
    try:
        results_df, review_text, saved_files = system.search_and_review(
            query=args.query,
            theme=args.theme,
            max_results=args.max_results,
            download_count=args.download_count,
            provider=args.provider,
            model=args.model if args.model else None,
            output_lang=args.output_lang,
            download_pdfs=args.download_pdfs,
            save_results=not args.no_save,
            output_dir=args.output_dir,
            date_range=date_range,
            sort_by=args.sort_by,
            sort_order=args.sort_order
        )
        if not args.no_save:
            logger.info("Files saved:")
            for file_path in saved_files:
                logger.info(f"- {file_path}")
        else:
            print("\nGenerated Review:")
            print("=" * 80)
            print(review_text)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
