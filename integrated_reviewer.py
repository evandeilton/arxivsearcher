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
    def generate_content(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
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


class AnthropicProvider(APIProvider):
    """
    Provedor de API Anthropic, atualizado para o novo modo de uso da biblioteca.
    """
    def __init__(self, config: APIConfig):
        # Inicializa o cliente Anthropic com a chave de API fornecida
        self.client = anthropic.Anthropic(api_key=config.key)
        self.config = config

    def generate_content(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
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
            max_tokens=4000,
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
        temperature: float = 0.3
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
            max_output_tokens=1000  # Ajuste conforme sua necessidade
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
    Integrated system for searching papers and generating literature reviews.
    """
    def __init__(self):
        self.arxiv_downloader = ArxivDownloader()
        self.review_agent = None

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
        Realiza apenas a busca no arXiv e retorna o DataFrame de resultados (sem gerar revisão).
        """
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
        logger.info(f"Found {len(results_df)} papers.")

        # Ordenação, se aplicável
        if sort_by and sort_order and sort_by != "relevance":
            if sort_by in results_df.columns:
                ascending = (sort_order == "ascending")
                results_df = results_df.sort_values(by=sort_by, ascending=ascending)
            else:
                logger.warning(f"Column '{sort_by}' not found in results. Skipping sorting.")
        
        return results_df

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
        Gera efetivamente a revisão de literatura a partir de um DataFrame filtrado.
        """
        # Inicializa o agente de revisão
        self.review_agent = LiteratureReviewAgent(provider=provider, model=model)
        logger.info("Generating literature review...")

        review_text, bibtex_entries = self.review_agent.process_papers(
            df=df,
            theme=theme,
            output_lang=output_lang
        )

        saved_files = []
        if save_results:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            review_path = output_dir_path / f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.Rmd"
            with open(review_path, 'w', encoding='utf-8') as f:
                f.write(review_text)
            saved_files.append(str(review_path))
            bib_path = output_dir_path / "references.bib"
            with open(bib_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(bibtex_entries))
            saved_files.append(str(bib_path))
            logger.info(f"Results saved to {output_dir_path}")
        
        return review_text, saved_files

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
        Fluxo completo usado no modo CLI: busca e já faz a revisão de TODOS os artigos encontrados.
        """
        # Passo 1: Busca
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
        # Passo 2: Revisão (usa todos os artigos)
        review_text, saved_files = self.review_papers(
            df=results_df,
            theme=theme,
            provider=provider,
            model=model,
            output_lang=output_lang,
            save_results=save_results,
            output_dir=output_dir
        )
        return results_df, review_text, saved_files


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


# #!/usr/bin/env python3
# import anthropic
# import logging
# import os
# import requests
# from dataclasses import dataclass
# from pathlib import Path
# from typing import List, Optional, Tuple
# from abc import ABC, abstractmethod
# from datetime import datetime
# from tenacity import retry, stop_after_attempt, wait_exponential

# from google import genai

# # Import do módulo arxivsearcher (deve estar instalado ou no path)
# from arxivsearcher import ArxivDownloader, run_arxiv_search

# # Configuração de logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(asctime)s] %(levelname)s: %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# logger = logging.getLogger(__name__)


# @dataclass
# class APIConfig:
#     key: str
#     base_url: Optional[str] = None
#     default_model: str = ""


# class APIProvider(ABC):
#     """Classe abstrata base para provedores de API."""
#     @abstractmethod
#     def generate_content(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
#         pass


# class DeepseekProvider(APIProvider):
#     """Provedor de API Deepseek."""
#     def __init__(self, config: APIConfig):
#         self.config = config
#         self.headers = {
#             "Authorization": f"Bearer {config.key}",
#             "Content-Type": "application/json"
#         }
#     def generate_content(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
#         response = requests.post(
#             self.config.base_url,
#             headers=self.headers,
#             json={
#                 "model": self.config.default_model,
#                 "messages": [
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 "temperature": temperature
#             }
#         )
#         response.raise_for_status()
#         return response.json()["choices"][0]["message"]["content"]


# class AnthropicProvider(APIProvider):
#     """
#     Provedor de API Anthropic, atualizado para o novo modo de uso da biblioteca.
#     """
#     def __init__(self, config: APIConfig):
#         # Inicializa o cliente Anthropic com a chave de API fornecida
#         self.client = anthropic.Anthropic(api_key=config.key)
#         self.config = config

#     def generate_content(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
#         """
#         Gera conteúdo usando a API da Anthropic, seguindo a interface de mensagens.
        
#         Args:
#             system_prompt (str): Instruções do sistema para definir comportamento, tom, etc.
#             user_prompt (str): Mensagem que representa a consulta/fala do usuário.
#             temperature (float): Parâmetro de amostragem que controla a aleatoriedade da saída.

#         Returns:
#             str: Conteúdo textual retornado pela API Anthropic.
#         """
#         message = self.client.messages.create(
#             model=self.config.default_model,
#             max_tokens=4000,
#             temperature=temperature,
#             # O parâmetro "system" contém as instruções de sistema
#             system=system_prompt,
#             # "messages" recebe uma lista de "turnos" de conversa,
#             # cada um com "role" (system/user) e "content" (lista de objetos)
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "text",
#                             "text": user_prompt
#                         }
#                     ]
#                 }
#             ]
#         )
#         return message.content[0].text

# class OpenAIProvider(APIProvider):
#     """Provedor de API OpenAI."""
#     def __init__(self, config: APIConfig):
#         self.config = config
#         self.headers = {
#             "Authorization": f"Bearer {config.key}",
#             "Content-Type": "application/json"
#         }
#     def generate_content(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
#         response = requests.post(
#             self.config.base_url,
#             headers=self.headers,
#             json={
#                 "model": self.config.default_model,
#                 "messages": [
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 "temperature": temperature
#             }
#         )
#         response.raise_for_status()
#         return response.json()["choices"][0]["message"]["content"]


# class GeminiProvider(APIProvider):
#     """
#     Provedor Google Generative AI (Gemini) atualizado para o novo SDK ("google-genai").
#     """
    
#     def __init__(self, config: APIConfig):
#         """
#         Inicializa o provedor Gemini.
        
#         Args:
#             config (APIConfig): Objeto de configuração com a chave de API e modelo padrão.
#         """
#         if genai is None:
#             raise ImportError(
#                 "Google AI package não está instalado. "
#                 "Instale via 'pip install google-genai' e 'pip install google-generativeai' se necessário."
#             )
#         self.config = config
#         # Cria um cliente do Google Generative AI, utilizando a chave do config
#         self.client = genai.Client(api_key=self.config.key)
        
#     def generate_content(
#         self,
#         system_prompt: str,
#         user_prompt: str,
#         temperature: float = 0.3
#     ) -> str:
#         """
#         Gera conteúdo chamando a API do Google Generative AI (modelo Gemini).
        
#         Args:
#             system_prompt (str): Instruções de sistema (tonalidade, papel etc.).
#             user_prompt (str): Pergunta ou comando do usuário.
#             temperature (float, optional): Controla a aleatoriedade da geração. 
#                                            Padrão em 0.3 para gerar textos consistentes.
                                           
#         Returns:
#             str: Texto retornado pela API do modelo Gemini.
#         """
#         # Monta a configuração de geração. Você pode adicionar mais parâmetros conforme a doc.
#         config_params = genai.types.GenerateContentConfig(
#             system_instruction=system_prompt,
#             temperature=temperature,
#             max_output_tokens=1000  # Ajuste conforme sua necessidade
#         )
        
#         # Chama o endpoint 'models.generate_content' com o modelo definido em config.default_model
#         response = self.client.models.generate_content(
#             model=self.config.default_model,
#             contents=user_prompt,
#             config=config_params
#         )
        
#         return response.text

# class LiteratureReviewAgent:
#     """Agente para geração de revisões de literatura usando vários provedores de IA."""
#     PROVIDER_CONFIGS = {
#         "deepseek": APIConfig(
#             key=os.getenv("DEEPSEEK_API_KEY", ""),
#             base_url="https://api.deepseek.com/v1/chat/completions",
#             default_model="deepseek-chat"
#         ),
#         "openai": APIConfig(
#             key=os.getenv("OPENAI_API_KEY", ""),
#             base_url="https://api.openai.com/v1/chat/completions",
#             default_model="gpt-4o"
#         ),
#         "anthropic": APIConfig(
#             key=os.getenv("ANTHROPIC_API_KEY", ""),
#             default_model="claude-3-5-haiku-20241022"
#         ),

#         "gemini": APIConfig(
#             key=os.getenv("GEMINI_API_KEY", ""),
#             default_model="gemini-2.0-flash"
#         )
#     }

#     PROVIDER_CLASSES = {
#         "deepseek": DeepseekProvider,
#         "anthropic": AnthropicProvider,
#         "openai": OpenAIProvider,
#         "gemini": GeminiProvider
#     }

#     def __init__(self, provider: str, model: Optional[str] = None):
#         config = self.PROVIDER_CONFIGS[provider]
#         if not config.key:
#             raise ValueError(f"API key not found for {provider}. Please set the {provider.upper()}_API_KEY environment variable.")
#         if model:
#             config.default_model = model
#         self.provider = self.PROVIDER_CLASSES[provider](config)

#     def _create_system_prompt(self, theme: str) -> str:
#         prompt = f"""
#         <role>
#             You are a researcher specialized in systematic literature reviews about {theme}. Your task is to analyze all articles provided by the user, extracting and synthesizing relevant information for the theme "{theme}".
#         <role\>
#         <important>
#             - Use ALL content from the provided articles. Do not ignore any relevant information.
#             - Read and process all provided articles carefully and extract relevant information
#             - Identify connections with the theme "{theme}"
#             - Extract relevant evidence, methods, and results
#             - Synthesize findings while maintaining focus on the theme
#             - write a comprehensive literature review with near 4000 words about the theme "{theme}"
#         <important\>

#         <guidelines>
#             "Atention": [Summary, Methodology, Results, Discussion, Conclusion]
#             "Analysis": [Theoretical frameworks, Methods used, Empirical evidence, Patterns and trends, Identified gaps, Relationship between papers and theme]
#         <guidelines\>

#         <output>
#             - Literature Review
#             - Methodology
#             - Results
#         <output\>
# """
#         return prompt.strip()

#     def process_papers(self, df, theme: str, n_papers: int, output_lang: str) -> Tuple[str, List[str]]:
#         try:
#             for col in ["title", "authors", "summary"]:
#                 if col not in df.columns:
#                     raise ValueError(f"Coluna '{col}' não encontrada nos resultados.")
#             formatted_papers = []
#             bibtex_entries = []
#             for _, paper in df.iterrows():
#                 authors_split = paper['authors'].split(',')
#                 first_author = authors_split[0].split()[-1] if authors_split else "Unknown"
#                 publication_year = str(paper.get('published', '2024'))[:4]
#                 citation_key = f"{first_author}{publication_year}"
#                 paper_info = (
#                     f"Title: {paper['title']}\n"
#                     f"Authors: {paper['authors']}\n"
#                     f"Summary: {paper['summary']}\n"
#                     f"Citation: @{citation_key}\n"
#                     "---"
#                 )
#                 formatted_papers.append(paper_info)
#                 bibtex = (
#                     f"@article{{{citation_key},\n"
#                     f"  title = {{{paper['title']}}},\n"
#                     f"  author = {{{paper['authors']}}},\n"
#                     f"  year = {{{publication_year}}},\n"
#                     f"  journal = {{arXiv}},\n"
#                     f"  url = {{{paper.get('pdf_url', '')}}}\n"
#                     f"}}\n"
#                 )
#                 bibtex_entries.append(bibtex)
#             content = "\n\n".join(formatted_papers)
#             system_prompt = self._create_system_prompt(theme)
#             user_prompt = f"""Create a comprehensive literature review based on the following papers. The review should be in {output_lang}.

# Papers:
# {content}

# Follow the structure in the system prompt while maintaining academic rigor and critical perspective."""
#             review = self.provider.generate_content(system_prompt, user_prompt)
#             return review, bibtex_entries
#         except Exception as e:
#             raise RuntimeError(f"Error processing papers: {str(e)}")


# class IntegratedReviewSystem:
#     """
#     Integrated system for searching papers and generating literature reviews.
#     """
#     def __init__(self):
#         self.arxiv_downloader = ArxivDownloader()
#         self.review_agent = None
        
#     def search_and_review(
#         self,
#         query: str,
#         theme: str,
#         max_results: int = 10,
#         download_count: int = 5,
#         provider: str = "anthropic",
#         model: Optional[str] = None,
#         output_lang: str = "en-US",
#         download_pdfs: bool = False,
#         save_results: bool = True,
#         output_dir: str = "reviews",
#         date_range: Optional[Tuple[datetime, datetime]] = None,
#         sort_by: Optional[str] = None,
#         sort_order: Optional[str] = None,
#     ) -> Tuple:
#         logger.info(f"Searching arXiv for: {query}")
#         results_df = run_arxiv_search(
#             query=query,
#             max_results=max_results,
#             download=download_pdfs,
#             save_csv=save_results,
#             output_dir=output_dir,
#             date_range=date_range,
#             download_count=download_count
#         )
#         logger.info(f"Found {len(results_df)} papers.")
#         if sort_by and sort_order and sort_by != "relevance":
#             if sort_by in results_df.columns:
#                 ascending = (sort_order == "ascending")
#                 results_df = results_df.sort_values(by=sort_by, ascending=ascending)
#             else:
#                 logger.warning(f"Column '{sort_by}' not found in results. Skipping sorting.")
#         self.review_agent = LiteratureReviewAgent(provider=provider, model=model)
#         logger.info("Generating literature review...")
#         review_text, bibtex_entries = self.review_agent.process_papers(
#             results_df,
#             theme,
#             max_results,
#             output_lang
#         )
#         saved_files = []
#         if save_results:
#             output_dir_path = Path(output_dir)
#             output_dir_path.mkdir(parents=True, exist_ok=True)
#             review_path = output_dir_path / f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.Rmd"
#             with open(review_path, 'w', encoding='utf-8') as f:
#                 f.write(review_text)
#             saved_files.append(str(review_path))
#             bib_path = output_dir_path / "references.bib"
#             with open(bib_path, 'w', encoding='utf-8') as f:
#                 f.write('\n'.join(bibtex_entries))
#             saved_files.append(str(bib_path))
#             logger.info(f"Results saved to {output_dir_path}")
#         return results_df, review_text, saved_files


# # Função para executar via CLI
# def parse_args():
#     import argparse
#     parser = argparse.ArgumentParser(description="Integrated Literature Review System")
#     parser.add_argument("--query", required=True, help="arXiv search query")
#     parser.add_argument("--theme", required=True, help="Theme for literature review")
#     parser.add_argument("--max_results", type=int, default=10, help="Maximum number of papers")
#     parser.add_argument("--download_count", type=int, default=5,
#                         help="Number of PDFs to download if --download is set.")
#     parser.add_argument("--provider", choices=["deepseek", "anthropic", "openai", "gemini"],
#                         default="anthropic", help="AI provider")
#     parser.add_argument("--model", help="Specific model to use")
#     parser.add_argument("--output_lang", default="en-US", help="Output language")
#     parser.add_argument("--download_pdfs", action="store_true", help="Download PDF files")
#     parser.add_argument("--output_dir", default="reviews", help="Output directory")
#     parser.add_argument("--no_save", action="store_true", help="Don't save results")
#     parser.add_argument("--start_date", help="Start date (YYYY-MM-DD)")
#     parser.add_argument("--end_date", help="End date (YYYY-MM-DD)")
#     parser.add_argument("--sort_by", choices=["relevance", "lastUpdatedDate", "submittedDate"],
#                         default="relevance", help="Sort the results by the selected field")
#     parser.add_argument("--sort_order", choices=["ascending", "descending"],
#                         default="descending", help="Sort order for the results")
#     return parser.parse_args()


# def main():
#     import sys
#     args = parse_args()
#     date_range = None
#     if args.start_date and args.end_date:
#         try:
#             start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
#             end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
#             date_range = (start_date, end_date)
#         except ValueError:
#             logger.error("Invalid date format. Use YYYY-MM-DD")
#             sys.exit(1)
#     system = IntegratedReviewSystem()
#     try:
#         results_df, review_text, saved_files = system.search_and_review(
#             query=args.query,
#             theme=args.theme,
#             max_results=args.max_results,
#             download_count=args.download_count,
#             provider=args.provider,
#             model=args.model if args.model else None,
#             output_lang=args.output_lang,
#             download_pdfs=args.download_pdfs,
#             save_results=not args.no_save,
#             output_dir=args.output_dir,
#             date_range=date_range,
#             sort_by=args.sort_by,
#             sort_order=args.sort_order
#         )
#         if not args.no_save:
#             logger.info("Files saved:")
#             for file_path in saved_files:
#                 logger.info(f"- {file_path}")
#         else:
#             print("\nGenerated Review:")
#             print("=" * 80)
#             print(review_text)
#     except Exception as e:
#         logger.error(f"Error: {str(e)}")
#         sys.exit(1)


# if __name__ == "__main__":
#     main()
