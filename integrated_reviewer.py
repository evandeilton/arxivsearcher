# integrated_reviewer.py
#!/usr/bin/env python3
import anthropic
from openai import OpenAI
import logging
import os
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Callable
from abc import ABC, abstractmethod
from datetime import datetime
import json
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential
import pandas as pd
from google import genai

# Import do módulo arxivsearcher
from arxivsearcher import ArxivDownloader, run_arxiv_search

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def validate_api_response(func):
    """Decorator para validar respostas de API."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            response = func(*args, **kwargs)
            if not response or not isinstance(response, str):
                raise ValueError("Resposta inválida da API")
            return response
        except Exception as e:
            logger.error(f"Erro na chamada da API: {str(e)}")
            raise
    return wrapper


@dataclass
class APIConfig:
    """Configuração para provedores de API."""
    key: str
    base_url: Optional[str] = None
    default_model: str = ""
    
    def __post_init__(self):
        """Validação após inicialização."""
        if not self.key:
            raise ValueError("API key é obrigatória")
        if self.base_url and not self.base_url.startswith(('http://', 'https://')):
            raise ValueError("URL base inválida")


class APIProvider(ABC):
    """Classe abstrata base para provedores de API."""
    @abstractmethod
    def generate_content(self, system_prompt:str, user_prompt: str, temperature: float = 0.0) -> str:
        """
        Gera conteúdo usando a API do provedor.
        
        Args:
            system_prompt (str): Instruções do sistema
            user_prompt (str): Prompt do usuário
            temperature (float): Parâmetro de temperatura para geração
            
        Returns:
            str: Conteúdo gerado
        """
        pass

class OpenRouterProvider(APIProvider):
    """
    Provedor OpenRouter usando SDK OpenAI.
    Documentação: https://openrouter.ai/docs
    """
    
    DEFAULT_MODEL = "google/gemini-2.0-pro-exp-02-05:free"
    MODELS = [
        "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
        "cognitivecomputations/dolphin3.0-mistral-24b:free",
        "openai/o3-mini-high",
        "openai/o3-mini",
        "openai/chatgpt-4o-latest",
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-001",
        "google/gemini-2.0-flash-thinking-exp:free",
        "google/gemini-2.0-flash-lite-preview-02-05:free",
        "google/gemini-2.0-pro-exp-02-05:free",
        "deepseek/deepseek-r1-distill-llama-70b:free",
        "deepseek/deepseek-r1-distill-qwen-32b",
        "deepseek/deepseek-r1:free",
        "qwen/qwen-plus",
        "qwen/qwen-max",
        "qwen/qwen-turbo",
        "mistralai/codestral-2501",
        "mistralai/mistral-small-24b-instruct-2501:free",
        "anthropic/claude-3.5-haiku-20241022:beta",
        "anthropic/claude-3.5-sonnet",
        "perplexity/sonar-reasoning",
        "perplexity/sonar",
        "perplexity/llama-3.1-sonar-large-128k-online"
    ]

    def __init__(
        self, 
        config: APIConfig,
        site_url: Optional[str] = None, 
        site_name: Optional[str] = None
    ):
        """
        Inicializa o provedor OpenRouter.
        
        Args:
            config: Configuração da API
            site_url: URL opcional do site para rankings
            site_name: Nome opcional do site para rankings
        """
        if not config.key:
            raise ValueError("API key é obrigatória")
            
        # Inicializa o cliente OpenAI com a URL base do OpenRouter
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.key
        )
        
        # Configura headers extras
        self.extra_headers = {}
        if site_url:
            self.extra_headers["HTTP-Referer"] = site_url
        if site_name:
            self.extra_headers["X-Title"] = site_name
            
        self.default_model = config.default_model if config.default_model in self.MODELS else self.DEFAULT_MODEL

    @validate_api_response
    def generate_content(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        temperature: float = 0.0
    ) -> str:
        """
        Gera conteúdo usando a API do OpenRouter.
        
        Args:
            system_prompt: Instruções do sistema
            user_prompt: Prompt do usuário
            temperature: Parâmetro de temperatura (0.0 a 1.0)
            
        Returns:
            str: Conteúdo gerado
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            completion = self.client.chat.completions.create(
                model=self.default_model,
                messages=messages,
                temperature=max(0.0, min(1.0, temperature)),
                extra_headers=self.extra_headers
            )
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenRouter API error: {str(e)}")
            raise

class DeepseekProvider(APIProvider):
    """Provedor Deepseek com tratamento de erros aprimorado."""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.key}",
            "Content-Type": "application/json"
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @validate_api_response
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
    """Provedor Anthropic com suporte à nova API Claude 3."""
    
    def __init__(self, config: APIConfig):
        if not config.key:
            raise ValueError("Anthropic API key é obrigatória")
        self.client = anthropic.Anthropic(api_key=config.key)
        self.config = config

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @validate_api_response
    def generate_content(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        """
        Gera conteúdo usando a API da Anthropic Claude 3.
        
        Args:
            system_prompt: Instruções do sistema
            user_prompt: Prompt do usuário
            temperature: Parâmetro de temperatura
            
        Returns:
            str: Conteúdo gerado
        """
        try:
            message = self.client.messages.create(
                model=self.config.default_model,
                max_tokens=10000,
                temperature=temperature,
                system=system_prompt,
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
        except Exception as e:
            logger.error(f"Erro na API Anthropic: {str(e)}")
            raise


class OpenAIProvider(APIProvider):
    """Provedor OpenAI com validação e retry."""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.key}",
            "Content-Type": "application/json"
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @validate_api_response
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
    """Provedor Google Generative AI (Gemini)."""
    
    def __init__(self, config: APIConfig):
        if genai is None:
            raise ImportError(
                "Google AI package não está instalado. "
                "Instale via 'pip install google-genai' e 'pip install google-generativeai'"
            )
        self.config = config
        genai.configure(api_key=self.config.key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @validate_api_response
    def generate_content(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        """
        Gera conteúdo usando a API do Google Generative AI.
        
        Args:
            system_prompt: Instruções do sistema
            user_prompt: Prompt do usuário
            temperature: Parâmetro de temperatura
            
        Returns:
            str: Conteúdo gerado
        """
        model = genai.GenerativeModel(model_name=self.config.default_model)
        chat = model.start_chat(system_prompt=system_prompt)
        response = chat.send_message(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=20000
            )
        )
        return response.text


class LiteratureReviewAgent:
    """Agente para geração de revisões de literatura usando vários provedores de IA."""
    
    PROVIDER_CONFIGS = {
        "openrouter": APIConfig(        
            key=os.getenv("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
            default_model="google/gemini-2.0-pro-exp-02-05:free"
        ),
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
        "openrouter": OpenRouterProvider,
        "deepseek": DeepseekProvider,
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider
    }

    def __init__(self, provider: str, model: Optional[str] = None):
        if provider not in self.PROVIDER_CONFIGS:
            raise ValueError(f"Provedor {provider} não suportado")
            
        config = self.PROVIDER_CONFIGS[provider]
        if not config.key:
            raise ValueError(
                f"API key não encontrada para {provider}. "
                f"Configure a variável de ambiente {provider.upper()}_API_KEY"
            )
            
        if model:
            config.default_model = model
            
        if provider == "openrouter":
            self.provider = OpenRouterProvider(
                config=config,
                site_url="",
                site_name=""
            )
        else:
            self.provider = self.PROVIDER_CLASSES[provider](config)

    def _create_system_prompt(self, theme: str) -> str:
        """
        Cria o prompt do sistema para a revisão.
        
        Args:
            theme: Tema da revisão
            
        Returns:
            str: Prompt formatado
        """
        prompt = f"""
        <role>
            You are a researcher specialized in systematic literature reviews about {theme}. 
            Your task is to analyze all articles provided by the user, extracting and synthesizing 
            relevant information for the theme "{theme}".
        </role>
        
        <important>
            - Use ALL content from the provided articles. Do not ignore any relevant information.
            - Read and process all provided articles carefully and extract relevant information
            - Identify connections with the theme "{theme}"
            - Extract relevant evidence, methods, and results
            - Synthesize findings while maintaining focus on the theme
            - Write a comprehensive literature review with near 4000 words about the theme "{theme}"
        </important>

        <guidelines>
            "Analysis": [
                Theoretical frameworks,
                Methods used,
                Empirical evidence,
                Patterns and trends,
                Identified gaps,
                Relationship between papers and theme
            ]
            
            "Structure": [
                Summary,
                Methodology,
                Results,
                Discussion,
                Conclusion
            ]
        </guidelines>

        <output>
            - Literature Review
            - Methodology
            - Results
            - References
        </output>
        """
        return prompt.strip()

    def process_papers(
        self,
        df: pd.DataFrame,
        theme: str,
        output_lang: str = "en-US"
    ) -> Tuple[str, List[str]]:
        """
        Gera o texto de revisão usando o conjunto de artigos selecionados.
        
        Args:
            df: DataFrame com os artigos
            theme: Tema da revisão
            output_lang: Idioma de saída
            
        Returns:
            Tuple[str, List[str]]: (texto da revisão, lista de entradas BibTeX)
        """
        try:
            required_columns = ["title", "authors", "summary"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Colunas ausentes: {', '.join(missing_columns)}")

            formatted_papers = []
            bibtex_entries = []

            for _, paper in df.iterrows():
                # Processamento dos autores
                authors_split = paper['authors'].split(',')
                first_author = authors_split[0].split()[-1] if authors_split else "Unknown"
                publication_year = str(paper.get('published', '2024'))[:4]
                citation_key = f"{first_author}{publication_year}"
                
                # Formatação do papel
                paper_info = (
                    f"Title: {paper['title']}\n"
                    f"Authors: {paper['authors']}\n"
                    f"Summary: {paper['summary']}\n"
                    f"Citation: @{citation_key}\n"
                    "---"
                )
                formatted_papers.append(paper_info)
                
                # Geração da entrada BibTeX
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
            logger.error(f"Erro no processamento dos artigos: {str(e)}")
            raise RuntimeError(f"Erro no processamento dos artigos: {str(e)}")


class IntegratedReviewSystem:
    """Sistema integrado para busca de artigos e geração de revisões de literatura."""
    
    def __init__(self):
        """Inicializa o sistema com downloader e sem agente de revisão."""
        self.arxiv_downloader = ArxivDownloader()
        self.review_agent = None
        self.progress_callback = None

    def set_progress_callback(self, callback: Callable[[int, int, str], None]) -> None:
        """
        Define a função de callback para atualizar o progresso.
        
        Args:
            callback: Função que recebe (current, total, message)
        """
        self.progress_callback = callback

    def update_progress(self, current: int, total: int, message: str = "") -> None:
        """
        Atualiza o progresso usando o callback se disponível.
        
        Args:
            current: Valor atual do progresso
            total: Valor total para 100%
            message: Mensagem descritiva opcional
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
    ) -> pd.DataFrame:
        """
        Realiza a busca no arXiv com feedback de progresso.
        
        Args:
            query: Termo de busca
            max_results: Número máximo de resultados
            download_count: Número de PDFs para download
            download_pdfs: Se deve baixar PDFs
            save_results: Se deve salvar resultados
            output_dir: Diretório de saída
            date_range: Intervalo de datas opcional
            sort_by: Campo para ordenação
            sort_order: Ordem da ordenação
            
        Returns:
            pd.DataFrame: Resultados da busca
        """
        try:
            self.update_progress(0, 100, "Iniciando busca no arXiv...")
            
            logger.info(f"Buscando no arXiv: {query}")
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
            logger.info(f"Encontrados {total_found} artigos.")
            self.update_progress(50, 100, f"Encontrados {total_found} artigos")

            # Ordenação dos resultados
            if sort_by and sort_order and sort_by != "relevance":
                if sort_by in results_df.columns:
                    self.update_progress(75, 100, "Ordenando resultados...")
                    ascending = (sort_order == "ascending")
                    results_df = results_df.sort_values(by=sort_by, ascending=ascending)
                else:
                    logger.warning(f"Coluna '{sort_by}' não encontrada. Ignorando ordenação.")
            
            self.update_progress(100, 100, "Busca concluída")
            return results_df
            
        except Exception as e:
            logger.error(f"Erro na busca: {str(e)}")
            raise

    def review_papers(
        self,
        df: pd.DataFrame,
        theme: str,
        provider: str = "anthropic",
        model: Optional[str] = None,
        output_lang: str = "en-US",
        save_results: bool = True,
        output_dir: str = "reviews"
    ) -> Tuple[str, List[str]]:
        """
        Gera a revisão de literatura com feedback detalhado de progresso.
        
        Args:
            df: DataFrame com os artigos
            theme: Tema da revisão
            provider: Provedor de IA
            model: Modelo opcional
            output_lang: Idioma de saída
            save_results: Se deve salvar resultados
            output_dir: Diretório de saída
            
        Returns:
            Tuple[str, List[str]]: (texto da revisão, lista de arquivos salvos)
        """
        try:
            total_papers = len(df)
            current_step = 0
            steps_per_paper = 3  # Análise, Processamento, Síntese
            total_steps = total_papers * steps_per_paper + 2  # +2 para inicialização e finalização
            
            # Inicialização
            self.update_progress(current_step, total_steps, "Inicializando agente de revisão...")
            self.review_agent = LiteratureReviewAgent(provider=provider, model=model)
            current_step += 1
            
            logger.info("Iniciando geração da revisão de literatura...")
            
            # Processamento dos artigos
            formatted_papers = []
            bibtex_entries = []
            
            for idx, (_, paper) in enumerate(df.iterrows(), 1):
                # Análise do artigo
                paper_title = paper['title'][:50] + "..." if len(paper['title']) > 50 else paper['title']
                self.update_progress(
                    current_step, 
                    total_steps,
                    f"Analisando artigo {idx}/{total_papers}: {paper_title}"
                )
                current_step += 1
                
                # Processamento das informações
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
                
                # Formatação e adição aos resultados
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
            
            # Geração da revisão
            self.update_progress(current_step, total_steps, "Gerando revisão de literatura...")
            
            review_text, _ = self.review_agent.process_papers(
                df=df,
                theme=theme,
                output_lang=output_lang
            )
            
            # Salvamento dos resultados
            saved_files = []
            if save_results:
                self.update_progress(current_step, total_steps, "Salvando resultados...")
                output_dir_path = Path(output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)
                
                # Salva a revisão
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                review_path = output_dir_path / f"review_{timestamp}.Rmd"
                with open(review_path, 'w', encoding='utf-8') as f:
                    f.write(review_text)
                saved_files.append(str(review_path))
                
                # Salva as referências
                bib_path = output_dir_path / f"references_{timestamp}.bib"
                with open(bib_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(bibtex_entries))
                saved_files.append(str(bib_path))
                
                logger.info(f"Resultados salvos em {output_dir_path}")
            
            # Finalização
            self.update_progress(total_steps, total_steps, "Revisão concluída com sucesso!")
            return review_text, saved_files
            
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
    ) -> Tuple[pd.DataFrame, str, List[str]]:
        """
        Executa o fluxo completo: busca e revisão com progresso detalhado.
        
        Args:
            query: Termo de busca
            theme: Tema da revisão
            max_results: Número máximo de resultados
            download_count: Número de PDFs para download
            provider: Provedor de IA
            model: Modelo opcional
            output_lang: Idioma de saída
            download_pdfs: Se deve baixar PDFs
            save_results: Se deve salvar resultados
            output_dir: Diretório de saída
            date_range: Intervalo de datas opcional
            sort_by: Campo para ordenação
            sort_order: Ordem da ordenação
            
        Returns:
            Tuple[pd.DataFrame, str, List[str]]: (resultados, texto da revisão, arquivos salvos)
        """
        try:
            # Busca
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
            
            # Revisão
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


def parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Sistema Integrado de Revisão de Literatura")
    
    parser.add_argument("--query", required=True, help="Query de busca no arXiv")
    parser.add_argument("--theme", required=True, help="Tema para a revisão")
    parser.add_argument("--max_results", type=int, default=10, help="Número máximo de artigos")
    parser.add_argument("--download_count", type=int, default=5,
                        help="Número de PDFs para download se --download for definido")
    parser.add_argument("--provider", choices=["deepseek", "anthropic", "openai", "gemini"],
                        default="anthropic", help="Provedor de IA")
    parser.add_argument("--model", help="Modelo específico")
    parser.add_argument("--output_lang", default="en-US", help="Idioma de saída")
    parser.add_argument("--download_pdfs", action="store_true", help="Baixar PDFs")
    parser.add_argument("--output_dir", default="reviews", help="Diretório de saída")
    parser.add_argument("--no_save", action="store_true", help="Não salvar resultados")
    parser.add_argument("--start_date", help="Data inicial (YYYY-MM-DD)")
    parser.add_argument("--end_date", help="Data final (YYYY-MM-DD)")
    parser.add_argument("--sort_by", choices=["relevance", "lastUpdatedDate", "submittedDate"],
                        default="relevance", help="Ordenar resultados pelo campo selecionado")
    parser.add_argument("--sort_order", choices=["ascending", "descending"],
                        default="descending", help="Ordem de classificação dos resultados")
    return parser.parse_args()


def main():
    """Função principal para execução via linha de comando."""
    import sys
    args = parse_args()
    
    # Processamento do intervalo de datas
    date_range = None
    if args.start_date and args.end_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
            if start_date > end_date:
                logger.error("Data inicial deve ser anterior à data final")
                sys.exit(1)
            date_range = (start_date, end_date)
        except ValueError:
            logger.error("Formato de data inválido. Use YYYY-MM-DD")
            sys.exit(1)

    # Inicialização do sistema
    system = IntegratedReviewSystem()
    
    try:
        # Execução do fluxo completo
        results_df, review_text, saved_files = system.search_and_review(
            query=args.query,
            theme=args.theme,
            max_results=args.max_results,
            download_count=args.download_count,
            provider=args.provider,
            model=args.model,
            output_lang=args.output_lang,
            download_pdfs=args.download_pdfs,
            save_results=not args.no_save,
            output_dir=args.output_dir,
            date_range=date_range,
            sort_by=args.sort_by,
            sort_order=args.sort_order
        )
        
        # Exibição dos resultados
        if not args.no_save:
            print("\nArquivos salvos:")
            for file_path in saved_files:
                print(f"- {file_path}")
        else:
            print("\nRevisão Gerada:")
            print("=" * 80)
            print(review_text)
            
    except Exception as e:
        logger.error(f"Erro: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
