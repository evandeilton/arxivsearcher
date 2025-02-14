# Integrated Literature Review System

This project provides an integrated system for searching academic papers on arXiv, generating literature reviews using AI, and interacting with the system through a user-friendly interface. It streamlines the research process by combining the functionality of `arxivsearcher.py`, `integrated_reviewer.py`, `reviewer_ui.py`, and `styles.py`.

## Table of Contents

- [Features](#features)
- [Project Structure and Architecture](#project-structure-and-architecture)
- [Configuration and API Providers](#configuration-and-api-providers)
- [Execution Flow](#execution-flow)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
  - [Streamlit UI (reviewer_ui.py)](#streamlit-ui-reviewer_uipy)
  - [Jupyter Notebook / Python API](#jupyter-notebook-python-api)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [`styles.py`](#stylespy)
- [`reviews` Directory](#reviews-directory)

## Features

- **ArXiv Search:** Search for papers on arXiv using keywords, date ranges, and sorting options.
- **AI-Powered Review Generation:** Generate comprehensive literature reviews using various AI providers (Anthropic, OpenAI, Gemini, Deepseek, OpenRouter).
- **Streamlit User Interface:** Interactive web interface for easy access to system features.
- **PDF Downloads:** Option to download PDFs of selected papers.
- **Result Export:** Save search results and generated reviews.
- **Caching and Rate Limiting:** Improve performance and adhere to API usage policies.

## Project Structure and Architecture

- **arxivsearcher.py:** Handles the search and download of articles from arXiv with rate limiting, caching, and API request handling.
- **integrated_reviewer.py:** Coordinates the literature review process by integrating search results, analyzing articles, and synthesizing review content via multiple AI providers. Includes progress callbacks and logging for feedback.
- **reviewer_ui.py:** Implements a Streamlit-based interface for a user-friendly experience.
- **styles.py:** Provides custom CSS and UI functions to style the Streamlit interface.

## Configuration and API Providers

Before using the system, set up the following environment variables with your API keys:

- `ANTHROPIC_API_KEY`: Obtain from [Anthropic's documentation](https://docs.anthropic.com/).
- `OPENAI_API_KEY`: Obtain from [OpenAI's documentation](https://platform.openai.com/docs/overview).
- `GEMINI_API_KEY`: Obtain from [Google AI Studio documentation](https://ai.google.dev/tutorials/setup).
- `DEEPSEEK_API_KEY`: Obtain from [Deepseek's documentation](https://platform.deepseek.com/docs).
- `OPENROUTER_API_KEY`: Obtain from [OpenRouter's documentation](https://openrouter.ai/docs).

Default models for each provider are configurable:
- **Anthropic:** `claude-3-5-haiku-20241022`
- **OpenAI:** `gpt-4o`
- **Gemini:** `gemini-2.0-flash`
- **Deepseek:** `deepseek-chat`
- **OpenRouter:** `google/gemini-2.0-pro-exp-02-05:free`

Available models for OpenRouter:

- `cognitivecomputations/dolphin3.0-r1-mistral-24b:free`
- `cognitivecomputations/dolphin3.0-mistral-24b:free`
- `openai/o3-mini-high`
- `openai/o3-mini`
- `openai/chatgpt-4o-latest`
- `openai/gpt-4o-mini`
- `google/gemini-2.0-flash-001`
- `google/gemini-2.0-flash-thinking-exp:free`
- `google/gemini-2.0-flash-lite-preview-02-05:free`
- `google/gemini-2.0-pro-exp-02-05:free`
- `deepseek/deepseek-r1-distill-llama-70b:free`
- `deepseek/deepseek-r1-distill-qwen-32b`
- `deepseek/deepseek-r1:free`
- `qwen/qwen-plus`
- `qwen/qwen-max`
- `qwen/qwen-turbo`
- `mistralai/codestral-2501`
- `mistralai/mistral-small-24b-instruct-2501:free`
- `anthropic/claude-3.5-haiku-20241022:beta`
- `anthropic/claude-3.5-sonnet`
- `perplexity/sonar-reasoning`
- `perplexity/sonar`
- `perplexity/llama-3.1-sonar-large-128k-online`

## Execution Flow

The system operates in four main phases:

1.  **Search Phase:**  
    `arxivsearcher.py` performs a query on arXiv, downloading articles while applying retries, caching, and rate limiting.

2.  **Review Phase:**  
    `integrated_reviewer.py` processes the search results, using progress callbacks to provide ongoing feedback and logging events for debugging.

3.  **Synthesis Phase:**  
    The system synthesizes a full literature review using AI providers, ensuring academic rigor and a critical perspective.

4.  **Output Phase:**  
    The generated review and references are saved in the output directory (typically within the `reviews` folder).

A simplified diagram of the workflow:

```
[User Query]
     │
     ▼
[Search Phase (arxivsearcher.py)]
     │
     ▼
[Review Phase (integrated_reviewer.py)]
     │
     ▼
[Synthesis Phase (AI Providers)]
     │
     ▼
[Output (reviews)]
```

## Installation

### Prerequisites

- Python 3.7 or higher
- [pip](https://pip.pypa.io/en/stable/)

### Steps

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/evandeilton/arxivsearcher.git
    cd findpaper
    ```

2.  **Create a Virtual Environment (Optional but Recommended)**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Command-Line Interface (CLI)

#### arxivsearcher.py

Run directly from your terminal:

```bash
python arxivsearcher.py --query "deep learning" --max_results 5 --download --download_count 2 --save_csv
```

For detailed options, use:

```bash
python arxivsearcher.py --help
```

#### integrated_reviewer.py

Execute to search arXiv and generate a literature review:

```bash
python integrated_reviewer.py --query "large language models" --theme "recent advances" --max_results 10 --provider openrouter --output_lang pt-BR --download_pdfs
```

### Streamlit UI (reviewer_ui.py)

Start the web interface:

```bash
streamlit run reviewer_ui.py
```

### Jupyter Notebook / Python API

Example usage in a Jupyter Notebook:

```python
from arxivsearcher import run_arxiv_search
results_df = run_arxiv_search(query="quantum computing", max_results=5)
print(results_df)

from integrated_reviewer import IntegratedReviewSystem
system = IntegratedReviewSystem()
results_df, review_text, saved_files = system.search_and_review(
    query="large language models",
    theme="ethical implications",
    max_results=10,
    provider="openrouter",
    output_lang="en-US"
)
print(review_text)
```

## Dependencies

- feedparser
- pandas
- requests
- tenacity
- tqdm
- streamlit
- matplotlib
- wordcloud
- anthropic
- google-genai
- google-generativeai
- plotly
- openai

## Contributing

Contributions are welcome! Please follow standard GitHub workflows to submit issues or pull requests.

## License

This project is licensed under the [MIT License](LICENSE).

## `styles.py`

The `styles.py` file contains functions and custom CSS to enhance the appearance and behavior of the Streamlit user interface.

## `reviews` Directory

The `reviews` directory stores the output of the literature review process. It contains R Markdown files (`review_*.Rmd`) with the generated literature reviews and BibTeX files (`references_*.bib`) with the corresponding bibliographic data. Each file includes a timestamp to indicate when it was generated.
