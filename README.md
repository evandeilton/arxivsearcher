# Integrated Literature Review System

This project provides an integrated system for searching academic papers on arXiv, generating literature reviews using AI, and interacting with the system through a user-friendly interface. It combines the functionality of `arxivsearcher.py`, `integrated_reviewer.py`, and `reviewer_ui.py` to streamline the research process.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
    - [arxivsearcher.py](#arxivsearcherpy)
    - [integrated_reviewer.py](#integrated_reviewerpy)
  - [Streamlit UI (reviewer_ui.py)](#streamlit-ui-reviewer_uipy)
  - [Jupyter Notebook / Python API](#jupyter-notebook--python-api)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Features

-   **ArXiv Search:** Search for papers on arXiv using keywords, date ranges, and sorting options.
-   **AI-Powered Review Generation:** Generate literature reviews using various AI providers (Anthropic, OpenAI, Gemini, Deepseek).
-   **Streamlit User Interface:** Interact with the system through a web-based UI.
-   **PDF Downloads:** Download PDFs of selected papers.
-   **Result Export:** Save search results and generated reviews.
-   **Caching:** Cache search results to improve performance.
-   **Rate Limiting:** Respect arXiv's API usage policies.

## Installation

### Prerequisites

-   Python 3.7 or higher
-   [pip](https://pip.pypa.io/en/stable/)

### Steps

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/evandeilton/arxivsearcher.git
    cd arxivsearcher
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
    **Additional Dependencies:**
    ```bash
    pip install anthropic google-genai
    ```

## Usage

### Command-Line Interface (CLI)

#### arxivsearcher.py

Run `arxivsearcher.py` directly from your terminal to perform searches and download papers.

**Example:**

```bash
python arxivsearcher.py --query "deep learning" --max_results 5 --download --download_count 2 --save_csv
```

Use `python arxivsearcher.py --help` to see all available options.

#### integrated_reviewer.py

Run `integrated_reviewer.py` to search arXiv and generate a literature review.

**Example:**

```bash
python integrated_reviewer.py --query "large language models" --theme "recent advances" --max_results 10 --provider anthropic --output_lang pt-BR --download_pdfs
```
Use `python integrated_reviewer.py --help` to see all available options. You will need to set API keys as environment variables for the providers you want to use (e.g., `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`).

### Streamlit UI (reviewer_ui.py)

Run `reviewer_ui.py` to start the Streamlit application:

```bash
streamlit run reviewer_ui.py
```

This will open a web interface in your browser where you can interact with the system. You will need to set API keys as environment variables for the providers you want to use (e.g., `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`).

### Jupyter Notebook / Python API

You can also import and use the functions from `arxivsearcher.py` and `integrated_reviewer.py` directly in your Python scripts or Jupyter notebooks.

**Example (arxivsearcher.py):**

```python
from arxivsearcher import run_arxiv_search

results_df = run_arxiv_search(query="quantum computing", max_results=5)
print(results_df)
```

**Example (integrated_reviewer.py):**

```python
from integrated_reviewer import IntegratedReviewSystem
from datetime import datetime

system = IntegratedReviewSystem()
results_df, review_text, saved_files = system.search_and_review(
    query="large language models",
    theme="ethical implications",
    max_results=10,
    provider="anthropic",
    output_lang="en-US",
    date_range=(datetime(2023, 1, 1), datetime(2024, 1, 1))
)

print(review_text)
```

## Configuration

-   **Rate Limiting:** The `ArxivDownloader` class in `arxivsearcher.py` implements rate limiting to avoid overwhelming the arXiv API.
-   **Caching:** Search results are cached to improve performance.
-   **API Keys:** You need to set environment variables for the AI providers you want to use in `integrated_reviewer.py` and `reviewer_ui.py`.  For example:
    -   `ANTHROPIC_API_KEY`
    -   `OPENAI_API_KEY`
    -   `GEMINI_API_KEY`
    -   `DEEPSEEK_API_KEY`

## Dependencies

-   requests
-   pandas
-   feedparser
-   tenacity
-   tqdm
-   streamlit
-   matplotlib
-   networkx
-   wordcloud
-   anthropic
-   google-genai
-   plotly

## License

This project is licensed under the [MIT License](LICENSE).
