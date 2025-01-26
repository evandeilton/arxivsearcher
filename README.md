# ArxivDownloader

ArxivSearcher is a powerful and flexible Python tool designed to search, retrieve, and download research papers from [arXiv](https://arxiv.org/). Whether you're a researcher, student, or enthusiast, ArxivSearcher simplifies the process of accessing the latest scientific papers in various fields.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
  - [Jupyter Notebook / Python API](#jupyter-notebook--python-api)
- [Functionality](#functionality)
  - [Search Articles](#search-articles)
  - [Download Papers](#download-papers)
  - [Retrieve Paper Details](#retrieve-paper-details)
  - [Save Results](#save-results)
- [Examples](#examples)
  - [CLI Example](#cli-example)
  - [Jupyter Notebook Example](#jupyter-notebook-example)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Flexible Search**: Perform advanced searches with keywords, date ranges, sorting options, and pagination.
- **Caching**: Efficiently cache search results to minimize redundant API calls.
- **PDF Downloads**: Download single or multiple PDF papers, with options for concurrency and skipping existing files.
- **Detailed Information**: Retrieve comprehensive details about specific papers using their arXiv IDs.
- **Command-Line Interface**: Run searches and downloads directly from the terminal with customizable arguments.
- **Notebook-Friendly**: Seamlessly integrate into Jupyter notebooks or Python scripts for interactive usage.
- **Result Export**: Save search results to CSV or JSON formats for easy analysis and record-keeping.
- **Robust Error Handling**: Implements rate limiting and retry logic to handle API limitations and network issues gracefully.

## Installation

### Prerequisites

- Python 3.7 or higher
- [pip](https://pip.pypa.io/en/stable/)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/arxiv-downloader.git
   cd arxiv-downloader
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *If you don't have a `requirements.txt`, here are the necessary packages:*

   ```bash
   pip install requests pandas feedparser tenacity tqdm
   ```

## Usage

ArxivSearcher can be used both as a command-line tool and as a Python library within scripts or Jupyter notebooks.

### Command-Line Interface (CLI)

Run the script directly from your terminal to perform searches and download papers.

### Jupyter Notebook / Python API

Import the `run_arxiv_search` function into your Python script or Jupyter notebook to integrate ArxivSearcher's functionality seamlessly.

## Functionality

### Search Articles

Search for articles on arXiv using keywords, with options to filter by date range, sort results, and paginate.

### Download Papers

Download PDFs of the papers found in your search. You can choose to download sequentially or concurrently, and skip existing files to save bandwidth.

### Retrieve Paper Details

Get detailed information about a specific paper using its arXiv ID, including title, authors, abstract, categories, and more.

### Save Results

Export your search results to CSV or JSON formats for further analysis or record-keeping.

## Examples

### CLI Example

Use the command-line interface to search for "deep learning" papers, download the top 5, and save the results to a CSV file.

```bash
python arxiv_downloader.py \
    --query "deep learning" \
    --max_results 5 \
    --sort_by "submittedDate" \
    --sort_order "descending" \
    --download \
    --download_count 5 \
    --concurrency 2 \
    --skip_existing \
    --output_dir "downloaded_papers" \
    --save_csv \
    --csv_file "deep_learning_papers.csv" \
    --start_date "2023-01-01" \
    --end_date "2023-12-31"
```

**Explanation of Arguments:**

- `--query`: The search terms (e.g., "deep learning").
- `--max_results`: Maximum number of search results to fetch.
- `--sort_by`: Field to sort by (`relevance`, `lastUpdatedDate`, `submittedDate`).
- `--sort_order`: Sort direction (`ascending`, `descending`).
- `--download`: Flag to download PDFs of the top results.
- `--download_count`: Number of PDFs to download.
- `--concurrency`: Number of concurrent downloads (e.g., 2 for parallel downloads).
- `--skip_existing`: Skip downloading files that already exist.
- `--output_dir`: Directory to save downloaded PDFs.
- `--save_csv`: Flag to save search results to a CSV file.
- `--csv_file`: Filename for the CSV output.
- `--start_date` & `--end_date`: Optional date range filter for submission dates.

### Jupyter Notebook Example

Use the Python API within a Jupyter notebook to search for "quantum computing" papers, download the first 3, and save the results to a JSON file.

```python
# Import the run_arxiv_search function
from arxiv_downloader import run_arxiv_search
import json

# Define search parameters
query = "quantum computing"
max_results = 10
download = True
download_count = 3
concurrency = 2
save_json = True
json_file = "quantum_computing_papers.json"

# Run the search and download process
results_df = run_arxiv_search(
    query=query,
    max_results=max_results,
    download=download,
    download_count=download_count,
    concurrency=concurrency,
    save_json=save_json,
    json_file=json_file
)

# Display the results
print(f"Found {len(results_df)} papers for query: '{query}'")
print(results_df[['title', 'authors', 'published']])
```

**Function Parameters:**

- `query` (str): Search query for arXiv articles.
- `max_results` (int): Maximum number of search results to fetch.
- `download` (bool): Whether to download PDFs of the top results.
- `download_count` (int): Number of PDFs to download if `download` is `True`.
- `concurrency` (int): Number of threads for parallel downloads (1 = sequential).
- `save_csv` (bool): Whether to save search results to a CSV file.
- `save_json` (bool): Whether to save search results to a JSON file.
- `csv_file` (str): CSV file path to save results if `save_csv` is `True`.
- `json_file` (str): JSON file path to save results if `save_json` is `True`.
- `start_date` & `end_date` (str, optional): Date range in `YYYY-MM-DD` format for submission dates.

## Configuration

ArxivSearcher offers various parameters to customize your searches and downloads. Below is a comprehensive list of arguments and their descriptions.

### Command-Line Arguments

| Argument          | Type    | Default             | Description                                                                 |
|-------------------|---------|---------------------|-----------------------------------------------------------------------------|
| `--query`         | String  | "Large Language Models" | Search query string for arXiv articles.                                     |
| `--max_results`   | Integer | 10                  | Maximum number of search results to fetch.                                  |
| `--sort_by`       | String  | "relevance"         | Sort field for results (`relevance`, `lastUpdatedDate`, `submittedDate`).  |
| `--sort_order`    | String  | "descending"        | Sort order for results (`ascending`, `descending`).                        |
| `--start`         | Integer | 0                   | Starting index for pagination.                                             |
| `--download`      | Flag    | False               | Whether to download the PDFs of the top results.                            |
| `--download_count`| Integer | 2                   | Number of PDFs to download if `--download` is set.                          |
| `--concurrency`   | Integer | 1                   | Number of concurrent downloads (1 = sequential).                           |
| `--output_dir`    | String  | "papers"            | Directory to save downloaded PDFs.                                         |
| `--skip_existing` | Flag    | False               | If set, skip downloading files that already exist locally.                |
| `--save_csv`      | Flag    | False               | If set, save search results to a CSV file.                                 |
| `--save_json`     | Flag    | False               | If set, save search results to a JSON file.                                |
| `--csv_file`      | String  | "arxiv_results.csv" | CSV file path to save results if `--save_csv` is set.                      |
| `--json_file`     | String  | "arxiv_results.json"| JSON file path to save results if `--save_json` is set.                     |
| `--start_date`    | String  | None                | Optional start date in `YYYY-MM-DD` format for submission dates.           |
| `--end_date`      | String  | None                | Optional end date in `YYYY-MM-DD` format for submission dates.             |

### Python Function Parameters (`run_arxiv_search`)

| Parameter        | Type                                     | Default             | Description                                                                 |
|------------------|------------------------------------------|---------------------|-----------------------------------------------------------------------------|
| `query`          | `str`                                    | **Required**        | Search query string for arXiv articles.                                     |
| `max_results`    | `int`                                    | `10`                | Maximum number of search results to fetch.                                  |
| `sort_by`        | `str`                                    | `"relevance"`       | Sort field for results (`relevance`, `lastUpdatedDate`, `submittedDate`).  |
| `sort_order`     | `str`                                    | `"descending"`      | Sort order for results (`ascending`, `descending`).                        |
| `start`          | `int`                                    | `0`                 | Starting index for pagination.                                             |
| `date_range`     | `Optional[Tuple[datetime, datetime]]`    | `None`              | Tuple of (`start_date`, `end_date`) for filtering by submission date.      |
| `download`       | `bool`                                   | `False`             | Whether to download the PDFs of the top results.                            |
| `download_count` | `int`                                    | `2`                 | Number of PDFs to download if `download` is `True`.                         |
| `concurrency`    | `int`                                    | `1`                 | Number of concurrent downloads (1 = sequential).                           |
| `skip_existing`  | `bool`                                   | `True`              | If `True`, skip downloading files that already exist locally.              |
| `output_dir`     | `str`                                    | `"papers"`          | Directory to save downloaded PDFs.                                         |
| `save_csv`       | `bool`                                   | `False`             | If `True`, save search results to a CSV file.                              |
| `save_json`      | `bool`                                   | `False`             | If `True`, save search results to a JSON file.                             |
| `csv_file`       | `str`                                    | `"arxiv_results.csv"` | CSV file path to save results if `save_csv` is `True`.                      |
| `json_file`      | `str`                                    | `"arxiv_results.json"`| JSON file path to save results if `save_json` is `True`.                     |

## Examples

### CLI Example

**Search for "deep learning" papers, download the top 5 PDFs, and save results to CSV:**

```bash
python arxiv_downloader.py \
    --query "deep learning" \
    --max_results 5 \
    --sort_by "submittedDate" \
    --sort_order "descending" \
    --download \
    --download_count 5 \
    --concurrency 2 \
    --skip_existing \
    --output_dir "downloaded_papers" \
    --save_csv \
    --csv_file "deep_learning_papers.csv" \
    --start_date "2023-01-01" \
    --end_date "2023-12-31"
```

**Explanation:**

- Searches for "deep learning" papers submitted between January 1, 2023, and December 31, 2023.
- Fetches the top 5 results sorted by submission date in descending order.
- Downloads the first 5 PDFs concurrently using 2 threads.
- Skips downloading PDFs that already exist in the `downloaded_papers` directory.
- Saves the search results to `deep_learning_papers.csv`.

### Jupyter Notebook Example

**Search for "quantum computing" papers, download the first 3, and save results to JSON:**

```python
# Import the run_arxiv_search function
from arxiv_downloader import run_arxiv_search
import json

# Define search parameters
query = "quantum computing"
max_results = 10
download = True
download_count = 3
concurrency = 2
save_json = True
json_file = "quantum_computing_papers.json"

# Run the search and download process
results_df = run_arxiv_search(
    query=query,
    max_results=max_results,
    download=download,
    download_count=download_count,
    concurrency=concurrency,
    save_json=save_json,
    json_file=json_file
)

# Display the results
print(f"Found {len(results_df)} papers for query: '{query}'")
print(results_df[['title', 'authors', 'published']])
```

**Output:**

```
Found 10 papers for query: 'quantum computing'
```

*(Followed by a table of titles, authors, and publication dates.)*

## Configuration

### Rate Limiting

ArxivSearcher implements a rate limiting mechanism to respect arXiv's API usage policies. By default, it ensures at least 3 seconds between API requests. You can adjust this by modifying the `RATE_LIMIT_DELAY` in the `ArxivConfig` class if necessary.

### Caching

Search results are cached to improve performance and reduce redundant API calls. The cache time-to-live (TTL) is set to 1 hour by default (`CACHE_TTL = 3600` seconds). This can be adjusted in the `ArxivConfig` class.

## Dependencies

ArxivSearcher relies on the following Python packages:

- [requests](https://pypi.org/project/requests/): For making HTTP requests.
- [pandas](https://pandas.pydata.org/): For handling and manipulating search results.
- [feedparser](https://pypi.org/project/feedparser/): For parsing arXiv's Atom feeds.
- [tenacity](https://pypi.org/project/tenacity/): For implementing retry logic.
- [tqdm](https://pypi.org/project/tqdm/): For displaying progress bars.
- [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html): For handling concurrency (part of Python's standard library).

*Install all dependencies using:*

```bash
pip install requests pandas feedparser tenacity tqdm
```

## Contributing

Contributions are welcome! Whether it's reporting a bug, suggesting an enhancement, or submitting a pull request, your input helps improve ArxivSearcher.

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

Please ensure that your code follows the existing style and includes appropriate documentation.

## License

This project is licensed under the [MIT License](LICENSE).

---

*Happy Researching!*