import argparse
import urllib.request
import feedparser
import time
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
import requests
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json
from functools import lru_cache
from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArxivConfig:
    """
    Configuration settings for ArxivDownloader.
    """
    BASE_URL: str = 'http://export.arxiv.org/api/query?'
    SEARCH_BASE: str = 'search_query={}&start={}&max_results={}'
    DEFAULT_OUTPUT_DIR: str = 'papers'
    RATE_LIMIT_DELAY: int = 3      # seconds
    MAX_RETRIES: int = 3
    CACHE_TTL: int = 3600          # 1 hour
    BATCH_SIZE: int = 10


class ArxivDownloader:
    """
    A class for searching and downloading papers from arXiv.

    Main features:
      - Searching arXiv by query and optional date range
      - Caching search results
      - Downloading PDF files (sequentially or in parallel)
      - Retrieving detailed paper information by arXiv ID

    Attributes:
        base_url (str): Base URL for the arXiv API.
        search_base (str): Base query string for searches.
        cache_ttl (int): Time-to-live (in seconds) for cached results.
        session (requests.Session): Reusable session for improved performance.
    """

    def __init__(self, cache_ttl: int = ArxivConfig.CACHE_TTL):
        """
        Initialize the ArxivDownloader.

        Args:
            cache_ttl (int): Time-to-live for cached search results (in seconds).
        """
        self.base_url: str = ArxivConfig.BASE_URL
        self.search_base: str = ArxivConfig.SEARCH_BASE
        self.cache_ttl: int = cache_ttl
        self._last_request_time: float = 0.0

        # Reusable session for improved performance
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ArxivDownloader/1.0 (+https://github.com/your-repo)"
        })

    def _rate_limit(self) -> None:
        """
        Implement rate limiting to avoid overwhelming the arXiv API.
        Ensures at least ArxivConfig.RATE_LIMIT_DELAY seconds between requests.
        """
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time

        if time_since_last_request < ArxivConfig.RATE_LIMIT_DELAY:
            time.sleep(ArxivConfig.RATE_LIMIT_DELAY - time_since_last_request)

        self._last_request_time = time.time()

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(ArxivConfig.MAX_RETRIES)
    )
    def _make_request(self, url: str) -> str:
        """
        Make an HTTP request with retry logic.

        Args:
            url (str): The URL to request.

        Returns:
            str: Response content from the requested URL.

        Raises:
            requests.exceptions.RequestException: If the request fails after retries.
        """
        self._rate_limit()
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to {url}: {str(e)}")
            raise

    def _rate_limited_download(
        self,
        pdf_url: str,
        output_path: Path
    ) -> None:
        """
        Internal helper to download a single PDF file with rate limiting.

        Args:
            pdf_url (str): Direct URL to the PDF file.
            output_path (Path): Target path where the PDF will be saved.

        Raises:
            requests.exceptions.RequestException: If the download fails.
        """
        # Each download request also respects the rate limit
        self._rate_limit()

        response = self.session.get(pdf_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192

        with open(output_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=f"Downloading {output_path.name}"
        ) as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))

    @lru_cache(maxsize=100)
    def search_articles(
        self,
        query: str,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = 'relevance',
        sort_order: str = 'descending',
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> pd.DataFrame:
        """
        Search for articles on arXiv with advanced filtering options.

        Args:
            query (str): Search terms (e.g. 'machine learning').
            max_results (int): Maximum number of results to return.
            start (int): Starting index for pagination.
            sort_by (str): Field to sort by ('relevance', 'lastUpdatedDate', 'submittedDate').
            sort_order (str): Sort direction ('ascending' or 'descending').
            date_range (Tuple[datetime, datetime], optional): Start/end datetime objects for submission date range.

        Returns:
            pd.DataFrame: DataFrame containing search results with columns:
                - 'title'
                - 'authors'
                - 'published'
                - 'updated'
                - 'summary'
                - 'pdf_url'
                - 'arxiv_id'
                - 'categories'
                - 'doi'
                - 'journal_ref'
                - 'comment'

        Raises:
            ValueError: If invalid sort options or date range is provided.
            requests.exceptions.RequestException: If the API request fails.
        """
        valid_sort_fields = {'relevance', 'lastUpdatedDate', 'submittedDate'}
        if sort_by not in valid_sort_fields:
            raise ValueError(f"sort_by must be one of {valid_sort_fields}")

        search_query = [f"all:{urllib.parse.quote(query)}"]

        # If the user provided a date range, append it to the search query
        if date_range:
            start_date, end_date = date_range
            if not (isinstance(start_date, datetime) and isinstance(end_date, datetime)):
                raise ValueError("date_range must contain datetime objects")
            search_query.append(
                f"submittedDate:[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"
            )

        final_query = " AND ".join(search_query)
        url = (
            f"{self.base_url}"
            f"search_query={final_query}&start={start}&max_results={max_results}"
            f"&sortBy={sort_by}&sortOrder={sort_order}"
        )

        try:
            response_text = self._make_request(url)
            feed = feedparser.parse(response_text)

            results = []
            for entry in feed.entries:
                paper = {
                    'title': entry.title,
                    'authors': ', '.join(author.name for author in entry.authors),
                    'published': entry.published,
                    'updated': entry.updated,
                    'summary': entry.summary,
                    'pdf_url': next(link.href for link in entry.links if link.type == 'application/pdf'),
                    'arxiv_id': entry.id.split('/abs/')[-1],
                    'categories': [tag.term for tag in entry.tags] if hasattr(entry, 'tags') else [],
                    'doi': entry.get('arxiv_doi', None),
                    'journal_ref': entry.get('arxiv_journal_ref', None),
                    'comment': entry.get('arxiv_comment', None)
                }
                results.append(paper)

            return pd.DataFrame(results)
        except Exception as e:
            logger.error(f"Error searching articles: {str(e)}")
            raise

    def download_papers(
        self,
        pdf_urls: Union[str, List[str]],
        output_dir: str = ArxivConfig.DEFAULT_OUTPUT_DIR,
        filenames: Optional[Union[str, List[str]]] = None,
        skip_existing: bool = False,
        concurrency: int = 1
    ) -> List[str]:
        """
        Download one or multiple papers from arXiv.

        Args:
            pdf_urls (Union[str, List[str]]): Single PDF URL or a list of URLs.
            output_dir (str): Directory to save PDFs (default: 'papers').
            filenames (Union[str, List[str]], optional): Custom filename(s) for each URL.
                If a list, must match the number of PDF URLs.
            skip_existing (bool): If True, skip downloading if the file already exists.
            concurrency (int): Number of concurrent downloads (default: 1).
                - 1 = sequential (original behavior)
                - >1 = parallel downloads (still rate-limited to avoid overwhelming the API)

        Returns:
            List[str]: Paths to successfully downloaded files.

        Raises:
            ValueError: If the length of filenames doesn't match pdf_urls.
            requests.exceptions.RequestException: If download fails.
        """
        if isinstance(pdf_urls, str):
            pdf_urls = [pdf_urls]
        if isinstance(filenames, str):
            filenames = [filenames]

        if filenames and len(filenames) != len(pdf_urls):
            raise ValueError("Number of filenames must match number of URLs")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        download_jobs = []
        for i, pdf_url in enumerate(pdf_urls):
            filename = filenames[i] if filenames else pdf_url.split('/')[-1]
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            output_path = Path(output_dir) / filename
            download_jobs.append((pdf_url, output_path))

        downloaded_files = []

        # Parallel downloads if concurrency > 1
        if concurrency > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {}
                for pdf_url, output_path in download_jobs:
                    # Skip if file exists
                    if skip_existing and output_path.exists():
                        logger.info(f"Skipping existing file: {output_path}")
                        downloaded_files.append(str(output_path))
                        continue
                    # Schedule a download
                    futures[executor.submit(self._rate_limited_download, pdf_url, output_path)] = (pdf_url, output_path)

                # Collect results
                for future in as_completed(futures):
                    pdf_url, output_path = futures[future]
                    try:
                        future.result()
                        downloaded_files.append(str(output_path))
                        logger.info(f"Successfully downloaded {output_path.name}")
                    except Exception as e:
                        logger.error(f"Error downloading {output_path.name}: {str(e)}")
                        raise
        else:
            # Sequential downloads (original behavior)
            for pdf_url, output_path in tqdm(download_jobs, desc="Downloading papers"):
                if skip_existing and output_path.exists():
                    logger.info(f"Skipping existing file: {output_path}")
                    downloaded_files.append(str(output_path))
                    continue

                try:
                    self._rate_limited_download(pdf_url, output_path)
                    downloaded_files.append(str(output_path))
                    logger.info(f"Successfully downloaded {output_path.name}")
                except Exception as e:
                    logger.error(f"Error downloading {output_path.name}: {str(e)}")
                    raise

        return downloaded_files

    def get_paper_details(self, arxiv_id: str) -> Dict:
        """
        Get detailed information about a specific paper by arXiv ID.

        Args:
            arxiv_id (str): A valid arXiv identifier (e.g., '2104.00001').

        Returns:
            Dict: Dictionary of paper details:
                - 'title'
                - 'authors'
                - 'published'
                - 'updated'
                - 'summary'
                - 'pdf_url'
                - 'categories'
                - 'doi'
                - 'journal_ref'
                - 'comment'
                - 'primary_category'

        Raises:
            ValueError: If no paper is found for the given arXiv ID.
            requests.exceptions.RequestException: If the API request fails.
        """
        url = f"{self.base_url}id_list={arxiv_id}"

        try:
            response_text = self._make_request(url)
            feed = feedparser.parse(response_text)

            if not feed.entries:
                raise ValueError(f"Paper not found: {arxiv_id}")

            entry = feed.entries[0]
            return {
                'title': entry.title,
                'authors': [author.name for author in entry.authors],
                'published': entry.published,
                'updated': entry.updated,
                'summary': entry.summary,
                'pdf_url': next(link.href for link in entry.links if link.type == 'application/pdf'),
                'categories': [tag.term for tag in entry.tags] if hasattr(entry, 'tags') else [],
                'doi': entry.get('arxiv_doi', None),
                'journal_ref': entry.get('arxiv_journal_ref', None),
                'comment': entry.get('arxiv_comment', None),
                'primary_category': entry.arxiv_primary_category['term'] if hasattr(entry, 'arxiv_primary_category') else None
            }
        except Exception as e:
            logger.error(f"Error getting paper details: {str(e)}")
            raise

    @staticmethod
    def save_results_to_csv(results: pd.DataFrame, file_path: str) -> None:
        """
        Save search results DataFrame to a CSV file.

        Args:
            results (pd.DataFrame): DataFrame containing the search results.
            file_path (str): Path to the CSV file.

        Raises:
            IOError: If unable to write the CSV file.
        """
        try:
            results.to_csv(file_path, index=False)
            logger.info(f"Search results saved to CSV: {file_path}")
        except Exception as e:
            logger.error(f"Error saving results to CSV: {str(e)}")
            raise IOError(f"Failed to save results to CSV: {e}")

    @staticmethod
    def save_results_to_json(results: pd.DataFrame, file_path: str) -> None:
        """
        Save search results DataFrame to a JSON file.

        Args:
            results (pd.DataFrame): DataFrame containing the search results.
            file_path (str): Path to the JSON file.

        Raises:
            IOError: If unable to write the JSON file.
        """
        try:
            results.to_json(file_path, orient='records', lines=True, force_ascii=False)
            logger.info(f"Search results saved to JSON: {file_path}")
        except Exception as e:
            logger.error(f"Error saving results to JSON: {str(e)}")
            raise IOError(f"Failed to save results to JSON: {e}")


def run_arxiv_search(
    query: str,
    max_results: int = 10,
    sort_by: str = 'relevance',
    sort_order: str = 'descending',
    start: int = 0,
    date_range: Optional[Tuple[datetime, datetime]] = None,
    download: bool = False,
    download_count: int = 2,
    concurrency: int = 1,
    skip_existing: bool = True,
    output_dir: str = ArxivConfig.DEFAULT_OUTPUT_DIR,
    save_csv: bool = False,
    save_json: bool = False,
    csv_file: str = "arxiv_results.csv",
    json_file: str = "arxiv_results.json"
) -> pd.DataFrame:
    """
    Friendly interface for Jupyter notebooks and Python scripts to search, optionally download, 
    and optionally save results.

    Args:
        query (str): Search query for arXiv (e.g., "large language models").
        max_results (int): Maximum number of results to return.
        sort_by (str): Field to sort by ('relevance', 'lastUpdatedDate', 'submittedDate').
        sort_order (str): Sort direction ('ascending', 'descending').
        start (int): Starting index for pagination.
        date_range (Optional[Tuple[datetime, datetime]]): A tuple of (start_date, end_date) for filtering by submission date.
        download (bool): If True, download the first `download_count` PDFs.
        download_count (int): Number of PDFs to download if `download` is True.
        concurrency (int): Number of threads for parallel downloads (1 = sequential).
        skip_existing (bool): If True, skip re-downloading files that already exist.
        output_dir (str): Directory to save PDF files.
        save_csv (bool): If True, save search results to a CSV file.
        save_json (bool): If True, save search results to a JSON file.
        csv_file (str): CSV file path to save results if `save_csv` is True.
        json_file (str): JSON file path to save results if `save_json` is True.

    Returns:
        pd.DataFrame: DataFrame containing the search results.

    Example:
        >>> from your_module import run_arxiv_search
        >>> df = run_arxiv_search(query="quantum computing", max_results=5, download=True, download_count=2)
    """
    downloader = ArxivDownloader()
    results = downloader.search_articles(
        query=query,
        max_results=max_results,
        start=start,
        sort_by=sort_by,
        sort_order=sort_order,
        date_range=date_range
    )

    logger.info(f"Found {len(results)} articles for query: {query}")

    # Download PDF files if requested
    if download and not results.empty:
        pdf_urls = results['pdf_url'].tolist()[:download_count]
        downloaded_files = downloader.download_papers(
            pdf_urls,
            output_dir=output_dir,
            skip_existing=skip_existing,
            concurrency=concurrency
        )
        logger.info(f"Downloaded files: {downloaded_files}")

    # Save results to CSV or JSON if requested
    if save_csv:
        downloader.save_results_to_csv(results, csv_file)
    if save_json:
        downloader.save_results_to_json(results, json_file)

    return results


def parse_args():
    """
    Parse command-line arguments for the ArxivDownloader script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Command-line interface for searching and downloading arXiv papers."
    )
    parser.add_argument("--query", type=str, default="Large Language Models",
                        help="Search query string for arXiv articles.")
    parser.add_argument("--max_results", type=int, default=10,
                        help="Maximum number of search results to fetch.")
    parser.add_argument("--sort_by", type=str, default="relevance",
                        help="Sort field for results (relevance, lastUpdatedDate, submittedDate).")
    parser.add_argument("--sort_order", type=str, default="descending",
                        help="Sort order for results (ascending, descending).")
    parser.add_argument("--start", type=int, default=0,
                        help="Starting index for pagination.")
    parser.add_argument("--download", action="store_true",
                        help="Whether to download the PDF of the top results.")
    parser.add_argument("--download_count", type=int, default=2,
                        help="Number of PDFs to download if --download is set.")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent downloads (1 = sequential).")
    parser.add_argument("--output_dir", type=str, default="papers",
                        help="Directory to save downloaded PDFs.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="If set, skip re-downloading existing PDFs.")
    parser.add_argument("--save_csv", action="store_true",
                        help="If set, save results to CSV.")
    parser.add_argument("--save_json", action="store_true",
                        help="If set, save results to JSON.")
    parser.add_argument("--csv_file", type=str, default="arxiv_results.csv",
                        help="CSV file path (used if --save_csv is set).")
    parser.add_argument("--json_file", type=str, default="arxiv_results.json",
                        help="JSON file path (used if --save_json is set).")
    parser.add_argument("--start_date", type=str, default=None,
                        help="Optional start date in YYYY-MM-DD format.")
    parser.add_argument("--end_date", type=str, default=None,
                        help="Optional end date in YYYY-MM-DD format.")

    return parser.parse_args()


def main():
    """
    Command-line entry point for the ArxivDownloader script.

    Example usage from the command line:
        python your_script.py --query "deep learning" --max_results 5 --download --download_count 2 \
            --save_csv --csv_file "my_results.csv"
    """
    args = parse_args()

    # Convert start_date and end_date if provided
    date_range = None
    if args.start_date and args.end_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
            date_range = (start_date, end_date)
        except ValueError:
            logger.error("Invalid date format. Please use YYYY-MM-DD.")
            raise

    # Run the search (and optionally download + save results)
    run_arxiv_search(
        query=args.query,
        max_results=args.max_results,
        sort_by=args.sort_by,
        sort_order=args.sort_order,
        start=args.start,
        date_range=date_range,
        download=args.download,
        download_count=args.download_count,
        concurrency=args.concurrency,
        skip_existing=args.skip_existing,
        output_dir=args.output_dir,
        save_csv=args.save_csv,
        save_json=args.save_json,
        csv_file=args.csv_file,
        json_file=args.json_file
    )


if __name__ == "__main__":
    main()
