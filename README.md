# FindPaper

A Python package for searching and downloading arXiv papers.

## Installation

```bash
pip install findpaper
```

## Usage

```python
from findpaper import search_papers

# Search for papers
papers = search_papers(query="machine learning", max_results=5)

# Download papers
for paper in papers:
    paper.download()
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
