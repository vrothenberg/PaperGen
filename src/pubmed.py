import asyncio
import aiohttp
import random
from typing import List, Dict
from rich import print as rprint
from xml.etree import ElementTree as ET
from asyncio import Lock
import time

class PubMedAPI:
    _lock = Lock()
    _last_request_time = time.monotonic()
    _global_delay = 1.0  # Minimum time (seconds) between requests globally


    def __init__(self, api_key: str, sleep_time: float = 2.0, max_retries: int = 10):
        """
        Initialize the PubMedAPI class.

        Args:
            api_key (str): Your PubMed API key.
            sleep_time (float): Time to wait between API requests to avoid rate-limiting.
            max_retries (int): Maximum number of retries for failed requests.
        """
        self.api_key = api_key
        self.sleep_time = sleep_time
        self.max_retries = max_retries
        self.search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    async def query(self, queries: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """
        Query the PubMed API using a list of queries and fetch detailed information.

        Args:
            queries (List[Dict[str, str]]): List of queries with sections and query text.

        Returns:
            Dict[str, List[Dict[str, str]]]: Results for each section with detailed metadata.
        """
        results = {}

        # Create a session for the entire query process
        async with aiohttp.ClientSession() as session:
            for query in queries:
                section = query["section"]
                search_query = query["query"]

                search_params = {
                    "db": "pubmed",
                    "term": search_query,
                    "retmax": 3,  # Limit to 3
                    "api_key": self.api_key,
                    "retmode": "json",
                }

                # Remove None values from params
                search_params = {k: v for k, v in search_params.items() if v is not None}

                # Perform the search with exponential backoff
                search_data = await self._request_with_backoff(
                    session=session,
                    method=session.get,
                    url=self.search_url,
                    params=search_params,
                )

                if search_data:
                    pmids = search_data.get("esearchresult", {}).get("idlist", [])
                    if pmids:
                        # Fetch detailed metadata for the PMIDs
                        fetch_params = {
                            "db": "pubmed",
                            "id": ",".join(pmids),
                            "retmode": "xml",
                            "api_key": self.api_key,
                        }

                        # Remove None values from params
                        fetch_params = {k: v for k, v in fetch_params.items() if v is not None}

                        fetch_data = await self._request_with_backoff(
                            session=session,
                            method=session.get,
                            url=self.fetch_url,
                            params=fetch_params,
                        )

                        if fetch_data:
                            results[section] = self._parse_response(fetch_data, search_query)
                        else:
                            results[section] = []
                    else:
                        results[section] = []
                else:
                    results[section] = []

        return results

    async def _request_with_backoff(self, session, method, url, **kwargs):
            """
            Perform an HTTP request with exponential backoff and global rate limit.
            """
            for attempt in range(self.max_retries):
                try:
                    # Ensure global throttling
                    async with self._lock:
                        now = time.monotonic()
                        elapsed = now - self._last_request_time
                        if elapsed < self._global_delay:
                            await asyncio.sleep(self._global_delay - elapsed)
                        self._last_request_time = time.monotonic()

                    # Send the request
                    async with method(url, **kwargs) as response:
                        if response.status == 200:
                            if "json" in kwargs.get("params", {}).get("retmode", ""):
                                return await response.json()
                            return await response.text()
                        elif response.status == 429:  # Handle rate limit
                            rprint(f"[yellow]Rate limit hit: {url}[/yellow]")

                except aiohttp.ClientError as e:
                    rprint(f"[red]Client error: {e}[/red]")

                # Exponential backoff with jitter
                delay = (2 ** attempt) + random.uniform(0, 1)
                rprint(f"[cyan]Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{self.max_retries})[/cyan]")
                await asyncio.sleep(delay)

            rprint(f"[red]Max retries reached for {url}. Skipping request.[/red]")
            return None

    def _parse_response(self, xml_response: str, search_query: str) -> List[Dict[str, str]]:
        """
        Parse the PubMed XML response to extract relevant details.

        Args:
            xml_response (str): XML response from PubMed efetch.
            search_query (str): The search query associated with the response.

        Returns:
            List[Dict[str, str]]: Parsed results with title, abstract, authors, DOI, URLs, and other metadata.
        """
        root = ET.fromstring(xml_response)
        articles = []

        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID")
            title = article.findtext(".//ArticleTitle")
            abstract = article.findtext(".//Abstract/AbstractText")

            # Extract authors
            authors = [
                f"{author.findtext('LastName')} {author.findtext('ForeName')}"
                for author in article.findall(".//Author")
                if author.findtext("LastName") and author.findtext("ForeName")
            ]

            # Extract journal info
            journal = article.findtext(".//Journal/Title")
            publication_date = article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate")

            # Extract DOI
            doi = None
            for id_elem in article.findall(".//ArticleId"):
                if id_elem.attrib.get("IdType") == "doi":
                    doi = id_elem.text
                    break

            # Construct the URL from PMID
            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            # Add the paper details
            articles.append({
                "query": search_query,
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "journal": journal,
                "publication_date": publication_date,
                "doi": doi,
                "pubmed_url": pubmed_url,
            })

        return articles


    def format_citation(self, paper: Dict) -> str:
        """
        Format the citation for a paper using available metadata from PubMed.

        Args:
            paper (Dict): The paper metadata returned by PubMed.

        Returns:
            str: Formatted citation with URL or DOI.
        """
        authors = ", ".join(paper.get("authors", [])[:3])
        if len(paper.get("authors", [])) > 3:
            authors += " et al."

        title = paper.get("title", "Unknown Title")
        year = paper.get("publication_date", "Unknown Year")
        journal = paper.get("journal", "Unknown Journal")
        doi = paper.get("doi", None)
        pubmed_url = paper.get("pubmed_url", None)

        citation = f"{authors}. \"{title}\" ({year}). Published in {journal}."

        if doi:
            citation += f" DOI: {doi}."
        if pubmed_url:
            citation += f" Available at: {pubmed_url}."

        return citation

    def format_results(self, results: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """
        Format PubMed results into JSON format suitable for LLM input.

        Args:
            results (Dict[str, List[Dict[str, str]]]): The detailed results from PubMed.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing abstracts and citations for LLM input.
        """
        formatted_results = []

        for section, papers in results.items():
            for paper in papers:
                abstract = paper.get("abstract")
                if not abstract:
                    continue

                citation = self.format_citation(paper)
                formatted_results.append({
                    "section": section,
                    "query": paper.get("query", "Missing Query"),
                    "title": paper.get("title", "Missing Title"),
                    "abstract": abstract,
                    "citation": citation,
                })

        return formatted_results
