import asyncio
import aiohttp
import random
from typing import List, Dict
from rich import print as rprint


class SemanticScholarAPI:
    _lock = asyncio.Lock()  # Global lock to enforce rate limits
    _last_request_time = asyncio.get_event_loop().time()
    _global_delay = 1.0  # Minimum time (in seconds) between requests globally

    def __init__(self, api_key: str, sleep_time: float = 2.0, max_retries: int = 10):
        """
        Initialize the SemanticScholarAPI class.

        Args:
            api_key (str): Your Semantic Scholar API key.
            sleep_time (float): Time to wait between API requests to avoid rate-limiting.
            max_retries (int): Maximum number of retries for failed requests.
        """
        self.api_key = api_key
        self.sleep_time = sleep_time
        self.max_retries = max_retries
        self.search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.batch_url = "https://api.semanticscholar.org/graph/v1/paper/batch"
        self.headers = {"x-api-key": self.api_key}

    async def query(self, queries: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """
        Query Semantic Scholar with search queries and fetch detailed information in batches.

        Args:
            queries (List[Dict[str, str]]): List of queries with sections and query text.

        Returns:
            Dict[str, List[Dict[str, str]]]: Results for each section with detailed paper information.
        """
        results = {}

        async with aiohttp.ClientSession(headers=self.headers) as session:
            for query in queries:
                section = query["section"]
                search_query = query["query"]
                search_params = {
                    "query": search_query,
                    "limit": 3,  # Adjust the limit based on your requirements
                }

                # Perform asynchronous GET request with retries
                search_data = await self._request_with_backoff(
                    session=session,
                    method=session.get,
                    url=self.search_url,
                    params=search_params
                )

                if search_data:
                    paper_ids = [paper["paperId"] for paper in search_data.get("data", [])]
                    if paper_ids:
                        details = await self._query_batch(session, paper_ids, search_query)
                        results[section] = details
                    else:
                        results[section] = []
                else:
                    results[section] = []

        return results

    async def _query_batch(self, session: aiohttp.ClientSession, paper_ids: List[str], search_query: str) -> List[Dict[str, str]]:
        """
        Query the Semantic Scholar batch endpoint for detailed information.

        Args:
            session (aiohttp.ClientSession): The active HTTP session.
            paper_ids (List[str]): List of paper IDs to query.

        Returns:
            List[Dict[str, str]]: A list of paper details with the requested fields.
        """
        fields = "title,abstract,authors,citationCount,referenceCount,url,venue,publicationVenue,year,openAccessPdf"
        payload = {"ids": paper_ids}
        params = {"fields": fields}

        response_data = await self._request_with_backoff(
            session=session,
            method=session.post,
            url=self.batch_url,
            params=params,
            json=payload
        )

        if response_data:
            for paper in response_data:
                paper["query"] = search_query

        return response_data if response_data else []

    async def _request_with_backoff(self, session: aiohttp.ClientSession, method, url, **kwargs) -> dict:
        """
        Perform an HTTP request with exponential backoff and global rate limiting.

        Args:
            session (aiohttp.ClientSession): The active HTTP session.
            method (callable): The HTTP method (e.g., `session.get`, `session.post`).
            url (str): The URL for the request.
            **kwargs: Additional parameters for the request.

        Returns:
            dict: The JSON response data or None if the request ultimately fails.
        """
        for attempt in range(self.max_retries):
            try:
                # Enforce global rate limiting
                async with self._lock:
                    now = asyncio.get_event_loop().time()
                    elapsed = now - self._last_request_time
                    if elapsed < self._global_delay:
                        await asyncio.sleep(self._global_delay - elapsed)
                    self._last_request_time = asyncio.get_event_loop().time()

                # Perform the request
                async with method(url, **kwargs) as response:
                    if response.status == 200:
                        return await response.json()
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

    def format_citation(self, paper: Dict) -> str:
        """
        Format the citation for a paper using available metadata from Semantic Scholar.

        Args:
            paper (Dict): The paper metadata returned by Semantic Scholar.

        Returns:
            str: Formatted citation with URL or DOI.
        """
        authors = ", ".join(author.get("name", "Unknown") for author in paper.get("authors", [])[:3])
        if len(paper.get("authors", [])) > 3:
            authors += " et al."

        title = paper.get("title", "Unknown Title")
        year = paper.get("year", "Unknown Year")
        venue = paper.get("venue", "Unknown Venue")

        publication_venue = paper.get("publicationVenue") or {}
        publication_name = publication_venue.get("name", venue)
        publication_url = publication_venue.get("url", "")

        doi = paper.get("externalIds", {}).get("DOI", None)
        open_access_pdf = paper.get("openAccessPdf") or {}
        pdf_url = open_access_pdf.get("url", None)
        general_url = paper.get("url", None)

        citation = f"{authors}. \"{title}\" ({year}). Published in {publication_name}."

        if doi:
            citation += f" DOI: {doi}."
        elif pdf_url:
            citation += f" Open Access PDF: {pdf_url}."
        elif general_url:
            citation += f" Available at: {general_url}."

        if publication_url:
            citation += f" Publication Info: {publication_url}."

        return citation

    def format_results(self, results: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """
        Format Semantic Scholar results into JSON format suitable for LLM input.

        Args:
            results (Dict[str, List[Dict[str, str]]]): The detailed results from Semantic Scholar.

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
