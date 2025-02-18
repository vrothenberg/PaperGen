# src/semanticscholar.py
import random
import asyncio
import aiohttp
import logging
import pandas as pd
from typing import List, Dict, Optional
from rich import print as rprint
from pydantic import ValidationError
from collections import defaultdict

from src.models import Paper


class SemanticScholarAPI:
    """
    A client to interact with Semantic Scholar's API, featuring:
      - Asynchronous querying with rate limiting.
      - Bulk (batch) fetching of detailed paper data.
      - Optional filtering based on SJR scores from a local CSV.
    """

    _lock = asyncio.Lock()  # Global lock to enforce rate limits
    _last_request_time = asyncio.get_event_loop().time()
    _global_delay = 1.0  # Minimum time (in seconds) between requests globally

    def __init__(
        self,
        api_key: str,
        sleep_time: float = 2.0,
        max_retries: int = 10,
        sjr_threshold: float = 1.0,
        min_citation_count: int = 50,
        logger: logging.Logger = None,
    ):
        """
        Initialize the SemanticScholarAPI class.

        Args:
            api_key (str): Your Semantic Scholar API key.
            sleep_time (float): Time to wait between API requests to avoid rate-limiting.
            max_retries (int): Maximum number of retries for failed requests.
            sjr_threshold (float): Minimum SJR score required to keep a paper.
        """
        self.api_key = api_key
        self.sleep_time = sleep_time
        self.max_retries = max_retries
        self.sjr_threshold = sjr_threshold
        self.min_citation_count = min_citation_count
        self.logger = logger

        # Endpoints
        self.search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.batch_url = "https://api.semanticscholar.org/graph/v1/paper/batch"
        self.headers = {"x-api-key": self.api_key}

        # Internal dictionary for ISSN -> { "sjr": float, "h_index": float }
        self._sjr_map: Dict[str, Dict[str, Optional[float]]] = {}

    def load_journal_sjr_data(self, csv_path: str) -> None:
        """
        Load a CSV file of journals and parse out SJR/H-Index info keyed by ISSN.

        Expected columns in CSV:
          - 'Issn1' and/or 'Issn2' for ISSNs
          - 'SJR'
          - 'H index' (optional, only if you want to store H-index as well)

        Args:
            csv_path (str): Path to the CSV file containing journal data.
        """
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            sjr = row.get("SJR")
            h_index = row.get("H index")

            # Some rows may have multiple ISSNs (e.g., "15424863, 00079235"),
            # so handle splitting if your data is structured that way.
            for col in ["Issn1", "Issn2"]:
                issn_vals = row.get(col, "")
                if isinstance(issn_vals, str):
                    for issn_raw in issn_vals.split(","):
                        issn_clean = issn_raw.replace("-", "").strip()
                        if issn_clean and pd.notnull(sjr):
                            self._sjr_map[issn_clean] = {
                                "sjr": float(sjr),
                                "h_index": float(h_index) if pd.notnull(h_index) else None,
                            }

    async def query(self, index, queries: List[Dict[str, str]]) -> Dict[str, List[Paper]]:
        """
        Query Semantic Scholar with search queries and fetch detailed information in batches.

        Args:
            queries (List[Dict[str, str]]): List of queries with sections, query text, and rationale.

        Returns:
            Dict[str, List[Paper]]: Results for each section with detailed paper information.
        """
        results: Dict[str, List[Paper]] = {}

        async with aiohttp.ClientSession(headers=self.headers) as session:
            for query in queries:
                section = query["section"]
                search_query = query["query"]
                rationale = query["rationale"]
                excerpt = query["excerpt"]

                # self.logger.info(f"[{index+1}] Attempting {search_query}")
                search_params = {
                    "query": search_query,
                    "limit": 100,  # Adjust the limit based on your requirements
                }

                if excerpt:
                    search_params["excerpt"] = excerpt

                # Perform asynchronous GET request with retries
                search_data = await self._request_with_backoff(
                    index=index,
                    session=session,
                    method=session.get,
                    url=self.search_url,
                    params=search_params,
                )

                # self.logger.info(f"[{index+1}] {len(search_data['data'])} results for {search_query} ")

                if search_data:
                    paper_ids = [paper["paperId"] for paper in search_data.get("data", [])]
                    if paper_ids:
                        details = await self._query_batch(
                            index, session, paper_ids, search_query, rationale, excerpt
                        )
                        # self.logger.info(f"[{index+1}] Paper Details for {search_query} {details}")
                        if section in results:
                            results[section].extend(details)  # Append to existing list
                        else:
                            results[section] = details  # Create a new list
                    else:
                        if section not in results:
                            results[section] = []
                else:
                    if section not in results:
                        results[section] = []

        return results

    async def _query_batch(
        self,
        index: int,
        session: aiohttp.ClientSession,
        paper_ids: List[str],
        search_query: str,
        rationale: str,
        excerpt: str,
    ) -> List[Paper]:
        """
        Queries the Semantic Scholar batch endpoint for detailed paper information.

        Args:
            index (int): Index of the current operation, used for logging.
            session (aiohttp.ClientSession): The active HTTP session.
            paper_ids (List[str]): List of paper IDs to query.
            search_query (str): The original search query used to find these papers.
            rationale (str): The rationale for the search query.

        Returns:
            List[Paper]: A list of validated Paper objects.
        """
        fields = (
            "title,abstract,authors,citationCount,referenceCount,"
            "url,venue,publicationVenue,year,openAccessPdf,externalIds"
        )
        payload = {"ids": paper_ids}
        params = {"fields": fields}

        response_data = await self._request_with_backoff(
            index=index,
            session=session,
            method=session.post,
            url=self.batch_url,
            params=params,
            json=payload,
        )

        if not response_data:
            return []

        validated_papers: List[Paper] = []
        for paper_data in response_data:
            if paper_data is None:
                continue  # Skip if empty
            paper_data["query"] = search_query
            paper_data["rationale"] = rationale
            paper_data["excerpt"] = excerpt
            paper_data["authors"] = paper_data.get("authors", [])[:3]

            pub_venue = paper_data.get("publicationVenue", {})
            issn = pub_venue.get("issn", "") if pub_venue else ""
            issn_clean = issn.replace("-", "").strip()

            if issn_clean in self._sjr_map:
                sjr_info = self._sjr_map[issn_clean]
                sjr = sjr_info.get("sjr")

                if sjr is not None and sjr > self.sjr_threshold:
                    paper_data["publicationVenue"]["SJR"] = sjr
                    paper_data["openAccessPdf"] = (
                        paper_data.get("openAccessPdf", {}).get("url")
                        if isinstance(paper_data.get("openAccessPdf"), dict)
                        else None
                    )

                    try:
                        validated_paper = Paper(
                            section="temp", citation="temp", **paper_data
                        )
                        if validated_paper.citationCount > self.min_citation_count:
                            validated_papers.append(validated_paper)

                    except ValidationError:
                        continue

        return validated_papers

    async def _request_with_backoff(
        self, index: int, session: aiohttp.ClientSession, method, url, **kwargs
    ) -> dict:
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
                        rprint(f"[yellow][{index+1}] Rate limit hit: {url}[/yellow]")
                        self.logger.warning(f"[{index+1}] Rate limit hit: {url}")

            except aiohttp.ClientError as e:
                rprint(f"[red][{index+1}] Client error: {e}[/red]")
                self.logger.error(f"[{index+1}] Client error: {e}")

            # Exponential backoff with jitter
            delay = (2**attempt) + random.uniform(0, 1)
            rprint(
                f"[cyan][{index+1}] Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{self.max_retries})[/cyan]"
            )
            self.logger.warning(
                f"[{index+1}] Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{self.max_retries})"
            )
            await asyncio.sleep(delay)

        rprint(f"[red][{index+1}]  Max retries reached for {url}. Skipping request.[/red]")
        self.logger.error(f"[{index+1}] Max retries reached for {url}. Skipping request.")
        return None

    def format_citation(self, paper: Dict) -> str:
        """
        Format the citation for a paper using available metadata from Semantic Scholar.

        Args:
            paper (Dict): The paper metadata returned by Semantic Scholar.

        Returns:
            str: Formatted citation with URL or DOI.
        """
        authors = ", ".join(
            author.get("name", "Unknown") for author in paper.get("authors", [])[:3]
        )
        if len(paper.get("authors", [])) > 3:
            authors += " et al."

        title = paper.get("title", "Unknown Title")
        year = paper.get("year", "Unknown Year")
        venue = paper.get("venue", "Unknown Venue")

        publication_venue = paper.get("publicationVenue") or {}
        publication_name = publication_venue.get("name", venue)
        publication_url = publication_venue.get("url", "")

        doi = paper.get("externalIds", {}).get("DOI", None)
        open_access_pdf = paper.get("openAccessPdf", None)
        general_url = paper.get("url", None)

        citation = f'{authors}. "{title}" ({year}). Published in {publication_name}.'

        if doi:
            citation += f" DOI: {doi}"
        if general_url:
            citation += f" Available at: {general_url}."
        if open_access_pdf:
            citation += f" Open Access PDF: {open_access_pdf}."
        if publication_url:
            citation += f" Publication Info: {publication_url}."

        return citation

    def format_results(self, results: Dict[str, List[Paper]]) -> List[Paper]:
        """
        Format Semantic Scholar results into JSON format suitable for LLM input,
        using the Paper model.
        """
        formatted_results: List[Paper] = []

        for section, papers in results.items():
            for paper in papers:
                if not paper.abstract:
                    continue

                paper.section = section
                paper.citation = self.format_citation(paper.model_dump())
                formatted_results.append(paper)

        return formatted_results

    def select_top_papers(
        self, results: List[Paper], max_papers_per_query: int = 5
    ) -> List[Paper]:
        """
        Selects the top papers, at most max_papers_per_query, based on citation count for each query.

        Args:
            results (List[Paper]): A list of Paper objects.
            max_papers_per_query (int): The maximum number of papers to return per query

        Returns:
            List[Paper]: A list of the most relevant and high-quality papers.
        """
        selected_papers: List[Paper] = []
        papers_by_query: Dict[str, List[Paper]] = defaultdict(
            list
        )  # key: query value: papers

        # 1. Group papers by query
        for paper in results:
            papers_by_query[paper.query].append(paper)

        # 2. For each query, sort by citation count and select top max_papers_per_query
        for query, papers in papers_by_query.items():
            sorted_papers = sorted(
                papers, key=lambda paper: paper.citationCount, reverse=True
            )
            selected_papers.extend(sorted_papers[:max_papers_per_query])

        return selected_papers