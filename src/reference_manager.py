# src/reference_manager.py

import json
from typing import List, Dict
import logging
import re
from pydantic import ValidationError


class ReferenceManager:
    def __init__(self):
        # Mapping from unique number to reference details
        self.number_to_ref: Dict[int, Dict] = {}
        self.ref_to_number: Dict[str, int] = {}
        self.next_number = 1
        self.logger = logging.getLogger("ReferenceManager")

    def add_references(self, references: List[str]):
        """
        Adds a list of reference strings to the manager, assigning unique numerical IDs.
        """
        for ref in references:
            # Strip existing numbering
            cleaned_ref = self._strip_numbering(ref)
            # Check if the reference already exists
            if cleaned_ref not in self.ref_to_number:
                unique_number = self.next_number
                self.number_to_ref[unique_number] = {"raw": cleaned_ref}
                self.ref_to_number[cleaned_ref] = unique_number
                self.next_number += 1
                self.logger.debug(f"Added new reference with number {unique_number}: {cleaned_ref}")
            else:
                existing_number = self.ref_to_number[cleaned_ref]
                self.logger.debug(f"Reference already exists with number {existing_number}: {cleaned_ref}")

    def add_papers(self, papers: List[Dict]):
        """
        Adds paper references fetched from Semantic Scholar.
        Assumes each paper dict contains necessary fields.
        """
        for paper in papers:
            ref = self._format_paper_reference(paper)
            # Strip existing numbering
            cleaned_ref = self._strip_numbering(ref)
            # Check if the reference already exists
            if cleaned_ref not in self.ref_to_number:
                unique_number = self.next_number
                self.number_to_ref[unique_number] = {"raw": cleaned_ref}
                self.ref_to_number[cleaned_ref] = unique_number
                self.next_number += 1
                self.logger.debug(f"Added new paper reference with number {unique_number}: {cleaned_ref}")
            else:
                existing_number = self.ref_to_number[cleaned_ref]
                self.logger.debug(f"Paper reference already exists with number {existing_number}: {cleaned_ref}")

    def _strip_numbering(self, reference: str) -> str:
        """
        Removes existing numbering from the reference string.
        E.g., "[1] Smith J..." becomes "Smith J..."
        """
        return re.sub(r'^\[\d+\]\s*', '', reference)

    def _format_paper_reference(self, paper: Dict) -> str:
        """
        Formats a paper dict into a reference string.
        """
        authors = ', '.join([author['name'] for author in paper.get('authors', [])])
        title = paper.get('title', 'No Title')
        journal = paper.get('venue', 'No Journal')
        year = paper.get('year', 'No Year')
        doi = paper.get('doi', '')
        url = f"https://doi.org/{doi}" if doi else 'No DOI'

        reference = f"{authors} ({year}). {title}. {journal}. {url}"
        return reference

    def get_reference_number(self, reference: str) -> int:
        """
        Retrieves the unique number assigned to a reference.
        """
        cleaned_ref = self._strip_numbering(reference)
        return self.ref_to_number.get(cleaned_ref, -1)  # Return -1 if not found

    def get_all_references(self) -> List[Dict]:
        """
        Returns all references sorted by their unique number.
        """
        sorted_refs = sorted(self.number_to_ref.items())
        return [{"reference_number": num, "authors": self._extract_authors(ref['raw']),
                 "year": self._extract_year(ref['raw']),
                 "title": self._extract_title(ref['raw']),
                 "journal_source": self._extract_journal(ref['raw']),
                 "url_doi": self._extract_url(ref['raw'])} for num, ref in sorted_refs]

    def _extract_authors(self, reference: str) -> str:
        """
        Extracts authors from the reference string.
        """
        return reference.split('(')[0].strip()

    def _extract_year(self, reference: str) -> str:
        """
        Extracts the publication year from the reference string.
        """
        match = re.search(r'\((\d{4})\)', reference)
        return match.group(1) if match else "No Year"

    def _extract_title(self, reference: str) -> str:
        """
        Extracts the title from the reference string.
        """
        try:
            title_part = reference.split('). ')[1]
            return title_part.split('. ')[0]
        except IndexError:
            return "No Title"

    def _extract_journal(self, reference: str) -> str:
        """
        Extracts the journal/source from the reference string.
        """
        try:
            journal_part = reference.split('. ')[1]
            return journal_part.split('. ')[0]
        except IndexError:
            return "No Journal"

    def _extract_url(self, reference: str) -> str:
        """
        Extracts the URL or DOI from the reference string.
        """
        try:
            return reference.split('. ')[-1]
        except IndexError:
            return "No DOI"

    def renumber_references(self, article: Dict) -> Dict:
        """
        Replaces existing reference citations in the article with unique numerical IDs.
        Updates both inline citations and the References section.
        """
        # Assign unique IDs based on first appearance
        def assign_ids(text: str) -> str:
            pattern = r'\[(.*?)\]'  # Assuming citations are in [ID] format

            def replace(match):
                ref_identifier = match.group(1)
                # Find the reference number
                ref_number = self.get_reference_number(ref_identifier)
                if ref_number == -1:
                    # Reference not found, handle as needed (e.g., skip or assign new ID)
                    self.logger.warning(f"Reference '{ref_identifier}' not found.")
                    return match.group(0)  # Keep original
                return f"[{ref_number}]"

            return re.sub(pattern, replace, text)

        # Update the article content with numerical IDs
        article['content'] = assign_ids(article.get('content', ''))

        # Create the References section
        references_list = self.get_all_references()

        # Update the References section
        article['References'] = "\n".join([
            f"[{ref['reference_number']}] {ref['authors']}. {ref['year']}. {ref['title']}. {ref['journal_source']}. {ref['url_doi']}"
            for ref in references_list
        ])
        return article
