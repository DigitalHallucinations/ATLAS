"""Medical research tooling for the MEDIC persona.

This package currently exposes helpers for searching the NCBI Entrez
programming interface (PubMed and PubMed Central).  Modules should export
async callables so they integrate cleanly with the tool runner's event
loop.
"""

from .PubMedCentral.ENTREZ_API import search_pubmed  # noqa: F401
from .PubMedCentral.PMC_API import search_pmc  # noqa: F401

__all__ = ["search_pubmed", "search_pmc"]
