---
audience: Persona authors
status: in_review
last_verified: 2025-12-21
source_of_truth: modules/Personas/MEDIC/Toolbox/medical_tools/PubMedCentral/ENTREZ_API.py; modules/Personas/MEDIC/Toolbox/medical_tools/_client.py; config.yaml
---

# Medical research tools

The MEDIC persona relies on the NCBI Entrez Programming Utilities (E-utilities)
for PubMed and PubMed Central searches. These tools are exposed as
`search_pubmed` and `search_pmc`, and are available through MEDIC's toolbox
(along with DocGenius when assigned the same research tasks).

## Configuration

Both tools accept an optional Entrez API key. Providing a key unlocks higher
rate limits (up to 10 requests per second) and allows ATLAS to stay within
NCBI's fair-use policy. Configure the credentials in one of the following ways:

- Set the `NCBI_API_KEY` environment variable.
- Add `NCBI_API_KEY` (and optionally `NCBI_API_EMAIL`) to `config.yaml` or your
  `.env` file managed by `ConfigManager`.

The optional `NCBI_API_EMAIL` value is used to populate the `User-Agent`
contact field recommended by NCBI.
Default behaviour throttles calls to approximately 0.34 seconds apart when no
API key is supplied (0.11 seconds with a key) and caps per-request timeouts at
15 seconds. Requests return the sanitized Entrez JSON payload, including query
translations and warning lists when present.

## Safety considerations

The tools only return identifier metadata (PubMed IDs and PMCIDs) along with
query translations. They do not download article bodies or modify local state.
Each call normalizes error responses and throttles outbound requests so that
multiple personas can share the same Entrez credentials safely.
