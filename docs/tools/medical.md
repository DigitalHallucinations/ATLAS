# Medical research tools

The MEDIC persona relies on the NCBI Entrez Programming Utilities (E-utilities)
for PubMed and PubMed Central searches. These tools are exposed as
`search_pubmed` and `search_pmc`.

## Configuration

Both tools accept an optional Entrez API key. Providing a key unlocks higher
rate limits (up to 10 requests per second) and allows ATLAS to stay within
NCBI's fair-use policy. Configure the credentials in one of the following ways:

- Set the `NCBI_API_KEY` environment variable.
- Add `NCBI_API_KEY` (and optionally `NCBI_API_EMAIL`) to `config.yaml` or your
  `.env` file managed by `ConfigManager`.

The optional `NCBI_API_EMAIL` value is used to populate the `User-Agent`
contact field recommended by NCBI.

## Safety considerations

The tools only return identifier metadata (PubMed IDs and PMCIDs) along with
query translations. They do not download article bodies or modify local state.
Each call normalizes error responses and throttles outbound requests so that
multiple personas can share the same Entrez credentials safely.
