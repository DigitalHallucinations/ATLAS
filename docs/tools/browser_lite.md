---
audience: Tool users and ops
status: in_review
last_verified: 2025-12-21
source_of_truth: modules/Tools/Base_Tools/browser_lite.py; modules/Tools/Base_Tools/webpage_fetch.py
---

# Browser Lite Tool

The **Browser Lite** tool provides constrained, policy-aware browsing for allowlisted domains.
It is designed to capture small amounts of information from trusted sites without giving the
assistant unrestricted access to the open web.

## Key features

- Enforces a configurable allowlist of domains and validates every navigation request before it
  is executed.
- Downloads and caches each site's `robots.txt` file to honour crawl directives automatically.
- Applies a configurable throttle between requests to avoid bursty traffic or rate-limit issues.
- Supports optional form submissions when both the tool call enables them and the active persona
  flags the action as safe.
- Can return sanitized page text and, when headless browser support is available, attach a PNG
  screenshot for every visited page.

## Usage

```json
{
  "url": "https://example.org/news",
  "actions": [
    {"type": "navigate", "url": "/more"}
  ],
  "take_screenshot": true,
  "extract_text": true
}
```

## Setup

Browser Lite will automatically enable screenshot capture when Playwright is installed and its
browser binaries are provisioned. After installing the project dependencies, run:

```bash
pip install -r requirements.txt
playwright install
```

The `playwright install` command downloads the Chromium runtime that Browser Lite relies on when a
page screenshot is requested.

### Form submissions

Form submissions are categorised as high risk. To enable them you must:

1. Set `"allow_forms": true` in the tool request, and
2. Ensure the persona context exposes `browser_high_risk_enabled: true`.

Attempts to submit forms without both conditions raise an explicit error so the assistant can
notify the user and stop the workflow.

## Safety considerations

- Requests are blocked when the domain is not in the allowlist or `robots.txt` disallows the path.
- Navigation stops once the session exceeds the configured page budget (defaults to five pages).
- Screenshot capture is best-effort; failures are logged but do not leak the raw exception details.
- The tool respects persona policy flags so administrators can gate high-risk behaviours on a
  per-persona basis.
