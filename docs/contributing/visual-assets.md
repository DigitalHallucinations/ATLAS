---
audience: Documentation maintainers and contributors
status: draft
last_verified: 2026-03-19
source_of_truth: docs/contributing/visual-assets.md
---

# Visual asset guidelines

Use this guide when adding diagrams, flow charts, or screenshots to ATLAS documentation.

## Authoring priority
- Start with **text-first diagrams** using Mermaid fenced code blocks for sequence, flow, or state diagrams so diffs stay readable and quick to review.
- When a graphical tool is necessary (for example, high-fidelity UI mockups), **export to `.svg`** to preserve scalability and searchability.
- Only fall back to **`.png`** when your tooling cannot export vector assets. If you must use `.png`, prefer high-DPI output and keep the source (for example, the design file) linked in the surrounding text.

### Mermaid usage example
Embed Mermaid directly in Markdown to keep diagrams versionable alongside the surrounding narrative:

```markdown
```mermaid
sequenceDiagram
    participant User
    participant GTK
    participant Atlas

    User->>GTK: Selects Persona
    GTK->>Atlas: Request persona load
    Atlas-->>GTK: Persona loaded
    GTK-->>User: Ready state
```
```

When a diagram requires a static image (for example, a complex UI mockup exported from Figma), place the exported asset under `docs/assets/` (see layout below) and embed it with an alt description:

```markdown
![Persona selection mockup](../assets/ui/persona-selection-mockup-v1.svg)
```

## Directory layout
- Store visuals under `docs/assets/` and group them by topic to keep related files discoverable.
  - UI diagrams: `docs/assets/ui/`
  - Server/API diagrams: `docs/assets/server/`
  - Add additional subfolders as needed (for example, `docs/assets/ops/` or `docs/assets/data/`) to match the documentation section you are updating.
- Keep assets near the pages they serve (for example, `docs/ui/gtk-overview.md` should pull from `docs/assets/ui/`).

## Naming and versioning
- Use **kebab-case** file names: `conversation-router-sequence.png`, `setup-wizard-overview.svg`.
- Include a version or date stamp when replacing an existing visual: `message-bus-topology-v2.png` or `persona-review-2026-03-09.svg`.
- Prefer vector or high-resolution sources when possible (`.svg` or high-DPI `.png`) and avoid embedding exported PDFs directly.
- Keep descriptive alt text in the filename when it improves searchability (for example, `setup-wizard-preflight-scorecard-v1.svg`).

## Embedding visuals in Markdown
Use relative paths that stay within the docs tree so links remain stable when moved.

```markdown
![GTK controller flow](../assets/ui/gtk-controller-flow-v1.svg)
![Server routing overview](../assets/server/http-routing-v2.png)
```

If a doc sits deeper in the tree, adjust the relative path accordingly (for example, `![...](../../assets/ui/...)`).

## Replacement checklist
- [ ] Place the new asset in the correct section folder under `docs/assets/`.
- [ ] Rename the file to kebab-case with a version or date suffix when supplanting an older diagram.
- [ ] Update the Markdown embed with the new path and alt text.
- [ ] Remove or archive superseded assets if they are no longer referenced.
