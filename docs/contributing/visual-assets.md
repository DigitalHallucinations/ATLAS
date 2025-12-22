---
audience: Documentation maintainers and contributors
status: draft
last_verified: 2026-03-09
source_of_truth: docs/contributing/visual-assets.md
---

# Visual asset guidelines

Use this guide when adding diagrams, flow charts, or screenshots to ATLAS documentation.

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
