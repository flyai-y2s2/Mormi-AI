# Independent service reference

`baseline_service.py` is the verbatim service at develop
`9708953bef9cff32fcd9af499b7cdcfa3b1bc67c`, before session-parent orchestration.
Load it only in tests under the `mormi_api` package for its relative imports.
Never import it in production or modify it to make a comparison pass.

The parent tests compare the independent service, not two adapters calling the
same new service routing code. Shared pedagogy/prompt/LLM/content code stays
unchanged and remains covered by the older independent turn-engine reference.
