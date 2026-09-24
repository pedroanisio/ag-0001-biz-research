Proposed improvements for bi-agent — inspected 2026-09-23

The highest-value improvements are to make evidence verification stricter and saved runs safer to resume. The project already has typed stage contracts, bounded model calls, multilingual reporting, and substantial automated coverage. The proposals below address specific gaps in those controls.

Inspection covered the README, package configuration, crawler, models, prompts, LLM wrapper, pipeline, CLI, report renderer, tests, and metadata from five saved runs. It included the staged changes present in the working tree. No live crawl or paid API call was made, and no application code was changed.

Validation: `.venv/bin/python -m pytest --cov=bi_agent --cov-report=term-missing -o cache_dir=/tmp/bi-agent-review-pytest` completed with **143 tests passing and 97.52% combined statement and branch coverage**. Separate in-memory and temporary-directory probes reproduced the behaviors identified below. Coverage is strong, but several tests currently endorse behavior these proposals would change.

Priority: **P1** addresses evidence integrity, incorrect results, or unreliable run accounting; **P2** improves recovery or research quality. Effort: **S** is a localized change, **M** crosses several modules, and **L** changes persisted contracts or stage architecture. Numbers are identifiers, not strict implementation dependencies.

1. **Require retrieval evidence for every accepted source.** Priority: **P1**. Effort: **M**.

   **Finding:** `verify_findings` accepts any URL on the company's domain, even if neither the crawler nor a server tool retrieved it. A local probe accepted `https://acme.test/never-fetched` with an empty search-hit list and empty ledger. In addition, `SearchHit` retains only URL, title, and age; the excerpt saved to the ledger comes from model output. A matching URL establishes discovery, not whether the retrieved content supports the statement.

   **Proposal:** Remove the domain-only exemption. Require an existing crawl record or a recorded successful search/fetch result. Persist retrieval method, timestamp, final URL, and available source content separately from model-written summaries. Distinguish search-result evidence from a fetched document, and attach supporting passages to important factual claims. When source text is unavailable, preserve that limitation instead of presenting a generated excerpt as retrieval proof.

   **Acceptance:** An invented on-site URL is rejected; an actually crawled URL remains usable without a new search; fetch errors create no retrieval record; and an unrelated retrieved page cannot verify a claim merely because its URL matches.

   **Code:** `bi_agent/pipeline.py:296–337`; `bi_agent/llm.py:115–119,285–306`; `bi_agent/models.py:67–76`.

2. **Tighten primary-source recognition and source independence.** Priority: **P1**. Effort: **M**.

   **Finding:** The official-host regular expression accepts names such as `gov.attacker.test` and `sec.attacker.test`. An on-site source labelled `company_filing` also keeps that label without a document check. Either can satisfy the single-primary-record rule. Separately, two subdomains of one publisher count as independent sources because the check compares full hostnames. All three behaviors were reproduced locally.

   **Proposal:** Use parsed hostnames and explicitly trusted domain boundaries for official registries. Require document-level support for a statutory filing designation, including company-hosted documents. Group sources by registrable domain and, where known, publisher ownership or original publication; do not count syndicated copies as independent confirmation. Apply verification to the particular claim supported by a record, rather than every claim citing that record.

   **Acceptance:** Spoofed official-looking hosts are downgraded; two sections of one publisher do not verify a fact; an ordinary company page cannot become a primary record through a model-provided label; and genuine records retain their appropriate evidentiary role.

   **Code:** `bi_agent/models.py:172–210,724–756`; `tests/test_models.py:106–128,202–213`.

3. **Preserve claim provenance through narrative generation and repairs.** Priority: **P1**. Effort: **L**.

   **Finding:** Narrative fields contain plain strings, and the narrative stage checks only whether citations that remain resolve. `repair_refs` removes unknown citations while leaving their assertions intact. A probe filled every narrative section with an invented financial statement citing `[E999]`; after repair, the narrative passed validation with an empty ledger. Structured sourced claims can similarly become unsupported “analytical inference” simply by losing their citations.

   **Proposal:** Give narrative factual statements links to validated claim IDs, classifications, and evidence. Require explicit premises for inferences. If repair removes a factual statement's last support, reject, omit, or regenerate that statement rather than silently retaining its wording. Save original/repaired values and reasons in a durable audit artifact, and render classification labels consistently from structured data. Revalidate loaded artifacts before report generation as well as at the model boundary.

   **Acceptance:** An unsupported factual sentence cannot survive by losing its citation; prose cannot strengthen the classification of its source claim; and every semantic repair is inspectable after the process exits.

   **Code:** `bi_agent/models.py:189–210,583–602,619–630,674–719`; `bi_agent/pipeline.py:539–577`; `bi_agent/report.py:41–42,86–89`.

4. **Separate retrieved website text from trusted model instructions.** Priority: **P1**. Effort: **M**.

   **Finding:** `_site_system` places crawled text directly in a system-message block for `identify` and `signals`. The analyst prompt does not explicitly establish that instructions appearing inside retrieved pages are untrusted. This is an architectural exposure to prompt injection; the inspection did not demonstrate an exploit against a live model.

   **Proposal:** Keep analyst instructions in the system prompt and supply pages as clearly delimited source data in a lower-trust message or document structure. State that source content cannot change the task, tool behavior, or evidence rules. Maintain the evidence checks independently of model compliance. Evaluate cache reuse after changing the message layout.

   **Acceptance:** Request-construction tests confirm that crawled text is absent from system instructions. An adversarial evaluation fixture containing a fake role declaration and an instruction to fabricate company facts does not alter the required output policy or produce accepted unsupported claims. Treat this evaluation as a regression check, not proof that prompt injection is eliminated.

   **Code:** `bi_agent/pipeline.py:211–225,258–287`; `bi_agent/prompts.py:8–35`; `tests/test_pipeline.py:168–175`.

5. **Track artifact dependencies and commit run state atomically.** Priority: **P1**. Effort: **L**.

   **Finding:** A new crawl overwrites metadata, pages, and the evidence ledger but leaves downstream artifacts and `research.partial.json` intact. A temporary-directory probe confirmed this. Evidence IDs restart at `E001`, so stale claims can resolve to different pages. Stage readers generally check file existence and schema shape, not whether inputs belong to the same run revision. JSON files are written directly, and related files are saved separately.

   **Proposal:** Record a run ID, schema version, model/prompt/configuration identifiers, and input hashes for each stage artifact. Mark dependent stages stale when their inputs change, and refuse incompatible checkpoints. Write individual files through temporary files plus atomic replacement; publish a stage manifest only after its related artifacts are complete. Use a per-run writer lock to avoid conflicting processes. A status command can then show completed, stale, incomplete, and runnable stages.

   **Acceptance:** Reusing a directory for another company cannot consume old research; changing crawl data invalidates dependent outputs; interrupted writes leave the last committed stage readable; and concurrent writers cannot silently overwrite one another.

   **Code:** `bi_agent/pipeline.py:76–113,165–193,267–268,380–381,427–428,563–577,619–637`; `bi_agent/models.py:130–158`.

6. **Keep failed research groups resumable after partial success.** Priority: **P2**. Effort: **M**.

   **Finding:** A transient failure updates progress in memory and returns before saving it. If other groups succeed, the stage eventually writes `findings.json` and deletes the checkpoint even when failed groups remain. Rerunning research then starts again instead of retrying only the gaps. The existing partial-failure test explicitly expects checkpoint deletion. Operational failures also appear in `not_found` alongside searches that genuinely found no evidence.

   **Proposal:** Persist a per-group status after every outcome: pending, completed, or failed, with attempt counts and failure details. Retain that history when publishing partial findings. Resume failed/pending groups without repeating completed work, and distinguish retrieval failure from an evidence gap in reports. Use bounded retries for transient errors and expose partial completion clearly.

   **Acceptance:** When one group fails and others succeed, a restart calls only unfinished groups and preserves existing findings. If every group fails, its status still survives. Reports distinguish “could not complete research” from “completed search found nothing reliable.”

   **Code:** `bi_agent/pipeline.py:351–381,427–470`; `tests/test_pipeline.py:178–234`; `bi_agent/report.py:252–254`.

7. **Accumulate usage across attempts and enforce a run budget.** Priority: **P1**. Effort: **M**.

   **Finding:** `metered` assigns a new summary to `data['stages'][stage]`, replacing previous usage for that stage. A probe recorded two research attempts but found only one call in the saved total. The top-level model field also represents only the latest invocation. Usage is flushed at stage exit, while CLI limits constrain individual responses and research calls rather than total expenditure.

   **Proposal:** Append usage records with run, stage, attempt, call ID, model, and the pricing assumptions used for each estimate. Persist completed-call usage immediately and derive totals from that history. Represent interrupted calls with unavailable final usage explicitly. Add a run-level token/search or estimated-cost budget that accounts for prior attempts and reserves a conservative allowance before another call. Unknown pricing must remain unknown rather than imply a zero cost.

   **Acceptance:** Rerunning or resuming a stage never reduces cumulative usage; mixed-model runs retain per-call attribution; a restart preserves recorded usage; and the budget stops additional calls with completed work safely saved. This inspection did not verify the current vendor price table.

   **Code:** `bi_agent/pipeline.py:135–159`; `bi_agent/llm.py:60–112,255–275`; `bi_agent/cli.py:35–45,69–78`.

8. **Enforce crawl policy before each network request.** Priority: **P1**. Effort: **M**.

   **Finding:** The crawler automatically follows redirects and checks the final host only after downloading the response. Sitemap destinations are not restricted to the site, the initial page is fetched before robots rules, and subdomains reuse the root's robots policy. Mock transport probes confirmed off-site redirect and sitemap requests plus a robots-disallowed initial-page request. The 3 MB response limit is also checked after the full body has been buffered.

   **Proposal:** Centralize fetching with HTTP(S) validation, explicit permitted-origin rules, per-origin robots handling, redirect-hop checks, pacing, and a shared request budget covering pages and sitemaps. Stream responses and stop at a decoded-byte cap, including compressed content. For public-site research, reject loopback/private/link-local destinations unless explicitly configured. Define a narrow policy for legitimate initial canonical-domain redirects.

   **Acceptance:** Tests inspect requests actually issued, not just stored pages. Disallowed redirects and sitemap targets are never requested; initial pages and subdomains obey the applicable robots policy; and oversized bodies are stopped while streaming.

   **Code:** `bi_agent/crawler.py:306–338,341–415`; `tests/test_crawler.py:64–74,103–118`.

9. **Use one URL identity policy that preserves resource semantics.** Priority: **P1**. Effort: **S–M**.

   **Finding:** `normalize_url` lowercases the entire URL, so `/Report?key=ABC` and `/report?key=abc` compare equal. The crawler's separate `canonical` function preserves that case but removes either port 80 or 443 regardless of scheme; a probe transformed `https://acme.test:80/` into `https://acme.test/`. These differences can merge distinct evidence or associate a finding with the wrong retrieved URL.

   **Proposal:** Share a parsed URL representation across crawling, evidence lookup, and source verification. Normalize scheme and hostname while preserving path and query case; remove only the default port for the actual scheme. Handle IPv6 and invalid URLs deliberately. Keep the requested and final URLs, and link aliases using observed redirects instead of assuming HTTP/HTTPS or `www` variants are interchangeable. Make tracking-parameter removal an explicit policy.

   **Acceptance:** Case-sensitive resources and non-default ports remain distinct, fragments do not create duplicate retrievals, invalid URLs yield clear errors, and a captured source can be matched consistently across all stages without broadening acceptance to another resource.

   **Code:** `bi_agent/models.py:85–90,172–175,213–218`; `bi_agent/crawler.py:125–145`; `bi_agent/pipeline.py:307–315`.

10. **Merge corroborating evidence and measure research progress beyond new wording.** Priority: **P2**. Effort: **M**.

    **Finding:** `_research_call` drops accepted findings whose normalized statement already exists without merging their evidence IDs. A probe returned the same statement from a second publisher: the new source entered the ledger, but the finding kept one citation and the call reported zero new findings. That count drives early stopping. The current deduplication also ignores topic and does not update its `seen` set while processing one batch, allowing duplicates within a response.

    **Proposal:** Upsert findings using an explicit claim identity that includes topic and relevant entity/time scope. For matching claims, merge evidence, reassess supported classification, and preserve disagreements rather than flattening them. Deduplicate within each batch. Measure follow-up value using new facts, independent corroboration, resolved gaps, and contradictions; count only supported improvements toward the stopping threshold.

    **Acceptance:** A follow-up can strengthen an existing claim without inventing different wording; its corroborating source remains attached; within-call duplicates are eliminated; and substantive verification progress prevents premature stopping even when no new statement is added.

    **Code:** `bi_agent/pipeline.py:343–348,370–383,447–462`; `bi_agent/prompts.py:237–244`; `tests/test_pipeline.py:431–447`.
