# O/A Levels Evaluator Pipeline – Flow

## High-level flow

```mermaid
flowchart TB
    subgraph entry [Entry]
        ModeA["Mode A: POST /oa-level/evaluate-from-image"]
        ModeB["Mode B: POST /oa-level/evaluate"]
    end

    subgraph modeA [Mode A path]
        Upload["Upload file (PDF/image)"]
        ValidateFile["Validate file type/size"]
        OCR["Mode A extraction: repo-local OA-Extraction"]
        BuildRequestA["Normalize extracted question/answer and build EvaluateRequest"]
    end

    subgraph modeB [Mode B path]
        JSON["JSON body: question + student_answer + optional filters"]
        BuildRequestB["Build EvaluateRequest"]
    end

    subgraph service [OALevelEvaluatorService.evaluate]
        Sanitize["Sanitize request"]
        SourceOrder["Get source order from config (e.g. primary = o_level_main_json, fallback = oa_main_dataset)"]
        Primary["_evaluate_from_repository (primary source)"]
        PrimaryOK{"Primary status != failed?"}
        Fallback["_evaluate_from_repository (fallback source)"]
        FallbackOK{"Fallback status != failed?"}
        CombinedFail["Return failed response with combined feedback"]
    end

    subgraph evalRepo [_evaluate_from_repository]
        GetRecords["repository.get_records()"]
        FilterRecords["repository.filter_records(all_records, request)"]
        NoRecords{"scoped_records empty?"}
        ResolveRecord["_resolve_record (match question)"]
        NoMatch{"best_record found?"}
        LookupMS["lookup_markscheme(record)"]
        EvalAnswer["evaluate_answer(student_answer, marking_scheme_answer, ...)"]
        BuildFeedback["build_feedback(grade_label, score_percent, ...)"]
        ReturnResponse["Return EvaluateResponse"]
        FailNoRecords["Return failed (no_records_feedback)"]
        FailNoMatch["Return failed (no_match_feedback, alternatives)"]
    end

    subgraph resolve [_resolve_record]
        HasQID{"request.question_id set?"}
        GetByID["repository.get_by_question_id(question_id)"]
        InScope["Record in scoped list?"]
        DirectMatch["Return MatchResult (confidence 1.0)"]
        Search["search_index.search(source, query, records)"]
        Rerank["Embedding similarity + rerank_search_results (lexical + hints)"]
        SearchResult["Return MatchResult + debug"]
    end

    subgraph searchIndex [SearchIndexManager.search]
        EnsureBuilt["ensure_built(source): build or load index"]
        GetEmbedder["_get_embedder(): hash or sentence_transformers"]
        EncodeQuery["Encode query text to vector"]
        Similarity["Dot-product similarity vs scoped embeddings"]
        TopK["Take top_k candidates"]
        RerankModule["rerank_search_results (question_matcher)"]
    end

    subgraph grading [evaluate_answer]
        MCQCheck["MCQ? (extract correct option from scheme)"]
        MCQMatch["Student option == correct option? Score 1 or 0"]
        GrokEnabled["use_grok_grading?"]
        GrokAPI["Call Grok API (grade against scheme)"]
        Deterministic["Deterministic: expected_points extraction, token overlap, score"]
        EvalResult["EvaluationResult (score, grade_label, feedback, ...)"]
    end

    ModeA --> Upload --> ValidateFile --> OCR --> BuildRequestA
    ModeB --> JSON --> BuildRequestB
    BuildRequestA --> service
    BuildRequestB --> service

    Sanitize --> SourceOrder --> Primary
    Primary --> PrimaryOK
    PrimaryOK -->|yes| ReturnResponse
    PrimaryOK -->|no| Fallback
    Fallback --> FallbackOK
    FallbackOK -->|yes| ReturnResponse
    FallbackOK -->|no| CombinedFail

    Primary --> GetRecords --> FilterRecords --> NoRecords
    NoRecords -->|yes| FailNoRecords
    NoRecords -->|no| ResolveRecord
    ResolveRecord --> NoMatch
    NoMatch -->|no| FailNoMatch
    NoMatch -->|yes| LookupMS --> EvalAnswer --> BuildFeedback --> ReturnResponse

    ResolveRecord --> HasQID
    HasQID -->|yes| GetByID --> InScope
    InScope -->|yes| DirectMatch
    InScope -->|no| SearchResult
    HasQID -->|no| Search
    Search --> EnsureBuilt --> GetEmbedder --> EncodeQuery --> Similarity --> TopK --> RerankModule --> SearchResult

    EvalAnswer --> MCQCheck
    MCQCheck -->|MCQ| MCQMatch --> EvalResult
    MCQCheck -->|not MCQ| GrokEnabled
    GrokEnabled -->|yes| GrokAPI --> EvalResult
    GrokEnabled -->|no| Deterministic --> EvalResult
```

## Data sources and source order

```mermaid
flowchart LR
    subgraph config [PipelineConfig]
        SourcePriority["source_priority e.g. o_level_main_first"]
    end

    subgraph sources [Data sources]
        MainJSON["o_level_main_json (MainJsonRepository)"]
        OAMain["oa_main_dataset (DatasetRepository)"]
        OLevel["o_level_json (FallbackDatasetRepository)"]
    end

    subgraph order [Source order]
        P1["Primary"]
        P2["Fallback"]
    end

    SourcePriority -->|o_level_main_first| P1
    P1 --> MainJSON
    P2 --> OAMain
    SourcePriority -->|o_level_json_first| P1
    P1 --> OLevel
    P2 --> OAMain
```

## Search index (embedding path)

```mermaid
flowchart TB
    subgraph config [Config]
        EmbedBackend["OA_EMBED_BACKEND: hash | sentence_transformers"]
        SearchMethod["OA_SEARCH_METHOD: embedding_local"]
    end

    subgraph getEmbedder [_get_embedder]
        CheckBackend["embed_backend == hash?"]
        HashEmbedder["_HashingEmbedder (384-dim, tokenize + hash)"]
        TryST["Try _SentenceTransformerEmbedder(embed_model)"]
        STFail["Fallback to _HashingEmbedder"]
    end

    subgraph build [ensure_built]
        ShouldRebuild["_should_rebuild (missing/stale manifest)"]
        LoadRecords["Load records from repository for source"]
        EncodeAll["embedder.encode(question_texts)"]
        SaveArtifacts["Write manifest.json, records.jsonl, embeddings.npy"]
        LoadIndex["_load_index (read artifacts into memory)"]
    end

    subgraph search [search]
        QueryVec["embedder.encode([query])[0]"]
        DotProduct["similarities = embeddings @ query_vec"]
        Normalize["Normalize to 0..1"]
        TopK["argpartition top_k"]
        Rerank["rerank_search_results (lexical + embedding blend)"]
    end

    EmbedBackend --> CheckBackend
    CheckBackend -->|yes| HashEmbedder
    CheckBackend -->|no| TryST
    TryST -->|fail| STFail
    TryST -->|ok| STFail

    HashEmbedder --> build
    ShouldRebuild -->|yes| EncodeAll --> SaveArtifacts
    ShouldRebuild -->|no| LoadIndex
    SaveArtifacts --> LoadIndex
    LoadIndex --> search
    QueryVec --> DotProduct --> Normalize --> TopK --> Rerank
```

## Startup and readiness

```mermaid
flowchart TB
    subgraph startup [API startup]
        Lifespan["Lifespan: startup"]
        WarmupThread["Start background thread: evaluator_service.warmup()"]
        WarmupDone["Set warmup_done Event when finished"]
        AppReady["Application startup complete"]
    end

    subgraph warmup [warmup]
        WarmupEmbedder["search_index.warmup_embedder()"]
        EnsurePrimary["search_index.ensure_built(primary_source)"]
    end

    subgraph endpoints [Endpoints]
        Health["GET /oa-level/health -> 200 + status ok"]
        Ready["GET /oa-level/ready -> 200 if warmup_done else 503"]
    end

    Lifespan --> WarmupThread --> AppReady
    WarmupThread --> WarmupEmbedder --> EnsurePrimary --> WarmupDone
    Health --> endpoints
    Ready --> WarmupState["Reads warmup_done state"]
```

## Different wording / paraphrasing

The pipeline does **not** require the question to be exactly the same as in the dataset. It can match when the question has the same meaning but different wording, depending on the embedding backend and thresholds.

**How matching works**

1. **Embedding similarity (75% of final score)**  
   The query question and each dataset question are turned into vectors; similarity is dot product (normalized).  
   - **With `OA_EMBED_BACKEND=sentence_transformers`**: Embeddings are semantic. Paraphrased questions (same meaning, different words) often get **high** similarity, so the pipeline can still match and grade correctly.  
   - **With `OA_EMBED_BACKEND=hash`**: Vectors are built from tokens (words). Different wording means different tokens, so similarity drops. Matching works best when the user’s question shares a lot of words with the dataset question.

2. **Lexical overlap (20%)**  
   Jaccard similarity on token sets. Same wording gives 1.0; paraphrasing reduces overlap, so this score is lower when wording is very different.

3. **Answer hint (5%)**  
   Small bonus when the student answer aligns with the marking scheme (e.g. same MCQ option).

**Thresholds (config)**

- `search_accepted_threshold` (default 0.78): best match above this → status `accepted`.
- `search_review_threshold` (default 0.62): above this but below accepted → `review_required`; below → `failed`.

**Practical takeaway**

- **Same meaning, different wording**: Use **sentence_transformers** (default or with a pre-downloaded model). The pipeline is designed to work in this case.  
- **Hash backend**: Prefer when you care about fast startup and no HF; matching is more sensitive to wording and works best when the question text is close to the dataset (e.g. same key phrases).  
- If paraphrased questions often get `review_required` or `failed`, you can lower `OA_SEARCH_ACCEPTED_THRESHOLD` (e.g. 0.72) or `OA_SEARCH_REVIEW_THRESHOLD` (e.g. 0.55), at the cost of more false positives.

---

## File and component reference

| Step | File / component |
|------|-------------------|
| API entry (Mode A/B) | [oa_main_pipeline/api.py](oa_main_pipeline/api.py) |
| OCR / extraction (Mode A) | [oa_main_pipeline/mode_a_oa_extraction.py](oa_main_pipeline/mode_a_oa_extraction.py) + [OA-Extraction/src/oa_extraction/pipeline.py](OA-Extraction/src/oa_extraction/pipeline.py) |
| Orchestration | [oa_main_pipeline/service.py](oa_main_pipeline/service.py) – `OALevelEvaluatorService.evaluate`, `_evaluate_from_repository`, `_resolve_record` |
| Repositories | `MainJsonRepository`, `DatasetRepository`, `FallbackDatasetRepository` |
| Search index | [oa_main_pipeline/search_index.py](oa_main_pipeline/search_index.py) – `SearchIndexManager.search`, `ensure_built`, `_get_embedder` |
| Rerank / match | [oa_main_pipeline/question_matcher.py](oa_main_pipeline/question_matcher.py) – `rerank_search_results` |
| Markscheme | [oa_main_pipeline/markscheme_lookup.py](oa_main_pipeline/markscheme_lookup.py) – `lookup_markscheme` |
| Grading | [oa_main_pipeline/answer_evaluator.py](oa_main_pipeline/answer_evaluator.py) – `evaluate_answer` (Grok or deterministic) |
| Feedback | [oa_main_pipeline/feedback_builder.py](oa_main_pipeline/feedback_builder.py) – `build_feedback` |
| Config | [oa_main_pipeline/config.py](oa_main_pipeline/config.py) – `PipelineConfig` |
