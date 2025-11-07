ragflow_repo – RAGFlow (infiniflow/ragflow) Workspace Guide

Overview
- This folder contains a fresh clone of RAGFlow at: ragflow_repo/ragflow
- RAGFlow is an open-source Retrieval-Augmented Generation engine with agentic workflows, deep document parsing (DeepDoc), and production-ready ingestion/orchestration.
- Upstream docs and quickstart: https://github.com/infiniflow/ragflow

Layout

  /scratch2/f004h1v/alphago_project/ragflow_repo
  └─ ragflow/                      # Upstream repo
     ├─ admin/                     # Admin service (server/client)
     ├─ agent/                     # Agent canvas, components, tool plugins
     ├─ agentic_reasoning/         # Deep research prompts and flows
     ├─ api/                       # Backend API (FastAPI), DB models, services, server entry
     ├─ chat_demo/                 # Static demo pages
     ├─ common/                    # Common helpers (string/float/time utils)
     ├─ conf/                      # Static conf (LLM factories, mappings, service_conf.yaml)
     ├─ deepdoc/                   # Deep document parsing (vision, parsers)
     ├─ docker/                    # Docker Compose stack, templates, nginx, scripts
     ├─ docs/                      # Docusaurus docs (quickstart, config, FAQs, dev guides)
     ├─ example/                   # HTTP/SDK usage samples
     ├─ graphrag/                  # GraphRAG utilities (entity resolution, search)
     ├─ helm/                      # Helm chart
     ├─ intergrations/             # Integrations (wechat, chrome extension, firecrawl)
     ├─ mcp/                       # Model Context Protocol client/server
     ├─ plugin/                    # Plugin framework and embedded plugins
     ├─ rag/                       # Core RAG pipeline (flows, LLM, NLP, prompts)
     ├─ sandbox/                   # Optional code-exec sandbox (gVisor); separate compose
     ├─ sdk/                       # SDKs (e.g., Python)
     ├─ web/                       # Frontend (TypeScript/React/Vite/Tailwind)
     ├─ Dockerfile*                # Docker build files
     ├─ download_deps.py           # Script to fetch runtime deps (dev/source mode)
     ├─ pyproject.toml / uv.lock   # Python project metadata/lock (uv)
     └─ README*.md                 # Readmes in multiple languages

Run – Docker (recommended)
Prereqs
- x86 host, Docker >= 24.0.0, Docker Compose >= v2.26.1
- Ensure vm.max_map_count >= 262144 (Elasticsearch requirement):
  sysctl vm.max_map_count
  sudo sysctl -w vm.max_map_count=262144
  # To persist across reboot, add to /etc/sysctl.conf:
  # vm.max_map_count=262144

Steps
  cd /scratch2/f004h1v/alphago_project/ragflow_repo/ragflow/docker
  # CPU path (embedding/DeepDoc on CPU):
  docker compose -f docker-compose.yml up -d

  # GPU path (embedding/DeepDoc accelerated):
  # sed -i '1i DEVICE=gpu' .env
  # docker compose -f docker-compose.yml up -d

Status & login
  docker logs -f docker-ragflow-cpu-1
  # When ready, open http://IP_OF_YOUR_MACHINE/

Configure models (LLMs, embeddings)
- Edit service_conf.yaml.template (auto-populated env vars) or use UI > Model providers.
- Set user_default_llm and API keys.

Key configs
- docker/.env: core settings (SVR_HTTP_PORT, MYSQL_PASSWORD, MINIO_PASSWORD, DOC_ENGINE, DEVICE)
- docker/service_conf.yaml.template: back-end services config, populated from env vars
- docker/docker-compose.yml: container stack

Doc engine switch (Elasticsearch -> Infinity)
  docker compose -f docker/docker-compose.yml down -v   # wipes volumes
  # In docker/.env set: DOC_ENGINE=infinity
  docker compose -f docker-compose.yml up -d

Run – from source (development)
Quick path (from repo docs):
  # Install uv and pre-commit (optional):
  pipx install uv pre-commit
  cd /scratch2/f004h1v/alphago_project/ragflow_repo/ragflow
  uv sync --python 3.10
  uv run download_deps.py
  pre-commit install
  # Start dependencies
  docker compose -f docker/docker-compose-base.yml up -d
  # optional mirror if HF blocked:
  # export HF_ENDPOINT=https://hf-mirror.com
  # jemalloc (if missing): apt-get install -y libjemalloc-dev
  # Backend service
  source .venv/bin/activate
  export PYTHONPATH=$(pwd)
  bash docker/launch_backend_service.sh
  # Frontend
  cd web && npm install && npm run dev

Important submodules and roles
- api/ (backend): FastAPI app (ragflow_server.py), DB models/services (MySQL), MinIO, Redis, Elasticsearch/Infinity.
- deepdoc/: Document layout analysis and parsers for complex formats; can use GPU.
- rag/: Flow orchestration for ingestion, chunking templates, re-ranking, recall strategies, prompts.
- agent/ and agentic_reasoning/: Agent templates and reasoning toolchain (web search, code exec, etc.).
- sandbox/: Optional code execution with gVisor; enable only if you need code tools.
- web/: UI to create datasets, parse files, test retrieval, and run chats.

Typical workflow (UI)
1) Start the stack (Docker) and open the UI.
2) Configure model providers and default models.
3) Create a dataset; pick embedding model and chunk template.
4) Upload files; run parsing; optionally adjust chunks/keywords.
5) Test retrieval; create a chat; select datasets and chat model.

Notes
- Images: v0.21.1 (full, ~9GB) includes embedding models; v0.21.1-slim (~2GB) relies on external embeddings. Nightly is smaller/unstable.
- Default HTTP port is 80; change via docker-compose.yml (HOST:CONTAINER mapping).
- ARM64 users must build images from source.

References
- RAGFlow upstream: https://github.com/infiniflow/ragflow


