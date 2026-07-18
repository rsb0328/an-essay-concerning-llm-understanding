# Third-party notices

This project does not redistribute model weights, third-party installers, or third-party source distributions. Python packages are resolved by the user's package installer and retain their own licenses.

Direct runtime dependencies:

| Component | Role | License | Project |
|---|---|---|---|
| FastAPI | HTTP API | MIT | https://github.com/fastapi/fastapi |
| Pydantic | Schema validation | MIT | https://github.com/pydantic/pydantic |
| Uvicorn | ASGI server | BSD 3-Clause | https://github.com/Kludex/uvicorn |
| HTTPX | Provider HTTP client | BSD 3-Clause | https://github.com/encode/httpx |
| NumPy | Vector mathematics | BSD 3-Clause | https://github.com/numpy/numpy |

Optional dependencies:

| Component | Role | License | Project |
|---|---|---|---|
| Sentence Transformers | Local embedding execution | Apache-2.0 | https://github.com/huggingface/sentence-transformers |
| Qdrant / qdrant-client | Vector storage | Apache-2.0 | https://github.com/qdrant/qdrant |
| Docling | Document conversion | MIT | https://github.com/docling-project/docling |

Models used only in the earlier prototype experiments:

| Model | Role | License | Model card |
|---|---|---|---|
| Qwen3 14B | Local generation | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-14B |
| BAAI/bge-m3 | Embeddings | MIT | https://huggingface.co/BAAI/bge-m3 |

Users who select other models, hosted APIs, parsers, or vector stores are responsible for reviewing their licenses, acceptable-use terms, privacy policies, and data-processing implications. The Apache-2.0 license of this repository does not relicense any dependency, model, service, source document, or generated output.

