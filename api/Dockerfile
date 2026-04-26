# Multi-stage build for ForSureLLM API.
#
# Build args :
#   MODEL_VARIANT  = pruned (default, 24 MB FR+EN) | full (113 MB multilingual)
#
# Examples :
#   docker build -t forsurellm-api .
#   docker build --build-arg MODEL_VARIANT=full -t forsurellm-api:full .
#
# Run :
#   docker run -p 8000:8000 forsurellm-api

# ---------- Stage 1 : builder (download/build artefacts, no compiler in final) ----------
FROM python:3.12-slim AS builder

WORKDIR /build

ARG MODEL_VARIANT=pruned
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for building wheels (kept in this stage only).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install runtime + web extras into a target directory we copy to the final stage.
COPY pyproject.toml README.md ./
COPY forsurellm ./forsurellm
RUN pip install --prefix=/install ".[web]"

# Download the ONNX model from HuggingFace Model repo (bake into image).
# Using huggingface_hub CLI for simplicity. Pruned by default.
RUN pip install --no-cache-dir "huggingface_hub>=0.20,<1.0"
RUN python -c "\
import os, shutil; \
from huggingface_hub import hf_hub_download; \
variant_arg = '$MODEL_VARIANT'; \
fname = 'forsurellm-int8_fr-en.onnx' if variant_arg == 'pruned' else 'forsurellm-int8.onnx'; \
print(f'fetching {fname}...'); \
src = hf_hub_download('jcfossati/ForSureLLM', fname); \
dst = '/build/forsurellm/models/' + fname; \
os.makedirs(os.path.dirname(dst), exist_ok=True); \
shutil.copy(src, dst); \
print(f'-> {dst}')"


# ---------- Stage 2 : runtime (slim, no compiler, no source) ----------
FROM python:3.12-slim AS runtime

ARG MODEL_VARIANT=pruned

WORKDIR /app

# Map MODEL_VARIANT (pruned|full) to FORSURELLM_VARIANT (_fr-en|empty) so the
# classifier picks the right ONNX. Resolved at build time, baked into image.
RUN if [ "$MODEL_VARIANT" = "pruned" ]; then \
        echo "_fr-en" > /etc/forsurellm_variant ; \
    else \
        echo "" > /etc/forsurellm_variant ; \
    fi

# Drop privileges
RUN groupadd -r app && useradd -r -g app -d /app -s /usr/sbin/nologin app

# Copy installed Python packages from builder.
COPY --from=builder /install /usr/local

# Copy application code.
COPY --from=builder /build/forsurellm /app/forsurellm
COPY api /app/api

USER app
EXPOSE 8000

# Health check uses /health (cheap, doesn't load model).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3); sys.exit(0 if r.status==200 else 1)" || exit 1

# Read the variant from the file written at build time, set env, exec uvicorn.
CMD ["/bin/sh", "-c", "export FORSURELLM_VARIANT=$(cat /etc/forsurellm_variant) && exec uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 1 --log-level warning"]
