FROM ghcr.io/ggml-org/llama.cpp:full-cuda

USER root

# llama-bench / llama-server などのバイナリは /app に置かれるが PATH に含まれていないため追加する
ENV PATH="/app:${PATH}"
# pip 版 gguf は llama.cpp 本体より古く新しい量子化型 (Q2_0 等) を読めないため、
# バイナリと同じコミットの /app/gguf-py を gguf-dump で優先して使う
ENV PYTHONPATH="/app/gguf-py"

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY start-llama.sh /usr/local/bin/start-llama.sh
COPY sync-model.sh /usr/local/bin/sync-model.sh
COPY start-llama.py /usr/local/bin/start-llama.py
COPY sync-model.py /usr/local/bin/sync-model.py
COPY model-list-utils.py /usr/local/bin/model-list-utils.py
COPY configure-model-preset.sh /usr/local/bin/configure-model-preset.sh
COPY configure-model-preset.py /usr/local/bin/configure-model-preset.py
COPY lazy-llama-proxy.py /usr/local/bin/lazy-llama-proxy.py
COPY model-preset-utils.py /usr/local/bin/model-preset-utils.py
COPY unload-model.sh /usr/local/bin/unload-model.sh
COPY unload-model.py /usr/local/bin/unload-model.py
RUN chmod 0755 /usr/local/bin/start-llama.sh /usr/local/bin/sync-model.sh /usr/local/bin/start-llama.py /usr/local/bin/sync-model.py /usr/local/bin/model-list-utils.py /usr/local/bin/configure-model-preset.sh /usr/local/bin/configure-model-preset.py /usr/local/bin/lazy-llama-proxy.py /usr/local/bin/model-preset-utils.py /usr/local/bin/unload-model.sh /usr/local/bin/unload-model.py

ENTRYPOINT ["/usr/local/bin/start-llama.sh"]
