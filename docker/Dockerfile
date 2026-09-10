FROM ghcr.io/ggml-org/llama.cpp:full-cuda

USER root

# llama-bench / llama-server などのバイナリは /app に置かれるが PATH に含まれていないため追加する
ENV PATH="/app:${PATH}"

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl jq \
    && rm -rf /var/lib/apt/lists/*

COPY start-llama.sh /usr/local/bin/start-llama.sh
COPY configure-model-preset.sh /usr/local/bin/configure-model-preset.sh
COPY lazy-llama-proxy.py /usr/local/bin/lazy-llama-proxy.py
RUN chmod 0755 /usr/local/bin/start-llama.sh /usr/local/bin/configure-model-preset.sh /usr/local/bin/lazy-llama-proxy.py

ENTRYPOINT ["/usr/local/bin/start-llama.sh"]
