# syntax=docker/dockerfile:1.18.0@sha256:dabfc0969b935b2080555ace70ee69a5261af8a8f1b4df97b9e7fbcf6722eddf

FROM --platform=$BUILDPLATFORM golang:1.26.5-alpine3.23@sha256:622e56dbc11a8cfe87cafa2331e9a201877271cbff918af53d3be315f3da88cc AS proxy-builder
ARG TARGETARCH
ARG RUNTIME_VERSION=1.0.0
ARG OLLAMA_VERSION=0.32.5
WORKDIR /src
COPY go.mod main.go ./
RUN CGO_ENABLED=0 GOOS=linux GOARCH="$TARGETARCH" go build \
    -buildvcs=false \
    -trimpath \
    -ldflags="-s -w -buildid= -X main.runtimeVersion=${RUNTIME_VERSION} -X main.ollamaVersion=${OLLAMA_VERSION}" \
    -o /out/kestrel-ollama-runtime ./

FROM golang:1.26.5-bookworm@sha256:1ecb7edf62a0408027bd5729dfd6b1b8766e578e8df93995b225dfd0944eb651 AS ollama-builder
ARG OLLAMA_COMMIT=eec8e0b9458b8a01be0c216a9cc53eefde24ef50
ADD --checksum=sha256:8c5030739b89aaa7eada7332c707cb3c6be27d9ec901b501a539fa57c576faae \
    https://github.com/ollama/ollama/archive/${OLLAMA_COMMIT}.tar.gz /tmp/ollama.tar.gz
RUN mkdir -p /src/ollama \
    && tar -xzf /tmp/ollama.tar.gz --strip-components=1 -C /src/ollama \
    && rm /tmp/ollama.tar.gz
WORKDIR /src/ollama
# The upstream 0.32.5 image predates fixed releases for these dependencies.
# Keep the application source pinned while rebuilding its network-facing binary
# with fixed, explicit module versions and the patched Go toolchain.
RUN go get \
      github.com/buger/jsonparser@v1.1.2 \
      golang.org/x/crypto@v0.52.0 \
      golang.org/x/image@v0.43.0 \
      golang.org/x/net@v0.55.0 \
      golang.org/x/text@v0.39.0 \
    && go mod download \
    && CGO_ENABLED=1 go build -buildvcs=false -trimpath -buildmode=pie \
      -ldflags="-s -w -buildid= -X=github.com/ollama/ollama/version.Version=0.32.5 -X=github.com/ollama/ollama/server.mode=release" \
      -o /out/ollama .

FROM ollama/ollama:0.32.5@sha256:4dea9fb511947e24a84237bb636b0203abcb2ff0d3fbc7b4ff865deb91362131
ARG RUNTIME_VERSION=1.0.0
ARG OLLAMA_VERSION=0.32.5
LABEL org.opencontainers.image.title="Kestrel private Ollama runtime" \
      org.opencontainers.image.description="Digest-pinned Ollama behind an authenticated, allowlisted streaming proxy" \
      org.opencontainers.image.source="https://github.com/KestrelSovereignAI/kestrel-cloud-runpod" \
      org.opencontainers.image.version="$RUNTIME_VERSION" \
      ai.kestrel.runtime.contract="ollama-runtime/v1" \
      ai.kestrel.ollama.version="$OLLAMA_VERSION" \
      ai.kestrel.ollama.source.commit="eec8e0b9458b8a01be0c216a9cc53eefde24ef50"

RUN groupadd --gid 10001 kestrel-ollama \
    && useradd --uid 10001 --gid 10001 --home-dir /var/lib/kestrel-ollama --shell /usr/sbin/nologin kestrel-ollama \
    && mkdir -p /models /var/lib/kestrel-ollama \
    && chown -R 10001:10001 /models /var/lib/kestrel-ollama
COPY --from=ollama-builder --chown=0:0 --chmod=0555 /out/ollama /usr/bin/ollama
COPY --from=proxy-builder --chown=0:0 --chmod=0555 /out/kestrel-ollama-runtime /usr/local/bin/kestrel-ollama-runtime

ENV HOME=/var/lib/kestrel-ollama \
    GIN_MODE=release \
    OLLAMA_DEBUG=false \
    OLLAMA_HOST=127.0.0.1:11435 \
    OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_MODELS=/models \
    OLLAMA_NOHISTORY=true \
    OLLAMA_NO_CLOUD=true \
    PORT=11434 \
    PORT_HEALTH=11434
# The entrypoint starts as root only to initialize a fresh Runpod volume, rejects
# symlinks in the cache tree, recursively assigns it to 10001:10001, and drops
# all user/group privilege before starting Ollama or opening the public port.
USER 0:0
VOLUME ["/models"]
EXPOSE 11434
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 CMD ["/usr/local/bin/kestrel-ollama-runtime", "healthcheck"]
ENTRYPOINT ["/usr/local/bin/kestrel-ollama-runtime"]
