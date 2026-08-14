FROM docker.m.daocloud.io/library/ubuntu:22.04 AS builder

ARG RUST_VERSION=1.88.0

# Ubuntu、rustup 和 crates.io 均使用中国大陆可访问的镜像源。
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
        amd64) ubuntu_mirror="http://mirrors.aliyun.com/ubuntu" ;; \
        arm64) ubuntu_mirror="http://mirrors.aliyun.com/ubuntu-ports" ;; \
        *) echo "unsupported architecture: $arch" >&2; exit 1 ;; \
    esac; \
    sed -i -E "s@https?://(archive.ubuntu.com|security.ubuntu.com)/ubuntu@${ubuntu_mirror}@g; s@https?://ports.ubuntu.com/ubuntu-ports@${ubuntu_mirror}@g" /etc/apt/sources.list; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl; \
    rm -rf /var/lib/apt/lists/*

ENV RUSTUP_DIST_SERVER=https://rsproxy.cn \
    RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup \
    PATH=/root/.cargo/bin:$PATH

RUN set -eux; \
    curl --proto '=https' --tlsv1.2 -sSf https://rsproxy.cn/rustup-init.sh \
        | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}"; \
    mkdir -p /root/.cargo; \
    printf '%s\n' \
        '[source.crates-io]' \
        'replace-with = "rsproxy-sparse"' \
        '' \
        '[source.rsproxy-sparse]' \
        'registry = "sparse+https://rsproxy.cn/index/"' \
        > /root/.cargo/config.toml

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

RUN set -eux; \
    cargo build --workspace --release --locked; \
    strip \
        target/release/caliper \
        target/release/caliper-runner \
        target/release/caliper-transfer


FROM docker.m.daocloud.io/library/ubuntu:24.04 AS runtime

RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
        amd64) ubuntu_mirror="http://mirrors.aliyun.com/ubuntu" ;; \
        arm64) ubuntu_mirror="http://mirrors.aliyun.com/ubuntu-ports" ;; \
        *) echo "unsupported architecture: $arch" >&2; exit 1 ;; \
    esac; \
    sed -i -E \
        "s@https?://(archive.ubuntu.com|security.ubuntu.com)/ubuntu@${ubuntu_mirror}@g; s@https?://ports.ubuntu.com/ubuntu-ports@${ubuntu_mirror}@g" \
        /etc/apt/sources.list.d/ubuntu.sources; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bash \
        libgomp1 \
        libnuma1 \
        libffi8 \
        python3 \
        python3-attr \
        python3-decorator \
        python3-dev \
        python3-numpy \
        python3-psutil \
        python3-scipy \
        python3-sympy; \
    python3 --version; \
    python3-config --prefix; \
    python3 -c 'import attr, ctypes, decorator, numpy, psutil, scipy, sympy'; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/target/release/caliper /usr/local/bin/caliper
COPY --from=builder /build/target/release/caliper-runner /usr/local/bin/caliper-runner
COPY --from=builder /build/target/release/caliper-transfer /usr/local/bin/caliper-transfer
COPY config ./config

ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest \
    LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver \
    PATH=/usr/local/Ascend/driver/tools:/usr/local/Ascend/driver/tools/msnpureport:$PATH \
    RUST_LOG=caliper=info,tower_http=info

VOLUME ["/app/storage"]
EXPOSE 7878

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD bash -c 'exec 3<>/dev/tcp/127.0.0.1/7878 && printf "GET /healthz HTTP/1.0\r\n\r\n" >&3 && read -r status <&3 && [[ "$status" == *" 200 "* ]]'

ENTRYPOINT ["/usr/local/bin/caliper"]
