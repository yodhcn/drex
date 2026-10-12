# syntax=docker/dockerfile:1
#
# drex-decision-models —— GPU 推理镜像（不内置任何模型权重）
#
# 本仓库只包含镜像定义，不含上游源码：构建时把 nace-ai/drex-decision-models 克隆进镜像。
# 权重在宿主机上取好，再挂进容器：
#
#   hf download nace-ai/drex-v1.5 --local-dir weights/drex-v1.5
#
#   docker run --rm --gpus all -p 8000:8000 \
#     -v "$PWD/weights/drex-v1.5:/models/drex-v1.5:ro" \
#     drex-decision-models:latest \
#     python serve.py --model /models/drex-v1.5 --host 0.0.0.0 --port 8000
#
# 【--host 0.0.0.0 不能省】上游 serve.py 的 --host 默认是 127.0.0.1。容器里监听回环地址
# 收不到 docker -p 转发来的流量（-p 是 DNAT 到容器 eth0），症状是服务启动正常、日志正常，
# 但宿主机就是连不上，两端都不报错。上游已经暴露了这个开关，所以镜像不需要任何 shim。
#
# 默认命令是 tail -f /dev/null 保活，用 docker run <镜像> <你的命令> 整体替换即可。

ARG BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ARG UPSTREAM_REPO=https://github.com/nace-ai/drex-decision-models.git
ARG UPSTREAM_REF=main
# Gated DeltaNet 快速内核。必须是「裸包名」，不要写成 flash-linear-attention[cuda]：
# 裸包的依赖只有 fla-core -> einops，动不了 torch；[cuda] extra 会引入 torch>=2.7 / triton>=3.3
# 约束，等于给未来的某版 fla 一个把 torch 顶出 requirements.txt 范围的机会。
# 置空即不装（--build-arg FLASH_LINEAR_ATTENTION_SPEC=），但 hybrid 骨干会静默退回慢速参考实现。
ARG FLASH_LINEAR_ATTENTION_SPEC=flash-linear-attention

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/drex \
    HF_HOME=/hf-cache

# Ubuntu 22.04 自带 Python 3.10，刚好够用：transformers 5.x 要求 >=3.10，
# torch 2.8 在 cp310-linux-x86_64 上有 manylinux_2_28 wheel（Ubuntu 22.04 的 glibc 是 2.35）。
# 所以不需要 deadsnakes PPA，也不需要另装一个 Python。
#
# 刻意不装 build-essential：需要编译的依赖（torch / tokenizers / pydantic-core / safetensors /
# regex / numpy）在 cp310 linux x86_64 上都有 wheel，构建期不需要编译器。
# 若将来某个包只剩 sdist，构建会因为缺 gcc 直接失败 —— 那时再补 build-essential，
# 而不是现在就为一个「万一」无声胖 250 MB。
# libgomp1 是 torch 运行时的 OpenMP 依赖，体积很小，装上省掉一整类运行期报错。
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        python3 \
        python3-venv \
        libgomp1; \
    rm -rf /var/lib/apt/lists/*

# 取上游源码用 git init + 单 ref 浅 fetch，而不是 git clone --branch：
# 后者对 commit SHA 会失败，前者同时支持分支 / tag / commit SHA，且始终只取一层历史。
RUN set -eux; \
    git init -q /opt/drex; \
    git -C /opt/drex remote add origin "${UPSTREAM_REPO}"; \
    git -C /opt/drex fetch -q --depth 1 origin "${UPSTREAM_REF}"; \
    git -C /opt/drex checkout -q --detach FETCH_HEAD; \
    printf 'repo=%s\nref=%s\ncommit=%s\n' \
        "${UPSTREAM_REPO}" "${UPSTREAM_REF}" "$(git -C /opt/drex rev-parse HEAD)" \
        > /opt/upstream.txt; \
    rm -rf /opt/drex/.git

# 依赖按上游文档装。上游只有 requirements.txt，没有 lockfile，所以就是 pip install -r。
# torch 从 PyPI 取，而 PyPI 的 Linux torch wheel 自带一整套 CUDA 12.8 用户态库
# （nvidia-*-cu12 依赖），运行时优先加载的是这一套 —— 基镜像的 CUDA 12.4 只是工具链布局和
# 驱动挂载点，不参与运算。因此这里不用 PyTorch 官方索引，也不该动它。见 README。
#
# flash-linear-attention 和 requirements.txt 放在同一次 pip 解析里：
# pip 会同时满足 torch<2.9 和它自己的约束，避免它把 torch 悄悄顶到范围外。
RUN set -eux; \
    python3 -m venv "${VIRTUAL_ENV}"; \
    "${VIRTUAL_ENV}/bin/pip" install --upgrade pip wheel; \
    if [ -n "${FLASH_LINEAR_ATTENTION_SPEC}" ]; then \
        "${VIRTUAL_ENV}/bin/pip" install -r /opt/drex/requirements.txt "${FLASH_LINEAR_ATTENTION_SPEC}"; \
    else \
        "${VIRTUAL_ENV}/bin/pip" install -r /opt/drex/requirements.txt; \
    fi; \
    mkdir -p "${HF_HOME}"

WORKDIR /opt/drex

# 构建期自检：只断言「装依赖这一步保证不了」的东西，不重复 requirements.txt 已经定死的版本号。
#
# serve.py / inference.py 的 --help 会走完整条 import 链
# （kev.api -> pydantic、kev.model -> transformers、kev.checkpoint -> torch），
# 但 argparse 的 -h 在 required=True 校验之前就退出，所以不会加载权重。
#
# fla 那条断言是这里最重要的检查，理由见下：
# transformers 的 Qwen3.5 Gated DeltaNet 用
#   @use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
# 解析内核，缺包时只 warning_once 一句就退回纯 torch 的参考实现 ——
# 功能正确、慢一个数量级，而构建期看起来一切正常。
# 它真正解析的路径是 fla.ops.gated_delta_rule.chunk_gated_delta_rule
# （hub_kernels.py 的 _KERNELS_INTERNAL_PATH_MAPPINGS 把 chunk_gated_delta_rule 映射到
# ops.gated_delta_rule，_PACKAGE_TO_DISTRIBUTION 把 fla 映射到 flash-linear-attention），
# 所以这里断言的就是同一个符号、同一条路径，而不是「顶层包能 import」这种自欺的检查。
RUN set -eux; \
    test -f /opt/drex/serve.py; \
    test -f /opt/drex/inference.py; \
    test -f /opt/drex/requirements.txt; \
    test -f /opt/drex/kev/model.py; \
    test -f /opt/drex/kev/checkpoint.py; \
    python -c "import torch, transformers, pydantic, huggingface_hub; from kev.checkpoint import load, read_meta; from kev.model import DecisionModel, load_backbone, is_hybrid; from kev.edlm import EfficientDLM, EfficientDLMConfig; from kev.api import SystemOneRequest; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| transformers', transformers.__version__)"; \
    if [ -n "${FLASH_LINEAR_ATTENTION_SPEC}" ]; then \
        python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule; assert chunk_gated_delta_rule.__module__.startswith('fla.'), chunk_gated_delta_rule.__module__; assert fused_recurrent_gated_delta_rule.__module__.startswith('fla.'), fused_recurrent_gated_delta_rule.__module__; print('fla kernel path OK:', chunk_gated_delta_rule.__module__)"; \
    fi; \
    python serve.py --help > /dev/null; \
    python inference.py --help > /dev/null

# NVIDIA 基镜像自带 ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]，必须清掉：
# 需求是「用户自己替换启动命令」，留着它会变成 nvidia_entrypoint.sh <你的命令>，
# 并且每次启动多打印一段 CUDA banner。该脚本本身只做 exec "$@" 透传，不做 GPU 注入
# （注入由 nvidia-container-toolkit 在容器启动前完成），所以清掉不影响 GPU 可用性。
#
# 另外全程没有覆盖 LD_LIBRARY_PATH —— 基镜像用它指向自己的 CUDA 路径，覆盖会导致
# 容器里找不到 CUDA 库。PATH 前置 /opt/venv/bin 是安全的。
ENTRYPOINT []

# 上游 serve.py 的默认端口。EXPOSE 只是文档，实际端口仍由你的 -p 决定。
EXPOSE 8000

# 只写 CMD，不写 ENTRYPOINT：docker run <镜像> <你的命令> 才能整体替换默认命令
# （加了 ENTRYPOINT 就必须用 --entrypoint 才能覆盖，不满足「用户自己替换」）。
# 不配 HEALTHCHECK：默认命令是 tail，探活必然失败。
CMD ["tail", "-f", "/dev/null"]
