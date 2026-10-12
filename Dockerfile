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

ARG BASE_IMAGE=nvidia/cuda:12.8.2-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ARG UPSTREAM_REPO=https://github.com/nace-ai/drex-decision-models.git
ARG UPSTREAM_REF=main
# Gated DeltaNet 快速内核。必须是「裸包名」，不要写成 flash-linear-attention[cuda]：
# 裸包的依赖只有 fla-core -> einops，动不了 torch；[cuda] extra 会引入 torch>=2.7 / triton>=3.3
# 约束，等于给未来的某版 fla 一个把 torch 顶出 requirements.txt 范围的机会。
# 置空即不装（--build-arg FLASH_LINEAR_ATTENTION_SPEC=），但 hybrid 骨干会静默退回慢速参考实现。
ARG FLASH_LINEAR_ATTENTION_SPEC=flash-linear-attention

# causal_conv1d：Gated DeltaNet 里另外那两个可选内核（causal_conv1d_fn / causal_conv1d_update）。
# PyPI 上这个包**只有 sdist**（1.7.0 也没有任何 wheel），`pip install causal-conv1d` 会走源码编译
# CUDA 扩展 —— 要 nvcc + g++ + ninja，等于把基镜像换成 -devel 变体。upstream 自己在 GitHub
# Release 上发布了预编译 wheel，所以这里直接下 wheel，**构建期永远不需要 nvcc**。
#
# wheel 名字里的五个可变位全部由「镜像里实际装了什么」决定，不是猜的，装之前有一条断言核对：
#   cu12     <- torch.version.cuda 的主版本（torch 2.8.0+cu128 -> 12.8 -> cu12）
#   torch2.8 <- torch.__version__ 的 major.minor
#   cp312    <- Ubuntu 24.04 的 python3 就是 3.12
#   abiTRUE  <- torch._C._GLIBCXX_USE_CXX11_ABI（从 PyPI torch wheel 里的 libc10.so 符号核实过：
#               std::__cxx11::basic_string 42 处、旧 ABI 的 _ZNSs 0 处）
#   x86_64   <- 只出 amd64
# 换 torch / 换 Python / 换架构时改下面两行即可 —— 断言失败时会把该用的新名字原样打出来。
# 关掉：--build-arg INSTALL_CAUSAL_CONV1D=false（那两个内核会静默退回参考实现）。
#
# 这里用一个显式布尔开关，而不是像 fla 那样用「空串哨兵」：fla 的 spec 本身是个包名，
# 一个字符串天然同时表达「装哪个」和「装不装」；而这个 wheel 的文件名很长且带版本，
# workflow 不该重复携带它（基镜像 / Python / 依赖版本一律由 Dockerfile 拥有），
# 所以「装不装」只能由布尔来表。
ARG INSTALL_CAUSAL_CONV1D=true
ARG CAUSAL_CONV1D_WHEEL=causal_conv1d-1.7.0+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
ARG CAUSAL_CONV1D_SHA256=a1fcd5d022b67ddfc9e1c291f890004c1c081a03d66b532f6eebe31219168e0e

# 不用虚拟环境：容器本身就是隔离边界，再套一层 venv 只是多一个目录、多一层 PATH 重写，
# 对「这个镜像里装了什么」没有任何帮助。直接装进系统 Python，于是 python / python3 是
# 同一个解释器（下面用 /usr/local/bin/python 软链保证 `python` 这个命令存在 ——
# Ubuntu 不提供它，而上游文档和自己的 README 都写 `python serve.py`）。
#
# PEP 668 的 EXTERNALLY-MANAGED 标记是 Debian/Ubuntu 用来保护**系统包管理器**的，
# 在一次性镜像里没有意义；用 PIP_BREAK_SYSTEM_PACKAGES=1 显式越过它，
# 而不是去删那个标记文件（保留它，让「有人误装到系统 Python」这件事仍然可见）。
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/drex \
    HF_HOME=/hf-cache

# 用 ubuntu24.04 而不是 22.04，就是为了 Python 3.12：24.04 的 python3 本身就是 3.12，
# 一行 apt 装完即可，不需要任何第三方源。jammy 的官方源里没有 3.12，为它在构建期引入
# deadsnakes PPA 不划算（多一个外部依赖，且 PPA 包不受发行版安全更新覆盖）。
# 顺带的好处：基镜像的 CUDA 与 torch 自带的 cu128 对齐了，不再有「tag 12.4 / 实际 12.8」的困惑；
# 12.8.2 也仍在 NVIDIA 的受支持标签列表里（12.4.1 已经不在）。
#
# 3.12 满足所有硬要求：transformers 5.x 要求 >=3.10；torch 2.8 在 cp312-linux-x86_64 上有
# manylinux_2_28 wheel（24.04 的 glibc 是 2.39）。FLA 也会把 3.11 列为「推荐」版本，
# 低于它只 warning，不影响正确性。
#
# gcc / libc6-dev / python3-dev 是**运行期**必需，不是为了编译依赖 ——
# 需要编译的依赖（torch / tokenizers / pydantic-core / safetensors / regex / numpy）
# 在 cp312 linux x86_64 上都有 wheel，装它们不需要编译器。
#
# 真正的理由是 Triton：它在**第一次真正 launch kernel 时**用 cc 现场编译一个 CPython
# 扩展模块（triton/runtime/build.py 的 _build()：cc <src> -O3 -shared -fPIC -o <name>.so，
# 还要 -I{sysconfig.get_paths()['include']} 去拿 Python.h；driver.py 的 CudaLauncher
# 调 make_launcher() 生成 C 再交给它，**每个 kernel 签名各一份**）。
# 缺编译器时它不会在任何地方提前失败：import 正常、权重正常加载、服务正常起来、
# /health 也是 200 —— 只有第一个请求 500，报
#   RuntimeError('Failed to find C compiler. Please specify via CC environment variable
#                 or set triton.knobs.build.impl.')
# 因为这份 C 是跑的时候才生成的，没法在构建期预编译缓存掉，也就不能
# 「装完 gcc 再卸掉」—— gcc 必须留在最终镜像里。
#
# 仍然不装 build-essential：Triton 只编译 C（不编 C++），所以 g++/make 是白带的；
# libc6-dev 给 stdio.h 之类（--no-install-recommends 下不会被 gcc 带进来），
# python3-dev 给 Python.h。Triton 自带 ptxas 和自己那份 CUDA 头
# （include_dirs = <site-packages>/triton/backends/nvidia/include），
# 所以 runtime 变体就够，不需要 devel、不需要 nvcc。代价约 150 MB。
#
# 这个判断后来被验证过一次：causal-conv1d 在 PyPI 上只剩 sdist，本来确实需要 nvcc，
# 但解法不是把工具链装进来，而是改从 upstream 的 Release 下预编译 wheel（见上面的 ARG）。
# curl 就是为那次下载准备的（基镜像里其实有，这里显式声明，不依赖基镜像的隐藏细节）。
# libgomp1 是 torch 运行时的 OpenMP 依赖，体积很小，装上省掉一整类运行期报错。
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip \
        gcc \
        libc6-dev \
        python3-dev \
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
# （nvidia-*-cu12 依赖），运行时优先加载的是这一套 —— 基镜像的 CUDA 版本只是工具链布局和
# 驱动挂载点，不参与运算。因此这里不用 PyTorch 官方索引，也不该动它。见 README。
#
# flash-linear-attention 和 requirements.txt 放在同一次 pip 解析里：
# pip 会同时满足 torch<2.9 和它自己的约束，避免它把 torch 悄悄顶到范围外。
#
# 装进系统 Python（没有 venv，见上面 ENV 处的说明）。顺带一个好处：Triton 编译自己那个
# C 扩展时用 sysconfig.get_paths()["include"] 找 Python.h，系统 Python 下就是标准的
# /usr/include/python3.12（由 python3-dev 提供），不用去猜 venv 里的 include 目录。
#
# **刻意不升级 pip**：pip 由 apt 的 python3-pip 提供（Ubuntu 24.04 → pip 24.0），它的文件
# 由 dpkg 拥有、没有 dist-info/RECORD，所以 `pip install --upgrade pip` 必然在卸载那一步
# 失败（实测）：
#     ERROR: Cannot uninstall pip 24.0, RECORD file not found. Hint: The package was installed by debian.
# 绕开它只能加 --ignore-installed，那会让新版落到 /usr/local 与 dpkg 那份并存 —— 镜像里留
# 两份 pip，只为换一个我们并没有用到的更新特性，不划算。直接用系统自带的那份。
RUN set -eux; \
    ln -sf /usr/bin/python3 /usr/local/bin/python; \
    if [ -n "${FLASH_LINEAR_ATTENTION_SPEC}" ]; then \
        python -m pip install -r /opt/drex/requirements.txt "${FLASH_LINEAR_ATTENTION_SPEC}"; \
    else \
        python -m pip install -r /opt/drex/requirements.txt; \
    fi; \
    mkdir -p "${HF_HOME}"

# 下 upstream 发布的预编译 wheel 装 causal_conv1d —— 完全不碰 pip 的 sdist 构建路径，
# 所以镜像里没有 nvcc / g++ / ninja，也不需要把基镜像换成 -devel 变体。
#
# 为什么要先断言：wheel 名里的 cu12 / torch2.8 / cp312 / abiTRUE 是「当前环境的推论」，
# 一旦 requirements.txt 把 torch 顶到 2.9、或基镜像换掉 Python 版本，这个名字就错了。
# 名字错了的症状本来是 curl 的 404（或者更糟：pip 404 后转去源码编译，报一个
# 「Failed to find C compiler」这种指不到真正原因的错误）。所以这里先用同一条命名规则
# 反推出该用的名字，不一致就当场退出并把新名字打出来。
#
# 下载 / 校验 / 安装 / 删包在同一个 RUN 里：187 MB 的 .whl 不会留在任何镜像层里。
# URL 里的 "+" 编码成 %2B：路径里裸 "+" 合规，但中间代理有把它当空格处理的。
# --no-deps：它的 install_requires 是 torch / packaging / ninja，前两个本来就已经装好，
#            ninja 只服务于「从源码编译」那条路，而那条路我们不走。
# 唯一性的保证来自 sha256：换了 wheel 却不换 sha 会直接构建失败，而不是装进一个来路不明的东西。
RUN set -eux; \
    if [ "${INSTALL_CAUSAL_CONV1D}" = "true" ]; then \
        python -c "import os, sys, platform, torch; w = os.environ['CAUSAL_CONV1D_WHEEL']; ver = w.split('-')[1].split('+')[0]; py = '%d%d' % sys.version_info[:2]; expect = 'causal_conv1d-%s+cu%storch%scxx11abi%s-cp%s-cp%s-linux_%s.whl' % (ver, torch.version.cuda.split('.')[0], '.'.join(torch.__version__.split('+')[0].split('.')[:2]), str(torch._C._GLIBCXX_USE_CXX11_ABI).upper(), py, py, 'aarch64' if platform.machine().lower() in ('aarch64', 'arm64') else 'x86_64'); assert expect == w, 'CAUSAL_CONV1D_WHEEL does not match this image; expected:\n  %s\ngot:\n  %s' % (expect, w); print('causal-conv1d wheel pin matches image:', w)"; \
        PKG_VERSION="$(printf '%s' "${CAUSAL_CONV1D_WHEEL}" | sed 's/^causal_conv1d-//; s/+.*$//')"; \
        URL_NAME="$(printf '%s' "${CAUSAL_CONV1D_WHEEL}" | sed 's/+/%2B/g')"; \
        curl -fsSL --retry 3 --retry-delay 2 -o /tmp/causal_conv1d.whl \
            "https://github.com/Dao-AILab/causal-conv1d/releases/download/v${PKG_VERSION}/${URL_NAME}"; \
        echo "${CAUSAL_CONV1D_SHA256}  /tmp/causal_conv1d.whl" | sha256sum -c -; \
        python -m pip install --no-deps /tmp/causal_conv1d.whl; \
        rm -f /tmp/causal_conv1d.whl; \
    fi

# 运行期工具链自检：Triton 的 _build() 需要「找得到 cc」+「Python.h」。
# 这里不写 `which gcc` 这种检查 —— gcc 在 PATH 里但 Python.h 不在、或者 sysconfig 的
# include 路径解析错，都会一路静默到第一个请求才 500。所以直接调 Triton 自己的编译入口
# 跑一次真编译：编出来的模块能被 import、函数能被调用，才算这条链是通的。
# （C 源码里的引号用 chr(34) 拼，避免在 Dockerfile + shell 双层引号里打结；
#   不传 library_dirs/libraries，因此不需要 libcuda —— 构建机没有 GPU 也能跑。）
RUN set -eux; \
    python -c "import triton; from triton.runtime.build import compile_module_from_src; q = chr(34); src = '#include <Python.h>\nstatic PyObject* f(PyObject* a, PyObject* b){ return PyLong_FromLong(42); }\nstatic PyMethodDef M[] = {{' + q + 'f' + q + ', f, METH_NOARGS, NULL}, {NULL, NULL, 0, NULL}};\nstatic struct PyModuleDef D = {PyModuleDef_HEAD_INIT, ' + q + 'triton_cc_probe' + q + ', NULL, -1, M};\nPyMODINIT_FUNC PyInit_triton_cc_probe(void){ return PyModule_Create(&D); }'; m = compile_module_from_src(src, 'triton_cc_probe'); assert m.f() == 42, m.f(); print('triton runtime toolchain OK | triton', triton.__version__)"

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
#
# causal_conv1d 那条断言同理，但它查的是两件不同的事：
# ① `import causal_conv1d_cuda` 会加载那个编译好的 .so 并解析它的全部未定义符号 ——
#    这是这道 ABI 的唯一实证：wheel 打得对不对（cxx11abiTRUE）、torch 能不能满足它，
#    错了就在这里爆 `undefined symbol`，而不是在用户的第一次请求里。
#    （先 import torch 再 import 它：torch 在 import 时把自己那套 nvidia-*-cu12 用户态库
#      以 RTLD_GLOBAL 预加载进来，.so 的 libcudart 依赖要靠它才能解析到。）
# ② 已装版本号必须等于固定的那个 wheel（含 +cu12torch2.8cxx11abiTRUE 这段 local version）——
#    证明装进去的确实是 INSTALL/下载那一步指定的那个文件，而不是被别的什么顶替了。
#    符号名同理：causal_conv1d_fn / causal_conv1d_update 就是 transformers 从
#    `causal_conv1d` 这个包里取的两个名字（它不在 _KERNELS_INTERNAL_PATH_MAPPINGS 里，
#    走的是 `{package}.{func_name}` 默认路径），所以断言里直接 import 这两个名字。
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
    if [ "${INSTALL_CAUSAL_CONV1D}" = "true" ]; then \
        python -c "import os, torch, importlib.metadata as M, causal_conv1d_cuda; from causal_conv1d import causal_conv1d_fn, causal_conv1d_update; want = os.environ['CAUSAL_CONV1D_WHEEL'].rsplit('.whl', 1)[0].split('-')[1]; got = M.version('causal_conv1d'); assert got.split('+')[0] == want.split('+')[0] and got.split('+')[-1].lower() == want.split('+')[-1].lower(), (got, want); print('causal_conv1d OK |', got, '|', causal_conv1d_cuda.__file__)"; \
    fi; \
    python serve.py --help > /dev/null; \
    python inference.py --help > /dev/null

# NVIDIA 基镜像自带 ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]，必须清掉：
# 需求是「用户自己替换启动命令」，留着它会变成 nvidia_entrypoint.sh <你的命令>，
# 并且每次启动多打印一段 CUDA banner。该脚本本身只做 exec "$@" 透传，不做 GPU 注入
# （注入由 nvidia-container-toolkit 在容器启动前完成），所以清掉不影响 GPU 可用性。
#
# 另外全程没有覆盖 LD_LIBRARY_PATH —— 基镜像用它指向自己的 CUDA 路径，覆盖会导致
# 容器里找不到 CUDA 库。PATH 也没有动，保持基镜像那一份。
ENTRYPOINT []

# 上游 serve.py 的默认端口。EXPOSE 只是文档，实际端口仍由你的 -p 决定。
EXPOSE 8000

# 只写 CMD，不写 ENTRYPOINT：docker run <镜像> <你的命令> 才能整体替换默认命令
# （加了 ENTRYPOINT 就必须用 --entrypoint 才能覆盖，不满足「用户自己替换」）。
# 不配 HEALTHCHECK：默认命令是 tail，探活必然失败。
CMD ["tail", "-f", "/dev/null"]
