# drex-decision-models Docker 镜像

把 [nace-ai/drex-decision-models](https://github.com/nace-ai/drex-decision-models) 打包成可直接运行的
Docker 镜像，由 GitHub Actions 在 Ubuntu runner 上构建，导出成 tar 后作为 **workflow artifact** 供你下载。

本仓库**只**包含镜像定义（`Dockerfile` + workflow），不含上游源码 —— 构建时会自动把上游仓库克隆进镜像。
镜像**不推送到任何镜像仓库**（GHCR、Docker Hub 都不用），CI 的产物就是那个 tar.gz。

> Drex 是一族决策模型：传一个 `state` 和若干具名问答（`noul` 是否 / `choice` 多选 / `score` 评分），
> 返回每个选项的概率，**不生成任何文本**。每个 runner 都对外提供 `POST /v1/systemone`。

---

## 拿到镜像

镜像在 CI 里构建，产出一个 `docker save` 格式的 tar.gz。流程：

1. 仓库页面 → **Actions** → 左侧选 **`docker_image_build_artifact`** → **Run workflow**
2. 按需填参数（见[工作流参数](#工作流参数)），然后运行
3. 等构建完成（首次约 20–40 分钟），在该次运行页面底部 **Artifacts** 区域下载 `drex-decision-models-latest-amd64`
4. 解压后是三个文件：`.tar.gz`（镜像本体）、`.sha256`（校验和）、`.manifest.txt`（构建信息）

```bash
unzip drex-decision-models-latest-amd64.zip -d image && cd image

# 校验完整性
sha256sum -c drex-decision-models-latest-amd64.tar.gz.sha256

# 导入到本地 docker，得到 drex-decision-models:latest
docker load -i drex-decision-models-latest-amd64.tar.gz
docker images drex-decision-models
```

> ⚠️ artifact 默认保留 **30 天**后自动删除。这个 tar 是镜像的唯一副本（没有镜像仓库兜底），
> 建议下载后自己留存一份到服务器或对象存储。

前置条件：宿主机装好 NVIDIA 驱动和 nvidia-container-toolkit。驱动版本要求见
[CUDA 版本与基镜像](#cuda-版本与基镜像) —— 简而言之**驱动 ≥ 525**。

```bash
nvidia-smi                                       # 确认驱动与 GPU 可见
docker run --rm --gpus all drex-decision-models:latest nvidia-smi
```

---

## 快速开始

镜像**不含权重**，先把 checkpoint 取到宿主机，再挂进容器。

```bash
# 1) 宿主机上取权重（bf16 约 18 GB）
hf download nace-ai/drex-v1.5 --local-dir weights/drex-v1.5

# 2) 启动服务
docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/weights/drex-v1.5:/models/drex-v1.5:ro" \
  drex-decision-models:latest \
  python serve.py --model /models/drex-v1.5 --host 0.0.0.0 --port 8000

# 3) 另开一个终端调用
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' -d @examples/request.json     # 见下方「例子从哪来」
curl http://127.0.0.1:8000/health
# {"status": "ok", "model": "drex-v1.5"}
```

只跑一次打分（不起服务）：

```bash
docker run --rm --gpus all \
  -v "$PWD/weights/drex-v1.5:/models/drex-v1.5:ro" \
  drex-decision-models:latest \
  python inference.py --model /models/drex-v1.5
```

### 后台运行（服务常驻）

```bash
docker run -d --name drex --restart unless-stopped --gpus all -p 8000:8000 \
  -v "$PWD/weights/drex-v1.5:/models/drex-v1.5:ro" \
  drex-decision-models:latest \
  python serve.py --model /models/drex-v1.5 --host 0.0.0.0 --port 8000

docker logs -f drex                                        # 首次加载约 18 GB 权重，屏住一会儿
curl http://127.0.0.1:8000/health                          # 通了才说明加载完成
docker stop drex && docker rm drex
```

多卡机器上限定用哪几张卡：

```bash
docker run -d --name drex --gpus '"device=0,1"' ...        # 注意这里需要一层引号
```

> 镜像默认命令是 `tail -f /dev/null`（只为保活）。**所有示例都必须显式给出要跑的命令**，
> 不写命令容器只会空转，不会有任何服务。

### 两个使用上的提醒

**① 镜像名之前才是 docker 的参数。** `docker run <镜像> <命令>` 里，镜像名之后的都是**容器内命令**。
下面这样写是把 `--gpus all` 当成 `serve.py` 的参数，不会生效：

```bash
# ✗ 错：--gpus all 给了 python，docker 看不到
docker run drex-decision-models:latest python serve.py --model /models/drex-v1.5 --gpus all

# ✓ 对
docker run --gpus all drex-decision-models:latest python serve.py --model /models/drex-v1.5
```

另外容器命令**不经过 shell**，要 `&&`、重定向或变量展开必须显式套一层：

```bash
docker run --rm --gpus all -p 8000:8000 -v "$PWD/weights/drex-v1.5:/models/drex-v1.5:ro" \
  drex-decision-models:latest \
  sh -c 'python serve.py --model /models/drex-v1.5 --host 0.0.0.0 --port 8000'
```

**② `--host 0.0.0.0` 不能省 —— 这是本镜像最容易踩的坑。**
上游 `serve.py` 的 `--host` 默认是 `127.0.0.1`。容器里监听回环地址，**收不到 `docker -p` 转发来的流量**
（`-p 8000:8000` 是 DNAT 到容器的 eth0 地址），症状是：服务启动正常、日志正常、宿主机却连不上，
**两端都不报错**。所以务必显式传 `--host 0.0.0.0`。

> 不想改命令也可以用 `--network host`（零改动，但失去端口隔离，且只有 Linux 有）。
> 上游本身暴露了 `--host`，所以本镜像**没有加任何 shim**，跑的就是上游原样的 `serve.py`。

### 例子从哪来

上游的 `examples/request.json` 已经在镜像里（`/opt/drex/examples/request.json`）：

```bash
# 容器内直接用
docker run --rm --gpus all -v "$PWD/weights/drex-v1.5:/models/drex-v1.5:ro" \
  drex-decision-models:latest \
  python inference.py --model /models/drex-v1.5 --request examples/request.json

# 或者拷到宿主机，方便边改边试
docker run --rm drex-decision-models:latest cat /opt/drex/examples/request.json > request.json
```

---

## 模型权重与挂载

### 只挂一个目录就够

Drex 的 checkpoint 是**合并好的**：backbone 的 safetensors 分片和 `head.pt` 放在同一个目录里，
**没有 adapter、没有指向别处的基座模型名字**。加载路径读的是本地目录里的 `config.json` 和权重文件，
全程不访问 Hugging Face。

所以「挂载模型」就是挂这一个目录，不存在「只挂 adapter 还得再下基座」那种坑：

```text
weights/drex-v1.5/
├── config.json              # backbone 配置
├── *.safetensors            # backbone 权重分片
├── head.pt                  # 指针头 + 元数据（head_dim / temperature / option_isolation / state_bidir）
└── tokenizer 相关文件        # 分词器
```

两种挂法都行：

```bash
# 只读挂载（推荐）：容器不需要写权重
-v /abs/path/weights/drex-v1.5:/models/drex-v1.5:ro

# 想在容器里跑 hf download 也可以：缓存挂到 /hf-cache（镜像里已设 HF_HOME）
-v /abs/path/hf-cache:/hf-cache
```

### `--model` 必须是本地目录

`kev/checkpoint.py` 直接读 `<--model>/config.json`，所以传 Hugging Face 仓库 ID 会直接报文件不存在。
先把权重 `hf download` 到本地，再传目录路径。

### 模型名来自目录名

`serve.py` 的 `--name` 默认取 `--model` 目录的**目录名**，也就是响应里的 `"model"` 字段和 `/health` 的
`"model"` 字段。上面例子挂在 `/models/drex-v1.5`，所以名字是 `drex-v1.5`，和上游文档一致。
挂到别的路径想保持同样的名字，就显式指定：

```bash
python serve.py --model /models/whatever --name drex-v1.5 --host 0.0.0.0
```

### 离线

镜像完全不依赖网络也能跑（checkpoint 自带 tokenizer 和 config），想彻底断网可以加：

```bash
-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
```

---

## GitHub Actions 工作流

`.github/workflows/docker-build-artifact.yml`。构建环境为 `ubuntu-latest`（这里的 Ubuntu 指的是
**runner**，不是镜像本身 —— 镜像基于 CUDA 基镜像）。

**只支持手动触发**：Actions → `docker_image_build_artifact` → Run workflow。
没有 push / PR 触发，所以合并代码不会自动构建。

### 工作流参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `image_tag` | `latest` | 给镜像打的 tag，决定 tar 里的镜像名 `drex-decision-models:<tag>` |
| `drex_ref` | `main` | 上游 `nace-ai/drex-decision-models` 的 ref：分支 / tag / commit SHA。要可复现的镜像就固定成 tag 或 SHA |
| `flash_linear_attention` | `true` | 是否安装 Gated DeltaNet 快速内核（见[下一节](#gated-deltanet-快速内核)）。**建议保持开启** |
| `causal_conv1d` | `true` | 是否安装同一段代码里的 conv 快速内核（同一节）。**建议保持开启** |
| `cleanup` | `true` | 构建前清理 runner 磁盘。hosted runner 默认空间紧张，**建议保持开启** |

基镜像、Python 版本、依赖版本都由 `Dockerfile` 决定，workflow 不再重复传，避免两处漂移。
（这两个开关也遵守这条：`flash_linear_attention` 映射成一个包名，`causal_conv1d` 只映射成
`true`/`false` —— causal-conv1d 用的是预编译 wheel，文件名长且带 torch 版本号，
写进 workflow 就等于多一个版本来源。）

产物：

- artifact 名 `drex-decision-models-<tag>-amd64`，内含 `.tar.gz` + `.sha256` + `.manifest.txt`
- `.manifest.txt` 里记了镜像 ID、大小、tar 的 sha256、镜像内实际的 Python / torch / torch 的 CUDA / transformers / fla-core / causal_conv1d 版本、上游仓库与 commit、以及产生它的那次 run 链接，方便日后追溯
- 上传时用 `compression-level: 0`（tar.gz 已经压过一遍，再压纯属浪费 CPU 和上传时间）
- 保留 30 天，同一 tag 重跑会直接覆盖旧 artifact

构建使用 Buildx + GitHub Actions 缓存（`type=gha`），CUDA / torch 那两层能被缓存住，重复构建省掉数 GB 下载。
注意 Actions 缓存每个仓库上限 10 GB，镜像偏大时缓存可能被挤掉，属正常现象。

---

## CUDA 版本与基镜像

默认：

```dockerfile
ARG BASE_IMAGE=nvidia/cuda:12.8.2-runtime-ubuntu24.04
FROM ${BASE_IMAGE}
```

### 基镜像的 CUDA 与 torch 的 CUDA

`requirements.txt` 只写了 `torch>=2.6,<2.9`，没有指定 PyTorch 官方索引，所以 torch 是从 PyPI 装的。
**PyPI 的 Linux torch wheel 自带一整套 CUDA 用户态库**（以 `nvidia-*-cu12` 依赖的形式，torch 2.8 带的是
CUDA 12.8 那一套，约 5 GB），运行时优先加载的就是这一套 —— **参与运算的 CUDA 来自 pip，不是基镜像**。
基镜像提供的是目录布局、`/usr/local/cuda`、`LD_LIBRARY_PATH`，以及驱动挂载点。

两者都写 12.8 只是让版本关系一眼看得明白；就算不一致（例如基镜像仍写 12.4）也不影响运行。
真正的约束在宿主机驱动上：CUDA 12.x 的次版本兼容规则允许 12.8 的用户态库跑在**驱动 ≥ 525** 上，
550 驱动的 A40 实测可用。

换基镜像只改这一行 —— 不需要动 workflow（基镜像版本只写在 Dockerfile 里，避免两处漂移）：

```dockerfile
ARG BASE_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04
```

> 换 base 等于同时换 Python：24.04 是 3.12，22.04 是 3.10。要留在 22.04 就得自己解决
> Python 3.12（jammy 官方源没有，得加 deadsnakes PPA），所以这里默认用 24.04。
> 也别换成 `-devel` 变体：那会多塞约 5 GB 工具链，而这里不需要 nvcc。

### 为什么不需要装 Python

Ubuntu 24.04 自带 Python 3.12，`python3` 就是最终用的那个解释器 —— `transformers 5.x` 要求
`>=3.10`，`torch 2.8` 在 `cp312-linux-x86_64` 上有 `manylinux_2_28` wheel（24.04 的 glibc 是 2.39），
FLA 也把 3.11 列为推荐版本（低于它只 warning，不影响正确性）。所以镜像里没有 deadsnakes PPA，
也没有第二个 Python。

### 为什么不用虚拟环境

容器本身就是隔离边界，再套一层 venv 只是多一个目录、多一层 `PATH` 重写，对「这个镜像里装了什么」
没有任何帮助。所以依赖直接装进系统 Python，`python` / `python3` 是同一个解释器。

两个随之而来的细节：

- **镜像里补了 `/usr/local/bin/python` 软链**。Ubuntu 默认不提供 `python` 这个命令（只有 `python3`），
  而上游文档和自己的 README 都写 `python serve.py`，所以把它接上，省的每次都要改写命令。
- **`PIP_BREAK_SYSTEM_PACKAGES=1`**。Ubuntu 24.04 带 PEP 668 的 `EXTERNALLY-MANAGED` 标记来保护
  系统包管理器，不越过它 pip 会直接拒绝安装。这里选择用这个变量显式越过，而**不删那个标记文件** ——
  万一有人真的误装到系统 Python，标记还在，行为依旧可见。

顺带一个好处：Triton 编译自己那个 C 扩展时用 `sysconfig.get_paths()["include"]` 找 `Python.h`，
系统 Python 下就是标准的 `/usr/include/python3.12`（由 `python3-dev` 提供），不用去猜 venv 里的目录。

### 为什么镜像里有一个 C 编译器

需要编译的依赖（`torch` / `tokenizers` / `pydantic-core` / `safetensors` / `regex` / `numpy`）
在 `cp312` Linux x86_64 上**都有 wheel**，装它们不需要编译器。但镜像里**装了 `gcc` +
`libc6-dev` + `python3-dev`（约 150 MB）**，因为 **Triton 在运行期需要编译 C**：

Triton 首次真正 launch 一个 kernel 时，会用 `cc` 现场编译一个 CPython 扩展模块
（`triton/runtime/build.py` 的 `_build()`：`cc <src> -O3 -shared -fPIC -o <name>.so`，
并用 `-I{sysconfig.get_paths()['include']}` 去拿 `Python.h`；那份 C 由 `driver.py` 的
`make_launcher()` 按**每个 kernel 签名**生成）。它不在 import 时编、也不在启动时编，
所以缺编译器时容器能起、`/health` 是 200，只有第一个请求 500。因为这份 C 是跑的时候才
生成的，没法在构建期预编译缓存，**`gcc` 必须留在最终镜像里**，不能「装完再卸掉」。

**没装 `build-essential`**：Triton 只编 C 不编 C++，`g++` / `make` 是白带的。
**也不需要 CUDA devel 变体**：Triton 自带 `ptxas` 和自己那份 CUDA 头
（`<site-packages>/triton/backends/nvidia/include`），`runtime` 基镜像就够，不用 nvcc。

> 这不是预防性猜测，它就是「服务能起来、`/health` 是 200、但每个请求都 500」的原因，
> 详见[故障排查](#故障排查)。

### 只有 amd64

`platforms: linux/amd64`。CUDA 12.8 的 arm64 变体虽然存在，但这套依赖在 arm64 上的 wheel 覆盖面
要单独核实（PyPI 的 torch aarch64 wheel 版本比 x86_64 落后不少），没核实清楚就不放一个「一选就失败」
的开关。需要 arm64 的话，要整体挪 CUDA 版本并同时改基镜像和 workflow 的 runner。

---

## Gated DeltaNet 快速内核

Drex v1.5 的骨干是 `Qwen3_5ForCausalLM`，**混合注意力**：每三层线性注意力（Gated DeltaNet）配一层全注意力。

这一段代码里有**两组**可选快速内核，都是「缺了只 warning、悄悄退回参考实现」，镜像里**默认都装**：
一组是 Gated DeltaNet 自己的递归内核（`fla`，见下），另一组是它 short-conv 用的
`causal_conv1d_fn` / `causal_conv1d_update`（[再下一节](#同一段代码里的-causal_conv1d_fn--causal_conv1d_update)）。

transformers 对这类层是这么取内核的：

```python
@use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
def torch_chunk_gated_delta_rule(...): ...
```

它真正解析的路径是 `fla.ops.gated_delta_rule.chunk_gated_delta_rule`
（映射表在 `transformers/integrations/hub_kernels.py`：`chunk_gated_delta_rule -> ops.gated_delta_rule`、
`fla -> flash-linear-attention`）。

**没有这个包不会报错**，只会 `warning_once` 一句然后退回纯 torch 的参考实现 ——
功能正确，但按 transformers 自己的注释，H100 上「慢一个数量级以上」，而这在构建期看起来一切正常。

所以镜像默认显式安装它，并且在**构建期断言的就是上面那个多级路径**（不是「顶层包能 import」这种
检查不了静默降级的写法）：

```dockerfile
python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule; ..."
```

两个实现细节：

- **装的是裸包名 `flash-linear-attention`，不是 `flash-linear-attention[cuda]`。**
  裸包的依赖只有 `fla-core -> einops`，动不了 torch；而 `[cuda]` extra 会引入
  `torch>=2.7` / `triton>=3.3` 约束，等于给未来的某版 fla 一个把 torch 顶出 `requirements.txt` 范围的机会。
  两个 wheel 装出来的顶层包都叫 `fla`（元包 + `fla-core` 各自装一半文件），所以断言里出现 `fla_core`
  是死代码 —— 断言用的是导入名 `fla`。
- **它和 `requirements.txt` 放在同一次 pip 解析里**（`pip install -r requirements.txt flash-linear-attention`），
  pip 会同时满足 `torch<2.9` 和它自己的约束；真有冲突会**直接构建失败**，而不是让 torch 被悄悄顶出范围。
  `triton` 由 torch 自己带（torch 2.8 依赖 `triton==3.4.0`），不需要额外装。

想出一个「纯参考实现」的对照镜像，把开关关掉即可（会慢很多）：

```
--build-arg FLASH_LINEAR_ATTENTION_SPEC=
```

### 同一段代码里的 `causal_conv1d_fn` / `causal_conv1d_update`

Gated DeltaNet 的 short-conv 部分还有两个同类可选内核，机制和上面完全一样（缺了只 warning、
退回参考实现）。它们由同一个装饰器解析：

```python
@use_kernel_func_from_hub_with_fallback("causal_conv1d_fn", "causal_conv1d")
def torch_causal_conv1d_fn(...): ...
```

`causal_conv1d_fn` / `causal_conv1d_update` 不在 `_KERNELS_INTERNAL_PATH_MAPPINGS` 里
（那张表只覆盖 `ops.*` 那批），所以走默认路径 `{package}.{func_name}` —— 直接从
`causal_conv1d` 包里取这两个名字。镜像里也是**默认装上**的。

**这里有个坑：`causal-conv1d` 在 PyPI 上只有 sdist。** 从 1.0.0 到 1.7.0 的每一版都只发布了
`.tar.gz`，一个 wheel 都没有。所以 `pip install causal-conv1d` 会走进 setuptools 的源码编译路径，
去编 CUDA 扩展 —— 需要 `nvcc` + `g++` + `ninja`，也就是把基镜像换成 `-devel` 变体（多约 5 GB）。

**但这个包其实有官方预编译 wheel，只是放在 GitHub Release 上而不是 PyPI。** 上游的 `setup.py`
自己就带这个逻辑（`CachedWheelsCommand`：先按规则拼出 wheel 名去 Release 下载，404 才回退源码编译）。
所以镜像的做法是**直接下那个 wheel**，完全不进 pip 的 sdist 构建路径 ——
构建期不需要 `nvcc`，也不需要动基镜像。

固定的那个版本：

```
causal_conv1d-1.7.0+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
sha256  a1fcd5d022b67ddfc9e1c291f890004c1c081a03d66b532f6eebe31219168e0e
```

名字里的五个可变位**全部是「镜像里已经确定的事实」的推论，不是猜的**，而且构建期有一条断言
用同一条规则反推一遍、不一致就当场失败：

| 片段 | 由什么决定 | 本项目取值 |
|---|---|---|
| `cu12` | `torch.version.cuda` 的主版本 | torch `2.8.0+cu128` → `12.8` → `cu12` |
| `torch2.8` | `torch.__version__` 的 major.minor | `2.8` |
| `cp312` | 镜像里的 Python | Ubuntu 24.04 自带的 `python3` 就是 3.12 |
| `cxx11abiTRUE` | `torch._C._GLIBCXX_USE_CXX11_ABI` | `True`（见下） |
| `linux_x86_64` | 架构 | 只出 amd64 |

其中只有 ABI 那一位没法从版本号看出来，所以专门查证过：把 PyPI 上 `torch-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl`
里的 `torch/lib/libc10.so` 取出来看符号表 —— `std::__cxx11::basic_string`（即 ABI=1）出现 42 处，
旧 ABI 的 `_ZNSs` 出现 0 处。所以是 `cxx11abiTRUE`。
**选错这一位的后果就是 `import` 时报 `undefined symbol`**，所以它值得单独查一次而不是默认。

装完之后构建期还会验两件事（`Dockerfile` 最后一层）：`import causal_conv1d_cuda` 能把这个 `.so`
载进来并解析掉全部未定义符号（这是那道 ABI 的唯一实证），以及已装版本号等于上面那个固定的
wheel 名（证明确实是它、没被别的东西顶替）。

### 为什么不用换基镜像

不必。`nvidia/cuda:12.8.2-runtime-ubuntu24.04` 已经同时满足这个 wheel 的两个前提：

- **Python 3.12 就是 `cp312` 的来源** —— 换基镜像会改掉 Python 版本，反而让 wheel 名对不上。
- **CUDA 12.8 与 torch 自带的 cu128 对齐**，也和 wheel 的 `cu12` 同大版本。`runtime` 变体提供
  的是 CUDA 目录布局和驱动挂载点；真正被加载的 `libcudart` / `cuBLAS` / `cuDNN` 用户态库来自
  torch 自己的 `nvidia-*-cu12` 依赖（torch 在 import 时以 `RTLD_GLOBAL` 预加载，
  `causal_conv1d_cuda.so` 的 `libcudart` 依赖靠它解析）。

换成 `-devel` 变体不会有任何好处 —— 那 5 GB 工具链只服务于「从源码编译」，而这里恰恰不编。

### 在 A40 上是哪份 kernel

上游发布 wheel 时是按 `setup.py` 里那份 `-gencode` 列表编的，包含
`sm_75 / sm_80 / sm_87 / sm_90 / sm_100 / sm_120`。**A40 是 compute capability 8.6，不在这个列表里**，
但不需要担心：CUDA 的二进制兼容规则允许同一大版本内、次版本不低于目标 cubin 的设备直接跑 ——
A40 会加载 **sm_80 那份 cubin**，不需要 JIT，也不需要 PTX。

驱动侧：550 ≥ 525，满足 CUDA 12.x 的次版本兼容要求（见 [CUDA 版本与基镜像](#cuda-版本与基镜像)）。

### 关掉 / 换版本

关掉（两个内核都会退到参考实现）：

```bash
--build-arg INSTALL_CAUSAL_CONV1D=false
```

想换别的版本（比如上游发了 1.8.0，或你把 torch 顶到了 2.9）：改 `Dockerfile` 里那两行
`CAUSAL_CONV1D_WHEEL` / `CAUSAL_CONV1D_SHA256`。**新名字不用自己推**：不改名字直接构建时，
断言会失败并把「按当前环境该用的名字」原样打出来，复制过去即可。
sha256 从 GitHub API 拿（`digest` 字段，形如 `sha256:...`）：

```bash
curl -s https://api.github.com/repos/Dao-AILab/causal-conv1d/releases/tags/v1.7.0 \
  | python -c "import json,sys; [print(a['name'], a['digest']) for a in json.load(sys.stdin)['assets'] if 'cu12torch2.8' in a['name'] and 'cp312' in a['name'] and 'x86_64' in a['name']]"
```

sha256 是强制的：换了 wheel 但忘了换 sha，构建会直接失败，而不是把一个来路不明的东西装进去。

---

## 上下文长度

默认 16,384 token。上限由骨干决定：**Drex v1.5（hybrid）可到 131,072，Drex DLM（非 hybrid）到 32,768**。

用环境变量调，注意这些值是在 **import 时**读取的，所以必须在进程启动前设好：

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e KEV_CONTEXT=131072 \
  -v "$PWD/weights/drex-v1.5:/models/drex-v1.5:ro" \
  drex-decision-models:latest \
  python serve.py --model /models/drex-v1.5 --host 0.0.0.0 --port 8000
```

| 变量 | 作用 |
|---|---|
| `KEV_CONTEXT` | 统一设定窗口，默认 16384 |
| `KEV_SERVE_MAX_STATE` | 单独覆盖 state 上限 |
| `KEV_SERVE_MAX_BRANCH` | 单独覆盖每题分支上限 |
| `KEV_SERVE_MAX_PACKED` | 单独覆盖整条打包记录上限 |

对非 hybrid 骨干（Drex DLM），把 `KEV_CONTEXT` 设到超过 32,768 会在加载时直接报错并提示降低它。
显存占用随文档长度增长。

---

## 故障排查

### 服务能起来，但每个请求 500（`Failed to find C compiler`）

日志长这样：

```text
fla/utils/_device.py:100: UserWarning: Triton is not supported on current platform, roll back to CPU.
fla/utils/_device.py:177: UserWarning: Triton is not supported on current platform, roll back to CPU.
System One scoring failed: RuntimeError('Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl.')
```

**和 GPU / 驱动 / CUDA 版本无关**，是镜像里缺 C 编译器。已在镜像中修掉（见
[为什么镜像里有一个 C 编译器](#为什么镜像里有一个-c-编译器)）。三件事分开看：

**① `roll back to CPU` 说的不是 torch 的设备。** 它是 `fla/utils/_device.py` 在 Triton 后端探测失败后的
降级警告（`get_available_device()` 和 `get_all_max_shared_mem()` 的 `except` 分支），和「推理跑在 CPU 上」
没有关系。决定模型在哪张卡上的是 `kev/checkpoint.py` 的 `torch.cuda.is_available()`。

**② 这个报错反而证明 GPU 是好的。** Triton 取驱动库目录前会先断言 `libcuda.so.1` 存在
（`driver.py` 里那句断言，失败时文案是 `libcuda.so cannot found!`）。你看到的是 `Failed to find
C compiler`，说明那一步已经过了 —— 驱动可见、`--gpus all` 生效。驱动 550 也满足要求（≥525）。

**③ 为什么只有第一个请求炸。** Triton 的 JIT 编译发生在第一次 kernel launch，不在 import、
也不在启动或权重加载时。所以容器起得来、权重加载正常、`/health` 返回 `{"status":"ok"}`，
然后每个 `/v1/systemone` 都 500。

想确认 GPU 侧确实没问题：

```bash
docker run --rm --gpus all drex-decision-models:latest \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### `cannot find -lcuda`

Triton 编译时用 `-lcuda` 链接，这需要容器里存在 **`libcuda.so` 这个名字**（不只是 `libcuda.so.1`）。
正常配置的 nvidia-container-toolkit（含 CDI）会建好这个软链；个别旧版本 / nixpkgs 场景只挂
`libcuda.so.1`，就会到这一步才失败。检查与临时补救（重启容器即失效）：

```bash
# 先看有没有
docker run --rm --gpus all drex-decision-models:latest \
  sh -c 'ls -l /usr/lib/x86_64-linux-gnu/libcuda.so*'

# 只有 .so.1 的话，起服务前补一个软链
docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/weights/drex-v1.5:/models/drex-v1.5:ro" \
  drex-decision-models:latest \
  sh -c 'ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so \
         && python serve.py --model /models/drex-v1.5 --host 0.0.0.0'
```

### 构建期的 `causal_conv1d` 断言失败

镜像最后一层有两条断言。分清这三种失败：

- **`AssertionError: CAUSAL_CONV1D_WHEEL does not match this image`** —— 环境漂移了（torch 换了
  小版本、基镜像换了 Python、或换了架构），所以固定的 wheel 名不再适用。报错里已经把「按当前环境
  该用的名字」原样打出来了，改 `Dockerfile` 里那两行 `CAUSAL_CONV1D_WHEEL` / `CAUSAL_CONV1D_SHA256` 即可。
- **`ImportError: ... undefined symbol ...`** —— wheel 装进去了但 ABI 不匹配，说明 `cxx11abi` 那一位选错了。
- **`ImportError: libcuda.so.1: cannot open shared object file`** —— 构建机是 runner（没有 GPU、没有驱动），
  而那个 `.so` 如果链接了 `libcuda.so.1` 就载不进来。这是本仓库唯一没法在本地预先验证的一点
  （生产镜像里有驱动，运行期 import 不受影响）。真遇到就从断言里去掉 `import causal_conv1d_cuda`、
  只保留版本号比对。

### 其他无害的日志

- `Current Python version 3.10 is below the recommended 3.11` —— FLA 的建议性提示。镜像现在是 3.12，
  真看到 3.10 说明基镜像被换成了 22.04 系列。
- `causal_conv1d_fn is falling back to its reference PyTorch implementation` —— 镜像默认是**装**的，
  真看到这句说明构建时把 `INSTALL_CAUSAL_CONV1D` 关掉了。（正常构建出来的镜像不会出现：
  构建期有一条 import 断言在挡「没装上但静默降级」这件事。）
- 启动时 `nvidia_entrypoint.sh` 那段 CUDA banner 不再出现，是 `ENTRYPOINT []` 生效了。

---

## 说明与限制

- **仅 GPU**：不提供 CPU 版本。`kev/checkpoint.py` 会按 CUDA → MPS → CPU 依次回退，但本镜像的用途是 NVIDIA GPU。
- **不推送镜像仓库**：CI 的产物只有 artifact，仓库里没有镜像副本，所以 artifact 过期前记得下载留存。
- **不内置模型**：镜像里没有任何权重，checkpoint 靠 `-v` 挂载。
- **驱动要求**：≥ 525（torch 自带的 CUDA 用户态库是 12.8，受 CUDA 12.x 次版本兼容规则约束）。
  550 驱动的 A40 已在跑，满足要求。
- **没有改上游任何源码**：`Dockerfile` 只是把上游克隆进来装依赖。上游 `serve.py` 本来就暴露了 `--host`，
  所以连 shim 都不需要，容器里跑的就是原样的 `serve.py` / `inference.py`。
- **`gcc` 在镜像里，但 `build-essential` 不在**：Triton 运行期要用 C 编译器，见
  [为什么镜像里有一个 C 编译器](#为什么镜像里有一个-c-编译器)。
- **镜像里没有 `nvcc`，也不需要**：Gated DeltaNet 那两个可选内核都不从源码编译 —— `fla` 装 PyPI 上的
  wheel，`causal_conv1d` 装 upstream **Release** 上的预编译 wheel（它只是没有发布到 PyPI）。
  所以基镜像保持在 `runtime` 变体，不必换成 `-devel`（省约 5 GB 工具链）。
- **不用虚拟环境**：依赖直接装在系统 Python 3.12 里，`python` / `python3` 是同一个解释器，
  见[为什么不用虚拟环境](#为什么不用虚拟环境)。
- **ENTRYPOINT 已清空**：NVIDIA 基镜像自带 `/opt/nvidia/nvidia_entrypoint.sh`，本镜像用 `ENTRYPOINT []` 清掉了，
  这样 `docker run <镜像> <你的命令>` 能干净地整体替换默认命令。
- **`LD_LIBRARY_PATH` 保持基镜像的默认值**，没有覆盖 —— 覆盖会导致容器内找不到 CUDA 库。
- **镜像体积**：CUDA 基镜像 + 自带 CUDA 库的 torch wheel，再加上 `causal_conv1d` 的 CUDA kernel
  （约 0.4 GB，因为一个 wheel 里塞了六份不同 compute capability 的 cubin），`docker save` 出的 tar
  未压缩约 8 GB 量级，压缩后约 3–4 GB。**单 artifact 上限 10 GB** 是 GitHub 的硬限制、不可调，
  因此镜像必须压缩后再传 —— 如果哪天压完还超过 10 GB，就得把 `BASE_IMAGE` 换成更瘦的变体，或改用镜像仓库。
- **串行推理**：上游服务用一个锁串行处理推理，但支持批量（请求里带 `requests` 列表，一次最多 1024 条），
  合得来的决策会共享一次前向。
- **运行用户**：容器内以 root 运行，方便直接写挂载目录和访问 GPU。
- **无 HEALTHCHECK**：默认命令是 `tail`，容器探活会失败，所以没有配置探活；真正启动服务后可自行
  `curl http://127.0.0.1:8000/health`。
- **无 VOLUME 声明**：避免 Docker 自动创建匿名卷，模型路径完全由你自己的 `-v` 决定。
- **首次启动很慢是正常的**：要先加载约 18 GB 的 bf16 权重，期间不监听端口。
- **checkpoint 只做安全加载**：`head.pt` 必须只含张量和基本元数据，自带的 Python 文件不会被执行；
  需要自定义 Python 实现的 checkpoint 不支持。
- 本仓库不跟踪上游代码变化，上游更新后需重新构建镜像（手动触发 workflow 即可）。
