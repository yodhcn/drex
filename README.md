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
| `cleanup` | `true` | 构建前清理 runner 磁盘。hosted runner 默认空间紧张，**建议保持开启** |

基镜像、Python 版本、依赖版本都由 `Dockerfile` 决定，workflow 不再重复传，避免两处漂移。

产物：

- artifact 名 `drex-decision-models-<tag>-amd64`，内含 `.tar.gz` + `.sha256` + `.manifest.txt`
- `.manifest.txt` 里记了镜像 ID、大小、tar 的 sha256、镜像内实际的 Python / torch / torch 的 CUDA / transformers / fla-core 版本、上游仓库与 commit、以及产生它的那次 run 链接，方便日后追溯
- 上传时用 `compression-level: 0`（tar.gz 已经压过一遍，再压纯属浪费 CPU 和上传时间）
- 保留 30 天，同一 tag 重跑会直接覆盖旧 artifact

构建使用 Buildx + GitHub Actions 缓存（`type=gha`），CUDA / torch 那两层能被缓存住，重复构建省掉数 GB 下载。
注意 Actions 缓存每个仓库上限 10 GB，镜像偏大时缓存可能被挤掉，属正常现象。

---

## CUDA 版本与基镜像

默认：

```dockerfile
ARG BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM ${BASE_IMAGE}
```

### 为什么基镜像写 12.4 而 torch 其实是 12.8

`requirements.txt` 只写了 `torch>=2.6,<2.9`，没有指定 PyTorch 官方索引，所以 torch 是从 PyPI 装的。
**PyPI 的 Linux torch wheel 自带一整套 CUDA 用户态库**（以 `nvidia-*-cu12` 依赖的形式，torch 2.8 带的是
CUDA 12.8 那一套，约 5 GB），运行时优先加载的就是这一套。

也就是说：基镜像的 CUDA 12.4 只是**工具链布局和驱动挂载点**，不参与运算。
「基镜像 tag 写 12.4、torch 实际跑 12.8」是这种情况下的正常现象，不是镜像标错了。

真正约束在宿主机驱动上：CUDA 12.x 的次版本兼容规则允许 12.8 的用户态库跑在**驱动 ≥ 525** 上。

想严格一致（基镜像 CUDA 也换成 12.8），改一行即可 —— 不需要动别的地方：

```dockerfile
ARG BASE_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu22.04
```

### 为什么不需要装 Python

Ubuntu 22.04 自带 Python 3.10，刚好够用：`transformers 5.x` 要求 `>=3.10`，
`torch 2.8` 在 `cp310-linux-x86_64` 上有 `manylinux_2_28` wheel（Ubuntu 22.04 的 glibc 是 2.35）。
所以镜像里没有 deadsnakes PPA，也没有第二个 Python。

### 为什么没有 build-essential

需要编译的依赖（`torch` / `tokenizers` / `pydantic-core` / `safetensors` / `regex` / `numpy`）
在 `cp310` Linux x86_64 上**都有 wheel**，构建期不需要编译器，所以没有把 `build-essential`
装进最终镜像（省掉约 250 MB）。若将来某个包只剩源码包，构建会因为缺 `gcc` 直接失败，那时再补即可。

### 只有 amd64

`platforms: linux/amd64`。CUDA 12.4 的 arm64 变体虽然存在，但这套依赖在 arm64 上的 wheel 覆盖面
要单独核实（PyPI 的 torch aarch64 wheel 版本比 x86_64 落后不少），没核实清楚就不放一个「一选就失败」
的开关。需要 arm64 的话，要整体挪 CUDA 版本并同时改基镜像和 workflow 的 runner。

---

## Gated DeltaNet 快速内核

Drex v1.5 的骨干是 `Qwen3_5ForCausalLM`，**混合注意力**：每三层线性注意力（Gated DeltaNet）配一层全注意力。

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

### 还有个可选的 `causal-conv1d`

Qwen3.5 的同一段代码里还有 `causal_conv1d_fn` / `causal_conv1d_update` 两个可选内核，
同样缺了就退回参考实现。**镜像默认没装**，因为 `causal-conv1d` 在 PyPI 上**只有源码包**
（1.7.0 只有 `.tar.gz`，没有任何 wheel），装它要现场编译 CUDA 扩展（需要 nvcc + `build-essential`），
成本远高于收益。真要用的话，在 `Dockerfile` 里补上工具链再装：

```dockerfile
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential ninja-build; \
    rm -rf /var/lib/apt/lists/*; \
    "${VIRTUAL_ENV}/bin/pip" install causal-conv1d
```

（放在基镜像的 CUDA devel 变体上更省事，因为 `runtime` 变体没有 nvcc。）

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

## 说明与限制

- **仅 GPU**：不提供 CPU 版本。`kev/checkpoint.py` 会按 CUDA → MPS → CPU 依次回退，但本镜像的用途是 NVIDIA GPU。
- **不推送镜像仓库**：CI 的产物只有 artifact，仓库里没有镜像副本，所以 artifact 过期前记得下载留存。
- **不内置模型**：镜像里没有任何权重，checkpoint 靠 `-v` 挂载。
- **驱动要求**：≥ 525（torch 自带的 CUDA 用户态库是 12.8，受 CUDA 12.x 次版本兼容规则约束）。
- **没有改上游任何源码**：`Dockerfile` 只是把上游克隆进来装依赖。上游 `serve.py` 本来就暴露了 `--host`，
  所以连 shim 都不需要，容器里跑的就是原样的 `serve.py` / `inference.py`。
- **`build-essential` 不在镜像里**：见[上面](#为什么没有-build-essential)。
- **ENTRYPOINT 已清空**：NVIDIA 基镜像自带 `/opt/nvidia/nvidia_entrypoint.sh`，本镜像用 `ENTRYPOINT []` 清掉了，
  这样 `docker run <镜像> <你的命令>` 能干净地整体替换默认命令。
- **`LD_LIBRARY_PATH` 保持基镜像的默认值**，没有覆盖 —— 覆盖会导致容器内找不到 CUDA 库。
- **镜像体积**：CUDA 基镜像 + 自带 CUDA 库的 torch wheel，`docker save` 出的 tar 未压缩约 8 GB 量级，
  压缩后约 3–4 GB。**单 artifact 上限 10 GB** 是 GitHub 的硬限制、不可调，因此镜像必须压缩后再传 ——
  如果哪天压完还超过 10 GB，就得把 `BASE_IMAGE` 换成更瘦的变体，或改用镜像仓库。
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
