# llama-api

一个用于管理 llama.cpp server 的 Web UI 和 API 服务，支持加载/卸载不同模型。

## 功能

- 🖥️ **Web UI** - 简洁的控制面板，一键加载/卸载模型
- 🔌 **OpenAI 兼容 API** - 无缝对接现有应用
- 📊 **系统监控** - CPU、内存、GPU VRAM 实时监控
- ⚡ **流式响应** - 支持 SSE 流式输出
- 🔧 **灵活配置** - YAML 配置文件，支持智能参数推断

## 快速开始

### 安装依赖

```bash
cd ~/llama-api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 配置

1. 创建配置目录：
```bash
sudo mkdir -p /var/containers/llama-api/models.d
sudo chown -R $USER:$USER /var/containers/llama-api
```

2. 复制主配置文件：
```bash
cp config.example.yaml /var/containers/llama-api/config.yaml
```

3. 创建模型配置（`/var/containers/llama-api/models.d/qwen.yaml`）：
```yaml
name: qwen
file: Qwen3.5-27B-Q4_K_M.gguf
args:
  ctx_size: 8192
  n_gpu_layers: 99
  threads: 8
```

### 启动服务

```bash
# 前台运行
uvicorn app.main:app --host 0.0.0.0 --port 8002

# 或使用 systemd（用户服务）
systemctl --user enable llama-api
systemctl --user start llama-api
```

## API 端点

### 管理 API

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/models` | GET | 已配置模型列表 |
| `/api/server/status` | GET | 服务状态 |
| `/api/server/metrics` | GET | 系统指标（CPU/内存/GPU） |
| `/api/server/load` | POST | 加载模型 |
| `/api/server/unload` | POST | 卸载模型 |

### OpenAI 兼容 API

| 端点 | 方法 | 功能 |
|------|------|------|
| `/v1/models` | GET | 模型列表 |
| `/v1/chat/completions` | POST | 对话补全 |
| `/v1/completions` | POST | 文本补全 |
| `/v1/embeddings` | POST | 向量嵌入 |

### 示例

**加载模型：**
```bash
curl -X POST http://localhost:8002/api/server/load \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen"}'
```

**对话补全：**
```bash
curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

**流式响应：**
```bash
curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```

**Python (OpenAI SDK)：**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8002/v1", api_key="none")

response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## 配置说明

### 主配置 (`/var/containers/llama-api/config.yaml`)

```yaml
# llama-server 可执行文件路径
llama_server: /home/holo/llama.cpp/rocm/bin/llama-server

# 动态库路径
ld_library_path: /home/holo/llama.cpp/rocm/bin

# 模型文件目录
models_dir: /var/models

# 默认端口（内部使用，对外统一 8002）
default_port: 8080
```

### 模型配置 (`/var/containers/llama-api/models.d/*.yaml`)

```yaml
name: qwen                    # 模型名称（API 调用时使用）
file: Qwen3.5-27B-Q4_K_M.gguf # GGUF 文件名
args:
  ctx_size: 8192              # 上下文长度
  n_gpu_layers: 99            # GPU 层数（全量卸载）
  threads: 8                  # CPU 线程数
```

未指定的参数会根据模型文件大小智能推断：
- `ctx_size`: <10B→8192, 10-50B→4096, >50B→2048
- `n_gpu_layers`: 99（全量 GPU 卸载）
- `threads`: CPU 核心数的一半

## 目录结构

```
~/llama-api/                  # 项目源码
├── app/
│   ├── main.py               # FastAPI 入口 + 反向代理
│   ├── config.py             # 配置管理
│   ├── models.py             # Pydantic 模型
│   ├── process_manager.py    # llama-server 进程管理
│   └── routes/
│       ├── models.py         # 模型 API
│       └── server.py         # 服务控制 API
├── webui/index.html          # Web UI
├── config.example.yaml       # 配置示例
├── models.d.example/         # 模型配置示例
└── requirements.txt

/var/containers/llama-api/    # 运行时配置
├── config.yaml
└── models.d/
    ├── qwen.yaml
    └── ...
```

## systemd 服务

```bash
# 安装服务
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/llama-api.service << 'EOF'
[Unit]
Description=llama-api - LLM model management service
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/holo/llama-api
ExecStart=/home/holo/llama-api/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8002
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF

# 启用并启动
systemctl --user daemon-reload
systemctl --user enable llama-api
systemctl --user start llama-api

# 查看状态
systemctl --user status llama-api
journalctl --user -u llama-api -f
```

## 防火墙

```bash
sudo firewall-cmd --permanent --add-port=8002/tcp
sudo firewall-cmd --reload
```

## 许可证

MIT