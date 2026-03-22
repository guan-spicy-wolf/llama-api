# llama-api

一个用于管理 llama.cpp server 的 Web UI 和 API 服务，支持**多模型并发运行**、自动空闲卸载和内存保护。

## 功能

- 🚀 **多模型并发** - 同时运行多个模型，按需加载，独立管理
- ⏱️ **空闲自动卸载** - 默认 5 分钟无请求后自动卸载，节省显存
- 🛡️ **内存保护** - 加载前检查可用内存，避免 OOM 崩溃
- 🖥️ **Web UI** - 简洁的控制面板，可视化监控和操作
- 🔌 **OpenAI 兼容 API** - 无缝对接现有应用
- 📊 **系统监控** - CPU、内存、GPU 实时监控
- ⚡ **流式响应** - 支持 SSE 流式输出
- 🔧 **灵活配置** - YAML 配置文件，支持智能参数推断

## 架构

```
                    ┌─────────────────────────────────────┐
                    │         llama-api (port 8002)        │
   Request ────────►│    ┌─────────────────────────┐       │
   {model: "qwen"}  │    │   ProcessManager        │       │
                    │    │   processes: {          │       │
                    │    │     "qwen": ProcessInfo │       │
                    │    │     "minimax": ...      │       │
                    │    │   }                     │       │
                    │    └──────────┬──────────────┘       │
                    └───────────────┼─────────────────────┘
                                    │ route by model name
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              llama-server    llama-server    llama-server
              (port 18080)    (port 18081)    (port 18082)
              model: qwen     model: minimax  model: ...
```

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

2. 创建模型配置（`/var/containers/llama-api/models.d/qwen.yaml`）：
```yaml
name: qwen
file: Qwen3.5-27B-Q4_K_M.gguf
idle_timeout: 300           # 可选，空闲超时（秒），默认 300
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
| `/api/server/status` | GET | 所有已加载模型状态 |
| `/api/server/status/{model}` | GET | 指定模型状态 |
| `/api/server/metrics` | GET | 系统指标（CPU/内存/GPU） |
| `/api/server/load` | POST | 加载模型 |
| `/api/server/unload` | POST | 卸载指定模型 |
| `/api/server/unload/all` | POST | 卸载所有模型 |

### OpenAI 兼容 API

| 端点 | 方法 | 功能 |
|------|------|------|
| `/v1/models` | GET | 已加载模型列表 |
| `/v1/chat/completions` | POST | 对话补全（需指定 model） |
| `/v1/completions` | POST | 文本补全 |
| `/v1/embeddings` | POST | 向量嵌入 |

### Anthropic 兼容 API

| 端点 | 方法 | 功能 |
|------|------|------|
| `/v1/messages` | POST | 对话补全（Anthropic 格式） |

### 示例

**加载模型：**
```bash
curl -X POST http://localhost:8002/api/server/load \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen"}'
```

**查看状态：**
```bash
curl http://localhost:8002/api/server/status | jq '.'
# {
#   "models": {
#     "qwen": {
#       "port": 18080,
#       "pid": 12345,
#       "idle_remaining": 280,
#       ...
#     }
#   },
#   "total_memory_gb": 15.2
# }
```

**卸载模型：**
```bash
curl -X POST http://localhost:8002/api/server/unload \
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

**Python (OpenAI SDK)：**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8002/v1", api_key="none")

# 请求会自动路由到已加载的 qwen 模型
response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**Anthropic 格式：**
```bash
curl http://localhost:8002/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

返回格式与 Anthropic API 兼容，支持 `thinking` 块、工具调用和流式响应。

## 配置说明

### 主配置 (`/var/containers/llama-api/config.yaml`)

```yaml
# llama-server 可执行文件路径
llama_server: /home/holo/llama.cpp/rocm/bin/llama-server

# 动态库路径
ld_library_path: /home/holo/llama.cpp/rocm/bin

# 模型文件目录
models_dir: /var/models

# 默认空闲超时（秒）
default_idle_timeout: 300
```

### 模型配置 (`/var/containers/llama-api/models.d/*.yaml`)

```yaml
name: qwen                    # 模型名称（API 调用时使用）
file: Qwen3.5-27B-Q4_K_M.gguf # GGUF 文件名
idle_timeout: 300             # 可选，空闲超时（秒），覆盖全局默认
args:
  ctx_size: 8192              # 上下文长度
  n_gpu_layers: 99            # GPU 层数（全量卸载）
  threads: 8                  # CPU 线程数
```

未指定的参数会根据模型文件大小智能推断：
- `ctx_size`: <10B→8192, 10-50B→4096, >50B→2048
- `n_gpu_layers`: 99（全量 GPU 卸载）
- `threads`: CPU 核心数的一半

### 运行时数据

`/var/containers/llama-api/routing.json` - 实时路由状态：

```json
{
  "qwen": {
    "port": 18080,
    "pid": 12345,
    "model": "qwen",
    "started_at": 1774192256.75,
    "last_used_at": 1774192300.12,
    "idle_timeout": 300
  }
}
```

## 目录结构

```
~/llama-api/                  # 项目源码
├── app/
│   ├── main.py               # FastAPI 入口 + 反向代理
│   ├── config.py             # 配置管理
│   ├── models.py             # Pydantic 模型
│   ├── process_manager.py    # 多模型进程管理
│   └── routes/
│       ├── models.py         # 模型 API
│       └── server.py         # 服务控制 API
├── webui/index.html          # Web UI
├── docs/plans/               # 设计文档
├── config.example.yaml       # 配置示例
└── requirements.txt

/var/containers/llama-api/    # 运行时配置和数据
├── config.yaml               # 主配置
├── routing.json              # 实时路由状态
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

## 错误码

| 状态码 | 说明 |
|--------|------|
| 400 | 请求体未指定 model |
| 404 | 模型配置不存在或模型未加载 |
| 503 | 模型未加载，需先调用 /api/server/load |
| 507 | 内存不足，无法加载模型 |

## 许可证

MIT