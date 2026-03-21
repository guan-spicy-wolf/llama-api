# llama-api 设计文档

## 概述

一个用于管理 llama.cpp server 的 Web UI 和 API 服务，支持加载/卸载不同模型。

## 技术栈

- **后端**：Python + FastAPI
- **前端**：简单静态 HTML + JavaScript
- **进程管理**：Python subprocess 管理 llama-server
- **配置格式**：YAML

## 目录结构

```
~/llama-api/                    # 项目源码
├── app/
│   ├── main.py                 # FastAPI 入口
│   ├── config.py               # 配置管理
│   ├── models.py               # Pydantic 模型
│   ├── process_manager.py      # llama-server 进程管理
│   └── routes/
│       ├── models.py           # 模型 CRUD API
│       └── server.py           # 加载/卸载 API
├── webui/                      # 静态前端
│   └── index.html
├── config.example.yaml         # 示例配置
└── requirements.txt

/var/containers/llama-api/      # 运行时数据
├── config.yaml                 # 实际配置
└── models.d/                   # 模型配置目录
    ├── minimax.yaml
    └── qwen.yaml
```

## API 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/models` | GET | 列出可用模型及状态 |
| `/api/models/{name}` | GET | 获取单个模型详情 |
| `/api/models/{name}` | PUT | 创建/更新模型配置 |
| `/api/models/{name}` | DELETE | 删除模型配置 |
| `/api/server/load` | POST | 加载指定模型 `{ "model": "minimax" }` |
| `/api/server/unload` | POST | 卸载当前模型 |
| `/api/server/status` | GET | 当前运行状态 |

## 配置管理

### 主配置文件

路径：`/var/containers/llama-api/config.yaml`

```yaml
llama_server: /home/holo/llama.cpp/rocm/bin/llama-server
models_dir: /var/models
default_port: 8080
```

### 模型配置文件

路径：`/var/containers/llama-api/models.d/{name}.yaml`

```yaml
name: minimax
file: MiniMax-139B-Merged.gguf
args:
  ctx_size: 4096
  n_gpu_layers: 99
  threads: 8
```

## 智能参数推断

当模型配置未指定参数时，根据模型文件大小自动推断：

| 参数 | 推断规则 |
|------|----------|
| `n_gpu_layers` | 99（全量卸载到 GPU） |
| `ctx_size` | <10B: 8192, 10-50B: 4096, >50B: 2048 |
| `threads` | CPU 核心数的一半 |

## 进程管理

- 使用 `asyncio.create_subprocess_exec` 启动 llama-server
- 跟踪进程状态（PID、启动时间、stdout/stderr）
- 提供 `/api/server/status` 查询当前状态
- 服务关闭时自动清理子进程

## 部署方式

- 原生运行：`uvicorn app.main:app --host 0.0.0.0 --port 8000`
- 可选：使用 systemd 管理（后续可添加）

## 依赖

```
fastapi
uvicorn
pyyaml
pydantic
aiofiles
```