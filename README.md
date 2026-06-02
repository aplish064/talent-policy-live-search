# Talent Policy Live Search

`talent-policy-live-search` 是一个“实时人才政策搜索”服务：

- 输入地区/学校关键词后，实时抓取候选页面并整理结果；
- 不落库、不存历史，不做数据库持久化；
- 以官方来源为主，并在抓取失败时自动切到 Scrapling 兜底。

---

## 1. 项目介绍（你在做什么）

这个项目的目标是把“查政策”从“试错式爬站”变成“可复用的检索流程”：

- 把用户查询映射到官方实体（省市、大学、机构）；
- 生成发现词和 URL 路径提示，做官方站点发现和 Web 搜索补充；
- 并发抓取候选页面并做正文抽取；
- 用 LLM 做相关性归纳与结构化输出（标题、适用对象、待遇、申报、日期）；
- 返回结果摘要（Markdown）和结构化卡片。

### 1.1 关键边界

- 结果不落地：`PERSIST_RESULTS=false` 固定约束；
- 不缓存历史：每次请求都是独立执行；
- 优先官方域名，非官方来源默认会被过滤/降权。

---

## 2. 技术栈与工具

### 后端

- `FastAPI`：HTTP 接口和页面服务；
- `Pydantic + pydantic-settings`：配置与数据校验；
- `httpx`：异步抓取核心通道；
- `BeautifulSoup (lxml)`：网页正文抽取和链接抽取；
- `PyMuPDF`：PDF 文本抽取；
- `python-docx`：Word 文档正文抽取；
- `yaml`：官方源配置加载。

### LLM 与检索

- MiniMax（兼容 Anthropic API 形态）用于：
  - 查询实体识别/候选关键词生成；
  - 页面内容判定与结构化归纳；
  - 查询结果 Markdown 汇总。
- `AnySearch`：公共检索补充（无匹配时会扩大覆盖面）。

### 兜底抓取

- `Scrapling`：当主抓取失败时的兜底抓取；
- 默认开启：`ENABLE_SCRAPLING_FALLBACK=true`。

### 前端

- 静态 `HTML/CSS/JS` 页面，直接由 FastAPI 静态文件挂载；
- 后端接口为 `/api/search`。

---

## 3. 项目流程（从一条查询到返回结果）

1. 前端提交查询（`query`）。
2. `SearchPipeline` 初始化环境与配置。
3. 读取 `official_sources.yaml` 进行实体匹配（地区、学校、机构）。
4. LLM 给出 `normalized_query` 与实体匹配候选。
5. 生成“发现词/URL 路径提示”。
6. 两类候选来源并行收集：
   - 官方源种子页（含 robots / sitemap / 页面内链接 / feed / 表单）
   - `web_search` 公共检索结果（无匹配时作为补充）
7. 对候选 URL 做去重、域名合法性过滤、优先级打分。
8. 并发抓取候选页：
   - 先用 `httpx` 拉取；
   - 抓取失败时尝试 legacy TLS 重试；
   - 仍失败时进入 `Scrapling` 兜底。
9. 抓取成功后抽取页面全文和结构信息：标题、正文、链接、政策条目、日期、附件、申报入口。
10. 对有效页面做判定与组织：
    - 可继续让 LLM 结构化组织，
    - 低价值/非政策页面会被过滤。
11. 汇总输出：
    - 按评分去重排序；
    - 返回 `results` + `summary_markdown` + 抓取告警。

---

## 4. 目录结构

```text
talent-policy-live-search/
├─ src/talent_policy_search/   # 代码（pipeline、抓取、LLM、抽取、路由）
├─ tests/                      # 测试
├─ config/                     # 官方源配置（YAML）
├─ .env.example                # 环境变量模板（含默认开启 Scrapling）
├─ pyproject.toml              # 依赖与运行定义
└─ README.md
```

---

## 5. 0 到 1 部署

### 5.1 前置条件

- Python 3.11+
- Git
- 可访问公网（用于抓取和 Web 搜索）

### 5.2 安装与配置

```bash
cd <你的工作目录>
git clone <项目仓库地址> talent-policy-live-search
cd talent-policy-live-search

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 默认会用到 Scrapling 兜底，建议直接装上抓取 extras
pip install -e ".[dev,scrape]"
```

```bash
cp .env.example .env
```

编辑 `.env`（至少确认）：

```ini
ANTHROPIC_API_KEY=<你的 MiniMax API Key>
ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic
ANTHROPIC_MODEL=MiniMax-M2.7-highspeed

ENABLE_SCRAPLING_FALLBACK=true
```

### 5.3 启动服务

```bash
source .venv/bin/activate
uvicorn talent_policy_search.server:app --reload --host 127.0.0.1 --port 8767
```

访问方式：

- 前端页：`http://127.0.0.1:8767/`
- 查询接口：`POST http://127.0.0.1:8767/api/search`

### 5.4 快速 API 测试

```bash
curl -X POST "http://127.0.0.1:8767/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"深圳"}'
```

---

## 6. 运行前校验

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 pytest -q
```

---

## 7. 关键返回字段

- `results_returned`：结果数量
- `duration_seconds`：耗时
- `official_sources_checked`：已检查官方源数量
- `candidate_pages_seen`：候选页抓取范围内数量
- `summary_markdown`：LLM 生成的中文摘要
- `results`：结构化政策条目（标题、适用对象、待遇、申报、日期）
- `warnings`：抓取/解析/LLM 相关告警

---

## 8. 生产部署（可选）

```bash
source .venv/bin/activate
uvicorn talent_policy_search.server:app --host 0.0.0.0 --port 8767
```

systemd 示例（`<project_root>` 替换为项目绝对路径）：

```ini
[Unit]
Description=Talent Policy Live Search
After=network.target

[Service]
Type=simple
WorkingDirectory=<project_root>
EnvironmentFile=<project_root>/.env
ExecStart=<project_root>/.venv/bin/uvicorn talent_policy_search.server:app --host 0.0.0.0 --port 8767
Restart=on-failure
RestartSec=3
User=<运行用户>
Group=<运行用户组>

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now talent-policy-live-search
sudo systemctl status talent-policy-live-search
```

