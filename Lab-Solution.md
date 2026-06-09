# Lab Solution — A2A Multi-Agent Codelab

**Ngày thực hiện:** 09/06/2026  
**Môi trường:** Python 3.13, LangGraph 1.x, LangChain, A2A SDK, OpenRouter API  
**Model:** `anthropic/claude-sonnet-4-5` (qua OpenRouter)

---

## Phần 3 — Single Agent (ReAct Loop)

### Bài Tập 3.1: Thêm tool `search_case_law`

**File:** `stages/stage_3_single_agent/main.py`

```python
@tool
def search_case_law(keywords: str) -> str:
    """Tìm kiếm án lệ theo từ khóa."""
    cases = {
        "breach":     "Hadley v. Baxendale (1854) - Consequential damages",
        "negligence": "Donoghue v. Stevenson (1932) - Duty of care",
        "contract":   "Carlill v. Carbolic Smoke Ball Co (1893) - Unilateral contract",
    }
    for key, case in cases.items():
        if key in keywords.lower():
            return case
    return "Không tìm thấy án lệ phù hợp"
```

Tool được thêm vào `TOOLS` list. Agent tự quyết định khi nào cần gọi nó trong ReAct loop.

### Bài Tập 3.2: Debug agent reasoning

`verbose=True` **không tương thích** với LangGraph v1.0+ (`create_react_agent`). Tham số này thuộc API cũ của LangChain agents. Giải pháp: dùng `stream_mode="updates"` thay thế để quan sát từng bước:

```python
async for chunk in graph.astream(inputs, stream_mode="updates"):
    for node_name, update in chunk.items():
        # node_name cho biết node nào vừa chạy xong
```

### Kết quả chạy Stage 3

Agent tự động gọi **5 tools song song** trong một lần THINK + ACT:

```
[Step 1] THINK + ACT
  Tool: search_legal_database  → data privacy statutes
  Tool: search_legal_database  → tax evasion statutes
  Tool: check_compliance_requirements → startup + technology
  Tool: calculate_penalty  → data_privacy, high, $5M
  Tool: calculate_penalty  → tax_evasion, high, $5M

[Step 2-6] OBSERVE  (5 tool results)

[Step 7] FINAL ANSWER  (tổng hợp đầy đủ)
```

**Điểm khác biệt với Stage 2:** Agent tự quyết định tool nào cần gọi, gọi bao nhiêu lần — không cần manual orchestration loop.

---

## Phần 4 — Multi-Agent In-Process

### Bài Tập 4.1: Thêm `privacy_agent`

**File:** `stages/stage_4_milti_agent/main.py`

Các thay đổi đã thực hiện:

1. **Thêm tool `search_privacy_law`** — knowledge base về GDPR, CCPA, breach notification
2. **Mở rộng `LegalState`** — thêm `needs_privacy: bool` và `privacy_result: Annotated[str, _last_wins]`
3. **Thêm node `call_privacy_specialist`** — ReAct agent chuyên GDPR/CCPA

```python
async def call_privacy_specialist(state: LegalState) -> dict:
    from langgraph.prebuilt import create_react_agent
    privacy_prompt = (
        "You are a data protection attorney specialising in GDPR, CCPA, and global privacy law. "
        "Use the search_privacy_law tool to ground your analysis. Keep your response under 200 words."
    )
    agent = create_react_agent(model=get_llm(), tools=[search_privacy_law], prompt=privacy_prompt)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})
    return {"privacy_result": result["messages"][-1].content}
```

### Bài Tập 4.2: Conditional routing

Cập nhật `check_routing` để LLM nhận diện `needs_privacy`, và `route_to_specialists` dispatch `Send("call_privacy_specialist", state)`:

```python
# check_routing — thêm needs_privacy vào JSON routing prompt
'needs_privacy = true → question involves data privacy, GDPR, CCPA, data breach, user data'

# route_to_specialists
if state.get("needs_privacy"):
    sends.append(Send("call_privacy_specialist", state))
```

### Kết quả chạy Stage 4

```
[Node: analyze_law]          Done (1547 chars)
[Node: check_routing]        needs_tax=True, needs_compliance=False, needs_privacy=True
[Node: call_privacy_specialist]  Done (1317 chars)   ← chạy song song
[Node: call_tax_specialist]      Done (1441 chars)   ← chạy song song
[Node: aggregate]            Done (3027 chars)
```

Privacy Agent và Tax Agent **chạy song song** nhờ LangGraph Send API — Privacy hoàn thành trước Tax trong lần chạy này.

**Bug đã fix:** Xóa import lỗi `from customer_agent import graph` và `IPython.display` (crash trong terminal).

---

## Phần 5 — Distributed A2A System

### Bước 1: Khởi động hệ thống (Windows)

`start_all.sh` không chạy trực tiếp trên Windows. Dùng Bash tool với background jobs:

```bash
mkdir -p logs
uv run python -m registry          > logs/registry.log          2>&1 &
sleep 2
uv run python -m tax_agent         > logs/tax_agent.log         2>&1 &
uv run python -m compliance_agent  > logs/compliance_agent.log  2>&1 &
sleep 4
uv run python -m law_agent         > logs/law_agent.log         2>&1 &
sleep 3
uv run python -m customer_agent    > logs/customer_agent.log    2>&1 &
sleep 4
uv run python test_client.py
```

**Thứ tự khởi động quan trọng:** Registry trước, leaf agents (Tax, Compliance) trước orchestrators (Law, Customer), vì leaf agents tự đăng ký với Registry khi start.

### Bài Tập 5.1: Trace request flow

Từ logs, `trace_id = 3b1ca03d-f707-4c0c-a299-45abed42fa0a` lan truyền xuyên suốt:

| Agent | Port | Depth | Timestamp |
|---|---|---|---|
| Customer Agent | 10100 | 0 | 16:17:57 |
| Law Agent | 10101 | 1 | 16:18:04 |
| Tax Agent | 10102 | 2 | 16:19:37 |
| Compliance Agent | 10103 | 2 | 16:19:38 |

**Sequence diagram:**
```
User → Customer Agent
         │ discover("legal_question") → Registry → Law Agent endpoint
         └→ Law Agent
               │ analyze_law  (LLM)
               │ check_routing (LLM) → needs_tax=True, needs_compliance=True
               ├→ discover("tax_question") → Tax Agent        ┐ parallel
               └→ discover("compliance_question") → Compliance │
               ←─── Tax Agent response (9,447 chars)           │
               ←─── Compliance Agent response (11,026 chars) ──┘
               │ aggregate (LLM)
         ←─── final_answer
       ←─── Customer Agent presents response
```

Cả `trace_id` và `context_id` đều được truyền vào mỗi A2A request header để phục vụ debugging và audit trail.

### Bài Tập 5.2: Test dynamic discovery khi Tax Agent bị dừng

Sau khi kill Tax Agent (port 10102), chạy lại `test_client.py`:

- **Registry:** Vẫn lưu endpoint cũ của Tax Agent (không tự xóa)
- **Law Agent:** `call_tax()` gọi endpoint → `httpx.ConnectError: All connection attempts failed`
- **Xử lý:** `except Exception` trong `call_tax()` trả về `[Tax analysis unavailable: ...]`
- **Kết quả:** Hệ thống vẫn trả về response đầy đủ từ Law + Compliance analysis

```
[law_agent] ERROR call_tax failed: All connection attempts failed
→ System continues with available analyses (graceful degradation)
```

**Kết luận:** A2A system có fault tolerance tự nhiên qua try/except — từng agent xử lý lỗi độc lập, không kéo sập toàn hệ thống.

### Bài Tập 5.3: Modify tax agent system prompt

**File:** `tax_agent/graph.py`

Thêm constraint vào `TAX_SYSTEM_PROMPT`:

```python
IMPORTANT: Keep your response SHORT and CONCISE — maximum 150 words.
Use bullet points only. No long paragraphs.
```

Restart Tax Agent (không cần restart các service khác — dynamic discovery tự tìm lại):

```bash
# kill port 10102
uv run python -m tax_agent > logs/tax_agent.log 2>&1 &
```

**Kết quả đo được:**

| Lần chạy | Tax Agent response |
|---|---|
| Trước khi sửa | 9,447 chars |
| Sau khi sửa | 1,294 chars |

Giảm **86%** output size — thể hiện rõ việc thay đổi behavior của một agent trong distributed system chỉ cần sửa 1 file và restart 1 service, không ảnh hưởng các service khác.

---

## Bài Tập Cộng Điểm — Latency Analysis & Optimization

### Câu 1: Latency baseline

**Đo bằng:** `date +%s%N` bao quanh `uv run python test_client.py`

**Kết quả: 111 giây**

Breakdown từ `law_agent.log`:

| Node | Thời gian | Ghi chú |
|---|---|---|
| `analyze_law` (LLM) | ~51s | Bottleneck lớn nhất — prompt không giới hạn độ dài |
| `aggregate` (LLM) | ~32s | Bottleneck thứ hai — tổng hợp văn bản dài |
| Tax + Compliance parallel | ~50s | Compliance xong trước Tax |
| `check_routing` (LLM) | ~2s | Gọi LLM chỉ để parse JSON routing |
| Customer Agent synthesis | ~15s | LLM call thêm ở Customer Agent |

**Nguyên nhân gốc rễ:** 4–5 LLM calls tuần tự, không giới hạn số token output → mỗi call tạo văn bản dài → latency cao.

### Câu 2: Phương án giảm latency

**Hai tối ưu hóa đã implement:**

#### Opt 1: Word limit cho tất cả LLM calls

**File:** `law_agent/graph.py`, `compliance_agent/graph.py`, `tax_agent/graph.py`

```python
# analyze_law — thêm vào system prompt:
"IMPORTANT: Keep your analysis under 80 words. Use bullet points only."

# aggregate — thêm vào system prompt:
"IMPORTANT: Keep your response under 150 words. Use bullet points."

# tax_agent và compliance_agent — tương tự
"IMPORTANT: Keep your response SHORT and CONCISE — maximum 150 words."
```

#### Opt 2: Keyword-based routing thay thế LLM routing

**File:** `law_agent/graph.py` — hàm `check_routing`

```python
async def check_routing(state: LawState) -> dict:
    q = state["question"].lower()
    needs_tax = any(kw in q for kw in [
        "tax", "irs", "evasion", "revenue", "penalty", "fbar", "fatca", "offshore",
    ])
    needs_compliance = any(kw in q for kw in [
        "compliance", "sec", "sox", "regulation", "fcpa", "aml",
    ])
    if not needs_tax and not needs_compliance:
        needs_tax = True
    return {"needs_tax": needs_tax, "needs_compliance": needs_compliance}
```

Loại bỏ hoàn toàn 1 LLM round-trip (~2s + network overhead).

### Kết quả đo được sau optimize

**Latency: 54 giây** (đo thực tế)

| Metric | Trước | Sau | Giảm |
|---|---|---|---|
| `analyze_law` | ~51s | ~6s | -88% |
| `check_routing` | ~2s | ~0s | -100% |
| `aggregate` | ~32s | ~8s | -75% |
| **Total** | **111s** | **54s** | **-51%** |

**Tóm tắt:** Giảm 57 giây (51%) bằng 2 thay đổi đơn giản — không thay đổi kiến trúc, không cần thêm infrastructure. Cách tiếp cận này áp dụng được cho bất kỳ LLM-heavy pipeline nào.

---

## Bonus — Live Visualization Tool

**File:** `viz_server.py`  
**Chạy:** `uv run python viz_server.py`  
**Mở:** http://localhost:8080

Xây dựng web app visualize luồng multi-agent system realtime:

### Kỹ thuật triển khai

**Backend (FastAPI + SSE):**

```python
def instrument(fn, name: str):
    """Wrap node để emit start/end events với elapsed time."""
    async def wrapper(state):
        t0 = time.perf_counter()
        await emit("node_start", node=name)
        result = await fn(state)
        elapsed = round(time.perf_counter() - t0, 1)
        await emit("node_end", node=name, elapsed=elapsed)
        return result
    return wrapper
```

Mỗi node được bọc bởi `instrument()` → emit SSE event ngay khi bắt đầu và kết thúc.

**SSE với heartbeat** để browser không buffer:

```python
async def _gen():
    while True:
        try:
            ev = await asyncio.wait_for(queue.get(), timeout=1.0)
            yield {"data": json.dumps(ev)}
            if ev.get("type") in ("done", "error"):
                break
        except asyncio.TimeoutError:
            yield {"data": json.dumps({"type": "ping"})}  # keepalive
```

**Frontend (JavaScript):**
- Node glow màu cam + pulse animation khi `active`
- Node chuyển xanh khi `done`
- Node mờ khi bị `skip` (routing không chọn)
- SVG bezier curves nối các node, đổi màu theo trạng thái
- Timer đếm realtime mỗi 100ms, freeze khi done

### Tính năng

| Tính năng | Mô tả |
|---|---|
| Realtime glow | Node sáng ngay khi agent bắt đầu (không chờ xong) |
| Parallel glow | Tax + Privacy cùng sáng đồng thời khi chạy song song |
| Timer per-agent | Đếm giây từ khi start, hiển thị `X.Xs ✓` sau khi done |
| Architecture badge | Badge cho biết pattern: Supervisor-Workers vs A2A |
| SVG connections | Bezier curves đổi màu: xám → cam (active) → xanh (done) |
| Skipped nodes | Nodes không được routing chọn tự động mờ đi |

---

## Tổng Kết So Sánh 5 Stages

| Stage | Pattern | Latency (ước tính) | Ưu điểm | Nhược điểm |
|---|---|---|---|---|
| 1 | Direct LLM | ~3s | Đơn giản, nhanh | Không có tools, không có context |
| 2 | LLM + Tools | ~8s | Tra cứu được data | Manual orchestration, single-pass |
| 3 | ReAct Agent | ~15s | Tự quyết định tools, multi-step | 1 agent xử lý tất cả domains |
| 4 | Multi-Agent In-Process | ~54s* | Parallel, chuyên môn hóa | Single process, tight coupling |
| 5 | Distributed A2A | ~111s* | Fault-tolerant, scalable, độc lập | Latency cao do HTTP overhead |

*Sau khi optimize với word limits + keyword routing.

### Câu Hỏi Ôn Tập — Đáp Án

**1. Khi nào dùng single agent thay vì multi-agent?**

Dùng single agent khi: bài toán thuộc 1 domain duy nhất, không cần phân tích song song, yêu cầu latency thấp, hoặc team nhỏ không có bandwidth duy trì nhiều agents. Multi-agent thích hợp khi cần nhiều chuyên môn khác nhau (tax + legal + compliance), cần xử lý song song để giảm thời gian, và từng domain đủ phức tạp để có system prompt riêng.

**2. Ưu điểm của A2A so với gRPC/REST thông thường?**

A2A được thiết kế đặc biệt cho AI agents: có Agent Card (service discovery + capability declaration), hỗ trợ streaming natively, propagate context (trace_id, context_id) qua toàn chain, và standardized message format (Task/Message/Part). gRPC và REST là general-purpose — dùng được nhưng phải tự build những tính năng này.

**3. Prevent infinite delegation loops trong A2A?**

Dùng `delegation_depth` counter trong state — mỗi lần delegate tăng lên 1. Khi `depth >= MAX_DELEGATION_DEPTH (=3)`, node `check_routing` return `{needs_tax: False, needs_compliance: False}` ngay lập tức. Xem `law_agent/graph.py:77-80`.

**4. Tại sao cần Registry? Có thể hardcode URLs không?**

Có thể hardcode nhưng không nên: mất khả năng scale (không thay được endpoint khi chuyển server), mất fault tolerance (không biết agent nào còn sống), và không thể load balance. Registry cho phép agents tự đăng ký khi start và discover lẫn nhau tại runtime — giống DNS cho microservices.

---

## Files Đã Tạo/Sửa

| File | Thay đổi |
|---|---|
| `stages/stage_3_single_agent/main.py` | Thêm `search_case_law` tool; xóa `verbose=True` |
| `stages/stage_4_milti_agent/main.py` | Thêm `privacy_agent`, conditional routing; fix import lỗi |
| `law_agent/graph.py` | Word limit prompts; keyword-based routing (không dùng LLM) |
| `compliance_agent/graph.py` | Thêm word limit vào system prompt |
| `tax_agent/graph.py` | Thêm word limit vào system prompt |
| `viz_server.py` | **Mới** — Live visualization server với SSE + realtime timers |
| `logs/` | **Mới** — Log files cho tất cả services |
