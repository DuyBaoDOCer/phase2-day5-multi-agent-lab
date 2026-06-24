# Benchmark Report: Single-Agent vs Multi-Agent

**Runs compared:** 2

---

## Results Table

| Run | Latency (s) | Cost (USD) | Quality (0-10) | Notes |
|---|---:|---:|---:|---|
| single-agent-baseline | 6.77 | N/A | 6.0 | words=442, errors=0 |
| multi-agent-workflow | 21.52 | $0.0028 | 8.8 | words=694, errors=0 |

---

## Analysis

- **Latency:** Multi-agent is 14.76s (218%) slower than baseline.
- **Quality:** multi-agent wins by 2.8 points (baseline=6.0, multi=8.8).

### Verdict

Multi-agent provides **higher quality** at the cost of higher **latency and cost** because each specialist agent (Researcher → Analyst → Writer) takes an additional LLM call. For simple queries, the baseline single-agent may be sufficient. For complex research tasks requiring synthesis, multi-agent wins on quality.

---

## Agent Route Traces

### single-agent-baseline

- Iterations: 0
- Sources found: 0
- Errors: none

### multi-agent-workflow

Route: `START → researcher → analyst → writer → done`
- Iterations: 4
- Sources found: 5
- Errors: none

---

## Failure Modes Observed

| Mode | Description | Mitigation |
|---|---|---|
| Empty search results | Tavily returns 0 results for niche topics | Retry with broader query |
| LLM timeout | Mistral API slow under load | Tenacity retry with backoff (3 attempts) |
| Hallucinated citations | Writer invents source titles | Critic agent flags unsupported claims |
| Max iterations hit | Supervisor loops if agents fail silently | Hard guardrail at `MAX_ITERATIONS` |

---

## Recommendations

1. Use **multi-agent** for queries requiring deep research and synthesis (> 2 sub-topics).
2. Use **single-agent baseline** for quick factual lookups where latency matters.
3. Always run the **Critic agent** in production to catch citation gaps.
4. Enable **LangSmith tracing** (`LANGSMITH_API_KEY`) for per-step debugging.

---

## Exit Ticket

### 1. Case nào nên dùng multi-agent? Vì sao?
- **Các tác vụ nghiên cứu chuyên sâu, tổng hợp thông tin đa chiều:** Ví dụ như nghiên cứu công nghệ mới, viết báo cáo thị trường, hoặc phân tích kỹ thuật.
- **Vì sao:** Phân rã bài toán phức tạp thành các vai trò chuyên biệt (Researcher, Analyst, Writer) giúp tận dụng tối đa năng lực LLM cho từng giai đoạn độc lập. Agent chuyên trách (Researcher) có thể tập trung cào dữ liệu và trích xuất thông tin thô, Analyst đánh giá logic và lọc bỏ nhiễu, còn Writer lo biên tập. Cách tiếp cận này giúp cải thiện đáng kể độ chính xác và chất lượng bài viết (điểm chất lượng tăng từ 6.0 lên 8.8) so với việc bắt một Single-Agent xử lý tất cả cùng lúc.

### 2. Case nào không nên dùng multi-agent? Vì sao?
- **Các truy vấn đơn giản, mang tính tra cứu nhanh hoặc yêu cầu phản hồi tức thời (Real-time/Low latency):** Ví dụ hỏi định nghĩa ngắn gọn, kiểm tra sự kiện thực tế đơn giản, hoặc dịch thuật cơ bản.
- **Vì sao:** Việc chuyển giao trạng thái (state handoff) và điều phối qua Supervisor tạo ra nhiều lượt gọi LLM tuần tự. Điều này làm tăng đáng kể độ trễ (latency tăng từ 6.77s lên 21.52s, tăng khoảng 218%) và chi phí vận hành API. Đối với các tác vụ đơn giản, single-agent baseline đáp ứng nhanh hơn, rẻ hơn mà chất lượng câu trả lời vẫn tương đương.
