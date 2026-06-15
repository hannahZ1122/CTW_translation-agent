# 🌐 Multilingual Translation Agent

> **Assessment Submission** · AI Product Manager Technical Test  
> Built with Vibe Coding — GPT-4o + Streamlit

---

## Live Demo

👉 **[https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)**  
*(Replace with your actual Streamlit Cloud URL after deployment)*

---

## 技术选型 Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| LLM | GPT-4o | Best multilingual quality; native Chinese/Japanese/Korean understanding without extra fine-tuning |
| Framework | Streamlit | Native `st.chat_message` + `st.chat_input` = authentic chat UX; single-file deploy in minutes |
| CSV Processing | pandas | Battle-tested; handles encoding edge cases (UTF-8 BOM for Excel compatibility) |
| Deployment | Streamlit Cloud | Free tier; GitHub-connected; auto-redeploy on push |

**Why not LangChain / Dify?**  
For a scoped 2-hour task with a single, linear workflow, adding an orchestration framework creates overhead without value. The agent logic is explicit Python state-machine — more readable, debuggable, and interview-reviewable than a framework DSL.

---

## Agent 设计思路 Agent Design

The agent follows **explicit stage-based decomposition** rather than a single "do everything" prompt. Each stage has one clear responsibility:

```
┌─────────────────────────────────────────────────────────────┐
│                    Translation Agent Flow                   │
├──────────┬────────────────────────────────────────────────┤
│  Stage   │  Responsibility                                │
├──────────┼────────────────────────────────────────────────┤
│  INIT    │  Greet user, request CSV + API key            │
│  PARSE   │  Read CSV, auto-detect Chinese columns         │
│          │  (regex: [一-鿿] ≥50% of sample rows) │
│  CONFIRM │  NLU via GPT-4o: extract columns + languages   │
│          │  from natural language input                   │
│  READY   │  Show translation plan, await user confirm     │
│  EXECUTE │  Batch translate (20 rows/call) with progress  │
│  DONE    │  Serve downloadable CSV, allow follow-up       │
└──────────┴────────────────────────────────────────────────┘
```

### Key Design Decisions

**1. Explicit column detection before translation**  
Rather than asking the user "which columns?", the agent proactively detects Chinese columns and proposes them. This mirrors good PM thinking: reduce friction by making intelligent defaults, then confirm.

**2. NLU for intent parsing (not rigid form input)**  
Instead of dropdowns for column/language selection, the agent parses free-form natural language ("把道具名和描述翻译成日语和韩语"). This is the core of the "chat" experience.

**3. Batched translation with retry**  
100+ rows in a single LLM call risks token limits and timeouts. Batching at 20 rows with 3-retry exponential backoff ensures reliability without complicating the UX.

**4. Graceful degradation**  
If a batch fails after 3 retries, the agent preserves the original text (no crash, no partial corrupt output). The user is informed and can retry.

**5. UTF-8 BOM output**  
The output CSV uses UTF-8 BOM encoding — a practical detail that ensures Chinese/Japanese characters display correctly when opened directly in Excel without manual encoding selection.

---

## 可进一步优化的方向 Future Improvements

### 短期 (Quick Wins)
- **Streaming progress**: Show translated text appearing in real-time rather than waiting for each full batch
- **Column type inference**: Distinguish "translateable text" from "codes/IDs/numbers" more precisely using LLM-assisted column classification
- **Language auto-suggest**: When the user uploads a CSV, suggest likely target languages based on detected file name or header patterns (e.g., a file named `game_jp.csv` → suggest English, Korean)

### 中期 (Product-level)
- **Translation memory / glossary**: For game content, maintain a terminology dictionary (e.g., skill names, item names) to ensure consistency across translates. Critical for franchise games.
- **Context-aware prompting**: Pass surrounding rows as context so the LLM understands narrative continuity (e.g., quest dialogue that references earlier lines)
- **Quality scoring**: After translation, run a lightweight back-translation check and flag rows where semantic similarity drops below threshold
- **Multi-file batch mode**: Accept a ZIP of CSVs for bulk localization pipelines

### 长期 (Platform-level, relevant to G123.jp)
- **Webhook integration**: Connect to game CMS so translated content auto-pushes to the game database — eliminating the download/re-upload step
- **Human-in-the-loop review**: Flag low-confidence translations for human review before finalization (especially for character names and culturally sensitive terms)
- **Cost optimization**: Semantic caching via vector DB — if the same item description appears across multiple game titles, reuse the cached translation instead of re-translating

---

## Local Development

```bash
git clone https://github.com/YOUR_USERNAME/translation-agent
cd translation-agent
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY=sk-...

streamlit run app.py
```

Or configure via `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-..."
```

---

## Sample Data

A sample CSV (`sample_game_items.csv`) with **105 rows** of game item data (names, descriptions, skill effects) is included for testing. This mirrors a real G123-style game localization use case.
