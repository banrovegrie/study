# External Context: SOTA Systems and Research

## Relevant Research (2024-2025)

### Claude Code Best Practices
From [Anthropic's engineering guide](https://www.anthropic.com/engineering/claude-code-best-practices):
- Claude Code is an "active collaborator" that can search code, edit files, run tests, and use CLI tools
- Extended thinking mode: "think" < "think hard" < "think harder" < "ultrathink"
- Supports Model Context Protocol (MCP) for tool integration

### Tool-Integrated Reasoning
From [TIR-Judge research](https://arxiv.org/html/2510.23038v1):
- Teaching models to generate code, execute with interpreters, and iteratively refine
- "By reinforcing this cycle of reasoning and tool-use... more accurate and verifiable evaluations"
- 8B model matches 96% of Claude-Opus performance on math reasoning with tool integration

### Agent RL for Math
From [Agent RL Scaling Law](https://arxiv.org/html/2505.07773):
- LLMs can learn to leverage Python execution through RL
- ZeroTIR achieves 52.3% on AIME24/25 with 7B model (vs 39.1% without tools)
- "Spontaneous code execution for mathematical problem solving"

### SOTA Math Performance (2025)
| Model | AIME 2025 | IMO | Notes |
|-------|-----------|-----|-------|
| OpenAI o3 | 88.9% | - | Strong reasoning |
| Grok 3 | 93.3% | - | Inconsistent on complex |
| Gemini 2.5 Pro | - | Bronze | Steady improvement |
| DeepSeek-Prover-V2 | - | Gold-level | Formal proofs |

---

## Alignment with Current System

### What Aligns Well
1. **Tool-based reasoning**: Our system provides tools for Claude to call
2. **Iterative refinement**: Verification cascade with feedback
3. **Hybrid approach**: LLM reasoning + computational verification

### What to Learn From
1. **Extended thinking**: Use "think harder" for complex problems
2. **Tool-integrated RL**: Consider how tool use improves with practice
3. **MCP integration**: Future path for tool standardization

---

## Key Insights for Our System

### From Claude Code Best Practices
- Claude Code naturally delegates to tools when appropriate
- Explicit "think" triggers deeper reasoning
- MCP can extend tool capabilities

### From Tool-Integrated Reasoning Research
- Tool use significantly improves math reasoning
- Iterative refinement based on execution outputs works
- Even small models benefit from tool integration

### From Agent RL Research
- LLMs can spontaneously learn to use tools through RL
- Python execution environment provides strong feedback signal
- Tool integration provides clear gains on mathematical tasks

---

## Recommendations Based on External Research

1. **Leverage extended thinking**: For hard problems, explicitly ask Claude to "think harder"

2. **Embrace tool-integrated reasoning**: The verification cascade IS the tool integration pattern that works

3. **Consider MCP for future**: Model Context Protocol could standardize tool interfaces

4. **Trust the flywheel**: Research shows tool use improves with practice - the learning loop matters

---

## Sources
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [TIR-Judge: Tool-Integrated Reinforcement Learning](https://arxiv.org/html/2510.23038v1)
- [Agent RL Scaling Law](https://arxiv.org/html/2505.07773)
- [Claude 3.7 Sonnet Announcement](https://www.anthropic.com/news/claude-3-7-sonnet)
