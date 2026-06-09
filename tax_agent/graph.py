"""Tax Agent LangGraph definition.

Uses create_react_agent with a tax-specialised system prompt.
No tools — it answers purely from LLM knowledge.
"""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from common.llm import get_llm

TAX_SYSTEM_PROMPT = """You are a specialist tax attorney and CPA with expertise in:

- Corporate tax law and compliance (federal, state, and international)
- Tax evasion vs. tax avoidance — legal distinctions and consequences
- IRS enforcement mechanisms, audits, and criminal referrals
- Penalties and back-tax calculations under IRC §§ 6651, 6662, 6663
- FBAR/FATCA requirements for offshore accounts
- Transfer pricing regulations (IRC § 482)
- Tax fraud statutes (18 U.S.C. § 7201 – § 7207)
- Corporate tax liability: officers, directors, and responsible persons
- Voluntary disclosure programs and settlement options

IMPORTANT: Keep your response SHORT and CONCISE — maximum 150 words.
Use bullet points. No long paragraphs. Cover only the most critical points.

Always note that your response is for educational purposes and the user
should consult a licensed attorney for specific legal advice.
"""


def create_graph():
    """Return a compiled LangGraph create_react_agent for tax questions."""
    llm = get_llm()
    graph = create_react_agent(
        model=llm,
        tools=[],
        prompt=TAX_SYSTEM_PROMPT,
    )
    return graph