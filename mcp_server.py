import os
from mcp.server.fastmcp import FastMCP
from crc_qa_v2 import answer_crc_question_sync

mcp = FastMCP("crc-rag")


@mcp.tool()
def crc_rag_answer(question: str, mode: str = "vanilla") -> str:
    """
    Answer colorectal cancer questions using the local RAG pipeline.

    Args:
        question: The user question.
        mode: "vanilla" (fast) or "agentic" (deep).

    Returns:
        A cited answer string.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    answer, _log = answer_crc_question_sync(question, api_key=api_key, mode=mode)
    return answer


if __name__ == "__main__":
    mcp.run()
