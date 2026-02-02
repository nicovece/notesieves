SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the user's personal notes.

You will be given relevant excerpts from the user's markdown notes, along with metadata about where each excerpt comes from.

Guidelines:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so clearly
- Always cite which source files informed your answer
- Use clear, educational explanations appropriate for someone learning the topic
- If concepts from multiple files connect, explain those connections

When citing sources, use this format at the end of your answer:

Sources:
  - [File Name] (Section: [Heading])
  - [File Name] (Section: [Heading])
"""


BROAD_SYSTEM_PROMPT = """You are a helpful assistant that helps the user navigate their personal notes collection.

You will be given a map of note files and their section headings that are relevant to the user's question.

Guidelines:
- Recommend which notes the user should read, based on the file names and headings
- Explain briefly why each note is relevant
- Group or order your recommendations by relevance (most relevant first)
- If headings suggest a reading order or progression, mention it
- You do NOT have the full note content — only file names and section headings
- Be honest about what you can and cannot infer from headings alone
"""


QUIZ_SYSTEM_PROMPT = """You are a tutor quizzing the user on concepts from their personal notes.

You will be given relevant excerpts from the user's notes as context. Use this material to generate questions and evaluate answers.

Guidelines:
- Ask ONE question at a time
- Questions should test understanding, not just recall — prefer "why" and "how" over "what"
- When the user answers, evaluate their response against the source material
- Explain what they got right, what they missed, and add any important context
- Then immediately ask the next question on a different aspect of the topic
- Do NOT repeat questions you have already asked in this session
- Keep a supportive but honest tone — correct misconceptions clearly
- Cite which note the question is based on

Format your responses like this:

When asking a question:
**Question:** [your question]

When evaluating + asking next:
**Feedback:** [evaluation of their answer]

**Question:** [next question]
"""


def build_quiz_start_prompt(topic: str, context_chunks: list[dict]) -> str:
    """Build the first quiz message with retrieved context."""
    context_parts = []

    for i, chunk in enumerate(context_chunks, 1):
        metadata = chunk["metadata"]
        heading = metadata.get("heading_hierarchy", "No heading")
        file_name = metadata.get("file_name", "Unknown")

        context_parts.append(
            f"--- Context {i} ---\n"
            f"Source: {file_name}\n"
            f"Section: {heading}\n\n"
            f"{chunk['text']}"
        )

    context_str = "\n\n".join(context_parts)

    return (
        f"Here are relevant excerpts from my notes:\n\n"
        f"{context_str}\n\n"
        f"---\n\n"
        f"Quiz me about: {topic}\n\n"
        f"Ask me one question to test my understanding."
    )


def build_broad_user_prompt(question: str, file_map: dict[str, list[str]]) -> str:
    """Build a prompt from the file→headings map for broad retrieval."""
    parts = []
    for file_name, headings in file_map.items():
        heading_list = "\n".join(f"    - {h}" for h in headings) if headings else "    (no headings)"
        parts.append(f"  {file_name}\n{heading_list}")

    map_str = "\n\n".join(parts)

    return (
        f"Here is a map of my notes that may be relevant:\n\n"
        f"{map_str}\n\n"
        f"---\n\n"
        f"My question: {question}\n\n"
        f"Based on the file names and section headings above, recommend which notes I should read and why."
    )


def build_user_prompt(question: str, context_chunks: list[dict]) -> str:
    """Build the user prompt with retrieved context."""
    context_parts = []

    for i, chunk in enumerate(context_chunks, 1):
        metadata = chunk["metadata"]
        heading = metadata.get("heading_hierarchy", "No heading")
        file_name = metadata.get("file_name", "Unknown")

        context_parts.append(
            f"--- Context {i} ---\n"
            f"Source: {file_name}\n"
            f"Section: {heading}\n\n"
            f"{chunk['text']}"
        )

    context_str = "\n\n".join(context_parts)

    return (
        f"Here are relevant excerpts from my notes:\n\n"
        f"{context_str}\n\n"
        f"---\n\n"
        f"My question: {question}\n\n"
        f"Please answer based on the context above, and cite which sources you used."
    )
