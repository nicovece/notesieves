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
