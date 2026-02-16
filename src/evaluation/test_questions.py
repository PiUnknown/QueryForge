"""Test questions for evaluating RAG system performance"""

EVALUATION_QUESTIONS = [
    {
        "id": 1,
        "question": "What is retrieval-augmented generation?",
        "category": "definition",
        "expected_sources": ["2005.11401.pdf", "2310.06825.pdf"],
        "difficulty": "easy"
    },
    {
        "id": 2,
        "question": "How does BERT use bidirectional attention?",
        "category": "technical",
        "expected_sources": ["1810.04805.pdf"],
        "difficulty": "medium"
    },
    {
        "id": 3,
        "question": "What are the main components of the transformer architecture?",
        "category": "technical",
        "expected_sources": ["2005.14165.pdf", "1810.04805.pdf"],
        "difficulty": "medium"
    },
    {
        "id": 4,
        "question": "How does Dense Passage Retrieval (DPR) work?",
        "category": "technical",
        "expected_sources": ["2104.08663.pdf"],
        "difficulty": "medium"
    },
    {
        "id": 5,
        "question": "What is the difference between self-RAG and standard RAG?",
        "category": "comparison",
        "expected_sources": ["2312.10997.pdf", "2005.11401.pdf"],
        "difficulty": "hard"
    },
    {
        "id": 6,
        "question": "Explain chain-of-thought prompting.",
        "category": "technical",
        "expected_sources": ["2201.11903.pdf"],
        "difficulty": "medium"
    },
    {
        "id": 7,
        "question": "What is LoRA and how does it work?",
        "category": "technical",
        "expected_sources": ["2106.09685.pdf"],
        "difficulty": "medium"
    },
    {
        "id": 8,
        "question": "How do language models scale with model size?",
        "category": "analysis",
        "expected_sources": ["2005.14165.pdf"],
        "difficulty": "hard"
    },
    {
        "id": 9,
        "question": "What are the key innovations in GPT-4?",
        "category": "technical",
        "expected_sources": ["2303.08774.pdf"],
        "difficulty": "medium"
    },
    {
        "id": 10,
        "question": "How does corrective RAG improve upon standard RAG?",
        "category": "comparison",
        "expected_sources": ["2401.15884.pdf", "2005.11401.pdf"],
        "difficulty": "hard"
    },
    {
        "id": 11,
        "question": "What is the purpose of embedding models in semantic search?",
        "category": "conceptual",
        "expected_sources": ["2212.09741.pdf", "2104.08663.pdf"],
        "difficulty": "easy"
    },
    {
        "id": 12,
        "question": "Explain the ReAct prompting framework.",
        "category": "technical",
        "expected_sources": ["2308.11432.pdf", "2210.03493.pdf"],
        "difficulty": "medium"
    },
    {
        "id": 13,
        "question": "How does Llama 2 differ from previous language models?",
        "category": "comparison",
        "expected_sources": ["2307.09288.pdf"],
        "difficulty": "medium"
    },
    {
        "id": 14,
        "question": "What are the challenges in building effective RAG systems?",
        "category": "analysis",
        "expected_sources": ["2310.06825.pdf", "2005.11401.pdf"],
        "difficulty": "hard"
    },
    {
        "id": 15,
        "question": "How does fine-tuning with human feedback work?",
        "category": "technical",
        "expected_sources": ["2203.02155.pdf"],
        "difficulty": "medium"
    },
]


def get_questions_by_category(category: str):
    """Filter questions by category"""
    return [q for q in EVALUATION_QUESTIONS if q["category"] == category]


def get_questions_by_difficulty(difficulty: str):
    """Filter questions by difficulty level"""
    return [q for q in EVALUATION_QUESTIONS if q["difficulty"] == difficulty]


def get_all_questions():
    """Return all evaluation questions"""
    return EVALUATION_QUESTIONS