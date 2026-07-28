"""Unit tests for the markdown-aware chunker (Task 2)."""

from kb.server.chunker import MAX_TOKENS, OVERLAP_TOKENS, Chunk, chunk


def test_returns_chunk_objects():
    chunks = chunk("# Title\n\nA short paragraph of text.")
    assert chunks
    assert all(isinstance(c, Chunk) for c in chunks)


def test_empty_content_yields_no_chunks():
    assert chunk("") == []
    assert chunk("   \n\n  ") == []


def test_splits_on_headers():
    content = (
        "# Alpha\n"
        "Body of alpha.\n"
        "## Beta\n"
        "Body of beta.\n"
        "### Gamma\n"
        "Body of gamma.\n"
    )
    chunks = chunk(content)
    paths = [c.header_path for c in chunks]
    assert "Alpha" in paths
    assert "Alpha > Beta" in paths
    assert "Alpha > Beta > Gamma" in paths


def test_header_path_nested_breadcrumb():
    content = (
        "# Key Insights\n"
        "Some framing text about the topic.\n"
        "## Beyond RAG\n"
        "Retrieval augmented generation is only the start; agents matter.\n"
    )
    chunks = chunk(content)
    beyond = [c for c in chunks if c.header_path == "Key Insights > Beyond RAG"]
    assert len(beyond) == 1
    assert beyond[0].header_path == "Key Insights > Beyond RAG"


def test_header_path_pops_sibling_headers():
    content = (
        "# One\n"
        "First.\n"
        "## Sub A\n"
        "Under A.\n"
        "## Sub B\n"
        "Under B.\n"
    )
    chunks = chunk(content)
    paths = {c.header_path for c in chunks}
    assert "One > Sub A" in paths
    assert "One > Sub B" in paths
    assert "One > Sub A > Sub B" not in paths


def test_h4_is_not_a_split_point():
    content = "# Top\nIntro.\n#### Deep\nDeep body stays in Top.\n"
    chunks = chunk(content)
    assert all(c.header_path == "Top" for c in chunks)


def test_content_before_first_header_has_empty_path():
    content = "Preamble text with no header.\n# Later\nBody.\n"
    chunks = chunk(content)
    assert chunks[0].header_path == ""


def test_chunk_index_and_of_are_consistent():
    content = "# A\nbody a\n## B\nbody b\n"
    chunks = chunk(content)
    total = len(chunks)
    assert [c.chunk_index for c in chunks] == list(range(total))
    assert all(c.chunk_of == total for c in chunks)


def test_long_section_falls_back_to_sliding_window():
    words = " ".join(f"w{i}" for i in range(1200))
    content = f"# Big\n{words}\n"
    chunks = chunk(content)
    assert len(chunks) > 1
    assert all(c.header_path == "Big" for c in chunks)
    # every window is bounded by MAX_TOKENS
    assert all(len(c.content.split()) <= MAX_TOKENS for c in chunks)


def test_sliding_window_overlaps():
    words = [f"w{i}" for i in range(1200)]
    content = "# Big\n" + " ".join(words) + "\n"
    chunks = chunk(content)
    first = chunks[0].content.split()
    second = chunks[1].content.split()
    # tail of the first window reappears at the head of the second
    assert first[-OVERLAP_TOKENS:] == second[:OVERLAP_TOKENS]


def test_short_section_preserves_original_text():
    content = "# H\nline one\nline two\n"
    chunks = chunk(content)
    assert chunks[0].content == "line one\nline two"


def test_section_with_only_header_produces_no_chunk():
    content = "# Empty\n## Filled\nhas body\n"
    chunks = chunk(content)
    assert all(c.header_path != "Empty" for c in chunks)
    assert any(c.header_path == "Empty > Filled" for c in chunks)
