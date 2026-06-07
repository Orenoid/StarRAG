import unittest

from langchain_core.documents import Document

from starrag import chat


class FakeStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query, k):
        self.calls.append((query, k))
        return [
            (
                Document(
                    page_content="global result",
                    metadata={"owner": "repo-owner", "name": "repo-name"},
                ),
                0.25,
            )
        ]


class ChatSearchTests(unittest.TestCase):
    def test_search_all_queries_one_global_store_and_uses_chunk_metadata(self):
        store = FakeStore()

        hits = chat._search_all(store, "needle")

        self.assertEqual(store.calls, [("needle", chat.TOP_K)])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][2:], ("repo-owner", "repo-name"))


if __name__ == "__main__":
    unittest.main()
