import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from starrag.tools import search_repo


class FakeStore:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def similarity_search_with_score(self, query, k):
        self.calls.append((query, k))
        return self.hits


class SearchRepoTests(unittest.TestCase):
    @patch("starrag.tools.load_global_store")
    def test_search_repo_loads_global_store_by_default(self, load_global_store):
        store = FakeStore(
            [
                (
                    Document(
                        page_content="match",
                        metadata={"chunk_id": "chunk-1", "repo_id": 1},
                    ),
                    0.1,
                )
            ]
        )
        load_global_store.return_value = (store, 1)

        results = search_repo("needle")

        load_global_store.assert_called_once_with()
        self.assertEqual(store.calls, [("needle", 10)])
        self.assertEqual(results, [{"chunkId": "chunk-1", "repoId": 1}])

    def test_search_repo_returns_top_k_chunk_and_repo_ids_sorted_by_score(self):
        store = FakeStore(
            [
                (
                    Document(
                        page_content="second",
                        metadata={"chunk_id": "chunk-2", "repo_id": 2},
                    ),
                    0.5,
                ),
                (
                    Document(
                        page_content="first",
                        metadata={"chunk_id": "chunk-1", "repo_id": 1},
                    ),
                    0.1,
                ),
            ]
        )

        results = search_repo("needle", top_k=2, store=store)

        self.assertEqual(store.calls, [("needle", 2)])
        self.assertEqual(
            results,
            [
                {"chunkId": "chunk-1", "repoId": 1},
                {"chunkId": "chunk-2", "repoId": 2},
            ],
        )

    @patch("starrag.tools.db.get_chunk_id", return_value="legacy-chunk")
    def test_search_repo_resolves_chunk_id_for_legacy_index(self, get_chunk_id):
        store = FakeStore(
            [
                (
                    Document(
                        page_content="legacy",
                        metadata={
                            "repo_id": 3,
                            "file_path": "src/app.py",
                            "chunk_index": 4,
                        },
                    ),
                    0.2,
                )
            ]
        )

        results = search_repo("legacy", store=store)

        self.assertEqual(results, [{"chunkId": "legacy-chunk", "repoId": 3}])
        get_chunk_id.assert_called_once_with(
            repo_id=3,
            file_path="src/app.py",
            chunk_index=4,
        )

    def test_search_repo_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "query"):
            search_repo("   ", store=FakeStore([]))
        with self.assertRaisesRegex(ValueError, "top_k"):
            search_repo("needle", top_k=0, store=FakeStore([]))


if __name__ == "__main__":
    unittest.main()
