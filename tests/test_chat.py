import unittest
from unittest.mock import patch

from click.testing import CliRunner
from langchain_core.documents import Document

from main import cli
from starrag import chat


class FakeDocstore:
    def search(self, chunk_id):
        return Document(
            page_content="full chunk content",
            metadata={
                "file_path": "src/app.py",
                "chunk_index": 3,
                "language": "python",
            },
        )


class FakeStore:
    def __init__(self):
        self.docstore = FakeDocstore()


class ChatTests(unittest.TestCase):
    @patch("starrag.chat.db.get_repo_by_id")
    @patch("starrag.chat.search_repo")
    @patch("starrag.chat.load_global_store")
    def test_chat_prints_repo_info_and_chunk_content(
        self,
        load_store,
        search_repo,
        get_repo_by_id,
    ):
        store = FakeStore()
        load_store.return_value = (store, 2)
        search_repo.return_value = [{"chunkId": "chunk-1", "repoId": 7}]
        get_repo_by_id.return_value = {
            "owner": "octocat",
            "name": "hello-world",
            "url": "https://github.com/octocat/hello-world",
        }

        result = CliRunner().invoke(cli, ["chat"], input="needle\nquit\n")

        self.assertEqual(result.exit_code, 0)
        search_repo.assert_called_once_with("needle", top_k=chat.TOP_K, store=store)
        get_repo_by_id.assert_called_once_with(7)
        self.assertIn("repoId=7", result.output)
        self.assertIn("chunkId=chunk-1", result.output)
        self.assertIn("octocat/hello-world", result.output)
        self.assertIn("https://github.com/octocat/hello-world", result.output)
        self.assertIn("src/app.py", result.output)
        self.assertIn("full chunk content", result.output)


if __name__ == "__main__":
    unittest.main()
