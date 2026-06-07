import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from starrag import vectorstore


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        return [float(len(text)), float(sum(ord(ch) for ch in text) % 97)]


class VectorStoreTests(unittest.TestCase):
    def test_all_repos_share_one_vector_store_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(vectorstore, "VECTOR_STORE_DIR", Path(tmp)):
                self.assertEqual(vectorstore.vector_store_path(1), vectorstore.vector_store_path(2))
                self.assertEqual(vectorstore.vector_store_path(1).name, "global")

    def test_build_and_save_appends_chunks_to_global_store(self):
        embeddings = FakeEmbeddings()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(vectorstore, "VECTOR_STORE_DIR", Path(tmp)):
                vectorstore.build_and_save(
                    repo_id=1,
                    documents=[
                        Document(
                            page_content="alpha chunk",
                            metadata={"repo_id": 1, "owner": "one", "name": "repo"},
                        )
                    ],
                    ids=["chunk-from-repo-1"],
                    embeddings=embeddings,
                )
                vectorstore.build_and_save(
                    repo_id=2,
                    documents=[
                        Document(
                            page_content="beta chunk",
                            metadata={"repo_id": 2, "owner": "two", "name": "repo"},
                        )
                    ],
                    ids=["chunk-from-repo-2"],
                    embeddings=embeddings,
                )

                store = FAISS.load_local(
                    str(vectorstore.vector_store_path(1)),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )

        self.assertEqual(
            set(store.index_to_docstore_id.values()),
            {"chunk-from-repo-1", "chunk-from-repo-2"},
        )

    def test_delete_ids_removes_only_matching_chunks_from_global_store(self):
        embeddings = FakeEmbeddings()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(vectorstore, "VECTOR_STORE_DIR", Path(tmp)):
                vectorstore.build_and_save(
                    repo_id=1,
                    documents=[Document(page_content="old alpha", metadata={"repo_id": 1})],
                    ids=["old-repo-1"],
                    embeddings=embeddings,
                )
                vectorstore.build_and_save(
                    repo_id=2,
                    documents=[Document(page_content="beta", metadata={"repo_id": 2})],
                    ids=["repo-2"],
                    embeddings=embeddings,
                )

                vectorstore.delete_ids(ids=["old-repo-1"], embeddings=embeddings)
                vectorstore.build_and_save(
                    repo_id=1,
                    documents=[Document(page_content="new alpha", metadata={"repo_id": 1})],
                    ids=["new-repo-1"],
                    embeddings=embeddings,
                )
                store = FAISS.load_local(
                    str(vectorstore.vector_store_path(1)),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )

        self.assertEqual(set(store.index_to_docstore_id.values()), {"repo-2", "new-repo-1"})

    def test_build_and_save_can_append_after_global_store_is_emptied(self):
        embeddings = FakeEmbeddings()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(vectorstore, "VECTOR_STORE_DIR", Path(tmp)):
                vectorstore.build_and_save(
                    repo_id=1,
                    documents=[Document(page_content="old alpha", metadata={"repo_id": 1})],
                    ids=["old-repo-1"],
                    embeddings=embeddings,
                )

                vectorstore.delete_ids(ids=["old-repo-1"], embeddings=embeddings)
                vectorstore.build_and_save(
                    repo_id=1,
                    documents=[Document(page_content="new alpha", metadata={"repo_id": 1})],
                    ids=["new-repo-1"],
                    embeddings=embeddings,
                )
                store = FAISS.load_local(
                    str(vectorstore.vector_store_path(1)),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )

        self.assertEqual(set(store.index_to_docstore_id.values()), {"new-repo-1"})


if __name__ == "__main__":
    unittest.main()
