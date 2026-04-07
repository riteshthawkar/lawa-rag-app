from collections import deque
from types import SimpleNamespace


class FakeDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class FakeRetriever:
    def __init__(self, docs):
        self.docs = list(docs)
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return list(self.docs)


def make_completion(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def make_stream_chunk(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
    )


class FakeStream:
    def __init__(self, chunks):
        self._chunks = deque(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return make_stream_chunk(self._chunks.popleft())


class FakeChatCompletions:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("No fake completion queued for this call")

        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


class FakeOpenAIClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeChatCompletions(responses))
