from collections import deque
from types import SimpleNamespace


class FakeDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class FakeRetriever:
    def __init__(self, docs, embeddings=None):
        self.docs = list(docs)
        self.queries = []
        self.embeddings = embeddings or SimpleNamespace(model_name="test-embedding-model")

    def invoke(self, query):
        self.queries.append(query)
        return list(self.docs)


class FakePineconeIndex:
    def __init__(self, stats=None):
        self.stats = stats or {"namespaces": {"default": {"vector_count": 1}}}

    def describe_index_stats(self):
        return self.stats


class FakePineconeInference:
    def __init__(self):
        self.calls = []

    def rerank(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=[])


class FakePineconeClient:
    def __init__(self, stats=None):
        self.stats = stats or {"namespaces": {"default": {"vector_count": 1}}}
        self.index_requests = []
        self.inference = FakePineconeInference()

    def Index(self, name):
        self.index_requests.append(name)
        return FakePineconeIndex(self.stats)


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


class FakeModelsAPI:
    def __init__(self, failures=None):
        self.failures = failures or {}
        self.calls = []

    async def retrieve(self, model):
        self.calls.append(model)
        result = self.failures.get(model)
        if isinstance(result, Exception):
            raise result
        return result or SimpleNamespace(id=model)


class FakeOpenAIClient:
    def __init__(self, responses, model_failures=None):
        self.chat = SimpleNamespace(completions=FakeChatCompletions(responses))
        self.models = FakeModelsAPI(model_failures)
