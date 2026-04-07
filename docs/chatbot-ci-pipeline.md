# Chatbot CI Pipeline Guide

This document describes a practical CI pipeline for RAG and chatbot services similar to this repository.

It is designed for apps that have:

- HTTP chat endpoints
- optional WebSocket chat flows
- retrieval dependencies such as Pinecone or another vector store
- LLM dependencies such as OpenAI
- a deployed environment that may not always be stable enough for hard blocking checks

## Is This Pipeline Good Enough?

Yes, with one important boundary:

- it is strong enough to prevent most broken or non-working code from reaching `main`
- it is not a guarantee of perfect answer quality or perfect production uptime

For this class of chatbot, that is the correct goal. CI should reliably catch engineering regressions, config mistakes, provider failures, bad request shapes, citation bugs, and integration failures. It should not pretend to prove factual perfection for every possible user query.

## What A Good Chatbot Pipeline Must Catch

For a production chatbot, the important failure points are:

- service does not boot
- health endpoints regress
- request validation breaks
- retrieval fails or changes shape
- reranking fails and fallback logic breaks
- LLM request parameters break after a model upgrade
- generation returns only fallback errors
- WebSocket auth/session flow breaks
- citation formatting or source mapping breaks
- provider credentials are missing or malformed
- deployed environment is reachable but not actually healthy

This repository covers those areas with a layered approach instead of one giant fragile end-to-end test.

## Recommended Pipeline Layers

### 1. Fast Unit And Mocked Integration Tests

Run these on every push and every pull request.

Purpose:

- catch code regressions fast
- keep feedback cheap and deterministic
- avoid paying for provider calls on every branch push

Typical coverage:

- `/`, `/api`, `/health`
- request validation failures
- successful chat generation with mocked retrieval and mocked LLM
- direct-response paths such as clarification or out-of-scope
- retrieval failure fallback
- rerank timeout fallback
- citation processing
- WebSocket message flow with mocked auth and mocked backend responses
- config defaults and forbidden model regressions

In this repo:

- workflow: [`../.github/workflows/ci.yml`](../.github/workflows/ci.yml)
- tests: [`../tests/test_app_endpoints.py`](../tests/test_app_endpoints.py), [`../tests/test_websocket.py`](../tests/test_websocket.py), [`../tests/test_query_rewriting.py`](../tests/test_query_rewriting.py), [`../tests/test_query_scenarios.py`](../tests/test_query_scenarios.py), [`../tests/test_citations.py`](../tests/test_citations.py), [`../tests/test_config_models.py`](../tests/test_config_models.py)

### 2. Live Provider Smoke Tests

Run these on `main` pushes or before deployment.

Purpose:

- verify that the real LLM and retrieval providers are reachable
- catch bad secrets, expired credentials, wrong model names, or broken provider networking

Typical coverage:

- OpenAI model reachability
- vector store reachability
- secret format sanity

In this repo:

- workflow: [`../.github/workflows/pre-deploy-validation.yml`](../.github/workflows/pre-deploy-validation.yml)
- tests: [`../tests/test_live_smoke.py`](../tests/test_live_smoke.py)

### 3. Local Live App Regression

Run the real app locally inside CI, but point it at real providers.

Purpose:

- validate the full app wiring, not just isolated provider calls
- catch startup/runtime failures that mocked tests cannot see
- catch request serialization, routing, and app-state initialization bugs

Typical coverage:

- start the app on an isolated port
- wait for `/health`
- send representative chat requests
- validate expected answer structure for a few key query types

In this repo:

- workflow job: `local-live-semantic-regression` in [`../.github/workflows/pre-deploy-validation.yml`](../.github/workflows/pre-deploy-validation.yml)
- tests: [`../tests/test_live_semantic_regression.py`](../tests/test_live_semantic_regression.py)

### 4. Optional Live WebSocket End-To-End Validation

Run this only when the backend auth/session dependency is stable and test credentials are available.

Purpose:

- validate the real client path for streaming chat
- catch auth/session integration failures between chatbot service and backend service

Typical coverage:

- mint or fetch a real backend token
- open WebSocket `/chat`
- send a real payload
- assert `processing`, `streaming`, and `completed` states
- assert final answer structure

In this repo:

- workflow job: `local-live-websocket-e2e` in [`../.github/workflows/pre-deploy-validation.yml`](../.github/workflows/pre-deploy-validation.yml)
- test: [`../tests/test_live_websocket_e2e.py`](../tests/test_live_websocket_e2e.py)

### 5. Manual Post-Deploy Smoke

Keep this manual unless your deployment system can reliably signal that rollout has finished.

Purpose:

- validate the deployed environment from outside the platform
- confirm DNS, TLS, routing, and runtime behavior after deployment

Why manual is often better:

- deployment platforms can report app health before the generation path is truly ready
- cloud cold starts and rollout timing can make automatic post-deploy checks noisy
- flaky post-deploy checks erode trust in the pipeline

In this repo:

- workflow: [`../.github/workflows/post-deploy-smoke.yml`](../.github/workflows/post-deploy-smoke.yml)
- test: [`../tests/test_deployed_smoke.py`](../tests/test_deployed_smoke.py)

## Recommended Gating Policy

For similar chatbot projects, use this policy:

- block merges on fast unit and mocked integration tests
- block `main` promotion on live provider smoke
- block `main` promotion on local live app regression
- block WebSocket E2E only if backend auth is stable and CI can mint a real token
- keep post-deploy smoke manual unless deployment completion is machine-verifiable

That balance gives strong protection without turning deployment into a flaky lottery.

## What This Repo Already Covers Well

This repository now has strong protection against:

- broken endpoints
- broken request validation
- broken retrieval and rerank fallbacks
- broken GPT request shapes after model changes
- broken query-routing paths
- broken citation generation
- missing or malformed provider secrets
- broken local live app startup with real providers
- broken WebSocket session path when backend credentials are available

## What It Does Not Prove

No chatbot CI pipeline should overclaim.

This pipeline does not prove:

- every answer is factually perfect
- every domain question returns the ideal wording
- the deployed environment will never have transient provider failures
- every external dependency is always available

If you need stronger semantic confidence, add a curated regression set with expected concepts rather than exact wording.

## Suggested Test Inventory For Similar Projects

Use a matrix like this:

| Layer | Trigger | Goal | Should Block? |
|---|---|---|---|
| Unit tests | every push, every PR | pure code behavior | yes |
| Mocked integration | every push, every PR | endpoint and flow behavior | yes |
| Live provider smoke | `main` push, release branch, manual | secrets and provider reachability | yes |
| Local live app regression | `main` push, release branch | real app with real providers | yes |
| Live WebSocket E2E | `main` push or manual | backend auth + stream flow | yes if stable |
| Post-deploy smoke | manual or deployment event | real deployed environment | usually no |

## Secret And Variable Strategy

Separate app runtime secrets from CI-only test inputs.

Runtime secrets usually include:

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `SECRET_KEY`
- `BACKEND_URL`

CI-only inputs may include:

- demo or test user credentials for minting a temporary token
- deployed app base URL
- optional staging-only backend URL

Do not store a short-lived access token as a long-term GitHub secret if you can mint it during CI.

## Template Workflow Structure

Use three workflows:

1. `ci.yml`
- trigger on every push and PR
- run only fast mocked tests

2. `pre-deploy-validation.yml`
- trigger on `main` pushes and manual runs
- run provider smoke
- run local live app regression
- optionally run WebSocket E2E

3. `post-deploy-smoke.yml`
- trigger manually, or from a real deployment completion signal
- run external smoke checks against the deployed URL

## Practical Rules For Other Chatbots

When reusing this pattern for another chatbot:

- keep `/health` simple and fast
- do not make a basic health endpoint depend on a live LLM call
- put deeper provider checks in dedicated probe endpoints or live smoke tests
- keep branch-push CI deterministic
- use real providers only where they add clear signal
- prefer representative semantic assertions over exact answer snapshots
- treat deployment smoke as operational validation, not ordinary unit CI

## Bottom Line

For chatbot applications like this one, this pipeline design is good enough to stop most bad or non-working code before it reaches `main` and before it is treated as release-ready.

That is the right standard for CI in an LLM-backed service.
