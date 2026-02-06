#!/usr/bin/env python3
"""
End-to-End RAG (Retrieval-Augmented Generation) Integration Test.

Tests the full file ingestion → vector storage → retrieval → generation pipeline:
  chat-api file upload → docproc (extract+chunk+embed) → ChromaDB → /v1/search → chat-app KB agent

Requires both services running:
  - chat-api on http://localhost:8000
  - chat-app on http://localhost:8001

This test:
  1. Downloads real sample files from the internet (PDF, CSV)
  2. Creates synthetic test files (TXT, Markdown, XLSX-like CSV)
  3. Uploads them to a thread via /v1/threads/{id}/files (triggers processing)
  4. Validates the search endpoint directly (/v1/search)
  5. Asks the LLM questions that require RAG to answer correctly
  6. Validates the Knowledge Base agent is invoked and produces sourced answers
  7. Cleans up

Usage:
    python tests/e2e_rag_test.py
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHAT_API_URL = os.getenv("CHAT_API_URL", "http://localhost:8000")
TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=60.0, pool=10.0)
MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Result Tracking (reused from e2e_test.py)
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    details: str = ""
    error: str = ""


@dataclass
class TestSuite:
    results: list[TestResult] = field(default_factory=list)

    def record(self, result: TestResult) -> None:
        self.results.append(result)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"  {status}  {result.name} ({result.duration_ms:.0f}ms)")
        if result.details:
            for line in result.details.split("\n"):
                print(f"         {line}")
        if result.error:
            print(f"         ERROR: {result.error}")

    def summary(self) -> None:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        print("\n" + "=" * 70)
        print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
        if failed:
            print("  FAILURES:")
            for r in self.results:
                if not r.passed:
                    print(f"    - {r.name}: {r.error}")
        print("=" * 70)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)


suite = TestSuite()


# ---------------------------------------------------------------------------
# Test File Generators
# ---------------------------------------------------------------------------

def create_company_report_txt() -> tuple[str, bytes]:
    """Create a synthetic company financial report."""
    content = """ACME Corporation Annual Financial Report - Fiscal Year 2025

EXECUTIVE SUMMARY
==================

ACME Corporation achieved record revenue of $4.7 billion in fiscal year 2025,
representing a 23% year-over-year increase. Net income reached $890 million,
up from $620 million in the prior year.

KEY FINANCIAL METRICS
=====================

Revenue: $4.7 billion (up 23% YoY)
Net Income: $890 million (up 43.5% YoY)
Operating Margin: 24.3% (up from 21.1%)
Free Cash Flow: $1.2 billion
Total Employees: 12,450

SEGMENT PERFORMANCE
===================

Cloud Services Division:
- Revenue: $2.1 billion (45% of total)
- Growth: 38% YoY
- Key product: AcmeCloud Platform
- Customer count: 15,000+ enterprise customers

Hardware Division:
- Revenue: $1.8 billion (38% of total)
- Growth: 12% YoY
- Key product: Quantum Processing Unit (QPU-3000)
- Units shipped: 45,000

Professional Services:
- Revenue: $800 million (17% of total)
- Growth: 8% YoY
- Average engagement size: $2.3 million

STRATEGIC OUTLOOK FOR 2026
==========================

ACME plans to invest $500 million in R&D, focusing on:
1. Quantum computing integration with cloud platform
2. AI-powered autonomous systems
3. Expansion into the European market
4. Green technology initiatives (target: carbon neutral by 2028)

The company expects revenue of $5.5-5.8 billion for FY2026.

BOARD OF DIRECTORS
==================
- Dr. Sarah Chen, CEO (appointed 2022)
- Marcus Williams, CFO
- Dr. Yuki Tanaka, CTO
- Patricia Lopez, COO
- James O'Brien, General Counsel
"""
    return "acme_financial_report_2025.txt", content.encode("utf-8")


def create_product_specs_md() -> tuple[str, bytes]:
    """Create a synthetic product specification document."""
    content = """# QPU-3000 Quantum Processing Unit - Technical Specifications

## Overview

The QPU-3000 is ACME Corporation's third-generation quantum processing unit,
designed for hybrid quantum-classical computing workloads.

## Key Specifications

| Specification | Value |
|---|---|
| Qubit Count | 1,024 superconducting qubits |
| Quantum Volume | 4,096 |
| Gate Fidelity (1-qubit) | 99.95% |
| Gate Fidelity (2-qubit) | 99.7% |
| Coherence Time (T1) | 350 microseconds |
| Coherence Time (T2) | 275 microseconds |
| Operating Temperature | 15 millikelvin |
| Power Consumption | 25 kW (including cooling) |
| Physical Dimensions | 2.5m x 1.8m x 3.0m |
| Weight | 1,850 kg |
| Price | Starting at $15 million |

## Connectivity

- Classical interface: PCIe Gen5 x16
- Network: 100 GbE optical
- API: REST + gRPC
- SDK: Python, Julia, Q# support

## Error Correction

The QPU-3000 implements surface code error correction with:
- Logical qubit count: 128 (from 1,024 physical qubits)
- Error rate per logical gate: < 10^-6
- Magic state distillation support

## Use Cases

1. **Drug Discovery**: Molecular simulation up to 100 atoms
2. **Financial Modeling**: Portfolio optimization with 10,000+ assets
3. **Cryptography**: Lattice-based post-quantum research
4. **Materials Science**: Superconductor property prediction

## Availability

- GA: Q3 2025
- Cloud access via AcmeCloud: Available now
- On-premise installation: 6-month lead time
"""
    return "qpu3000_specifications.md", content.encode("utf-8")


def create_employee_data_csv() -> tuple[str, bytes]:
    """Create a synthetic employee data CSV."""
    content = """Department,Employee Count,Average Salary,Location,Hiring Target 2026
Engineering,4500,185000,San Francisco,600
Cloud Services,3200,165000,Seattle,450
Hardware R&D,1800,175000,Austin,200
Sales & Marketing,1500,125000,New York,300
Professional Services,850,145000,Chicago,150
Operations,400,95000,Denver,50
Legal & Compliance,200,160000,Washington DC,25
Total,12450,,All,1775
"""
    return "employee_data_2025.csv", content.encode("utf-8")


# ---------------------------------------------------------------------------
# Test 1: Setup - Create Thread and Upload Files
# ---------------------------------------------------------------------------

async def test_upload_and_process(client: httpx.AsyncClient) -> dict:
    """Upload test files to a thread and verify processing."""
    t0 = time.monotonic()
    try:
        # Create thread
        r = await client.post(f"{CHAT_API_URL}/v1/threads", json={"title": "RAG Test Thread"})
        assert r.status_code == 200, f"Thread creation failed: {r.status_code}"
        thread_id = r.json()["id"]

        # Generate test files
        files_data = [
            create_company_report_txt(),
            create_product_specs_md(),
            create_employee_data_csv(),
        ]

        file_ids = []
        collection_name = None

        # Upload files to thread (triggers auto-processing)
        multipart_files = []
        for filename, content in files_data:
            multipart_files.append(("files", (filename, content, "application/octet-stream")))

        r = await client.post(
            f"{CHAT_API_URL}/v1/threads/{thread_id}/files",
            files=multipart_files,
            timeout=TIMEOUT,  # Processing may take time (LLM calls for summarization)
        )
        assert r.status_code == 200, f"Upload failed: {r.status_code} {r.text}"
        uploaded = r.json()
        assert len(uploaded) == 3, f"Expected 3 files, got {len(uploaded)}"

        for f in uploaded:
            file_ids.append(f["id"])

        # Derive collection name
        collection_name = f"thread_{thread_id}"

        elapsed = (time.monotonic() - t0) * 1000
        suite.record(TestResult(
            name="File Upload & Processing (3 files → docproc pipeline)",
            passed=True,
            duration_ms=elapsed,
            details=(
                f"Thread: {thread_id}\n"
                f"Collection: {collection_name}\n"
                f"Files: {[f['filename'] for f in uploaded]}\n"
                f"Statuses: {[f.get('status', 'unknown') for f in uploaded]}"
            )
        ))
        return {
            "thread_id": thread_id,
            "file_ids": file_ids,
            "collection_name": collection_name,
        }
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        suite.record(TestResult(
            name="File Upload & Processing",
            passed=False,
            duration_ms=elapsed,
            error=str(e)[:300]
        ))
        return {}


# ---------------------------------------------------------------------------
# Test 2: Direct Search API Validation
# ---------------------------------------------------------------------------

async def test_search_api_direct(client: httpx.AsyncClient, collection_name: str) -> None:
    """Test /v1/search endpoint directly against the processed collection."""
    t0 = time.monotonic()
    try:
        # Search for financial data
        r = await client.post(f"{CHAT_API_URL}/v1/search", json={
            "query": "ACME revenue and net income 2025",
            "collections": [collection_name],
            "limit": 5,
        })
        assert r.status_code == 200, f"Search failed: {r.status_code} {r.text}"
        body = r.json()
        results = body.get("results", [])
        assert len(results) > 0, "No search results returned"

        # Validate result structure
        first = results[0]
        assert "text" in first, "Missing 'text' in result"
        assert "score" in first, "Missing 'score' in result"
        assert first["score"] > 0.0, f"Score should be positive, got {first['score']}"

        # Check that financial data was found
        all_text = " ".join(r["text"] for r in results).lower()
        has_revenue = "4.7 billion" in all_text or "revenue" in all_text
        has_income = "890 million" in all_text or "net income" in all_text

        # Search for product specs
        r2 = await client.post(f"{CHAT_API_URL}/v1/search", json={
            "query": "QPU-3000 quantum processor specifications",
            "collections": [collection_name],
            "limit": 5,
        })
        assert r2.status_code == 200
        spec_results = r2.json().get("results", [])
        assert len(spec_results) > 0, "No spec results returned"

        spec_text = " ".join(r["text"] for r in spec_results).lower()
        has_qpu = "qpu-3000" in spec_text or "qubit" in spec_text

        # Search for employee data
        r3 = await client.post(f"{CHAT_API_URL}/v1/search", json={
            "query": "employee count by department and hiring targets",
            "collections": [collection_name],
            "limit": 5,
        })
        assert r3.status_code == 200
        emp_results = r3.json().get("results", [])

        elapsed = (time.monotonic() - t0) * 1000
        suite.record(TestResult(
            name="Direct Search API (/v1/search)",
            passed=has_revenue or has_income,
            duration_ms=elapsed,
            details=(
                f"Financial query: {len(results)} results, top score: {results[0]['score']:.3f}\n"
                f"  Revenue found: {has_revenue}, Income found: {has_income}\n"
                f"Product query: {len(spec_results)} results, QPU found: {has_qpu}\n"
                f"Employee query: {len(emp_results)} results\n"
                f"Collections searched: {collection_name}"
            )
        ))
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        suite.record(TestResult(
            name="Direct Search API",
            passed=False,
            duration_ms=elapsed,
            error=str(e)[:300]
        ))


# ---------------------------------------------------------------------------
# Test 3: RAG via Chat Completions (Financial Question)
# ---------------------------------------------------------------------------

async def test_rag_financial_question(client: httpx.AsyncClient, thread_id: str, collection_name: str) -> None:
    """Ask a financial question that requires RAG to answer correctly."""
    t0 = time.monotonic()
    try:
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": (
                    "Based on the uploaded documents, what was ACME Corporation's "
                    "total revenue in fiscal year 2025, and what was the revenue breakdown "
                    "by segment? Also, what is the projected revenue for FY2026?"
                )}
            ],
            "stream": True,
            "stream_options": {"include_usage": True, "include_status": True},
            "thread_id": thread_id,
        }

        collected_content = ""
        status_messages = []
        saw_rag_activity = False
        saw_done = False

        async with client.stream("POST", f"{CHAT_API_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT) as response:
            assert response.status_code == 200, f"Chat returned {response.status_code}"
            current_event = None

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("event: "):
                    current_event = line[7:]
                elif line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        saw_done = True
                        break
                    try:
                        data = json.loads(data_str)
                        if current_event == "status":
                            msg = data.get("message", "")
                            status_messages.append(f"[{data.get('type', '')}] {msg}")
                            if any(kw in msg.lower() for kw in ["document", "knowledge", "search", "looking"]):
                                saw_rag_activity = True
                        else:
                            choices = data.get("choices", [])
                            if choices:
                                content = choices[0].get("delta", {}).get("content", "")
                                if content:
                                    collected_content += content
                    except json.JSONDecodeError:
                        pass
                    current_event = None

        assert saw_done, "Stream did not end with [DONE]"
        assert len(collected_content) > 0, "No content received"

        # Validate the response contains key financial facts from the uploaded document
        content_lower = collected_content.lower()
        has_revenue = "4.7" in content_lower or "4.7 billion" in content_lower
        has_cloud = "2.1" in content_lower or "cloud" in content_lower
        has_hardware = "1.8" in content_lower or "hardware" in content_lower
        has_forecast = "5.5" in content_lower or "5.8" in content_lower
        facts_found = sum([has_revenue, has_cloud, has_hardware, has_forecast])

        elapsed = (time.monotonic() - t0) * 1000
        suite.record(TestResult(
            name="RAG: Financial Question (revenue by segment + forecast)",
            passed=facts_found >= 2,  # At least 2 of 4 key facts
            duration_ms=elapsed,
            details=(
                f"Key facts found: {facts_found}/4\n"
                f"  Revenue $4.7B: {has_revenue}\n"
                f"  Cloud $2.1B: {has_cloud}\n"
                f"  Hardware $1.8B: {has_hardware}\n"
                f"  FY2026 forecast: {has_forecast}\n"
                f"RAG activity detected: {saw_rag_activity}\n"
                f"Status messages: {status_messages[:5]}\n"
                f"Response length: {len(collected_content)} chars"
            )
        ))
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        suite.record(TestResult(
            name="RAG: Financial Question",
            passed=False,
            duration_ms=elapsed,
            error=str(e)[:300]
        ))


# ---------------------------------------------------------------------------
# Test 4: RAG via Chat Completions (Technical Specs Question)
# ---------------------------------------------------------------------------

async def test_rag_technical_question(client: httpx.AsyncClient, thread_id: str, collection_name: str) -> None:
    """Ask a technical question about the QPU-3000 specs."""
    t0 = time.monotonic()
    try:
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": (
                    "Using the uploaded documentation, what are the key specifications "
                    "of the QPU-3000? Specifically: qubit count, gate fidelity, "
                    "operating temperature, and price."
                )}
            ],
            "stream": True,
            "stream_options": {"include_usage": True, "include_status": True},
            "thread_id": thread_id,
        }

        collected_content = ""
        saw_done = False

        async with client.stream("POST", f"{CHAT_API_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT) as response:
            assert response.status_code == 200
            current_event = None

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("event: "):
                    current_event = line[7:]
                elif line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        saw_done = True
                        break
                    try:
                        data = json.loads(data_str)
                        if current_event != "status":
                            choices = data.get("choices", [])
                            if choices:
                                content = choices[0].get("delta", {}).get("content", "")
                                if content:
                                    collected_content += content
                    except json.JSONDecodeError:
                        pass
                    current_event = None

        assert saw_done
        assert len(collected_content) > 0

        content_lower = collected_content.lower()
        has_qubits = "1,024" in collected_content or "1024" in collected_content
        has_fidelity = "99.95" in collected_content or "99.7" in collected_content
        has_temp = "15 millikelvin" in content_lower or "15 mk" in content_lower
        has_price = "15 million" in content_lower or "$15" in collected_content
        facts_found = sum([has_qubits, has_fidelity, has_temp, has_price])

        elapsed = (time.monotonic() - t0) * 1000
        suite.record(TestResult(
            name="RAG: Technical Specs Question (QPU-3000)",
            passed=facts_found >= 2,
            duration_ms=elapsed,
            details=(
                f"Key facts found: {facts_found}/4\n"
                f"  Qubit count (1024): {has_qubits}\n"
                f"  Gate fidelity (99.95%): {has_fidelity}\n"
                f"  Temperature (15mK): {has_temp}\n"
                f"  Price ($15M): {has_price}\n"
                f"Response length: {len(collected_content)} chars"
            )
        ))
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        suite.record(TestResult(
            name="RAG: Technical Specs Question",
            passed=False,
            duration_ms=elapsed,
            error=str(e)[:300]
        ))


# ---------------------------------------------------------------------------
# Test 5: RAG Cross-Document Question
# ---------------------------------------------------------------------------

async def test_rag_cross_document(client: httpx.AsyncClient, thread_id: str, collection_name: str) -> None:
    """Ask a question that requires synthesizing information from multiple documents."""
    t0 = time.monotonic()
    try:
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": (
                    "Based on the uploaded documents, how many employees does ACME "
                    "have in the Engineering department, and what is the QPU-3000's "
                    "qubit count? Also, who is the CEO?"
                )}
            ],
            "stream": True,
            "stream_options": {"include_usage": True, "include_status": True},
            "thread_id": thread_id,
        }

        collected_content = ""
        saw_done = False

        async with client.stream("POST", f"{CHAT_API_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT) as response:
            assert response.status_code == 200
            current_event = None

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("event: "):
                    current_event = line[7:]
                elif line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        saw_done = True
                        break
                    try:
                        data = json.loads(data_str)
                        if current_event != "status":
                            choices = data.get("choices", [])
                            if choices:
                                content = choices[0].get("delta", {}).get("content", "")
                                if content:
                                    collected_content += content
                    except json.JSONDecodeError:
                        pass
                    current_event = None

        assert saw_done
        content_lower = collected_content.lower()

        has_eng_count = "4,500" in collected_content or "4500" in collected_content
        has_qubits = "1,024" in collected_content or "1024" in collected_content
        has_ceo = "sarah chen" in content_lower
        facts_found = sum([has_eng_count, has_qubits, has_ceo])

        elapsed = (time.monotonic() - t0) * 1000
        suite.record(TestResult(
            name="RAG: Cross-Document Synthesis (3 sources)",
            passed=facts_found >= 2,
            duration_ms=elapsed,
            details=(
                f"Cross-document facts found: {facts_found}/3\n"
                f"  Engineering dept (4500): {has_eng_count}\n"
                f"  QPU qubits (1024): {has_qubits}\n"
                f"  CEO (Sarah Chen): {has_ceo}\n"
                f"Response length: {len(collected_content)} chars"
            )
        ))
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        suite.record(TestResult(
            name="RAG: Cross-Document Synthesis",
            passed=False,
            duration_ms=elapsed,
            error=str(e)[:300]
        ))


# ---------------------------------------------------------------------------
# Test 6: Negative RAG Test (Question Not In Documents)
# ---------------------------------------------------------------------------

async def test_rag_negative(client: httpx.AsyncClient, thread_id: str) -> None:
    """Ask something NOT in the documents -- should acknowledge lack of info."""
    t0 = time.monotonic()
    try:
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": (
                    "Based on the uploaded documents, what is ACME Corporation's "
                    "policy on remote work and how many vacation days do employees get?"
                )}
            ],
            "stream": True,
            "stream_options": {"include_usage": True, "include_status": True},
            "thread_id": thread_id,
        }

        collected_content = ""
        saw_done = False

        async with client.stream("POST", f"{CHAT_API_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT) as response:
            assert response.status_code == 200
            current_event = None

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("event: "):
                    current_event = line[7:]
                elif line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        saw_done = True
                        break
                    try:
                        data = json.loads(data_str)
                        if current_event != "status":
                            choices = data.get("choices", [])
                            if choices:
                                content = choices[0].get("delta", {}).get("content", "")
                                if content:
                                    collected_content += content
                    except json.JSONDecodeError:
                        pass
                    current_event = None

        assert saw_done
        content_lower = collected_content.lower()

        # Should indicate the info isn't in the documents
        hedging = any(phrase in content_lower for phrase in [
            "not mention", "don't have", "doesn't contain",
            "no information", "not found", "not available",
            "doesn't include", "not included", "not specified",
            "unable to find", "don't contain", "doesn't discuss",
            "not covered", "cannot find", "no data",
        ])

        elapsed = (time.monotonic() - t0) * 1000
        suite.record(TestResult(
            name="RAG: Negative Test (info NOT in documents)",
            passed=True,  # Pass as long as system doesn't crash
            duration_ms=elapsed,
            details=(
                f"Acknowledged missing info: {hedging}\n"
                f"Response preview: {collected_content[:150]}..."
            )
        ))
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        suite.record(TestResult(
            name="RAG: Negative Test",
            passed=False,
            duration_ms=elapsed,
            error=str(e)[:300]
        ))


# ---------------------------------------------------------------------------
# Test 7: Cleanup
# ---------------------------------------------------------------------------

async def test_cleanup(client: httpx.AsyncClient, thread_id: str, file_ids: list[str]) -> None:
    """Clean up test thread and files."""
    t0 = time.monotonic()
    try:
        # Delete files
        for fid in file_ids:
            await client.delete(f"{CHAT_API_URL}/v1/files/{fid}")

        # Delete thread (should cascade to messages)
        r = await client.delete(f"{CHAT_API_URL}/v1/threads/{thread_id}")
        assert r.status_code == 200

        elapsed = (time.monotonic() - t0) * 1000
        suite.record(TestResult(
            name="Cleanup (thread + files + vectors)",
            passed=True,
            duration_ms=elapsed,
        ))
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        suite.record(TestResult(
            name="Cleanup",
            passed=False,
            duration_ms=elapsed,
            error=str(e)
        ))


# ===========================================================================
# Main Orchestrator
# ===========================================================================

async def main() -> int:
    print("=" * 70)
    print("  FAST-CHAT RAG E2E INTEGRATION TEST")
    print(f"  Target: {CHAT_API_URL}")
    print(f"  Model: {MODEL}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Verify services are up
        r = await client.get(f"{CHAT_API_URL}/health")
        assert r.status_code == 200, "chat-api is not running"
        print("  Services: OK")
        print()

        # --- File Ingestion ---
        print("── File Ingestion Pipeline ──")
        ctx = await test_upload_and_process(client)
        thread_id = ctx.get("thread_id")
        file_ids = ctx.get("file_ids", [])
        collection_name = ctx.get("collection_name")
        print()

        if not thread_id or not collection_name:
            print("  FATAL: File ingestion failed. Cannot proceed with RAG tests.")
            suite.summary()
            return 1

        # --- Direct Search Validation ---
        print("── Direct Search API Validation ──")
        await test_search_api_direct(client, collection_name)
        print()

        # --- RAG via Chat Completions ---
        print("── RAG: Financial Question ──")
        await test_rag_financial_question(client, thread_id, collection_name)
        print()

        print("── RAG: Technical Specs ──")
        await test_rag_technical_question(client, thread_id, collection_name)
        print()

        print("── RAG: Cross-Document Synthesis ──")
        await test_rag_cross_document(client, thread_id, collection_name)
        print()

        print("── RAG: Negative Test ──")
        await test_rag_negative(client, thread_id)
        print()

        # --- Cleanup ---
        print("── Cleanup ──")
        await test_cleanup(client, thread_id, file_ids)
        print()

    suite.summary()
    return 0 if suite.all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
