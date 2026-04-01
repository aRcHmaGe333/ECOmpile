from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from .contracts import load_contract, validate
from .manifest import ManifestArtifact, read_jsonl, write_json, write_public_manifest
from .paths import PipelinePaths
from .redaction import SID_RE, WINDOWS_PATH_RE, redact_text


PROMOTION_THRESHOLD = 0.72
MIN_OCCURRENCES = 3
SCORE_FORMULA = "0.45*recurrence + 0.25*acceptance_signal + 0.20*cross_session_reuse + 0.10*determinism_shape"
TOKEN_RE = re.compile(r"[a-zA-Z0-9_\\/-]+")
STOPWORDS = {
    "the",
    "and",
    "for",
    "from",
    "with",
    "that",
    "this",
    "your",
    "into",
    "then",
    "also",
    "will",
    "would",
    "when",
    "what",
    "where",
    "which",
    "should",
    "could",
    "have",
    "about",
    "mode",
    "task",
    "path",
    "file",
}


@dataclass
class Cluster:
    key: str
    outcomes: list[dict]


def mine_candidates(paths: PipelinePaths, *, public_safe: bool = True) -> int:
    outcomes = read_jsonl(paths.outcomes_path)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for outcome in outcomes:
        grouped[_cluster_key(outcome)].append(outcome)

    clusters = [Cluster(key=key, outcomes=value) for key, value in grouped.items()]
    if not clusters:
        payload = _result_payload([], total_outcomes=0)
        write_json(paths.candidates_path, payload)
        public_path = paths.public_artifacts_dir / "ranked_candidates.public.json"
        write_json(public_path, _public_payload(payload) if public_safe else payload)
        write_public_manifest(
            paths,
            artifacts=[
                ManifestArtifact("ranked_candidates", str(paths.candidates_path), "internal"),
                ManifestArtifact("ranked_candidates_public", str(public_path), "public"),
            ],
            public_safe=public_safe,
        )
        return 0

    max_occurrences = max(len(cluster.outcomes) for cluster in clusters)
    all_sessions = {outcome["session_id"] for outcome in outcomes}
    max_sessions = max(1, len(all_sessions))
    contract = load_contract("kernel_candidate", paths.root)
    candidates: list[dict] = []

    for cluster in clusters:
        occurrences = len(cluster.outcomes)
        recurrence = occurrences / max_occurrences
        acceptance = _average(cluster.outcomes, "acceptance_signal")
        determinism = _average(cluster.outcomes, "determinism_shape")
        session_count = len({outcome["session_id"] for outcome in cluster.outcomes})
        cross_session = session_count / max_sessions
        score = (
            0.45 * recurrence
            + 0.25 * acceptance
            + 0.20 * cross_session
            + 0.10 * determinism
        )
        candidate = _build_candidate(
            cluster=cluster,
            recurrence=recurrence,
            acceptance_signal=acceptance,
            cross_session_reuse=cross_session,
            determinism_shape=determinism,
            occurrences=occurrences,
            score=score,
        )
        validate(contract, candidate, where=f"kernel_candidate:{candidate['candidate_id']}")
        candidate["promotion_eligible"] = bool(score >= PROMOTION_THRESHOLD and occurrences >= MIN_OCCURRENCES)
        candidates.append(candidate)

    candidates.sort(key=lambda item: (-item["score"], -item["metrics"]["occurrences"], item["candidate_id"]))
    payload = _result_payload(candidates, total_outcomes=len(outcomes))
    write_json(paths.candidates_path, payload)
    public_payload = _public_payload(payload) if public_safe else payload
    public_path = paths.public_artifacts_dir / "ranked_candidates.public.json"
    write_json(public_path, public_payload)
    write_public_manifest(
        paths,
        artifacts=[
            ManifestArtifact("ranked_candidates", str(paths.candidates_path), "internal"),
            ManifestArtifact("ranked_candidates_public", str(public_path), "public"),
        ],
        public_safe=public_safe,
    )
    return len(candidates)


def _result_payload(candidates: list[dict], *, total_outcomes: int) -> dict:
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "score_formula": SCORE_FORMULA,
        "promotion_threshold": PROMOTION_THRESHOLD,
        "min_occurrences": MIN_OCCURRENCES,
        "total_outcomes": total_outcomes,
        "candidates": candidates,
    }


def _build_candidate(
    *,
    cluster: Cluster,
    recurrence: float,
    acceptance_signal: float,
    cross_session_reuse: float,
    determinism_shape: float,
    occurrences: int,
    score: float,
) -> dict:
    representative = cluster.outcomes[0]
    intent_tokens = _top_intent_tokens(cluster.outcomes)
    emit_template = [
        line.strip()
        for line in str(representative["candidate_payload"]).splitlines()
        if line.strip()
    ]
    if not emit_template:
        emit_template = [str(representative["candidate_payload"]).strip()]
    provenance = [
        {"session_id": item["session_id"], "source_anchor": item["source_anchor"]}
        for item in cluster.outcomes[:8]
    ]
    digest = hashlib.sha1(cluster.key.encode("utf-8")).hexdigest()[:12]
    return {
        "candidate_id": f"KERN_{digest}".upper(),
        "intent_signature_tokens": intent_tokens,
        "context": representative["context"],
        "emit_template": emit_template,
        "stop_condition": "If intent signature and context match, stop exploration and emit template.",
        "forbid": [
            "Speculative detours before known deterministic template.",
            "Claims of missing native capability without direct check.",
            "Unbounded multi-path exploration when candidate kernel matches."
        ],
        "provenance": provenance,
        "metrics": {
            "recurrence": round(recurrence, 4),
            "acceptance_signal": round(acceptance_signal, 4),
            "cross_session_reuse": round(cross_session_reuse, 4),
            "determinism_shape": round(determinism_shape, 4),
            "occurrences": int(occurrences),
        },
        "score": round(score, 4),
    }


def _average(rows: list[dict], field: str) -> float:
    values = [float(item[field]) for item in rows]
    return sum(values) / max(1, len(values))


def _cluster_key(outcome: dict) -> str:
    payload = str(outcome.get("candidate_payload", "")).lower()
    payload = SID_RE.sub("[sid]", payload)
    payload = WINDOWS_PATH_RE.sub("[path]", payload)
    payload = re.sub(r"\d+", "[n]", payload)
    payload = re.sub(r"\s+", " ", payload).strip()
    return f"{outcome.get('candidate_type')}|{payload}"


def _top_intent_tokens(outcomes: list[dict]) -> list[str]:
    counter: Counter[str] = Counter()
    for outcome in outcomes:
        text = str(outcome.get("intent_text", "")).lower()
        for token in TOKEN_RE.findall(text):
            if len(token) < 3:
                continue
            if token in STOPWORDS:
                continue
            counter[token] += 1
    tokens = [token for token, _count in counter.most_common(8)]
    if len(tokens) == 0:
        return ["generic", "intent"]
    if len(tokens) == 1:
        return [tokens[0], "intent"]
    return tokens


def _public_payload(payload: dict) -> dict:
    clone = json.loads(json.dumps(payload))
    for candidate in clone.get("candidates", []):
        candidate["emit_template"] = [redact_text(line) for line in candidate.get("emit_template", [])]
    return clone
