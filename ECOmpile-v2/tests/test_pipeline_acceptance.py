import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))

from ecov2.compile_kernels import compile_staging_kernels
from ecov2.extract import extract_outcomes
from ecov2.ingest import ingest_linked_sessions
from ecov2.manifest import read_json, read_jsonl
from ecov2.mine import mine_candidates
from ecov2.paths import build_paths
from ecov2.review import apply_review_decisions, build_review_dilemmas
from ecov2.router import route_prompt


def _session_markdown(user_text: str, *, family: str, sid: str) -> str:
    return (
        f"# Session {family}\n"
        f"Provider: test-provider\n"
        f"Family: {family}\n"
        "Updated: 2026-04-01T10:00:00Z\n"
        "## Conversation\n"
        "**User:**\n"
        f"{user_text}\n"
        "\n"
        "**Assistant:**\n"
        "```powershell\n"
        f"cmd /c 'icacls C:\\ /remove *{sid} /t /c /q'\n"
        "```\n"
        "\n"
        "**User:**\n"
        "accepted\n"
    )


class TestECOmpileV2Acceptance(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        self.output_root = self.work / "ECOmpile-v2"
        self.output_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT / "schemas", self.output_root / "schemas")
        self.source_root = self.work / "linked_sessions"
        for family in ("codex", "vscode", "claude"):
            (self.source_root / family).mkdir(parents=True, exist_ok=True)

        sid = "S-1-5-21-1111-2222-3333-4444"
        (self.source_root / "codex" / "run1__s1.md").write_text(
            _session_markdown(
                "sid remove please use C:\\Users\\example\\secret and mail me at user@example.com",
                family="codex",
                sid=sid,
            ),
            encoding="utf-8",
        )
        (self.source_root / "codex" / "run1_dup__s1b.md").write_text(
            _session_markdown(
                "sid remove please use C:\\Users\\example\\secret and mail me at user@example.com",
                family="codex",
                sid=sid,
            ),
            encoding="utf-8",
        )
        (self.source_root / "vscode" / "run2__s2.md").write_text(
            _session_markdown("sid remove now", family="vscode", sid=sid),
            encoding="utf-8",
        )
        (self.source_root / "claude" / "run3__s3.md").write_text(
            _session_markdown("sid remove now", family="claude", sid=sid),
            encoding="utf-8",
        )

        # Invalid / metadata-only file (should be skipped)
        (self.source_root / "codex" / "metadata_only.md").write_text(
            "# No Conversation\nProvider: test\nFamily: codex\n",
            encoding="utf-8",
        )

        self.paths = build_paths(self.output_root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run_pipeline_to_staging(self) -> None:
        ingest_linked_sessions(self.paths, source_root=self.source_root, public_safe=True)
        extract_outcomes(self.paths, public_safe=True)
        mine_candidates(self.paths, public_safe=True)
        compile_staging_kernels(self.paths, public_safe=True)

    def test_01_ingestion_accepts_clean_families_and_skips_noise(self) -> None:
        result = ingest_linked_sessions(self.paths, source_root=self.source_root, public_safe=True)
        self.assertEqual(result.records_count, 3)
        self.assertEqual(result.skipped_files, 1)
        self.assertEqual(result.duplicate_sessions_skipped, 1)

        rows = read_jsonl(self.paths.normalized_path)
        self.assertEqual({row["family"] for row in rows}, {"codex", "vscode", "claude"})

    def test_02_extraction_emits_provenance_and_valid_payloads(self) -> None:
        ingest_linked_sessions(self.paths, source_root=self.source_root, public_safe=True)
        count = extract_outcomes(self.paths, public_safe=True)
        self.assertGreaterEqual(count, 3)
        rows = read_jsonl(self.paths.outcomes_path)
        self.assertTrue(all(row["source_anchor"].startswith("line:") for row in rows))
        self.assertTrue(all(row["candidate_payload"] for row in rows))

    def test_03_scoring_is_deterministic_for_identical_inputs(self) -> None:
        ingest_linked_sessions(self.paths, source_root=self.source_root, public_safe=True)
        extract_outcomes(self.paths, public_safe=True)

        mine_candidates(self.paths, public_safe=True)
        first = read_json(self.paths.candidates_path)
        first_order = [(item["candidate_id"], item["score"]) for item in first["candidates"]]

        mine_candidates(self.paths, public_safe=True)
        second = read_json(self.paths.candidates_path)
        second_order = [(item["candidate_id"], item["score"]) for item in second["candidates"]]

        self.assertEqual(first_order, second_order)

    def test_04_hitl_gate_blocks_unreviewed_kernels(self) -> None:
        self._run_pipeline_to_staging()

        decisions_file = self.work / "decisions-empty.json"
        decisions_file.write_text(json.dumps({"decisions": []}), encoding="utf-8")
        report = apply_review_decisions(self.paths, decisions_file=decisions_file, public_safe=True)

        self.assertEqual(report["promoted_count"], 0)
        self.assertTrue(self.paths.kernels_index_path.exists())
        lines = [line for line in self.paths.kernels_index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)

    def test_05_route_hits_active_kernel_after_approval(self) -> None:
        self._run_pipeline_to_staging()
        candidates = read_json(self.paths.candidates_path)["candidates"]
        self.assertTrue(candidates)
        candidate_id = candidates[0]["candidate_id"]

        decisions = {
            "decisions": [
                {
                    "candidate_id": candidate_id,
                    "decision": "approve",
                    "reviewer": "test-reviewer",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "reason": "Acceptance scenario",
                }
            ]
        }
        decisions_file = self.work / "decisions-approve.json"
        decisions_file.write_text(json.dumps(decisions), encoding="utf-8")
        apply_review_decisions(self.paths, decisions_file=decisions_file, public_safe=True)

        index_lines = self.paths.kernels_index_path.read_text(encoding="utf-8").splitlines()
        header = index_lines[0].split("\t")
        row = index_lines[1].split("\t")
        row_map = {header[i]: row[i] for i in range(len(header))}
        prompt = " ".join(token for token in row_map["tokens"].split(",") if token)

        routed = route_prompt(self.paths, prompt=prompt, platform=row_map["platform"], surface=row_map["context"])
        self.assertEqual(routed.status, "kernel_hit")
        self.assertEqual(routed.kernel_id, candidate_id)
        self.assertTrue(routed.emitted_template)

    def test_06_route_fallback_for_unknown_intent(self) -> None:
        self._run_pipeline_to_staging()
        candidates = read_json(self.paths.candidates_path)["candidates"]
        candidate_id = candidates[0]["candidate_id"]
        decisions = {
            "decisions": [
                {
                    "candidate_id": candidate_id,
                    "decision": "approve",
                    "reviewer": "test-reviewer",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
            ]
        }
        decisions_file = self.work / "decisions-approve2.json"
        decisions_file.write_text(json.dumps(decisions), encoding="utf-8")
        apply_review_decisions(self.paths, decisions_file=decisions_file, public_safe=True)

        routed = route_prompt(self.paths, prompt="completely unrelated unknown prompt", platform="windows", surface="cli")
        self.assertEqual(routed.status, "fallback")
        self.assertIn("No kernel matched", routed.reason)

    def test_07_public_outputs_are_redacted(self) -> None:
        self._run_pipeline_to_staging()

        public_norm = (self.paths.public_artifacts_dir / "normalized_sessions.public.jsonl").read_text(encoding="utf-8")
        public_outcomes = (self.paths.public_artifacts_dir / "extracted_outcomes.public.jsonl").read_text(encoding="utf-8")
        public_candidates = (self.paths.public_artifacts_dir / "ranked_candidates.public.json").read_text(encoding="utf-8")

        for blob in (public_norm, public_outcomes, public_candidates):
            self.assertNotIn("user@example.com", blob)
            self.assertNotIn("C:\\Users\\example", blob)
            self.assertNotIn("S-1-5-21-1111-2222-3333-4444", blob)

    def test_08_independence_when_experiment_is_temporarily_renamed(self) -> None:
        root = Path.home() / "code" / "ECOmpile"
        experiment = root / "Experiment"
        renamed = root / "Experiment.__independence_tmp__"
        if not experiment.exists():
            self.skipTest("Experiment path not present in environment")
        if renamed.exists():
            self.skipTest("Temporary Experiment rename target already exists")

        experiment.rename(renamed)
        try:
            result = ingest_linked_sessions(self.paths, source_root=self.source_root, public_safe=True)
            self.assertEqual(result.records_count, 3)
            extract_outcomes(self.paths, public_safe=True)
            mine_candidates(self.paths, public_safe=True)
            compile_staging_kernels(self.paths, public_safe=True)
            self.assertTrue(self.paths.staging_index_path.exists())
        finally:
            renamed.rename(experiment)

    def test_09_provider_packet_uses_v2_artifacts_only(self) -> None:
        provider_doc = ROOT / "docs" / "provider_packet.md"
        self.assertTrue(provider_doc.exists())
        text = provider_doc.read_text(encoding="utf-8")
        self.assertNotIn(str(Path.home() / "code" / "ECOmpile" / "Experiment"), text)
        self.assertNotIn("../Experiment", text)
        self.assertNotIn("../repo", text)

    def test_10_grouped_dilemma_review_reduces_manual_decisions(self) -> None:
        self._run_pipeline_to_staging()
        pack = build_review_dilemmas(self.paths, public_safe=True, include_non_eligible=False, similarity_threshold=0.5)
        self.assertGreaterEqual(pack["candidate_count"], 1)
        self.assertGreaterEqual(pack["dilemma_count"], 1)

        dilemmas = read_json(Path(pack["dilemmas_path"]))["dilemmas"]
        self.assertTrue(dilemmas)
        first = dilemmas[0]
        self.assertTrue(first["candidate_ids"])

        decisions_payload = {
            "dilemmas": [
                {
                    "dilemma_id": first["dilemma_id"],
                    "candidate_ids": first["candidate_ids"],
                    "decision": "approve",
                    "reviewer": "group-reviewer",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "reason": "Grouped decision test",
                }
            ]
        }
        decisions_file = self.work / "decisions-grouped.json"
        decisions_file.write_text(json.dumps(decisions_payload), encoding="utf-8")
        report = apply_review_decisions(self.paths, decisions_file=decisions_file, public_safe=True)
        self.assertGreaterEqual(report["promoted_count"], 1)


if __name__ == "__main__":
    unittest.main()
