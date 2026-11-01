#!/usr/bin/env python3
"""CPU-only positive/negative controls for Spark measurement attribution (not GPU evidence)."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spark25_kld_tail as tail
import spark25_qos as qos
import spark25_quality as quality
from spark25_validation import fnv1a64, plugin_hashes, sha256, validate_report


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / "home"
        for directory in ("kernels", "architectures", "quantizers"):
            (self.home / directory).mkdir(parents=True)
            (self.home / directory / "fixture.so").write_bytes(directory.encode())
        self.reference = self.root / "bf16.rad"
        self.candidate = self.root / "candidate.rad"
        self.runtime = self.root / "radiance"
        self.corpus = self.root / "corpus.jsonl"
        self.reference.write_bytes(b"BF16 checkpoint")
        self.candidate.write_bytes(b"IQ8R checkpoint")
        self.runtime.write_bytes(b"not executed; subprocess is mocked")
        self.corpus.write_text('{"prompt":"a b", "score_from":0}\n')
        self.out = self.root / "quality"
        self.score_kl = .1

    def fake_runtime(self, command, **kwargs):
        def arg(name):
            return command[command.index(name) + 1]
        if "--kld-record" in command:
            directory = Path(arg("--kld-record"))
            meta = {"format": "radiance-kld-top-1", "model": arg("--model"),
                    "corpus": arg("--kld-corpus"), "corpus_fnv1a64": fnv1a64(self.corpus),
                    "top": 128, "rows": 1,
                    "docs": [{"tokens": 2, "score_from": 0, "row0": 0, "source": "fixture"}]}
            (directory / "kld.json").write_text(json.dumps(meta))
            (directory / "tokens.i32").write_bytes(struct.pack("<ii", 1, 2))
            (directory / "rows.f32").write_bytes(struct.pack("<fff", 0, 0, 0))
            (directory / "top.bin").write_bytes(b"".join(struct.pack("<if", i, -float(i + 1))
                                                          for i in range(128)))
        else:
            directory = Path(arg("--kld-ref"))
            meta = json.loads((directory / "kld.json").read_text())
            report = {"model": arg("--model"), "reference": meta["model"],
                      "ref_dir": str(directory), "corpus": meta["corpus"],
                      "ref_form": "top-128 + tail", "all": {"positions": 1,
                      "kld": {key: self.score_kl for key in ("mean", "p99", "p99_9", "max")}}}
            path = Path(arg("--kld-out"))
            path.write_text(json.dumps(report))
            Path(str(path) + ".rows").write_bytes(struct.pack("<ffff", self.score_kl, 0, 0, 1))

    def quality_args(self, *extra):
        return ["spark25_quality.py", "--radiance", str(self.runtime), "--home", str(self.home),
                "--reference", str(self.reference), "--corpus", str(self.corpus),
                "--candidate", "full=" + str(self.candidate), "--out", str(self.out), *extra]

    def record_quality(self, *extra):
        with patch.object(sys, "argv", self.quality_args(*extra)), \
                patch.object(quality.subprocess, "run", side_effect=self.fake_runtime) as execute, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(quality.main(), 0)
        return execute

    def test_plugin_inventory_includes_architecture_and_optional_quantizer(self):
        inventory = plugin_hashes(self.home)
        for directory, key in (("kernels", "kernel_shared_objects"),
                               ("architectures", "architecture_shared_objects"),
                               ("quantizers", "quantizer_shared_objects")):
            self.assertEqual(inventory[key][directory + "/fixture.so"],
                             sha256(self.home / directory / "fixture.so"))

    def test_missing_architecture_inventory_rejected(self):
        (self.home / "architectures/fixture.so").unlink()
        with self.assertRaisesRegex(ValueError, "no shared objects"):
            plugin_hashes(self.home)

    def test_valid_quality_receipt_and_reuse_are_bound(self):
        self.record_quality()
        run = json.loads((self.out / "run.json").read_text())
        self.assertIn("architectures/fixture.so", run["architecture_shared_objects"])
        self.assertEqual(run["reference_provenance"]["inputs"]["reference_sha256"], sha256(self.reference))
        self.assertEqual(run["candidates"]["full"]["report_sha256"], sha256(self.out / "full-kld.json"))
        execute = self.record_quality("--reuse-reference")
        self.assertEqual(execute.call_count, 1)  # score only, no new reference recording

    def test_recording_input_mutations_do_not_publish_provenance_or_receipt(self):
        paths = (self.reference, self.runtime, self.corpus, self.home / "architectures/fixture.so")
        for index, path in enumerate(paths):
            with self.subTest(path=path):
                self.out = self.root / f"record-mutation-{index}"
                original = path.read_bytes()

                def mutate_during_record(command, **kwargs):
                    self.fake_runtime(command, **kwargs)
                    if "--kld-record" in command:
                        path.write_bytes(original + b"changed")

                try:
                    with patch.object(sys, "argv", self.quality_args()), \
                            patch.object(quality.subprocess, "run", side_effect=mutate_during_record) as execute:
                        with self.assertRaisesRegex(ValueError, "reference inputs changed"):
                            quality.main()
                    self.assertEqual(execute.call_count, 1)
                    self.assertFalse((self.out / "bf16-top128-ref/provenance.json").exists())
                    self.assertFalse((self.out / "run.json").exists())
                finally:
                    path.write_bytes(original)

    def test_scoring_input_mutations_prevent_success_receipt(self):
        paths = (self.reference, self.runtime, self.corpus, self.home / "architectures/fixture.so")
        for index, path in enumerate(paths):
            with self.subTest(path=path):
                self.out = self.root / f"score-mutation-{index}"
                original = path.read_bytes()

                def mutate_during_score(command, **kwargs):
                    self.fake_runtime(command, **kwargs)
                    if "--kld-ref" in command:
                        path.write_bytes(original + b"changed")

                try:
                    with patch.object(sys, "argv", self.quality_args()), \
                            patch.object(quality.subprocess, "run", side_effect=mutate_during_score), \
                            contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(ValueError, "reference inputs changed"):
                            quality.main()
                    self.assertFalse((self.out / "run.json").exists())
                finally:
                    path.write_bytes(original)

    def test_failed_retry_removes_previous_success_receipt(self):
        self.record_quality()
        self.reference.write_bytes(b"changed")
        with patch.object(sys, "argv", self.quality_args("--reuse-reference")):
            with self.assertRaises(ValueError):
                quality.main()
        self.assertFalse((self.out / "run.json").exists())

    def test_signed_kl_roundoff_is_preserved(self):
        self.score_kl = -1e-10
        self.record_quality()
        self.bind_tail()
        raw = (self.out / "full-kld.json.rows").read_bytes()
        self.assertLess(struct.unpack_from("<f", raw)[0], 0)

    def test_nonfinite_kl_does_not_emit_success_receipt(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                self.out = self.root / str(value)
                self.score_kl = value
                with patch.object(sys, "argv", self.quality_args()), \
                        patch.object(quality.subprocess, "run", side_effect=self.fake_runtime), \
                        contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        quality.main()
                self.assertFalse((self.out / "run.json").exists())

    def test_reuse_rejects_changed_baseline_bytes_at_same_path(self):
        self.record_quality()
        self.reference.write_bytes(b"different BF16 checkpoint at the same path")
        with patch.object(sys, "argv", self.quality_args("--reuse-reference")), \
                patch.object(quality.subprocess, "run") as execute:
            with self.assertRaisesRegex(ValueError, "baseline, runtime, settings or data"):
                quality.main()
        execute.assert_not_called()

    def test_reuse_rejects_changed_architecture_before_scoring(self):
        self.record_quality()
        (self.home / "architectures/fixture.so").write_bytes(b"stale or rebuilt graph")
        with patch.object(sys, "argv", self.quality_args("--reuse-reference")), \
                patch.object(quality.subprocess, "run") as execute:
            with self.assertRaises(ValueError):
                quality.main()
        execute.assert_not_called()

    def test_reuse_rejects_unbound_legacy_reference(self):
        self.record_quality()
        (self.out / "bf16-top128-ref/provenance.json").unlink()
        with patch.object(sys, "argv", self.quality_args("--reuse-reference")):
            with self.assertRaisesRegex(ValueError, "lacks provenance"):
                quality.main()

    def test_reuse_rejects_changed_reference_payload(self):
        self.record_quality()
        (self.out / "bf16-top128-ref/top.bin").write_bytes(b"changed")
        with patch.object(sys, "argv", self.quality_args("--reuse-reference")):
            with self.assertRaises(ValueError):
                quality.main()

    def test_report_mismatches_rejected_even_with_equal_row_counts(self):
        self.record_quality()
        report = json.loads((self.out / "full-kld.json").read_text())
        for key in ("ref_dir", "reference", "corpus", "model"):
            with self.subTest(key=key):
                bad = copy.deepcopy(report)
                bad[key] = str(self.root / "different-but-same-row-count")
                with self.assertRaisesRegex(ValueError, "does not match"):
                    validate_report(bad, self.out / "bf16-top128-ref", self.reference,
                                    self.corpus, self.candidate, 1)

    def bind_tail(self):
        return tail.bound_run(self.out / "run.json", self.out / "bf16-top128-ref",
                              self.out / "full-kld.json")

    def test_tail_binds_report_reference_candidate_and_runtime(self):
        self.record_quality()
        run, candidate, _, _, common = self.bind_tail()
        self.assertEqual(candidate["model"], str(self.candidate))
        self.assertIn(str(self.home), common)
        self.assertEqual(run["architecture_shared_objects"], plugin_hashes(self.home)["architecture_shared_objects"])

    def test_tail_rejects_report_from_other_candidate(self):
        self.record_quality()
        path = self.out / "full-kld.json"
        report = json.loads(path.read_text())
        report["model"] = str(self.root / "other.rad")
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "report model"):
            self.bind_tail()

    def test_tail_rejects_changed_candidate_or_row_scores(self):
        for changed in ("candidate", "rows"):
            with self.subTest(changed=changed):
                # Each subcase starts from one valid run, then invalidates exactly one artifact.
                if self.out.exists():
                    import shutil
                    shutil.rmtree(self.out)
                self.record_quality()
                path = self.candidate if changed == "candidate" else self.out / "full-kld.json.rows"
                path.write_bytes(path.read_bytes() + b"changed")
                with self.assertRaisesRegex(ValueError, "artifact hash"):
                    self.bind_tail()

    def test_private_server_uses_bound_candidate_and_rejects_foreign_alias(self):
        self.record_quality()
        run, candidate, _, _, common = self.bind_tail()
        for foreign in (False, True):
            with self.subTest(foreign=foreign):
                proc = Mock(pid=123)
                proc.poll.return_value = None
                commands = []
                def start(command, **kwargs):
                    commands.append(command)
                    return proc
                def get(url, timeout):
                    return {"data": [{"id": "different-server" if foreign else commands[0][-1]}]}
                with patch.object(tail.subprocess, "Popen", side_effect=start), patch.object(tail, "get", side_effect=get):
                    if foreign:
                        with self.assertRaisesRegex(ValueError, "private server identity"):
                            with tail.candidate_server(run, candidate, common, self.root / f"server-{foreign}.log"):
                                self.fail("must not expose an unrelated server")
                    else:
                        with tail.candidate_server(run, candidate, common, self.root / f"server-{foreign}.log") as server:
                            command = server["command"]
                            self.assertEqual(command[command.index("--model") + 1], str(self.candidate))
                            self.assertTrue(server["model"].startswith("spark-tail-"))
                proc.terminate.assert_called_once()

    def response(self, data):
        return io.BytesIO(json.dumps(data).encode())

    def test_qos_requires_complete_positive_usage(self):
        valid = {"usage": {"prompt_tokens": 3, "completion_tokens": 2},
                 "choices": [{"finish_reason": "length"}]}
        with patch.object(qos.urllib.request, "urlopen", return_value=self.response(valid)):
            measured = qos.request("http://localhost", "spark", "a b", 2, 1)
        self.assertEqual(measured["completion_tokens"], 2)
        variants = [{}, {"usage": {}}, {"usage": None}]
        for missing in ("prompt_tokens", "completion_tokens"):
            bad = copy.deepcopy(valid)
            del bad["usage"][missing]
            variants.append(bad)
        for field, value in (("prompt_tokens", 0), ("prompt_tokens", "3"),
                              ("prompt_tokens", True), ("completion_tokens", 0),
                              ("completion_tokens", 1), ("completion_tokens", 3),
                              ("completion_tokens", "2"), ("completion_tokens", True)):
            bad = copy.deepcopy(valid)
            bad["usage"][field] = value
            variants.append(bad)
        bad = copy.deepcopy(valid)
        bad["choices"][0]["finish_reason"] = "stop"
        variants.append(bad)
        for bad in variants:
            with self.subTest(response=bad), \
                    patch.object(qos.urllib.request, "urlopen", return_value=self.response(bad)):
                with self.assertRaises(ValueError):
                    qos.request("http://localhost", "spark", "a b", 2, 1)

    def test_complete_qos_records_architecture_and_expected_work(self):
        output = self.root / "complete-qos"
        args = ["spark25_qos.py", "--radiance", str(self.runtime), "--home", str(self.home),
                "--model", str(self.candidate), "--prompts", str(self.corpus), "--out", str(output),
                "--requests", "1", "--repeats", "1", "--concurrency", "1", "--max-tokens", "2"]
        proc = Mock()
        proc.poll.return_value = None
        response = {"usage": {"prompt_tokens": 3, "completion_tokens": 2},
                    "choices": [{"finish_reason": "length"}]}
        with patch.object(sys, "argv", args), patch.object(qos.subprocess, "Popen", return_value=proc), \
                patch.object(qos.urllib.request, "urlopen", side_effect=lambda *a, **k: self.response(response)), \
                patch.object(qos, "verify_server"), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(qos.main(), 0)
        receipt = json.loads((output / "qos.json").read_text())
        self.assertIn("architectures/fixture.so", receipt["architecture_shared_objects"])
        self.assertEqual(receipt["results"]["1"]["generated_tokens"], 2)
        proc.terminate.assert_called_once()

    def test_reuse_rejects_stale_or_partial_scorer_output(self):
        self.record_quality()
        for output in ('none', 'report', 'rows'):
            with self.subTest(output=output):
                # Restore an old valid report, then simulate a successful incomplete scorer.
                if not (self.out / 'run.json').exists():
                    self.record_quality('--reuse-reference')
                def incomplete(command, **kwargs):
                    report = self.out / 'full-kld.json'
                    rows = Path(str(report) + '.rows')
                    old_report, old_rows = self.saved_outputs
                    if output == 'report':
                        report.write_bytes(old_report)
                    if output == 'rows':
                        rows.write_bytes(old_rows)
                self.saved_outputs = ((self.out / 'full-kld.json').read_bytes(),
                                      (self.out / 'full-kld.json.rows').read_bytes())
                with patch.object(sys, 'argv', self.quality_args('--reuse-reference')), \
                        patch.object(quality.subprocess, 'run', side_effect=incomplete):
                    with self.assertRaises(FileNotFoundError):
                        quality.main()
                self.assertFalse((self.out / 'run.json').exists())
                self.assertEqual((self.out / 'full-kld.json').read_bytes(), self.saved_outputs[0])
                self.assertEqual((self.out / 'full-kld.json.rows').read_bytes(), self.saved_outputs[1])

    def test_quality_rejects_output_collisions_before_deleting_inputs(self):
        for name in ('full-kld.json', 'full-kld.json.rows', 'run.json'):
            with self.subTest(name=name):
                self.out = self.root / ('collision-' + name)
                self.out.mkdir()
                colliding = self.out / name
                colliding.write_bytes(b'irreplaceable input')
                self.candidate = colliding
                with patch.object(sys, 'argv', self.quality_args()), \
                        patch.object(quality.subprocess, 'run') as execute, \
                        contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        quality.main()
                self.assertEqual(colliding.read_bytes(), b'irreplaceable input')
                execute.assert_not_called()

    def test_quality_protects_external_plugin_symlink_targets(self):
        for name in ('run.json', 'full-kld.json', 'full-kld.json.rows'):
            with self.subTest(name=name):
                self.out = self.root / ('plugin-' + name)
                self.out.mkdir()
                target = self.out / name
                target.write_bytes(b'plugin target')
                plugin = self.home / 'kernels/fixture.so'
                plugin.unlink()
                plugin.symlink_to(target)
                with patch.object(sys, 'argv', self.quality_args()), \
                        patch.object(quality.subprocess, 'run') as execute, \
                        contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        quality.main()
                self.assertEqual(target.read_bytes(), b'plugin target')
                execute.assert_not_called()

    def test_quality_rolls_back_scores_on_promotion_or_receipt_failure(self):
        for failure_name in ('full-kld.json.rows', 'run.json'):
            with self.subTest(failure_name=failure_name):
                self.out = self.root / ('promotion-' + failure_name)
                self.record_quality()
                paths = (self.out / 'full-kld.json', self.out / 'full-kld.json.rows')
                original = [path.read_bytes() for path in paths]
                real_replace = os.replace
                failed = False
                def fail_once(src, dst):
                    nonlocal failed
                    if not failed and Path(dst) == self.out / failure_name:
                        failed = True
                        raise OSError('simulated promotion failure')
                    return real_replace(src, dst)
                with patch.object(sys, 'argv', self.quality_args('--reuse-reference')), \
                        patch.object(quality.subprocess, 'run', side_effect=self.fake_runtime), \
                        patch.object(quality.os, 'replace', side_effect=fail_once), \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(OSError, 'simulated promotion failure'):
                        quality.main()
                self.assertTrue(failed)
                self.assertEqual([path.read_bytes() for path in paths], original)
                self.assertFalse((self.out / 'run.json').exists())

    def test_rollback_failure_retains_backup_and_restores_other_scores(self):
        self.record_quality()
        paths = (self.out / 'full-kld.json', self.out / 'full-kld.json.rows')
        original = [path.read_bytes() for path in paths]
        real_replace = os.replace
        failures = []
        def fail_publish_and_one_restore(src, dst):
            src, dst = Path(src), Path(dst)
            if dst == self.out / 'run.json':
                failures.append('publish')
                raise OSError('publication blocked')
            if src.name == 'backup-1':
                failures.append('restore')
                raise OSError('restore blocked')
            return real_replace(src, dst)
        with patch.object(sys, 'argv', self.quality_args('--reuse-reference')), \
                patch.object(quality.subprocess, 'run', side_effect=self.fake_runtime), \
                patch.object(quality.os, 'replace', side_effect=fail_publish_and_one_restore), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'publication blocked.*recover backups') as error:
                quality.main()
        self.assertEqual(failures, ['publish', 'restore'])
        self.assertIsInstance(error.exception.__cause__, OSError)
        self.assertEqual(paths[0].read_bytes(), original[0])
        self.assertFalse(paths[1].exists())
        retained = list(self.out.glob('.spark25-score-*'))
        self.assertEqual(len(retained), 1)
        self.assertEqual((retained[0] / 'backup-1').read_bytes(), original[1])
        self.assertIn(str(retained[0]), str(error.exception))
        self.assertFalse((self.out / 'run.json').exists())

    def test_tail_log_cannot_truncate_model_or_follow_symlink(self):
        self.record_quality()
        run, candidate, _, _, common = self.bind_tail()
        before = self.candidate.read_bytes()
        log = self.root / 'tail.server.log'
        log.symlink_to(self.candidate)
        for target in (log, self.candidate):
            with self.subTest(target=target), patch.object(tail.subprocess, 'Popen') as start:
                with self.assertRaises(FileExistsError):
                    with tail.candidate_server(run, candidate, common, target):
                        self.fail('must not start')
                start.assert_not_called()
                self.assertEqual(self.candidate.read_bytes(), before)

    def test_qos_log_symlink_cannot_truncate_model(self):
        output = self.root / 'symlink-qos'
        output.mkdir()
        (output / 'server.log').symlink_to(self.candidate)
        before = self.candidate.read_bytes()
        args = ['spark25_qos.py', '--radiance', str(self.runtime), '--home', str(self.home),
                '--model', str(self.candidate), '--prompts', str(self.corpus), '--out', str(output),
                '--requests', '1', '--concurrency', '1']
        with patch.object(sys, 'argv', args), patch.object(qos.subprocess, 'Popen') as start:
            with self.assertRaises(FileExistsError):
                qos.main()
        self.assertEqual(self.candidate.read_bytes(), before)
        self.assertFalse((output / 'qos.json').exists())
        start.assert_not_called()

    def test_qos_rejects_fewer_requests_than_concurrency(self):
        args = ['spark25_qos.py', '--radiance', str(self.runtime), '--home', str(self.home),
                '--model', str(self.candidate), '--prompts', str(self.corpus), '--out', str(self.root / 'qos'),
                '--requests', '1', '--concurrency', '4']
        with patch.object(sys, 'argv', args), contextlib.redirect_stderr(io.StringIO()), \
                patch.object(qos.subprocess, 'Popen') as start:
            with self.assertRaises(SystemExit):
                qos.main()
        start.assert_not_called()

    def test_qos_rejects_foreign_server_and_dead_child(self):
        proc = Mock()
        proc.poll.return_value = None
        with patch.object(qos.urllib.request, 'urlopen', return_value=self.response({'data': [{'id': 'foreign'}]})):
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                qos.verify_server('http://localhost', 'unique-id', proc)
        proc.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, 'exited'):
            qos.verify_server('http://localhost', 'unique-id', proc)
        proc.poll.return_value = None
        with patch.object(qos.urllib.request, 'urlopen', return_value=self.response({'data': [{'id': 'unique-id'}]})):
            qos.verify_server('http://localhost', 'unique-id', proc)

    def test_qos_rejects_prompt_file_replacement_after_snapshot(self):
        output = self.root / 'changed-qos'
        args = ['spark25_qos.py', '--radiance', str(self.runtime), '--home', str(self.home),
                '--model', str(self.candidate), '--prompts', str(self.corpus), '--out', str(output),
                '--requests', '1', '--repeats', '1', '--concurrency', '1', '--max-tokens', '2']
        proc = Mock()
        proc.poll.return_value = None
        response = {'usage': {'prompt_tokens': 3, 'completion_tokens': 2},
                    'choices': [{'finish_reason': 'length'}]}
        def change(*args):
            self.corpus.write_text('{"prompt":"different"}\n')
        with patch.object(sys, 'argv', args), patch.object(qos.subprocess, 'Popen', return_value=proc), \
                patch.object(qos.urllib.request, 'urlopen', side_effect=lambda *a, **k: self.response(response)), \
                patch.object(qos, 'verify_server', side_effect=change), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'prompt file'):
                qos.main()
        self.assertFalse((output / 'qos.json').exists())

    def test_failed_qos_does_not_emit_success_receipt(self):
        output = self.root / "qos"
        args = ["spark25_qos.py", "--radiance", str(self.runtime), "--home", str(self.home),
                "--model", str(self.candidate), "--prompts", str(self.corpus), "--out", str(output),
                "--requests", "1", "--repeats", "1", "--concurrency", "1", "--max-tokens", "2"]
        proc = Mock()
        proc.poll.return_value = None
        with patch.object(sys, "argv", args), patch.object(qos.subprocess, "Popen", return_value=proc), \
                patch.object(qos.urllib.request, "urlopen", side_effect=lambda *a, **k: self.response({})), \
                patch.object(qos, "verify_server"):
            with self.assertRaisesRegex(ValueError, "missing usage"):
                qos.main()
        proc.terminate.assert_called_once()
        self.assertFalse((output / "qos.json").exists())


if __name__ == "__main__":
    unittest.main()
