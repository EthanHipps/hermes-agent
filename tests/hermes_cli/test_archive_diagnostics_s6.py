"""S6 archive suite E: every diagnostic bundle carries the disposition (§9.8 L1627, §12.1 L2044)."""

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from tests.agent.memory_service.native_sentinel import native_memory_sentinel


def _block(text: str) -> dict:
    lines = text.splitlines()
    start = lines.index("curated_memory:")
    body = [lines[start]] + [line for line in lines[start + 1:start + 7] if line.startswith("  ")]
    return yaml.safe_load("\n".join(body))


def _authoritative_home(tmp_path) -> Path:
    from hermes_cli.config import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    exe = tmp_path / "p.exe"
    exe.write_bytes(b"MZ")
    (home / "config.yaml").write_text(
        f"memory:\n  provider: example\n  provider_mode: authoritative\n"
        f"  provider_executable: '{exe}'\n  principal_id: ethan\n", encoding="utf-8")
    native = home / "memories"
    native.mkdir(exist_ok=True)
    (native / "MEMORY.md").write_text("dormant\n", encoding="utf-8")
    return home


def test_dump_carries_the_block_and_touches_no_native_file(monkeypatch, capsys, tmp_path):
    from hermes_cli import dump
    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")
    home = _authoritative_home(tmp_path)
    with native_memory_sentinel(home / "memories") as sentinel:
        dump.run_dump(SimpleNamespace(show_keys=False))
    sentinel.assert_untouched()
    out = capsys.readouterr().out
    assert _block(out)["curated_memory"] == {"authority": "provider", "provider": "example", "provider_api": 1,
                                             "included": False, "disposition": "provider-managed",
                                             "restore_action": "reconnect-provider"}
    assert "provider-managed and not included" in out


def test_additive_dump_has_no_block(monkeypatch, capsys, tmp_path):
    from hermes_cli import dump
    from hermes_cli.config import get_hermes_home
    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("display:\n  streaming: true\n", encoding="utf-8")
    dump.run_dump(SimpleNamespace(show_keys=False))
    assert "curated_memory:" not in capsys.readouterr().out


def test_share_bundle_and_nous_envelope_carry_the_block(monkeypatch, tmp_path):
    from hermes_cli import debug, dump
    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")
    _authoritative_home(tmp_path)
    bundle = debug.collect_share_bundle(log_lines=5, redact=True)
    assert _block(bundle["report"])["curated_memory"]["included"] is False
    envelope = json.loads(gzip.decompress(debug.build_nous_bundle(bundle)))
    assert "curated_memory:" in envelope["files"]["report"]
