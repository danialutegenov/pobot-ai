from pathlib import Path

from app.config import AppConfig


def test_config_uses_repo_relative_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = AppConfig.from_env()

    assert config.raw_dir == tmp_path / "data" / "raw"
    assert config.processed_dir == tmp_path / "data" / "processed"
    assert config.top_k == 5
    assert config.chat_model == "deepseek-chat"

