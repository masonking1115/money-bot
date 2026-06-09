from moneybot.config import Settings
from moneybot.risk.kill_switch import kill_switch_active


def test_inactive_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    s = Settings(kill_switch_file=str(tmp_path / "nope"))
    assert kill_switch_active(s) is False


def test_active_when_env_flag_truthy(tmp_path, monkeypatch):
    monkeypatch.setenv("MONEYBOT_KILL_SWITCH", "1")
    s = Settings(kill_switch_file=str(tmp_path / "nope"))
    assert kill_switch_active(s) is True


def test_env_flag_false_string_stays_inactive(tmp_path, monkeypatch):
    monkeypatch.setenv("MONEYBOT_KILL_SWITCH", "false")
    s = Settings(kill_switch_file=str(tmp_path / "nope"))
    assert kill_switch_active(s) is False


def test_active_when_file_exists(tmp_path, monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    flag = tmp_path / "KILL_SWITCH"
    flag.write_text("halt")
    s = Settings(kill_switch_file=str(flag))
    assert kill_switch_active(s) is True
