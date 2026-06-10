from moneybot.config import Settings


def test_paper_starting_cash_default():
    s = Settings()
    assert s.paper_starting_cash == 100_000.0


def test_paper_starting_cash_overridable(monkeypatch):
    monkeypatch.setenv("MONEYBOT_PAPER_STARTING_CASH", "250000")
    s = Settings()
    assert s.paper_starting_cash == 250_000.0
