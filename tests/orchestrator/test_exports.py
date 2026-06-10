import moneybot.orchestrator as orch


def test_public_exports():
    assert orch.Orchestrator is not None
    assert orch.build_orchestrator is not None
    assert set(["Orchestrator", "build_orchestrator"]).issubset(orch.__all__)
