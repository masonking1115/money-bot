import moneybot.execution as ex


def test_public_exports():
    assert ex.ExecutionAdapter is not None
    assert ex.build_execution_adapter is not None
    assert set(["ExecutionAdapter", "build_execution_adapter"]).issubset(ex.__all__)
