def test_the_suite_never_writes_to_the_real_feature_layer():
    from src.features import config as FC

    assert "ch8_test_" in str(FC.DATA_PROCESSED), FC.DATA_PROCESSED
    assert "ch8_test_" in str(FC.DATA_INTERIM), FC.DATA_INTERIM