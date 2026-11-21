from scalar_eagle_traceability import validate


def test_all_78_normative_requirements_mapped_once_to_existing_executable_tests():
    rows=validate()
    assert len(rows)==78
