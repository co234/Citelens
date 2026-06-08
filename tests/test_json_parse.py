from citation_cleaner.llm.json_parse import coerce_json_string_list, parse_llm_json, strip_json_fences


def test_strip_json_fences():
    assert strip_json_fences("```json\n[1]\n```") == "[1]"


def test_parse_llm_json_repairs_unterminated_string():
    broken = '["Vaswani et al. 2017", "Devlin et al. 2019 with no close'
    parsed = parse_llm_json(broken)
    assert isinstance(parsed, list)
    assert len(parsed) >= 1


def test_parse_llm_json_object_with_fences():
    parsed = parse_llm_json('```json\n{"verdict": "same", "confidence": 0.9}\n```')
    assert parsed["verdict"] == "same"


def test_coerce_json_string_list_from_wrapper_object():
    parsed = {"references": ["Vaswani et al. 2017", "Devlin et al. 2019"]}
    assert coerce_json_string_list(parsed) == ["Vaswani et al. 2017", "Devlin et al. 2019"]


def test_coerce_json_string_list_from_object_items():
    parsed = [{"raw": "Smith et al. 2020"}, {"reference": "Jones, 2021"}]
    assert coerce_json_string_list(parsed) == ["Smith et al. 2020", "Jones, 2021"]
