from pathlib import Path


def test_home_teaching_catalog_loader_declares_utf8() -> None:
    content_source = (
        Path(__file__).parents[1] / "src" / "mormi_api" / "content.py"
    ).read_text(encoding="utf-8")

    assert 'path.read_text(encoding="utf-8")' in content_source
