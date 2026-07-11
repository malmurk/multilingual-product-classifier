from src.preprocess import build_input_text, clean_text


def test_clean_text_lowercases():
    assert clean_text("SAMSUNG Galaxy") == "samsung galaxy"


def test_clean_text_strips_special_chars():
    assert clean_text("product!!! @#$%") == "product"


def test_clean_text_preserves_cyrillic():
    assert clean_text("Компрессор автомобильный") == "компрессор автомобильный"


def test_clean_text_preserves_romanian():
    result = clean_text("Telefon Ă â Î ș ț mobil")
    assert "ă" in result
    assert "â" in result
    assert "î" in result
    assert "ș" in result
    assert "ț" in result


def test_build_input_text_title_only():
    result = build_input_text(title="Samsung A20")
    assert result == "samsung a20"


def test_build_input_text_with_attributes():
    result = build_input_text(title="Samsung A20", attributes={"color": "Black", "storage": "64GB"})
    assert "samsung a20" in result
    assert "black" in result
    assert "64gb" in result


def test_build_input_text_with_description():
    result = build_input_text(
        title="Samsung A20",
        description="A great smartphone with large screen and good battery life",
    )
    assert "samsung a20" in result
    assert "great smartphone" in result


def test_build_input_text_truncates_description():
    long_desc = "word " * 500
    result = build_input_text(title="X", description=long_desc)
    assert len(result) < 350


def test_build_input_text_handles_missing_fields():
    result = build_input_text(title="Test product")
    assert result == "test product"


def test_build_input_text_separates_with_pipe():
    result = build_input_text(title="Phone", attributes={"brand": "Samsung"})
    assert "|" in result
