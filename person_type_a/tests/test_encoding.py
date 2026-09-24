from person_type_a.encoding import LETTERS, assign_letters, check_letters_single_token


class CharTokenizer:
    """测试用假分词器：每字符一个 token，id = ord(char)。"""
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_assign_letters_in_order():
    opts = ("airport ground staff", "flight attendant", "passenger", "__insufficient_evidence__")
    m = assign_letters(opts)
    assert m == {"airport ground staff": "A", "flight attendant": "B",
                 "passenger": "C", "__insufficient_evidence__": "D"}


def test_assign_letters_caps_at_26():
    m = assign_letters(tuple(f"c{i}" for i in range(26)))
    assert len(m) == 26 and m["c25"] == "Z"


def test_check_letters_single_token_ok():
    bad = check_letters_single_token(CharTokenizer())
    assert bad == []


class WeirdTokenizer(CharTokenizer):
    def encode(self, text, add_special_tokens=False):
        # 把 "Q" 拆成两个 token，模拟词表异常
        if text == "Q":
            return [81, 82]
        return [ord(c) for c in text]


def test_check_letters_reports_multitoken():
    assert check_letters_single_token(WeirdTokenizer()) == ["Q"]


class VariantTokenizer(CharTokenizer):
    """模拟 SentencePiece 变体：裸 "A" 是 id 65，但接在 "：" 后续写成 999（▁A 形态）。"""
    def encode(self, text, add_special_tokens=False):
        if text.endswith("A") and "：" in text:
            return [ord(c) for c in text[:-1]] + [999]
        return [ord(c) for c in text]


def test_letter_token_id_resolves_in_context_variant():
    from person_type_a.encoding import letter_token_id
    assert letter_token_id(VariantTokenizer(), "A", context="：") == 999


def test_letter_token_id_defaults_to_bare_when_context_clean():
    from person_type_a.encoding import letter_token_id
    assert letter_token_id(CharTokenizer(), "A", context="：") == ord("A")
