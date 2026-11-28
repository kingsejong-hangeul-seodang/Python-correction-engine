"""자모 쓰는 법 문장(stroke_guide): 정답 획에서 뽑은 순서·방향이 맞는지.

정답 획 사전이 있어야 해서, 없으면(CI) 건너뛴다.
"""

import pytest

from app import pipeline
from app.services.stroke_guide import guide

needs_data = pytest.mark.skipif(not pipeline.reference_available(), reason="정답 획 사전 없음")


@needs_data
@pytest.mark.parametrize(
    ("char", "jamo", "role", "want"),
    [
        (
            "보",
            "ㅂ",
            "cho",
            "1 왼쪽 세로획 (위→아래), 2 오른쪽 세로획 (위→아래), 3 위 가로획 (왼→오), 4 아래 가로획 (왼→오)",
        ),
        ("다", "ㄷ", "cho", "1 가로획 (왼→오), 2 꺾인 획 (아래로 꺾어 오른쪽으로)"),
        ("가", "ㄱ", "cho", "1 꺾인 획 (오른쪽으로 꺾어 아래로)"),
        ("어", "ㅇ", "cho", "1 동그라미 (왼쪽으로 돌려)"),
        ("지", "ㅈ", "cho", "1 가로획 (왼→오), 2 왼쪽 비스듬한 획 (왼쪽 아래로), 3 오른쪽 비스듬한 획 (오른쪽 아래로)"),
        ("어", "ㅓ", "jung", "1 가로획 (왼→오), 2 세로획 (위→아래)"),
    ],
)
def test_guide_follows_reference_strokes(char, jamo, role, want):
    assert guide(char, jamo, role) == want


@needs_data
def test_guide_in_english():
    assert guide("다", "ㄷ", "cho", en=True) == "1 horizontal stroke (left to right), 2 bent stroke (down then right)"


@needs_data
def test_guide_without_role_uses_first_place():
    assert guide("앙", "ㅇ", None) == guide("앙", "ㅇ", "cho")


def test_unknown_char_or_jamo_gives_none(monkeypatch):
    from app.services import stroke_guide

    def missing(char):
        raise KeyError(char)

    monkeypatch.setattr(stroke_guide, "get_reference", missing)
    assert guide("A", "ㄱ", "cho") is None
