"""글자 인식 2차 읽기(gemini_reader): Gemini 대신 가짜 답을 돌려주게 바꿔 규칙을 확인한다."""

import time

import pytest

from app import config, pipeline
from app.schemas import GradeRequest
from app.services import gemini_reader
from app.services.recognizer import Candidate, CellKind, Recognition
from tests.helpers import client, load_request


@pytest.fixture
def req() -> GradeRequest:
    return GradeRequest.model_validate(load_request("advanced"))


@pytest.fixture
def gemini(monkeypatch):
    """Gemini 대신 answer를 돌려준다. answer가 예외면 그 예외를 낸다. 부른 기록은 calls에."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(config, "LLM_READ", True)

    class Fake:
        answer: object = []
        calls: list[list[bytes]] = []

    def fake_ask(pngs):
        Fake.calls.append(pngs)
        if isinstance(Fake.answer, Exception):
            raise Fake.answer
        if callable(Fake.answer):
            return Fake.answer(pngs)
        return Fake.answer

    Fake.calls = []
    monkeypatch.setattr(gemini_reader, "_ask", fake_ask)
    return Fake


def _rec(index: int, chars: str, by_strokes: bool = False) -> Recognition:
    cands = [Candidate(char=c, score=0.5 - 0.05 * i) for i, c in enumerate(chars)]
    return Recognition(cell_index=index, kind=CellKind.CHAR, candidates=cands, by_strokes=by_strokes)


def _written(req: GradeRequest) -> list[int]:
    return [c.index for c in req.cells if c.strokes]


def test_read_as_target_and_in_engine_candidates_is_accepted(req, gemini):
    i = _written(req)[0]
    gemini.answer = ["보"]
    recs = [_rec(i, "고보짚싶곰")]
    got = gemini_reader.reread(req.model_copy(update={"cells": [req.cells[0]]}), recs, [["보"]])
    assert got[0].top.char == "보" and got[0].by_strokes and got[0].accepts("보")
    assert len(gemini.calls) == 1 and gemini.calls[0][0].startswith(b"\x89PNG")


def test_read_outside_engine_candidates_is_not_accepted(req, gemini):
    """'밧'을 '보'로 읽어도 엔진 후보에 '보'가 없으면 인정하지 않는다 (작은 받침을 놓친 경우)."""
    gemini.answer = ["보"]
    recs = [_rec(req.cells[0].index, "밧발봇밥뽀")]
    got = gemini_reader.reread(req.model_copy(update={"cells": [req.cells[0]]}), recs, [["보"]])
    assert got == recs


def test_read_other_than_target_keeps_engine_result(req, gemini):
    gemini.answer = ["고"]
    recs = [_rec(req.cells[0].index, "고보짚싶곰")]
    got = gemini_reader.reread(req.model_copy(update={"cells": [req.cells[0]]}), recs, [["보"]])
    assert got == recs


def test_only_ambiguous_cells_are_asked_in_one_call(req, gemini):
    """확실한 칸(획 비교로 정답 글자를 고름)은 묻지 않고, 애매한 칸들은 한 번에 묻는다."""
    cells = [c for c in req.cells if c.strokes][:3]
    sure = _rec(cells[0].index, "나너", by_strokes=True)
    vague = [_rec(cells[1].index, "녀너"), _rec(cells[2].index, "조좋", by_strokes=True)]
    gemini.answer = ["너", "좋"]
    got = gemini_reader.reread(req.model_copy(update={"cells": cells}), [sure, *vague], [["나"], ["너"], ["좋"]])
    assert len(gemini.calls) == 1 and len(gemini.calls[0]) == 2
    assert got[0] is sure and [r.top.char for r in got[1:]] == ["너", "좋"]


def test_nothing_ambiguous_means_no_call(req, gemini):
    gemini.answer = AssertionError("부르면 안 됨")
    recs = [_rec(req.cells[0].index, "나너", by_strokes=True)]
    assert gemini_reader.reread(req.model_copy(update={"cells": [req.cells[0]]}), recs, [["나"]]) == recs


@pytest.mark.parametrize("setting", ["no-key", "off"])
def test_no_key_or_turned_off_means_no_call(req, gemini, monkeypatch, setting):
    monkeypatch.setattr(
        config, "GEMINI_API_KEY" if setting == "no-key" else "LLM_READ", "" if setting == "no-key" else False
    )
    gemini.answer = AssertionError("부르면 안 됨")
    recs = [_rec(req.cells[0].index, "고보")]
    assert gemini_reader.reread(req.model_copy(update={"cells": [req.cells[0]]}), recs, [["보"]]) == recs


@pytest.mark.parametrize(
    "answer",
    [RuntimeError("429 RESOURCE_EXHAUSTED"), ["보", "보"], []],
    ids=["error", "too-many-answers", "no-answer"],
)
def test_failure_or_odd_answer_keeps_engine_result(req, gemini, answer):
    gemini.answer = answer
    recs = [_rec(req.cells[0].index, "고보")]
    assert gemini_reader.reread(req.model_copy(update={"cells": [req.cells[0]]}), recs, [["보"]]) == recs


def test_slow_answer_is_not_waited_for(req, gemini, monkeypatch):
    monkeypatch.setattr(config, "LLM_READ_TIMEOUT_MS", 50)

    def slow(pngs):
        time.sleep(0.5)
        return ["보"]

    gemini.answer = slow
    recs = [_rec(req.cells[0].index, "고보")]
    t0 = time.perf_counter()
    got = gemini_reader.reread(req.model_copy(update={"cells": [req.cells[0]]}), recs, [["보"]])
    assert got == recs and time.perf_counter() - t0 < 0.4


def test_cell_picture_is_a_png_of_fixed_size(req):
    png = gemini_reader._png(req.cells[0].strokes)
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and png[16:24] == (256).to_bytes(4, "big") * 2


@pytest.mark.skipif(not pipeline.reference_available(), reason="정답 획 사전 없음")
def test_grade_still_answers_when_second_read_breaks(monkeypatch):
    def broken(*args):
        raise RuntimeError("고장")

    monkeypatch.setattr(gemini_reader, "reread", broken)
    assert client.post("/grade", json=load_request("advanced")).status_code == 200
