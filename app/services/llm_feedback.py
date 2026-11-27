"""[혁윤] E-07 킹세종 LLM 한마디 (P1, 실패 시 템플릿).

Gemini Flash로 결과 화면 위 킹세종 말풍선을 학습자에게 맞는 말투로 바꾼다.
판정은 엔진이 하고 LLM은 말투만 바꾼다. 표정(expression)도 템플릿 기준 그대로 쓴다.

입출력
- 입력: 요청(req), 묶은 결과(AggregateResult)
  LLM에는 틀린 곳 설명(엔진이 칸별 errors로 만든 템플릿 문장), 정답 획에서 뽑은 바른 획순, 언어만 넘긴다
  (정진님 명세 4번).
  중수·고수는 정답 문장과 글자·띄어쓰기 설명을 보내지 않는다 (답을 말풍선에 말해 버리지 않게).
  학습자 손글씨·세션 정보는 보내지 않는다.
- 출력: SejongMessage(source = llm). 아래 경우에는 None → feedback_templates.sejong_message를 쓴다
    - GEMINI_API_KEY가 없음 (config.py)
    - 설명할 틀린 곳이 없음 (다 맞음, 균형만 통과 못 함, 획 판정을 못 함) → 미리 써 둔 템플릿 문구
    - 시간 초과(LLM_TIMEOUT_MS)·호출 실패·형식이 이상한 답 (빈 문장, 너무 긴 문장)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as WaitTimeout
from typing import Any

from pydantic import BaseModel

from app import config
from app.schemas import Explanation, GradeRequest, Mark, SejongMessage
from app.services import feedback_templates, stroke_guide
from app.services.aggregate import AggregateResult

log = logging.getLogger(__name__)

MAX_LEN = 60  # 말풍선 한 줄 최대 글자 수 (넘으면 화면이 깨질 수 있어 템플릿을 쓴다)

SYSTEM = """You are King Sejong, the creator of Hangul, a warm teacher on the result screen of a Korean handwriting \
app for foreign learners who trace K-pop lyrics. Write the speech bubble above the result as JSON with two fields.

text (at most 35 characters): first say concretely what is right and what is wrong, then end with a short \
inviting question. Use the facts given (which letters, how many words differ).
sub_text (at most 45 characters): one concrete next step.
- Stroke order or missing/extra strokes: tell how to write that jamo using ONLY the "Correct strokes" line \
given (you may mention just the first one or two strokes). Never change that order or direction, never add \
strokes. If no "Correct strokes" line is given, do not describe strokes; tell them to follow the orange box.
- Wrong letters or spacing at stage intermediate or advanced: do NOT reveal the answer. Never write any word \
or letter of the sentence, neither the wrong one nor the right one. Tell the learner where to look (the red boxes).
- Wrong letters or spacing at stage beginner: you may say which letter to fix.

Rules: mention only the mistakes given, never invent others. Never say the writing is perfect. \
No quotation marks around the whole line, no emoji. Write in the language requested. \
Polite, friendly modern Korean (해요체) when the language is Korean.

Examples (Korean):
- beginner, letters all right, stroke order wrong in 다 (ㄷ):
  Correct strokes of ㄷ in 다: 1 가로획 (왼→오), 2 꺾인 획 (아래로 꺾어 오른쪽으로)
  {"text": "글자는 다 맞았어요. 다의 ㄷ 쓰는 순서만 볼까요?",
   "sub_text": "ㄷ은 가로획을 먼저, 그다음 아래로 꺾어 써요."}
- intermediate, 1 of 2 words wrong:
  {"text": "한 어절이 달라요. 다시 써 볼까요?", "sub_text": "빨간 상자 안에서 어느 글자가 다른지 찾아보세요."}"""


class _Bubble(BaseModel):
    """LLM이 돌려줄 형식."""

    text: str
    sub_text: str


SERVER_MIN_TIMEOUT_MS = 10_000  # Gemini 서버는 10초보다 짧은 제한 시간을 받지 않는다 (400 INVALID_ARGUMENT)

# 기다리기는 우리 쪽에서 한다: LLM_TIMEOUT_MS가 지나면 답을 기다리지 않고 템플릿을 쓴다.
# 이미 보낸 요청은 뒤에서 끝까지(최대 SERVER_MIN_TIMEOUT_MS) 가고 결과는 버린다.
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm")
_client: Any = None


def _gemini() -> Any:
    """Gemini 클라이언트 (처음 부를 때 한 번 만든다). 재시도 없음."""
    global _client
    if _client is None:
        from google import genai
        from google.genai import types

        _client = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options=types.HttpOptions(
                timeout=max(config.LLM_TIMEOUT_MS, SERVER_MIN_TIMEOUT_MS),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
    return _client


def _ask(prompt: str) -> _Bubble | None:
    """Gemini에 묻고 답을 _Bubble로 받는다."""
    from google.genai import types

    response = _gemini().models.generate_content(
        model=config.LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            response_mime_type="application/json",
            response_schema=_Bubble,
            temperature=0.7,
        ),
    )
    parsed = response.parsed
    return parsed if isinstance(parsed, _Bubble) else None


MAX_GUIDES = 2  # 획순 쓰는 법은 앞쪽 칸 두 개까지만 넘긴다 (말풍선 한 줄에 다 못 담음)


def _prompt(req: GradeRequest, result: AggregateResult, mistakes: list[Explanation]) -> str:
    """LLM에 줄 사실들: 단계, 어절 몇 개 중 몇 개가 틀렸는지, 글자·띄어쓰기가 맞았는지, 틀린 곳, 바른 획순.

    중수·고수는 글자·띄어쓰기 답을 숨기는 단계라, 그 설명 문장(정답이 들어 있음)은 보내지 않고 개수만 보낸다.
    획순 쓰는 법은 Gemini가 지어내지 않게 정답 획에서 뽑아 준다 (stroke_guide).
    """
    en = req.locale == "en"
    beginner = req.stage.value == "beginner"
    wrong_words = [w.text for w in result.words if not w.correct]
    hidden = [m for m in mistakes if m.kind != Mark.STROKE and not beginner]
    lines = [
        f"Language: {'English' if en else 'Korean'}",
        f"Stage: {req.stage.value}",
        f"Words: {len(result.words)}, wrong words: {len(wrong_words)}"
        + (f" ({', '.join(wrong_words)})" if beginner and wrong_words else ""),
        "Letters and spacing: " + ("all right" if result.correct else "some are wrong"),
    ]
    if beginner:
        lines.insert(2, f"Target sentence: {req.target}")
    if hidden:
        kinds = sorted({"letters" if m.kind == Mark.CHAR else "spacing" for m in hidden})
        lines.append(f"Hidden mistakes (answer must stay secret): {len(hidden)} in {' and '.join(kinds)}, in red boxes")
    lines.append("Mistakes:")
    lines += [f"- {m.text}" for m in mistakes if m not in hidden]
    errors = {c.index: c.errors for c in result.cells}
    guides = 0
    for m in mistakes:
        if m.kind != Mark.STROKE or m.jamo is None or m.right is None or guides >= MAX_GUIDES:
            continue
        role = next((e.role.value for e in errors.get(m.cell_indexes[0], []) if e.jamo == m.jamo and e.role), None)
        if (how := stroke_guide.guide(m.right, m.jamo, role, en)) is not None:
            lines.append(f"Correct strokes of {m.jamo} in {m.right}: {how}")
            guides += 1
    return "\n".join(lines)


def sejong_line(req: GradeRequest, result: AggregateResult) -> SejongMessage | None:
    """킹세종 말투 한마디. 실패하거나 쓸 필요가 없으면 None."""
    if not config.GEMINI_API_KEY:
        return None
    mistakes = feedback_templates.explanations(req, result)
    if not mistakes:
        return None
    prompt = _prompt(req, result, mistakes)
    future = _pool.submit(_ask, prompt)
    try:
        bubble = future.result(timeout=config.LLM_TIMEOUT_MS / 1000)
    except WaitTimeout:
        # 아직 줄 서 있던 호출은 취소한다 (아무도 안 볼 답에 Gemini 사용량을 쓰지 않게). 이미 보낸 호출은 못 멈춘다
        future.cancel()
        log.warning("LLM 말풍선이 %dms 안에 오지 않아 템플릿을 씀", config.LLM_TIMEOUT_MS)
        return None
    except Exception as e:  # 키 오류·결제·네트워크 등: 화면은 템플릿으로. 이유(Google 오류 문장)도 남긴다
        log.warning("LLM 말풍선 실패, 템플릿을 씀: %s: %s", type(e).__name__, e)
        return None
    if bubble is None:
        log.warning("LLM 말풍선 형식이 이상해 템플릿을 씀")
        return None
    text, sub_text = bubble.text.strip(), bubble.sub_text.strip()
    if not text or not sub_text or len(text) > MAX_LEN or len(sub_text) > MAX_LEN:
        log.warning("LLM 말풍선이 비었거나 너무 길어 템플릿을 씀")
        return None
    expression = feedback_templates.sejong_message(req, result).expression  # 표정은 판정 결과로 정한다
    return SejongMessage(text=text, sub_text=sub_text, expression=expression, source="llm")
