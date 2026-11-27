"""[혁윤] D-01 글자 인식 2차 읽기: 엔진이 애매하게 본 칸만 Gemini에게 그림으로 한 번 더 읽힌다.

흘려 쓰거나 획을 이어 쓴 글씨는 그림·획 비교(recognizer)로 정답 글자를 못 고르는 일이 있다.
그런 칸만 모아 Gemini에게 '무슨 글자인지' 물어, 정답 글자로 읽히면 맞게 쓴 것으로 인정한다.

입출력
- 입력: 요청(req), 칸마다 인식 결과(recognitions), 칸마다 올 수 있는 정답 글자(expected)
- 출력: 인식 결과 목록 (같은 순서·개수). 인정한 칸만 Gemini가 읽은 글자를 1순위로 올린다 (by_strokes = True)

규칙
- 애매한 칸: 글자 칸인데 획 비교로 고르지 못했거나(예전 그림 판정으로 돌아감), 1순위가 이 칸의 정답 글자가 아닌 칸
- 채점 한 번에 Gemini는 한 번만 부른다: 애매한 칸 그림을 한 요청에 모아 보내고 글자 목록으로 받는다
  (무료 등급은 분당 요청 수로 막혀서, 칸마다 부르면 고수 30칸 채점 한 번에 4~6번이 된다)
- Gemini에는 칸 그림만 보낸다. 정답 글자는 알려 주지 않는다 (알려 주면 그 글자로 맞춰 읽는다)
- 인정 조건: Gemini가 읽은 글자가 ① 이 칸의 정답 글자이고 ② 엔진 후보(최대 5개) 안에 있을 때.
  ②가 없으면 Gemini가 작은 받침을 놓쳐 '밧'을 '보'로 읽는 식으로 틀린 글씨를 맞다고 하는 일이 있다
- Gemini가 다른 글자로 읽었다고 엔진이 맞다고 본 칸을 틀리게 만들지는 않는다
- 키가 없음·꺼 둠(LLM_READ=off)·시간 초과(LLM_READ_TIMEOUT_MS)·실패·이상한 답이면 엔진 결과 그대로

패드 샘플(2026-11-28, gemini-3.5-flash-lite): 애매하지만 제대로 쓴 22칸 중 11칸을 구하고,
일부러 틀리게 쓴 3칸·가짜 틀린 글씨 18칸은 하나도 맞다고 하지 않았다. 호출 한 번 약 1초.
"""

from __future__ import annotations

import json
import logging
import re
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as WaitTimeout
from dataclasses import replace

import numpy as np

from app import config
from app.schemas import GradeRequest, Stroke
from app.services import llm_feedback
from app.services.recognizer import MAX_CANDIDATES, MIN_ACCEPT_SCORE, Candidate, CellKind, Recognition

log = logging.getLogger(__name__)

MAX_CELLS = 10  # 한 번에 물어보는 칸 수 상한 (요청이 너무 커지지 않게. 넘는 칸은 엔진 결과 그대로)
SIZE, PAD_PX, RADIUS = 256, 24, 3.5  # 그림 크기, 가장자리 여백, 선 굵기(반지름) (픽셀)

PROMPT = (
    "Each image shows ONE Korean Hangul character handwritten by a beginner learner in a single grid cell. "
    "The writing may be messy, tilted or have strokes joined together. "
    "For each image, in order, give the Hangul syllable it shows. "
    'Answer only a JSON array of strings, one character per image, e.g. ["가", "나"].'
)

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm-read")


def ambiguous(rec: Recognition, expected: list[str]) -> bool:
    """Gemini에게 다시 물어볼 칸인지."""
    if rec.kind != CellKind.CHAR or rec.top is None or not expected:
        return False
    return not rec.by_strokes or rec.top.char not in expected


def reread(req: GradeRequest, recognitions: list[Recognition], expected: list[list[str]]) -> list[Recognition]:
    """애매한 칸을 Gemini에게 한 번에 읽혀, 인정한 칸만 바꾼 인식 결과 목록."""
    if not config.GEMINI_API_KEY or not config.LLM_READ:
        return recognitions
    cells = {c.index: c for c in req.cells}
    asks = [k for k, (rec, exp) in enumerate(zip(recognitions, expected, strict=True)) if ambiguous(rec, exp)]
    asks = [k for k in asks if cells[recognitions[k].cell_index].strokes][:MAX_CELLS]
    if not asks:
        return recognitions
    pngs = [_png(cells[recognitions[k].cell_index].strokes) for k in asks]
    future = _pool.submit(_ask, pngs)
    try:
        reads = future.result(timeout=config.LLM_READ_TIMEOUT_MS / 1000)
    except WaitTimeout:
        future.cancel()
        log.warning("글자 2차 읽기가 %dms 안에 오지 않아 엔진 결과를 씀 (%d칸)", config.LLM_READ_TIMEOUT_MS, len(asks))
        return recognitions
    except Exception as e:  # 키 오류·한도 초과·네트워크 등
        log.warning("글자 2차 읽기 실패, 엔진 결과를 씀: %s: %s", type(e).__name__, e)
        return recognitions
    if len(reads) != len(asks):
        log.warning("글자 2차 읽기 답의 개수가 달라 엔진 결과를 씀 (%d칸에 %d개)", len(asks), len(reads))
        return recognitions

    out = list(recognitions)
    for k, read in zip(asks, reads, strict=True):
        rec = recognitions[k]
        names = [c.char for c in rec.candidates]
        if read in expected[k] and read in names[:MAX_CANDIDATES]:
            top = Candidate(char=read, score=max(rec.score_of(read), MIN_ACCEPT_SCORE))
            others = [c for c in rec.candidates if c.char != read]
            out[k] = replace(rec, candidates=[top, *others][:MAX_CANDIDATES], by_strokes=True)
        log.info(
            "글자 2차 읽기: 칸 %d 엔진 %s → Gemini %s%s",
            rec.cell_index,
            names[0],
            read or "?",
            " (인정)" if out[k] is not rec else "",
        )
    return out


def _ask(pngs: list[bytes]) -> list[str]:
    """그림들을 한 요청으로 보내고 그림마다 읽은 글자(한글 음절 하나, 못 읽으면 '')를 받는다."""
    from google.genai import types

    parts: list[object] = []
    for i, png in enumerate(pngs, 1):
        parts += [f"Image {i}:", types.Part.from_bytes(data=png, mime_type="image/png")]
    parts.append(PROMPT)
    response = llm_feedback._gemini().models.generate_content(
        model=config.LLM_MODEL,
        contents=parts,
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
    )
    try:
        got = json.loads(response.text or "")
    except json.JSONDecodeError:
        return []
    if not isinstance(got, list):
        return []
    return [m.group(0) if (m := re.search(r"[가-힣]", str(x))) else "" for x in got]


def _png(strokes: list[Stroke]) -> bytes:
    """칸의 획을 흰 바탕 검은 선 그림(PNG, 회색조)으로. 쓴 자국에 맞춰 가운데 놓고 크게 그린다."""
    pts = [[(p[0], p[1]) for p in st.points] for st in strokes]
    xs = [x for line in pts for x, _ in line]
    ys = [y for line in pts for _, y in line]
    x0, y0 = min(xs), min(ys)
    span = max(max(xs) - x0, max(ys) - y0, 1e-6)
    s = (SIZE - 2 * PAD_PX) / span
    ox, oy = (SIZE - (max(xs) - x0) * s) / 2, (SIZE - (max(ys) - y0) * s) / 2

    img = np.full((SIZE, SIZE), 255, dtype=np.uint8)
    r = int(np.ceil(RADIUS))
    dy, dx = np.mgrid[-r : r + 1, -r : r + 1]
    disk = dx**2 + dy**2 <= RADIUS**2
    for line in pts:
        xy = np.array([(ox + (x - x0) * s, oy + (y - y0) * s) for x, y in line])
        if len(xy) > 1:  # 점 사이를 0.5px 간격으로 채운다
            seg = [np.linspace(a, b, max(2, int(np.hypot(*(b - a)) * 2))) for a, b in zip(xy, xy[1:], strict=False)]
            xy = np.concatenate(seg)
        for cx, cy in np.rint(xy).astype(int):
            ya, yb, xa, xb = cy - r, cy + r + 1, cx - r, cx + r + 1
            if ya < 0 or xa < 0 or yb > SIZE or xb > SIZE:
                continue  # 여백(PAD_PX)이 선 굵기보다 커서 생기지 않지만, 혹시 몰라 건너뛴다
            img[ya:yb, xa:xb][disk] = 0

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = b"".join(b"\x00" + row.tobytes() for row in img)
    header = struct.pack(">IIBBBBB", SIZE, SIZE, 8, 0, 0, 0, 0)  # 8비트 회색조
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
