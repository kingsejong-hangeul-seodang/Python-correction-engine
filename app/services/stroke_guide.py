"""[혁윤] 자모 쓰는 법 문장: 정답 획 데이터에서 바른 획순을 말로 바꾼다.

킹세종 말풍선(llm_feedback)이 획순 팁을 Gemini 지식으로 지어내면 틀린 순서를 말할 수 있어서,
엔진이 판정에 쓰는 정답 획(core/reference)에서 직접 순서·방향을 뽑아 사실로 넘긴다.

입출력
- 입력: 글자, 자모, 자리(cho/jung/jong), 언어
- 출력: "1 위 가로획(왼→오), 2 왼쪽 세로획(위→아래), ..." 같은 한 줄. 정답 획이 없으면 None

획 모양은 시작점·끝점·꺾인 점으로 고른다 (좌표 0~1, y는 아래로 갈수록 커짐).
- 점: 아주 짧은 획 / 동그라미: 시작과 끝이 거의 같고 길게 도는 획
- 가로·세로: 한쪽으로 곧게 / 삐침: 비스듬히 / 꺾임: 가운데서 크게 꺾이면 두 토막으로 말한다
- 같은 자모 안에 같은 모양 획이 2~3개면 서로 비교해 위치(위·가운데·아래, 왼쪽·가운데·오른쪽)를 붙인다
"""

from __future__ import annotations

import math

from app.core.reference import get_reference

Point = tuple[float, float]

_WORDS = {
    # 키: (한국어, 영어)
    "dot": ("점", "dot"),
    "circle": ("동그라미", "circle"),
    "h": ("가로획", "horizontal stroke"),
    "v": ("세로획", "vertical stroke"),
    "slant": ("비스듬한 획", "slanted stroke"),
    "bent": ("꺾인 획", "bent stroke"),
    "left": ("왼쪽", "left"),
    "right": ("오른쪽", "right"),
    "top": ("위", "top"),
    "bottom": ("아래", "bottom"),
    "middle": ("가운데", "middle"),
    "ccw": ("왼쪽으로 돌려", "counterclockwise"),
    "cw": ("오른쪽으로 돌려", "clockwise"),
}
_ARROW = {"right": "왼→오", "left": "오→왼", "down": "위→아래", "up": "아래→위"}
_ARROW_EN = {"right": "left to right", "left": "right to left", "down": "top to bottom", "up": "bottom to top"}
_GO = {  # 비스듬한·꺾인 획의 토막 방향
    "right": ("오른쪽으로", "right"),
    "left": ("왼쪽으로", "left"),
    "down": ("아래로", "down"),
    "up": ("위로", "up"),
    "down-left": ("왼쪽 아래로", "down-left"),
    "down-right": ("오른쪽 아래로", "down-right"),
    "up-left": ("왼쪽 위로", "up-left"),
    "up-right": ("오른쪽 위로", "up-right"),
}

DOT_LEN = 0.06  # 이보다 짧으면 점
STRAIGHT = 2.0  # 한쪽 성분이 다른 쪽의 이 배 이상이면 가로·세로
BEND = 0.3  # 시작·끝을 이은 선에서 이만큼(선 길이 대비) 벗어난 점이 있으면 꺾인 획


def _length(pts: list[Point]) -> float:
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:], strict=False))


def _heading(a: Point, b: Point) -> str:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) >= STRAIGHT * abs(dy):
        return "right" if dx > 0 else "left"
    if abs(dy) >= STRAIGHT * abs(dx):
        return "down" if dy > 0 else "up"
    return ("down" if dy > 0 else "up") + "-" + ("right" if dx > 0 else "left")


def _corner(pts: list[Point]) -> int | None:
    """시작·끝을 이은 선에서 가장 멀리 벗어난 점의 번호 (크게 꺾였을 때만)."""
    a, b = pts[0], pts[-1]
    chord = math.dist(a, b)
    if chord < 1e-6 or len(pts) < 3:
        return None

    def off(p: Point) -> float:
        return abs((b[0] - a[0]) * (a[1] - p[1]) - (a[0] - p[0]) * (b[1] - a[1])) / chord

    k = max(range(1, len(pts) - 1), key=lambda i: off(pts[i]))
    return k if off(pts[k]) > BEND * chord else None


def _shape(pts: list[Point]) -> tuple[str, list[str]]:
    """(모양, 방향 토막들). 모양: dot | circle | h | v | slant | bent"""
    length = _length(pts)
    if length < DOT_LEN:
        return "dot", []
    if math.dist(pts[0], pts[-1]) < 0.25 * length:  # 거의 제자리로 돌아오는 긴 획
        area = sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(pts, pts[1:] + pts[:1], strict=False))
        return "circle", ["ccw" if area < 0 else "cw"]  # y가 아래로 커지는 좌표라 부호가 수학과 반대
    k = _corner(pts)
    if k is not None:
        return "bent", [_heading(pts[0], pts[k]), _heading(pts[k], pts[-1])]
    h = _heading(pts[0], pts[-1])
    return ("h" if h in ("left", "right") else "v" if h in ("up", "down") else "slant"), [h]


def _places(shapes: list[str], pts: list[list[Point]]) -> list[str | None]:
    """같은 모양 획이 2~3개면 서로 비교해 자리를 붙인다 (가로·꺾인 획은 위·아래, 나머지는 왼쪽·오른쪽)."""
    out: list[str | None] = [None] * len(shapes)
    for shape in set(shapes):
        same = [i for i, x in enumerate(shapes) if x == shape]
        if shape == "circle" or not 2 <= len(same) <= 3:
            continue
        axis = 1 if shape in ("h", "bent") else 0
        at = {i: sum(p[axis] for p in pts[i]) / len(pts[i]) for i in same}
        ranked = sorted(same, key=lambda i: at[i])
        if at[ranked[-1]] - at[ranked[0]] < 0.1:  # 거의 같은 자리면 자리 말을 붙이지 않는다
            continue
        names = ("top", "middle", "bottom") if axis == 1 else ("left", "middle", "right")
        picks = (names[0], names[2]) if len(same) == 2 else names
        for i, name in zip(ranked, picks, strict=True):
            out[i] = name
    return out


def _say(shape: str, heads: list[str], where: str | None, en: bool) -> str:
    i = 1 if en else 0
    place = f"{_WORDS[where][i]} " if where else ""
    if shape == "dot":
        return f"{place}{_WORDS['dot'][i]}"
    if shape == "circle":
        return f"{_WORDS['circle'][i]} ({_WORDS[heads[0]][i]})"
    if shape in ("h", "v"):
        arrow = (_ARROW_EN if en else _ARROW)[heads[0]]
        return f"{place}{_WORDS[shape][i]} ({arrow})"
    goes = (" then " if en else " 꺾어 ").join(_GO[h][i] for h in heads)
    return f"{place}{_WORDS[shape][i]} ({goes})"


def guide(char: str, jamo: str, role: str | None, en: bool = False) -> str | None:
    """char 안의 jamo를 쓰는 바른 순서. 정답 획이 없거나 자모를 못 찾으면 None."""
    try:
        strokes = get_reference(char)
    except (KeyError, FileNotFoundError):
        return None
    mine = [s for s in strokes if s.jamo == jamo and (role is None or s.role.value == role)]
    if role is None and mine:  # 자리를 모르면 처음 나온 자리 하나만
        mine = [s for s in mine if s.role == mine[0].role]
    if not mine:
        return None
    shapes = [_shape(list(s.points)) for s in mine]
    places = _places([sh for sh, _ in shapes], [list(s.points) for s in mine])
    said = [_say(shape, heads, where, en) for (shape, heads), where in zip(shapes, places, strict=True)]
    return ", ".join(f"{n} {s}" for n, s in enumerate(said, 1))
