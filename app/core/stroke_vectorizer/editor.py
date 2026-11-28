"""
③ 수동 조정 편집기 (로컬 웹 UI)

  python -m app.core.stroke_vectorizer.editor [--font <폰트>] [--port 8765]   → 브라우저에서 http://localhost:8765

화면에서 획을 고르고 점을 끌어 옮긴 뒤 저장하면 data/extracted/<폰트>/tweaks.json에 이동량이 저장된다.
범위: "이 글자만" 또는 "같은 자리 전체"(예: 받침 있는 ㅝ 전부). 반영 방식: docs/extracted_data.md
"자모 방식" (tweaks.json "mode", 고르면 바로 저장): 그 자모를 모든 글자에서
  자모 모양 그대로 (jamo)      자모 단독 모양 그대로 자리 박스에 넣는다
  부품마다 단독 자모 (base)    겹자모만. 부품 박스마다 부품의 단독 자모 모양 (ㅀ → ㄹ, ㅎ)
다 고친 뒤 python -m app.core.stroke_vectorizer.compose --all 로 정답 획 사전을 다시 만들어야 서버에 반영된다.
"""

import argparse
import json
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.core.stroke_vectorizer.compose import (
    Profile,
    compose,
    deltas_for,
    fixed_mode,
    get_profile,
    set_delta,
    set_fixed_mode,
    slot_jamo,
    sub_base,
    tweak_summary,
)
from app.core.stroke_vectorizer.jamo_templates import PIECES, all_syllables, decompose, ksx1001

HERE = Path(__file__).parent
KSX = set(ksx1001())
EDITOR_MODES = (None, "jamo", "base")  # 편집기에서 고를 수 있는 자모 방식 (None = 자동, box_mode.json)


def slot_of(syl):
    """글자의 자리 id 목록 (초·중·종)"""
    parts = decompose(syl)
    jung, has_jong = parts[1][1], len(parts) == 3
    return [sub_base(r, j, jung, has_jong) for r, j in parts]


def syl_view(prof, syl):
    """편집용: 획마다 base(①② 결과, 자모 방식 고정 반영) + 자리 조정 + 글자 조정"""
    strokes = compose(syl, prof.id, meta=True, use_tweaks=False, use_fixed_mode=True)
    for s in strokes:
        s["slot_d"], s["syl_d"] = deltas_for(prof.tweaks, s["slot"], s["idx"], syl, s["order"])
        s["fixed_mode"] = fixed_mode(prof.tweaks, s["jamo"])  # None / "jamo" / "base"
        s["compound"] = len(PIECES.get(s["jamo"], [s["jamo"]])) > 1  # 겹자모 (base를 고를 수 있음)
        del s["points"]
    return {"syl": syl, "strokes": strokes}


def slot_tweaks_of(prof, jamo):
    """그 자모의 자리 조정(점 이동량)이 있는 자리 id들"""
    return [k for k in prof.tweaks.get("slot", {}) if slot_jamo(k) == jamo]


def same_slot(slot, n, exclude):
    """같은 자리를 쓰는 글자 (완성형 먼저)"""
    hits = [c for c in all_syllables() if c != exclude and slot in slot_of(c)]
    hits.sort(key=lambda c: c not in KSX)
    ksx = [c for c in hits if c in KSX]
    pick = random.Random(slot).sample(ksx, min(n, len(ksx)))
    if len(pick) < n:
        pick += [c for c in hits if c not in KSX][: n - len(pick)]
    return pick, len(hits)


class Handler(BaseHTTPRequestHandler):
    prof: Profile  # 실행할 때 main에서 정함

    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json; charset=utf-8", code=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        p = self.prof
        try:
            if u.path == "/":
                self.send((HERE / "editor.html").read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/font":
                self.send(Path(p.meta["file"]).read_bytes(), "font/ttf")
            elif u.path == "/api/meta":
                self.send(
                    {
                        "font": p.id,
                        "name": p.meta.get("name", p.id),
                        "upm": p.meta["upm"],
                        "em_top": p.meta["em_top"],
                    }
                )
            elif u.path == "/api/syl":
                self.send(syl_view(p, q["c"]))
            elif u.path == "/api/many":
                self.send([syl_view(p, c) for c in q["c"] if 0xAC00 <= ord(c) <= 0xD7A3])
            elif u.path == "/api/slot_chars":
                chars, total = same_slot(q["slot"], int(q.get("n", 12)), q.get("exclude"))
                self.send({"chars": "".join(chars), "total": total})
            elif u.path == "/api/tweaks":
                self.send(tweak_summary(p.tweaks))
            elif u.path == "/api/slot_tweaks":
                self.send({"slots": slot_tweaks_of(p, q["jamo"])})
            else:
                self.send({"error": "not found"}, code=404)
        except Exception as e:  # noqa: BLE001
            self.send({"error": f"{type(e).__name__}: {e}"}, code=400)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        p = self.prof
        try:
            if u.path == "/api/save":
                # body: {"syl", "edits": [{"order", "slot_d": [[dx,dy]×32] | null, "syl_d": ... }]}
                syl = body["syl"]
                meta = {s["order"]: s for s in compose(syl, p.id, meta=True, use_tweaks=False)}
                for e in body["edits"]:
                    s = meta[e["order"]]
                    if "slot_d" in e:
                        set_delta(p.tweaks, "slot", s["slot"], s["idx"], e["slot_d"] or [[0, 0]])
                    if "syl_d" in e:
                        set_delta(p.tweaks, "syllable", syl, s["order"], e["syl_d"] or [[0, 0]])
                p.save_tweaks()
                self.send(syl_view(p, syl))
            elif u.path == "/api/mode":
                # body: {"syl", "jamo", "mode": null|"jamo"|"base", "clear_slot": bool} → 자모 방식 고정/해제, 바로 저장
                jamo, mode = body["jamo"], body.get("mode")
                if mode not in EDITOR_MODES:
                    raise ValueError(f"mode는 {EDITOR_MODES} 중 하나: {mode}")
                if mode == "base" and len(PIECES.get(jamo, [jamo])) < 2:
                    raise ValueError(f"{jamo}는 겹자모가 아니라 base를 쓸 수 없음")
                set_fixed_mode(p.tweaks, jamo, mode)
                if body.get("clear_slot"):  # 기준 모양이 바뀌어 기존 자리 이동량이 어긋나므로
                    for k in slot_tweaks_of(p, jamo):
                        del p.tweaks["slot"][k]
                p.save_tweaks()
                self.send(syl_view(p, body["syl"]))
            else:
                self.send({"error": "not found"}, code=404)
        except Exception as e:  # noqa: BLE001
            self.send({"error": f"{type(e).__name__}: {e}"}, code=400)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", help="폰트 (data/fonts/ 안의 파일 이름 또는 경로, 기본 DONGGUK-UNIVERSITY)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    Handler.prof = get_profile(args.font)
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as e:
        raise SystemExit(
            f"포트 {args.port}을 이미 다른 프로세스가 쓰는 중입니다.\n"
            f"  확인: ss -ltnp | grep :{args.port}\n"
            f"  다른 포트로: python -m app.core.stroke_vectorizer.editor --port {args.port + 1}"
        ) from e
    print(f"편집기: http://localhost:{args.port}  (폰트 {Handler.prof.id}, 저장 → {Handler.prof.dir / 'tweaks.json'})")
    print("끝내려면 Ctrl+C")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
