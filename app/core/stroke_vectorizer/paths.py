"""Paths shared by the reference reader and stroke data tools."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FONT_FILES_DIR = ROOT / "data" / "fonts"  # 폰트 파일 (.ttf/.otf)
FONTS_DIR = ROOT / "data" / "extracted"  # 폰트별 추출 결과
OUTPUTS_DIR = ROOT / "outputs"  # 검수용 그림 등 생성물 (git에 올리지 않음)
DEFAULT_FONT_FILE = FONT_FILES_DIR / "NotoSansKR-Light.ttf"  # 정답 획 기본 폰트
FONT_EXTS = (".ttf", ".otf")


def project_path(p):
    """프로젝트 루트 기준 상대 경로 → 실제 경로"""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def relative_to_root(p):
    """화면에 보여 줄 경로: 프로젝트 안이면 루트 기준 상대 경로"""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


# ═════════ 폰트 ═════════
def font_id(font=None):
    """폰트 → id (파일 이름에서 확장자 뺀 것). None = 기본 폰트.
    'NotoSansKR-Light', 'NotoSansKR-Light.ttf', 'data/fonts/NotoSansKR-Light.ttf' 모두 'NotoSansKR-Light'"""
    if font is None:
        return DEFAULT_FONT_FILE.stem
    p = Path(font)
    return p.stem if p.suffix.lower() in FONT_EXTS else p.name


def font_file(font=None):
    """폰트 → 폰트 파일 경로. 경로를 주면 그대로, 이름만 주면 data/fonts/에서 찾는다"""
    if font is not None and Path(font).suffix.lower() in FONT_EXTS:
        p = Path(font)
        if not p.is_absolute() and not p.exists():
            p = project_path(p)
        if p.exists():
            return p
    fid = font_id(font)
    for ext in FONT_EXTS:
        if (FONT_FILES_DIR / f"{fid}{ext}").exists():
            return FONT_FILES_DIR / f"{fid}{ext}"
    have = sorted(p.name for p in FONT_FILES_DIR.glob("*") if p.suffix.lower() in FONT_EXTS)
    raise SystemExit(f"폰트 파일 '{fid}' 없음. data/fonts/에 넣어 주세요. 있는 폰트: {', '.join(have) or '없음'}")


def font_dir(font=None):
    """폰트의 추출 결과 폴더 data/extracted/<폰트 id>/"""
    return FONTS_DIR / font_id(font)


def composed_path(font=None):
    """정답 획 사전 파일 (compose.py --all이 만들고 core/reference.py가 읽음)"""
    return font_dir(font) / "composed_all.json"
