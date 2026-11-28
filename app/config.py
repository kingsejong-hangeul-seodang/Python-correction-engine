"""환경변수 설정."""

import os

# 응답의 engineVersion. 판정의 일부(글자 인식 등)가 아직 임시인 동안은 "-stub"을 붙여 백엔드가 구분할 수 있게 한다.
ENGINE_VERSION = os.getenv("ENGINE_VERSION", "0.1.0-stub")

# 킹세종 LLM 말풍선 (services/llm_feedback.py). 키가 없으면 LLM을 쓰지 않고 템플릿 말풍선만 쓴다.
# 키는 Google AI Studio에서 발급한다. 절대 깃에 올리지 않는다 (서버 환경변수로만 넣는다).
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = os.getenv(
    "LLM_MODEL", "gemini-3.5-flash-lite"
)  # 무료 등급이 있는 Flash-Lite (2026-11 확인). 바꾸려면 이름을 넣는다
# 이 시간 안에 Gemini 답이 없으면 템플릿 말풍선. Gemini가 붐비면 3초를 넘기도 해서 5초로 둔다
# (서버가 엔진을 최대 10초 기다리므로, 판정 시간(글자 인식 획 비교로 30칸이면 약 3초)·첫 요청 사전 읽기를 더해도
# 그 안에 끝나게)
LLM_TIMEOUT_MS = int(os.getenv("LLM_TIMEOUT_MS", "5000"))

# 글자 인식 2차 읽기 (services/gemini_reader.py): 엔진이 애매하게 본 칸만 Gemini에게 그림으로 다시 읽힌다.
# 키가 있으면 켜져 있다. 끄려면 LLM_READ=off. 칸 그림(손글씨)이 Google로 간다 (무료 등급은 Google이 개선에 쓸 수 있음)
LLM_READ = os.getenv("LLM_READ", "on").lower() != "off"
# 이 시간 안에 답이 없으면 엔진 결과 그대로. 보통 약 1초.
# 판정(30칸 약 3초) + 이것 + 말풍선(LLM_TIMEOUT_MS)이 서버 대기 시간 안에 들어오게 둔다
LLM_READ_TIMEOUT_MS = int(os.getenv("LLM_READ_TIMEOUT_MS", "3000"))
