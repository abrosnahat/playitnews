"""
planner.py — превращает текст сценария озвучки в СТРУКТУРИРОВАННЫЙ план сцен.

Вместо передачи matcher'у сырого текста, сценарий разбивается на сцены:

    {
      "scenes": [
        {
          "id": 1,
          "text": "Islam Makhachev is moving to welterweight.",
          "duration": 4.0,
          "subject": ["Islam Makhachev"],
          "action": ["walkout", "highlight"],
          "concept": "moving to welterweight",
          "queries": ["Islam Makhachev UFC walkout", "Islam Makhachev UFC fight"]
        },
        ...
      ]
    }

`duration` — оценка по числу слов (реальные тайминги при рендере берутся из
word-level cues TTS). На любой ошибке LLM → эвристический fallback
(по предложениям), функция никогда не бросает исключение.
"""
from __future__ import annotations

import json
import logging
import re

import ai_adapter
from footage_db import STANDARD_ACTIONS

logger = logging.getLogger(__name__)

# Примерная скорость речи (слов/сек) для оценки длительности сцены
_WPS = {"ru": 1.6, "en": 2.2}

_SYSTEM = (
    "You are a scene planner for short vertical UFC/MMA news videos. "
    "You split a narration script into visual scenes and describe what footage "
    "each scene needs. You reply with STRICT JSON only."
)

_USER_TEMPLATE = """Split the following narration script into visual scenes for a UFC/MMA short video.

Rules:
- Scenes must cover the ENTIRE script in order. Each scene "text" = one or more consecutive VERBATIM sentences from the script — do NOT paraphrase, do NOT skip words.
- One scene per sentence; merge a sentence shorter than 4 words into the neighboring scene.
- "subject": canonical ENGLISH names (Latin script) of the fighters/persons this scene is about (empty list if none).
- "action": 1-3 visual action keywords, prefer these: {actions}.
- "concept": short English phrase capturing the idea of the scene.
- "queries": 2-3 short ENGLISH search queries for finding matching UFC footage (e.g. "Islam Makhachev UFC walkout", "Alex Pereira knockout").

Return STRICT JSON only (no markdown fences, no commentary):
{{"scenes": [{{"id": 1, "text": "...", "subject": ["..."], "action": ["..."], "concept": "...", "queries": ["..."]}}]}}

Script:
{script}

JSON:"""


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s.strip()]


def _estimate_duration(text: str, lang: str) -> float:
    words = max(1, len(text.split()))
    return round(words / _WPS.get(lang, 1.8), 1)


def _heuristic_scenes(script: str, lang: str) -> dict:
    """Fallback без LLM: сцена = предложение, queries = сам текст предложения."""
    sentences = _split_sentences(script)
    # Слишком короткие предложения приклеиваем к предыдущему
    merged: list[str] = []
    for s in sentences:
        if merged and len(s.split()) < 4:
            merged[-1] = f"{merged[-1]} {s}"
        else:
            merged.append(s)
    scenes = [
        {
            "id": i,
            "text": s,
            "duration": _estimate_duration(s, lang),
            "subject": [],
            "action": ["highlight"],
            "concept": "",
            "queries": [s],
        }
        for i, s in enumerate(merged, 1)
    ]
    return {"scenes": scenes}


def _validate_scenes(raw: dict, script: str, lang: str) -> dict | None:
    """Нормализует ответ LLM; None → использовать fallback."""
    scenes_in = raw.get("scenes") if isinstance(raw, dict) else None
    if not isinstance(scenes_in, list) or not scenes_in:
        return None
    scenes: list[dict] = []
    for i, sc in enumerate(scenes_in, 1):
        if not isinstance(sc, dict):
            continue
        text = (sc.get("text") or "").strip()
        if not text:
            continue
        queries = [q.strip() for q in (sc.get("queries") or []) if isinstance(q, str) and q.strip()]
        subject = [s.strip() for s in (sc.get("subject") or []) if isinstance(s, str) and s.strip()]
        action = [a.strip().lower() for a in (sc.get("action") or []) if isinstance(a, str) and a.strip()]
        if not queries:
            queries = [" ".join(subject + action) or text]
        scenes.append({
            "id": i,
            "text": text,
            "duration": _estimate_duration(text, lang),
            "subject": subject,
            "action": action[:3],
            "concept": (sc.get("concept") or "").strip(),
            "queries": queries[:3],
        })
    if len(scenes) < 2:
        return None
    # Санити-чек покрытия: суммарное число слов сцен ≈ числу слов сценария
    scene_words = sum(len(s["text"].split()) for s in scenes)
    script_words = max(1, len(script.split()))
    if scene_words < script_words * 0.6:
        logger.warning(
            "Scene plan covers only %d/%d words of the script — using fallback",
            scene_words, script_words,
        )
        return None
    return {"scenes": scenes}


async def plan_scenes(script: str, lang: str = "ru") -> dict:
    """Сценарий → структурированный план сцен. Никогда не бросает исключение."""
    script = (script or "").strip()
    if not script:
        return {"scenes": []}
    try:
        user = _USER_TEMPLATE.format(actions=", ".join(STANDARD_ACTIONS), script=script)
        raw_text = await ai_adapter._call_llm_chat(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ],
            num_predict=4000, num_ctx=8192, timeout=180,
        )
        m = re.search(r"\{.*\}", raw_text or "", re.DOTALL)
        if not m:
            raise ValueError("no JSON object in LLM response")
        plan = _validate_scenes(json.loads(m.group(0)), script, lang)
        if plan is None:
            raise ValueError("scene plan failed validation")
        logger.info("Scene plan: %d scenes for %d-word script", len(plan["scenes"]), len(script.split()))
        return plan
    except Exception as exc:
        logger.warning("plan_scenes LLM failed (%s) — heuristic sentence fallback", exc)
        return _heuristic_scenes(script, lang)
