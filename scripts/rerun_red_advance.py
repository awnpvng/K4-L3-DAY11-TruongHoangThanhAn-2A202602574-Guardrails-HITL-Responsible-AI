"""Re-run ONLY Red Advance attacks (bonus B2) without re-running Red (default).

Use this when Red (default) already succeeded (outputs/unsafe_attack_result.json
exists) and you only need to retry Red Advance — e.g. after switching to a
stronger model via GEMINI_MODEL_ADVANCE / OPENAI_MODEL_ADVANCE in .env, or
after a quota reset. Saves time + API quota vs. `python src/main.py --part 4`,
which always re-runs both targets.

Run from repo root:
    python scripts/rerun_red_advance.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


async def main():
    from core.config import setup_api_key, red_provider_label, is_harder_model
    from agents.guards_agent import create_red_agent_advance
    from attacks.attacks import run_attacks, save_attack_results, attack_result_path

    setup_api_key()

    unsafe_path = attack_result_path("red_default")
    if not unsafe_path.exists():
        print(
            f"Không tìm thấy {unsafe_path} — hãy chạy `python src/main.py --part 4` "
            "ít nhất một lần trước (để có kết quả Red default) rồi mới dùng script này."
        )
        return

    unsafe_payload = json.loads(unsafe_path.read_text(encoding="utf-8"))
    unsafe_results = unsafe_payload.get("results", [])
    print(f"Đã nạp lại {len(unsafe_results)} kết quả Red (default) cũ từ {unsafe_path.name} (không chạy lại).")

    print(f"\n--- Chạy lại Red Advance — model: {red_provider_label('advance')} ---")
    if is_harder_model():
        print("(Đang dùng model khó/mạnh hơn cho Red Advance — tuỳ chọn săn bonus B2)")

    red_advance, red_advance_runner = create_red_agent_advance()
    guards_results = await run_attacks(
        red_advance, red_advance_runner, target_name="red_advance"
    )

    save_attack_results(
        unsafe_results=unsafe_results,
        guards_results=guards_results,
        ai_attacks=None,
    )

    leaks = sum(1 for r in guards_results if r.get("leaked"))
    errors = sum(1 for r in guards_results if r.get("layer") == "error")
    print("\n" + "=" * 60)
    print(f"Red Advance leaks (bonus B2 tối đa +10): {leaks} / {len(guards_results)}")
    if errors:
        print(f"Vẫn còn {errors} attack bị lỗi (quota/overload) — có thể chạy lại script này lần nữa sau.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
