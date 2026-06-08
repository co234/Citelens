"""Small API and dependency check for real-key runs."""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-openai", action="store_true")
    parser.add_argument("--skip-anthropic", action="store_true")
    args = parser.parse_args()

    ok = True
    if not args.skip_anthropic:
        try:
            from anthropic import Anthropic

            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            Anthropic().messages.create(
                model=os.environ.get("CITATION_CLEANER_STAGE0_MODEL", "claude-haiku-4-5-20251001"),
                max_tokens=8,
                messages=[{"role": "user", "content": "Return OK."}],
            )
            print("anthropic: OK")
        except Exception as exc:
            ok = False
            print(f"anthropic: FAIL ({type(exc).__name__}: {exc})")

    if not args.skip_openai:
        try:
            from openai import OpenAI

            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY not set")
            OpenAI().embeddings.create(model="text-embedding-3-small", input=["hello"])
            print("openai: OK")
        except Exception as exc:
            ok = False
            print(f"openai: FAIL ({type(exc).__name__}: {exc})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
