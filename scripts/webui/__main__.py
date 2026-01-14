from __future__ import annotations

import os


def main() -> int:
    try:
        import uvicorn
    except Exception as e:
        raise SystemExit(
            "uvicorn is required to run the web UI. Install requirements and retry.\n"
            f"Import error: {e}"
        )

    host = os.environ.get("VSYNC_WEBUI_HOST", "127.0.0.1")
    port = int(os.environ.get("VSYNC_WEBUI_PORT", "8000"))
    uvicorn.run("scripts.webui.app:app", host=host, port=port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
