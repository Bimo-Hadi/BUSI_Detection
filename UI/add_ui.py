"""Menambahkan halaman web ke app.py hasil notebook (aman dijalankan berulang).
Pakai:  python add_ui.py      (jalankan di folder beeonic_api yang berisi app.py)"""
from pathlib import Path

app = Path("app.py")
code = app.read_text(encoding="utf-8")
MARK = "# --- Beeonic web UI ---"
if MARK in code:
    print("UI sudah terpasang.")
else:
    code += f'''

{MARK}
from fastapi.responses import FileResponse


@app.get("/", include_in_schema=False)
def web_ui():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
'''
    app.write_text(code, encoding="utf-8")
    print("UI terpasang di app.py")
assert Path("static/index.html").exists(), "Folder static/index.html belum ada di sini"
