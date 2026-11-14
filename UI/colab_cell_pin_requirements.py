# Tempel sebagai sel TERAKHIR di notebook (setelah sel zip), jalankan sekali.
# Menulis requirements.txt dengan versi PERSIS dari Colab, supaya API di server
# memakai TensorFlow/Keras yang sama dengan saat training (reproducible).
from importlib.metadata import version
pkgs = {'tensorflow-cpu': 'tensorflow', 'keras': 'keras', 'fastapi': 'fastapi', 'uvicorn[standard]': 'uvicorn',
        'python-multipart': 'python-multipart', 'pillow': 'pillow', 'numpy': 'numpy'}
lines = [f"{name}=={version(dist)}" for name, dist in pkgs.items()]
Path(PKG_DIR, 'requirements.txt').write_text('\n'.join(lines) + '\n')
shutil.make_archive('beeonic_api', 'zip', PKG_DIR)          # zip ulang, sekarang dengan versi terkunci
print('\n'.join(lines))
try:
    from google.colab import files; files.download('beeonic_api.zip')
except Exception:
    pass
