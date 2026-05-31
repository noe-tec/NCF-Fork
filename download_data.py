from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError

DATA_DIR = Path("./")
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "hexiangnan/neural_collaborative_filtering/master/Data"
)

FILES = [
    "ml-1m.test.negative",
    "ml-1m.test.rating",
    "ml-1m.train.rating",
    "pinterest-20.test.negative",
    "pinterest-20.test.rating",
    "pinterest-20.train.rating",
]

for filename in FILES:
    url = f"{BASE_URL}/{filename}"
    destination = DATA_DIR / filename

    print(f"Descargando {filename}...")

    try:
        urlretrieve(url, destination)
        print(f"Guardado en {destination}")
    except HTTPError as e:
        print(f"Error HTTP al descargar {filename}: {e.code}")
    except URLError as e:
        print(f"Error de conexión al descargar {filename}: {e.reason}")

print("Proceso terminado.")