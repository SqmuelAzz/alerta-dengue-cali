# Instalación y ejecución

Guía para instalar el proyecto desde cero y regenerar todos los resultados.

## 1. Requisitos

- Python 3.11 o superior (probado con 3.13). En Ubuntu 22.04 (WSL), que trae Python 3.10, se
  puede instalar una versión más reciente con `sudo apt install python3.12 python3.12-venv`
  (repositorio `ppa:deadsnakes/ppa`) y crear el entorno con `python3.12 -m venv .venv`.
- `make` (en Windows se recomienda WSL; sin `make`, ver la sección 6).
- Unos 2 GB libres: los datos de origen pesan 374 MB y los microdatos del INS se leen completos.

## 2. Clonar e instalar

```bash
git clone <url-del-repositorio> dengue_cali
cd dengue_cali
python3 -m venv .venv
source .venv/bin/activate          # en Windows (PowerShell): .venv\Scripts\Activate.ps1
make install                       # pip install -e ".[dev,app]"
cp .env.example .env               # opcional: solo si se quiere cambiar algún parámetro
```

Para guardar las versiones exactas con las que se obtuvieron los resultados:

```bash
pip freeze > requirements.lock.txt
```

## 3. Datos de origen

Los datos **no** están en el repositorio (pesan más de 350 MB). Se descargan de los portales
abiertos y se ponen en la carpeta `Dataset_Dengue/` en la raíz del proyecto (o en otra ruta
indicada con `DENGUE_SOURCE_DIR` en el `.env`). Si se tiene el archivo `Dataset_Dengue.zip`:

```bash
unzip Dataset_Dengue.zip -d .      # crea ./Dataset_Dengue con los 13 archivos
```

| Archivo | Fuente | Contenido |
|---|---|---|
| `bd_dengue.csv` | [Datos Abiertos Cali – Dengue](https://datos.cali.gov.co/dataset/casos-notificados-en-sivigila-para-dengue-en-el-distrito-de-santiago-de-cali) | Casos de dengue 2022 |
| `53.-bd_casos-notificados-en-sivigila-dengue-corte-6-dic.csv` | Mismo portal | Casos de dengue 2023 |
| `54.-bd_casos-notificados-en-sivigila-para-dengue_grave-corte-6-dic.csv` | [Datos Abiertos Cali – Dengue grave](https://datos.cali.gov.co/dataset/casos-notificados-en-sivigila-para-dengue-grave-en-el-distrito-de-santiago-de-cali) | Dengue grave 2023 |
| `55.-bd_casos-notificados-en-sivigila-mortalidad_dengue-corte-6-dic.csv` | [Datos Abiertos Cali – Mortalidad](https://datos.cali.gov.co/dataset/casos-notificados-en-sivigila-para-mortalidad-por-dengue-en-el-distrito-de-santiago-de-cali) | Muertes por dengue 2023 |
| `Datos_2020_210.xlsx`, `Datos_2021_210.xlsx`, `Datos_2024_210.xlsx`, `Datos_2025_210.xlsx` | [Portal SIVIGILA – INS, buscador de microdatos](https://portalsivigila.ins.gov.co/Paginas/Buscador.aspx) (evento 210) | Casos de dengue de todo el país |
| `proyecciones_de_poblaci_n_de_cali_por_comuna_y_corregimiento__2006-2020.csv` | [Datos Abiertos Cali – Planeación](https://datos.cali.gov.co/dataset/proyecciones-de-poblacion-de-cali-por-comuna-y-corregimiento-2006-2020) | Población por comuna |
| `14.-unificado-centinela-gsa-sspm-2010-2020.csv` | [Datos Abiertos Cali – Centinela](https://datos.cali.gov.co/dataset/centinela-de-hospitalizaciones-menores-hasta-5-anos-grupo-salud-ambiental) | Hospitalizaciones de menores de 5 años |
| `s54a-sgyg.csv` | [IDEAM – Precipitación](https://www.datos.gov.co/d/s54a-sgyg) | Lluvia, estaciones de Cali |
| `sbwg-7ju4.csv` | [IDEAM – Temperatura del aire](https://www.datos.gov.co/d/sbwg-7ju4) | Temperatura, estaciones de Cali |
| `Cat_logo_Nacional_de_Estaciones_del_IDEAM.csv` | [IDEAM – Catálogo de estaciones](https://www.datos.gov.co/d/hp9r-jxuu) | Altitud, estado y ubicación |

**Fuentes complementarias.** Tres archivos más se descargan de las API de datos abiertos con:

```bash
make descargar     # necesita internet; unos 5 minutos
```

| Archivo | Fuente | Contenido |
|---|---|---|
| `ideam_precipitacion_cali_2020-2025.csv` | [API IDEAM – Precipitación](https://www.datos.gov.co/d/s54a-sgyg) | Lluvia de 4 estaciones de Cali, 2020-2025 |
| `ideam_temperatura_cali_2020-2025.csv` | [API IDEAM – Temperatura](https://www.datos.gov.co/d/sbwg-7ju4) | Temperatura de las mismas estaciones |
| `PPED-AreaMun-2018-2042_VP.xlsx` | [DANE – Proyecciones de población municipal](https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion) | Población oficial por municipio y área, 2018-2042 |

Si no están, el pipeline funciona igual con las demás fuentes: usa la tendencia lineal de
Planeación para la población y solo el clima de la descarga manual.

La ingesta busca cada archivo por un patrón de nombre (`flows/00_ingestion.py`, lista `SOURCES`),
así que se pueden usar descargas más recientes siempre que conserven el nombre y las columnas.

## 4. Ejecutar

```bash
make pipeline      # las cuatro etapas (unos 2 minutos; leer los Excel del INS es lo más lento)
make test          # 136 pruebas con datos inventados (no necesitan los datos reales)
make lint          # revisión de estilo con ruff
```

Etapas por separado: `make ingest`, `make silver`, `make gold`, `make train`. La ingesta no vuelve
a copiar un archivo si su huella SHA-256 no cambió.

Cuadernos (después de `make pipeline`):

```bash
jupyter lab notebooks/
```

Antes de hacer commit de un cuaderno, limpiar sus salidas: `nbstripout notebooks/*.ipynb`
(o `nbstripout --install` una vez para que Git lo haga solo).

## 5. API y aplicación web

Necesitan los modelos (`make pipeline` o `make train`).

```bash
make api           # http://127.0.0.1:5000/salud
make app           # http://localhost:8501 (Streamlit)
```

El diseño se adapta al tamaño de la pantalla. Para ver cómo se ve en un celular sin salir del
computador, en el navegador se presiona `F12` y luego `Ctrl + Shift + M` (vista de dispositivo
móvil). El tema de colores está en `.streamlit/config.toml`.

Ejemplos con la API:

```bash
curl http://127.0.0.1:5000/pronostico
curl -X POST http://127.0.0.1:5000/pronostico -H "Content-Type: application/json" \
     -d '{"casos_recientes": [45, 52]}'
curl -X POST http://127.0.0.1:5000/riesgo -H "Content-Type: application/json" \
     -d '{"edad": 70, "sexo": "F", "dias_sintomas_consulta": 5, "estrato": 1, "comuna": 6,
          "afiliacion": "Subsidiado", "fiebre": true}'
curl http://127.0.0.1:5000/comunas/6
curl "http://127.0.0.1:5000/clima?anio=2024"
```

## 6. Sin `make`

| `make` | Comando equivalente |
|---|---|
| `make install` | `pip install -e ".[dev,app]"` |
| `make descargar` | `python -m flows.descarga` |
| `make pipeline` | `python -m flows.job --stage all` |
| `make train` | `python -m flows.job --stage train` |
| `make test` | `pytest -q` |
| `make lint` | `ruff check .` |
| `make api` | `python -m app.api` |
| `make app` | `streamlit run app/dashboard.py` |
| `make clean` | borrar `data/`, `models/`, `.pytest_cache/`, `.ruff_cache/` y `job.log` |

## 7. Problemas frecuentes

- **`FileNotFoundError` en la ingesta:** falta algún archivo en `Dataset_Dengue/` o la variable `DENGUE_SOURCE_DIR` apunta a otra carpeta.
- **La API o la aplicación dicen que faltan archivos:** ejecutar primero `make pipeline`.
- **Aviso de versión de scikit-learn al cargar un modelo:** el modelo se entrenó con otra versión; volver a ejecutar `make train`.
