# Alerta Dengue Cali

Producto de datos para la estimación semanal de casos de dengue y la predicción de brotes y de
casos con signos de alarma en Santiago de Cali, a partir de datos epidemiológicos, poblacionales
y climáticos de 2020 a 2025. Está pensado para la Secretaría de Salud Pública Distrital (grupo de
vigilancia epidemiológica y programa de enfermedades transmitidas por vectores).

**Autores:** Samuel Acevedo Mambuscay (2216646) y Luis Carlos Tez (2216702)
Ingeniería de Sistemas — Universidad Libre, seccional Cali · Inteligencia Artificial y Ciencia de Datos

## Qué responde

| Pregunta | Resultado |
|---|---|
| **Estimar:** ¿cuántos casos e incidencia hubo por semana, año y comuna? ¿En qué zona del canal endémico está la ciudad? | Serie de 314 semanas, incidencia anual y por comuna, canal endémico por cuartiles |
| **Predecir:** ¿cuántos casos habrá en 1 a 4 semanas? ¿Habrá brote? | Pronóstico con intervalo y semáforo; en la prueba 2024-2025, MAE de 33 casos a 1 semana y 72 a 4, alerta con sensibilidad de 96-98 % |
| **Predecir:** ¿este caso tendrá signos de alarma? ¿Será hospitalizado? | Random Forest con datos de la primera consulta; AUC 0,62 (casos del Distrito 2022-2023) y 0,65 (71.586 casos 2020-2025, probado en 2024-2025), con la explicación de qué pesa en cada resultado |
| **Descubrir:** ¿cómo se agrupan las comunas? ¿Quién se agrava? ¿Influye el clima? | Tres perfiles de comunas (K-means), estacionalidad, severidad por grupo, oportunidad de consulta y clima de semanas anteriores frente al crecimiento de casos |
| **Anomalías:** ¿semanas inusuales, registros inconsistentes, lecturas climáticas erróneas? | Semanas sobre el cuartil 3 del canal y tabla de calidad de datos |

## Cómo ejecutarlo

```bash
make install       # instala el proyecto y sus herramientas
make descargar     # series del IDEAM y población del DANE desde las API (necesita internet)
make pipeline      # fuente -> Bronze -> Silver -> Gold -> modelos
make test          # pruebas automáticas
make api           # API REST en http://127.0.0.1:5000
make app           # aplicación web "Alerta Dengue Cali" (Streamlit)
```

Antes de `make pipeline` hay que poner los archivos de datos en `Dataset_Dengue/` (no se suben a
Git). Los pasos completos, la lista de archivos y cómo ejecutar sin `make` están en
[`docs/INSTALACION.md`](docs/INSTALACION.md).

## Aplicación web

`make app` abre **Alerta Dengue Cali** en el navegador (`http://localhost:8501`):

| Pestaña | Qué muestra |
|---|---|
| 📊 Situación | Zona del canal endémico de la última semana, resumen de cada año (casos, incidencia, semanas en brote y semana pico), el canal endémico año por año y la lluvia y temperatura del IDEAM junto a los casos, con el cambio de sensores de 2024 y la prueba de si el clima de semanas anteriores anticipa los casos |
| 📈 Pronóstico | Casos esperados en las próximas 4 semanas con intervalo, semáforo y alerta de brote; permite agregar los casos de semanas recientes y muestra el desempeño del modelo |
| 🏘️ Comunas | Los tres perfiles de comunas, la incidencia de cada una y la ficha de la comuna elegida |
| 🩺 Riesgo de un caso | Probabilidad de signos de alarma o dengue grave y de hospitalización con los datos de la primera consulta, comparada con el promedio de los casos, y qué características la suben o la bajan |

Cada pestaña permite descargar sus tablas en CSV, y el diseño se adapta al ancho de la pantalla
(computador, tableta o celular).

## Estructura

```
dengue_cali/
├── flows/                       # Pipeline (código reutilizable)
│   ├── common/                  # Rutas, configuración, logging, huellas y herramientas epidemiológicas
│   ├── job.py                   # Orquestador: python -m flows.job --stage all
│   ├── descarga.py              # API del IDEAM y del DANE -> carpeta de fuentes (make descargar)
│   ├── 00_ingestion.py          # Fuentes -> Bronze (copia fiel + SHA-256 + reporte de calidad)
│   ├── 01_data_engineering.py   # Bronze -> Silver (casos, serie semanal, población, clima, centinela)
│   ├── 03_aggregation.py        # Silver -> Gold (incidencia, canal endémico, perfiles, pruebas)
│   └── 05_ml_models.py          # Silver/Gold -> modelos con sus metadatos
├── app/                         # API (Flask) y aplicación web (Streamlit)
├── notebooks/                   # 02_exploracion.ipynb y 04_modelado.ipynb
├── tests/                       # Una prueba por módulo (datos inventados)
├── reports/                     # Conclusiones, problema de ML, ficha de los modelos y figuras
├── docs/                        # Instalación, diccionario de datos y decisiones de diseño
├── data/  models/               # Se generan con make pipeline (no se versionan)
└── pyproject.toml  Makefile  .env.example
```

## Documentos

- [`reports/conclusiones_eda.md`](reports/conclusiones_eda.md): hallazgos del análisis exploratorio y lo que no se puede afirmar.
- [`reports/problema_ml.md`](reports/problema_ml.md): definición de los problemas, métricas, validación y reflexión ética.
- [`reports/model_card.md`](reports/model_card.md): ficha de los cuatro modelos, con métricas por subgrupo y limitaciones.
- [`docs/diccionario_datos.md`](docs/diccionario_datos.md): columnas de cada capa.
- [`docs/decisiones.md`](docs/decisiones.md): decisiones de diseño que cambian los resultados.

## Fuentes de datos

- Alcaldía de Santiago de Cali, Secretaría de Salud Pública (2023). *Casos notificados en SIVIGILA para dengue, dengue grave y mortalidad por dengue en el Distrito de Santiago de Cali* [conjuntos de datos]. Datos Abiertos Cali. https://datos.cali.gov.co
- Instituto Nacional de Salud (2026). *Portal SIVIGILA, microdatos del evento 210 Dengue, años 2020, 2021, 2024 y 2025*. https://portalsivigila.ins.gov.co/Paginas/Buscador.aspx
- Departamento Administrativo de Planeación de Cali (2018). *Proyecciones de población de Cali por comuna y corregimiento 2006-2020*. Datos Abiertos Cali.
- DANE (2025). *Proyecciones de población municipal por área 2018-2042* (actualización posterior a la pandemia). https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion
- Secretaría de Salud Pública de Cali, Grupo de Salud Ambiental. *Vigilancia centinela de hospitalizaciones en menores de 5 años, 2010-2020*. Datos Abiertos Cali.
- IDEAM (2026). *Precipitación*, *Temperatura ambiente del aire* y *Catálogo nacional de estaciones*. Datos Abiertos Colombia, consultados por la API. https://www.datos.gov.co
- Bortman, M. (1999). Elaboración de corredores o canales endémicos mediante planillas de cálculo. *Revista Panamericana de Salud Pública, 5*(1), 1-8.

Se usan solo datos anonimizados y publicados en portales abiertos (Ley 1581 de 2012). Los
modelos apoyan la vigilancia y la priorización; no reemplazan el criterio clínico ni las cifras
oficiales de los boletines epidemiológicos.
