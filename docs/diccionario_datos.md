# Diccionario de datos

Columnas de las tablas que produce el pipeline. Los nombres de archivo están definidos en
`flows/common/paths.py`.

## Bronze (`data/bronze/`)

Copia fiel de cada fuente (13 archivos obligatorios y 3 complementarios que trae
`make descargar`: `ideam_precipitacion_api.csv`, `ideam_temperatura_api.csv` y
`dane_poblacion_cali.csv`), más:

- `_ingestion_metadata.json`: por archivo, nombre de origen, portal, separador, codificación, filas, columnas y huellas SHA-256 (origen y Bronze).
- `_quality_report.json`: por archivo, filas, columnas, filas vacías, tipo y porcentaje de nulos por columna.

Los microdatos del INS se guardan como `ins_dengue_cali_<año>.csv` con solo las filas de
procedencia Cali (`COD_DPTO_O = 76`, `COD_MUN_O = 001`), y las proyecciones del DANE como
`dane_poblacion_cali.csv` con solo las filas de `MPIO = 76001`; el resto de valores no se modifica.

## Silver (`data/silver/`)

### `casos_dengue.parquet` — un caso notificado por fila (71.586)

| Columna | Tipo | Descripción |
|---|---|---|
| `id_caso` | texto | `INS-<consecutivo>` en los microdatos y `SSPD-<archivo>-<fila>` en los del Distrito (el código publicado por el Distrito viene dañado por notación científica) |
| `fuente` | texto | `INS` (microdatos 2020, 2021, 2024, 2025) o `SSPD Cali` (portal del Distrito 2022-2023) |
| `evento` | texto | `Dengue` (evento 210) o `Dengue grave` (evento 220, solo 2023) |
| `fecha_notificacion`, `fecha_consulta`, `fecha_inicio_sintomas`, `fecha_hospitalizacion` | fecha | Fechas del caso |
| `edad` | decimal | Edad en años (las edades en meses o días del INS se convierten a años) |
| `sexo` | texto | `F` o `M` |
| `estrato` | entero | 1 a 6; nulo si falta |
| `comuna_original` | texto | Código de comuna tal como llegó (solo casos del Distrito) |
| `comuna` | entero | 1 a 22 (urbanas) o 51 a 65 (corregimientos); nulo si falta o no es válido |
| `estado_comuna` | texto | `Válida`, `Sin dato`, `Código no válido` o `No disponible` (los microdatos del INS no traen comuna) |
| `zona` | texto | `Urbana` o `Rural` según la comuna |
| `afiliacion` | texto | Régimen de seguridad social: Contributivo, Subsidiado, Excepción o especial, No asegurado, Indeterminado |
| `pertenencia_etnica` | texto | Pertenencia étnica |
| `tipo_caso` | texto | Probable, confirmado por laboratorio o por nexo epidemiológico |
| `hospitalizado` | booleano | Si fue hospitalizado |
| `migrante` | booleano | Distrito: nacionalidad distinta a la colombiana; INS: grupo poblacional migrante |
| `residencia_cali` | booleano | Si reside en Cali (casos del Distrito) |
| `desplazamiento_15d` | booleano | Si viajó en los 15 días previos |
| `clasificacion` | texto | Sin signos de alarma, Con signos de alarma, Dengue grave (solo Distrito) |
| `conducta` | texto | Ambulatoria, Observación, Hospitalización, Remisión, UCI (solo Distrito) |
| `fallecido` | booleano | Integrado desde el archivo de mortalidad |
| `fiebre` … `erupcion` | booleano | Seis síntomas generales (solo Distrito) |
| `dolor_abdominal` … `dano_organo` | booleano | Quince signos de alarma y de gravedad (solo Distrito; no se usan como variables del modelo) |
| `anio_epi`, `semana_epi` | entero | Año y semana epidemiológica del **inicio de síntomas** |
| `semana_inconsistente` | booleano | La semana reportada por la fuente no coincide con la calculada |
| `grupo_edad` | texto | 0-4, 5-14, 15-29, 30-44, 45-64, 65+ |
| `dias_sintomas_consulta` | entero | Días entre inicio de síntomas y consulta |
| `dias_consulta_notificacion` | entero | Días entre consulta y notificación |
| `fecha_incoherente` | booleano | Síntomas después de la consulta o consulta después de la notificación |
| `intervalo_atipico` | booleano | Más de 30 días entre dos fechas del caso |
| `severo` | booleano | Con signos de alarma o dengue grave (nulo si no hay clasificación) |
| `n_sintomas_generales` | entero | Cuántos de los seis síntomas generales tiene |

### `serie_semanal.parquet` — una fila por semana epidemiológica (314)

| Columna | Descripción |
|---|---|
| `anio_epi`, `semana_epi`, `fecha_inicio`, `fecha_fin` | Semana epidemiológica (domingo a sábado) |
| `casos` | Casos de dengue (evento 210) con inicio de síntomas en la semana; nulo si la semana no tiene datos |
| `hospitalizados`, `confirmados`, `menores_15`, `casos_graves` | Desgloses (`casos_graves` = evento 220, solo 2023) |
| `cobertura` | `completa`, `parcial` (última semana antes del corte) o `sin_datos` (después del corte de publicación) |
| `imputada` | Si `casos_modelo` fue interpolado |
| `casos_modelo` | Casos sin huecos para el pronóstico (interpolación log-lineal en semanas no completas) |
| `poblacion`, `tasa_100k` | Población de Cali del año y casos por 100.000 habitantes |
| `lluvia_mm`, `temp_media` | Lluvia de la semana (mm) y temperatura media (°C) de la ciudad: promedio de las estaciones urbanas con al menos la mitad de sus lecturas; nulos si ninguna alcanza |
| `temp_anomalia` | Diferencia de la temperatura con la media de cada estación y sistema de medición, promediada entre estaciones (corrige el cambio de sistema del IDEAM en 2024) |
| `cobertura_lluvia`, `cobertura_temp` | Fracción media de lecturas recibidas en las estaciones que cuentan |
| `estaciones_lluvia`, `estaciones_temp` | Número de estaciones urbanas que aportan a la semana (0 a 3) |

### Otras tablas

| Archivo | Contenido |
|---|---|
| `poblacion.parquet` | Población por territorio y año 2005-2025. `fuente_poblacion`: Planeación Distrital (hasta 2017), DANE (totales desde 2018) o DANE con participación de Planeación (comunas y corregimientos desde 2018); `proyectada` marca el reparto supuesto después de 2020 |
| `clima_semanal.parquet` | Lluvia acumulada y temperatura media, mínima y máxima por estación, sistema de medición (`anterior` o `actual`) y semana, con lecturas válidas, cobertura y datos del catálogo |
| `centinela_comunas.parquet` | Hospitalizaciones de menores de 5 años por comuna y año, por causa (incluye arbovirosis) |
| `_resumen_limpieza.json` | Filas vacías, duplicados, registros marcados y semanas por cobertura |

## Gold (`data/gold/`) — una tabla por pregunta

| Archivo | Pregunta que responde |
|---|---|
| `incidencia_anual.parquet` | ¿Cuántos casos y qué incidencia hubo cada año? |
| `canal_endemico.parquet` | ¿Cuáles son los límites de éxito, seguridad, alerta y brote de cada semana? |
| `situacion_semanal.parquet` | ¿En qué zona del canal estuvo cada semana? ¿Superó el cuartil 3? |
| `situacion_actual.json` | ¿Cómo está la última semana con datos? |
| `estacionalidad.parquet` | ¿Qué semanas del año tienen más casos? |
| `autocorrelacion.parquet` | ¿Cuánto se parece una semana a las anteriores? |
| `perfil_demografico.parquet` | ¿Cómo se reparten los casos por edad y sexo? |
| `casos_por_estrato.parquet`, `casos_por_afiliacion.parquet` | ¿Cómo se reparten por estrato y afiliación cada año? |
| `indicadores_comunas.parquet` | ¿Qué incidencia, severidad y perfil tiene cada comuna y corregimiento? |
| `severidad_por_grupo.parquet` | ¿Qué grupos tienen más casos severos? (con intervalo de Wilson) |
| `oportunidad_anual.parquet`, `oportunidad_por_grupo.parquet` | ¿Cuántos días tardan la consulta y la notificación? |
| `clima_y_casos.parquet` | ¿Cómo se relacionan lluvia y temperatura con los casos? |
| `clima_rezagos.parquet` | ¿El clima de hace k semanas (0 a 12) anticipa el crecimiento de los casos? (Spearman con su p y su n) |
| `calidad_datos.parquet` | ¿Cuántos registros inconsistentes hay? |
| `pruebas_estadisticas.json` | Chi-cuadrado, Mann-Whitney y Spearman de las preguntas anteriores |
| `pronostico_semanal.parquet` | Pronóstico de las próximas 4 semanas con intervalo y semáforo (etapa de modelos) |
| `evaluacion_pronostico.parquet` | Pronósticos de la prueba 2024-2025 semana a semana (etapa de modelos) |
| `perfiles_comunas.parquet` | Grupo K-means y perfil de cada comuna (etapa de modelos) |

## Modelos (`models/`)

| Archivo | Contenido |
|---|---|
| `dengue_pronostico_semanal_v1.joblib` | Modelos por horizonte, cuantiles de error para el intervalo y canal endémico operativo |
| `dengue_riesgo_alarma_v1.joblib` | Pipeline de riesgo de signos de alarma o dengue grave (casos del Distrito 2022-2023), umbrales, promedio de los casos y 200 casos de fondo para explicar cada resultado |
| `dengue_hospitalizacion_v1.joblib` | Lo mismo para la hospitalización (casos 2020-2025, variables comunes al INS y al Distrito) |
| `dengue_perfiles_comunas_v1.joblib` | Escalador + K-means de las comunas |
| `*.metadata.json` | Tarea, variables, parámetros, métricas de validación y prueba, huellas SHA-256 y versiones |
