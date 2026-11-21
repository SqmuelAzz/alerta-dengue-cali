# Decisiones de diseño

Registro de las decisiones que cambian los resultados, con su justificación. Cuando una decisión
cambia, se actualiza aquí, en `.env.example` o en las constantes del módulo correspondiente.

## 1. Estructura del proyecto

Se sigue la plantilla del curso: arquitectura medallion y `flows/` por etapas.

| Capa | Carpeta | Contenido |
|---|---|---|
| raw | `Dataset_Dengue/` (fuera de Git) | Archivos tal como se descargan |
| bronze | `data/bronze/` | Copia fiel con huella SHA-256; de los microdatos del INS, solo filas de Cali |
| silver | `data/silver/` | Datos limpios, tipados y con variables derivadas |
| gold | `data/gold/` y `models/` | Tablas para el informe, el tablero y los modelos |

Además de las cuatro etapas, `flows/common/epidemiologia.py` reúne las herramientas que usan
varias etapas (calendario epidemiológico, canal endémico, tasas, grupos de edad) y `app/` contiene
la API y la aplicación web, que solo leen las capas y los modelos.

## 2. Periodo 2020-2025

Se trabaja con los seis años para los que hay microdatos o casos individuales: 2020, 2021, 2024 y
2025 del INS y 2022-2023 del Distrito, en total 314 semanas epidemiológicas. Extender la serie
hacia atrás (2017-2019) requiere los conjuntos de datos agregados del SIVIGILA de esos años y
queda como trabajo futuro.

## 3. Semana epidemiológica de inicio de síntomas

Todos los casos se ubican en la semana epidemiológica (domingo a sábado; la semana 1 contiene el
4 de enero) de su **fecha de inicio de síntomas**. Es el criterio de la variable `SEMANA` de los
microdatos del INS: en 57.586 de sus 57.618 casos coincide con la semana calculada desde
`INI_SIN`. En los 32 restantes se respeta el dato oficial y quedan marcados como
`semana_inconsistente`. Los casos del Distrito, que no traen la semana, se ubicaron con el mismo
criterio para que la serie sea comparable. Las cifras
pueden diferir de los boletines que cuentan por semana de notificación.

## 4. Solo procedencia Cali y solo evento 210 en la serie

- De los microdatos nacionales se toman los casos con procedencia Cali (`COD_DPTO_O = 76`, `COD_MUN_O = 001`), que es el criterio del proyecto (57.618 casos).
- Los registros con ajuste 6 o D (descartados o error de digitación) se eliminan; en los archivos entregados no hay ninguno.
- La serie cuenta solo el evento 210. Los 106 casos de dengue grave de 2023 (evento 220) quedan en Silver y en `casos_graves`, pero no en `casos`, porque el dengue grave solo está publicado para 2023 y sumarlo haría los años no comparables.

## 5. Semanas sin datos

Los archivos del Distrito se publicaron con corte el 6 de diciembre. Las semanas posteriores a la
última notificación se marcan como `sin_datos` y la semana anterior al corte como `parcial`
(`SEMANAS_REZAGO=1`). Para el pronóstico, esas 9 semanas se interpolan en escala logarítmica
(`casos_modelo`) y se excluyen del entrenamiento y la evaluación: la interpolación usa datos
posteriores y filtraría información del futuro.

## 6. Población

- Desde 2018 los totales de la ciudad, de la cabecera y de la zona rural son las proyecciones
  oficiales del DANE (censo 2018, actualización de julio de 2025), que se descargan con
  `make descargar`. Antes de 2018 se usa la proyección de Planeación Distrital (censo 2005).
- El DANE no publica población por comuna. Cada comuna recibe el total de la cabecera del DANE
  multiplicado por su participación en la proyección de Planeación del mismo año; cada
  corregimiento, el total rural por su participación entre los corregimientos. Planeación llega
  a 2020, así que de 2021 en adelante se conserva la participación de 2020 y la fila queda
  marcada en `proyectada` (el total es oficial; el reparto, supuesto).
- La columna `fuente_poblacion` dice de dónde sale cada cifra. Si el archivo del DANE no está, el
  pipeline vuelve al método anterior (tendencia lineal de Planeación).
- El DANE estima unos 2,28 millones de habitantes en 2024, cerca de 300.000 menos que la
  extrapolación de Planeación; por eso las tasas son más altas (2024: 1.561,5 por 100.000).
- Las tasas por comuna usan la población media del periodo.

## 7. Canal endémico

- Método de cuartiles (Bortman, 1999) sobre conteos semanales, con una ventana de ±2 semanas.
- Años base: 2021, 2022 y 2025, los años de baja transmisión disponibles (`CANAL_ANIOS_BASE`). Solo entran semanas completas.
- Para **evaluar** la alerta en 2024-2025 se usa un canal con años base anteriores a la prueba (2021 y 2022); el canal operativo usa los tres.

## 8. Clima

- `make descargar` trae de la API de Datos Abiertos la lluvia y la temperatura de 2020-2025 de
  cuatro estaciones automáticas del IDEAM en Cali (Universidad del Valle, Base Aérea, Siloé y
  Farallones), mes a mes. Se unen con la descarga manual de 2023; las lecturas repetidas se marcan
  y no se cuentan dos veces.
- El clima de la ciudad es el promedio de las tres estaciones urbanas que recibieron al menos la
  mitad de sus lecturas en la semana (Farallones está en la zona de montaña y no representa la
  ciudad). La lluvia de cada estación se lleva a la semana completa antes de promediar.
- En la segunda mitad de 2024 el IDEAM cambió la descripción de los sensores y la temperatura
  registrada bajó entre 1 y 4 °C. Cada estación y sistema se trata como una serie distinta: la
  variable `temp_anomalia` es la diferencia con su propia media, y es la que usan los análisis y
  el pronóstico.
- Hay lluvia en 159 semanas y temperatura en 121 (casi nada en 2021 y 2023); las semanas sin
  lecturas suficientes quedan vacías.

## 9. Modelos

- **Pronóstico:** el objetivo es el cambio logarítmico; candidatos simples, con temperatura y de aprendizaje automático; selección por horizonte con la validación 2023 (`PRONOSTICO_MODELO_POR_HORIZONTE`). Si varios modelos quedan a menos del 1 % del mejor MAE (`TOLERANCIA_EMPATE`), gana el más simple: ingenuo, tendencia, tendencia con clima y por último aprendizaje automático. El amortiguamiento de la tendencia es un hiperparámetro elegido en validación.
- **Riesgo de signos de alarma:** solo los casos del Distrito 2022-2023 traen clasificación clínica y síntomas; partición temporal 80/20, Random Forest con los hiperparámetros de la búsqueda en malla (`RIESGO_PARAMS`), umbral de Youden y niveles por tercios con predicciones fuera de muestra.
- **Hospitalización:** el dato viene en todas las fuentes, así que se entrena con los casos de 2020-2023 y se prueba con los de 2024-2025, solo con las variables comunes al INS y al Distrito (`TAREAS_CLASIFICACION`, `HOSPITALIZACION_PARAMS`). Con los seis años el AUC no sube (0,650 frente a 0,655 entrenando solo con 2022-2023), pero la prueba pasa de 2.833 a 38.662 casos.
- **Explicación de cada resultado:** cada grupo de variables se reemplaza por los valores de 200 casos reales y se mide cuánto cambia la probabilidad (`contribuciones`). Se prefirió a un caso de referencia único porque ese cambiaba mucho con valores raros, como la edad de 100 años.
- **Comunas:** K-means con k = 3 (`N_GRUPOS_COMUNAS`), elegido por silueta.
- `make train` compara la elección de la validación con la congelada y deja una advertencia en el registro si dejan de coincidir.
