# Conclusiones del análisis exploratorio — Dengue en Cali 2020-2025

**Autores:** Samuel Acevedo Mambuscay (2216646) y Luis Carlos Tez (2216702) — Universidad Libre, seccional Cali
**Cuaderno:** `notebooks/02_exploracion.ipynb` · **Datos:** capas Silver y Gold generadas con `make pipeline`

Cada hallazgo va con su cifra y su tamaño de muestra (n). Las figuras están en `reports/figuras/`.

## 1. Datos analizados

| Fuente | Años | Casos de Cali en Silver |
|---|---|---|
| Microdatos del evento 210 del Portal SIVIGILA (INS) | 2020, 2021, 2024, 2025 | 57.618 |
| Casos de dengue y dengue grave del portal de Datos Abiertos de Cali (SSPD) | 2022, 2023 | 13.968 |
| **Total** | 2020-2025 | **71.586** |

Complementos: proyecciones oficiales de población municipal del DANE (2018-2042, actualización de
2025) y proyecciones por comuna de Planeación Distrital (2005-2020), vigilancia centinela de
hospitalizaciones en menores de 5 años (2010-2020) y lluvia y temperatura de cuatro estaciones
automáticas del IDEAM en Cali (2020-2025, descargadas de la API de Datos Abiertos con
`make descargar`).

## 2. Decisiones de limpieza y el sesgo que introducen

| Decisión | Registros afectados | Efecto o sesgo posible |
|---|---|---|
| Descartar filas vacías de los archivos del portal | 104.878 (51.088 en 2022 y 53.790 en mortalidad 2023) | Ninguno: no tienen datos. |
| Eliminar notificaciones duplicadas (misma persona, fechas y lugar) | 3 | Ninguno apreciable. |
| Integrar la única fila de mortalidad al caso de dengue grave correspondiente | 1 | Una sola muerte en los datos: no se puede analizar la letalidad. |
| Usar la **semana de inicio de síntomas** en todos los años | 71.586 | Es el criterio de la variable `SEMANA` del INS; los archivos del Distrito se recalcularon igual para que la serie sea comparable. Las cifras pueden diferir de los boletines por semana de notificación. |
| Contar solo el evento 210 (dengue) en la serie | 106 casos de dengue grave de 2023 quedan aparte | 2023 suma 11.160 casos en la serie (11.266 con dengue grave). Así los años son comparables, porque el dengue grave solo está publicado para 2023. |
| Marcar como sin datos las semanas posteriores al corte de publicación | 7 semanas (2022: 50-52; 2023: 49-52) y 2 parciales | 2022 y 2023 están **subestimados**. Para el modelo de pronóstico se interpolan (log-lineal) y quedan marcadas. |
| Usar la población oficial del DANE desde 2018 y repartir el total de la cabecera entre comunas según su participación en la proyección de Planeación | 2018-2025 | El DANE (censo 2018) estima unos 2,28 millones de habitantes, cerca de 300.000 menos que la proyección de Planeación basada en el censo 2005: las tasas son más altas que con esa proyección. Después de 2020 se supone que la participación de cada comuna no cambia. |
| Clima de la ciudad como promedio de las tres estaciones urbanas con al menos la mitad de sus lecturas | 159 semanas con lluvia y 121 con temperatura | Las semanas sin lecturas suficientes quedan vacías; no se rellenan. |
| No borrar registros con fechas o códigos dudosos; solo marcarlos | ver sección 7 | Cada análisis decide si los usa. |

## 3. Magnitud y tendencia (n = 314 semanas)

- **Dos epidemias en el periodo.** 2020: 13.230 casos (582,2 por 100.000 habitantes). 2024: 35.604 casos (**1.561,5 por 100.000**), el año con más transmisión, con un pico de 1.295 casos en la semana 21. En 2025 bajó a 3.060 casos (134,4 por 100.000). Ver `fig01_serie_semanal.png` y `fig02_incidencia_anual.png`.
- Las cifras de 2022 (2.702) y 2023 (11.160) son mínimos, porque esos archivos se cortaron en diciembre.
- **La serie es muy persistente:** la autocorrelación es 0,99 a una semana y sigue en 0,95 a cuatro semanas. Los casos de la próxima semana se parecen mucho a los de esta, así que el pronóstico ingenuo es una referencia difícil de superar.
- **La estacionalidad es débil frente a la diferencia entre años.** Las semanas 1 a 11 tienen en promedio un índice estacional de 1,2 a 1,5 (20 % a 50 % más casos que la semana promedio del año) y las últimas semanas del año de 0,5 a 0,7, pero el rango entre años es muy amplio (`fig04_estacionalidad_autocorrelacion.png`).

## 4. Canal endémico y semanas inusuales (anomalías)

Canal por cuartiles (Bortman, 1999) con los años de baja transmisión 2021, 2022 y 2025 como base
y una ventana de ±2 semanas. Semanas completas en zona de brote (por encima del cuartil 3):

| Año | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| Semanas en brote | 51 de 53 | 38 de 52 | 1 de 48 | 32 de 47 | 52 de 52 | 3 de 53 |

- 2024 estuvo en brote todo el año y 2020 casi todo el año (`fig03_canal_endemico.png`).
- La última semana con datos (semana 53 de 2025) tuvo 30 casos, en **zona de éxito**: Cali cerró 2025 dentro de su canal endémico.
- **Limitación:** que 2021 tenga 38 semanas en brote siendo año base muestra que tres años base son pocos y que el primer semestre de 2021 todavía arrastraba la epidemia de 2020. El canal debe recalcularse cuando haya más años sin epidemia.

## 5. ¿Quién se enferma y quién se agrava?

**Edad (n = 71.586):** mediana de 19 años y 40,0 % menores de 15 años; en los casos del Distrito
2022-2023 (n = 13.968) la mediana es 18 años y 41,1 % son menores de 15. El grupo de 5 a 14 años es
el que más casos aporta (`fig05_casos_edad_sexo.png`).

**Severidad** (con signos de alarma o dengue grave). Solo se puede medir con los casos del Distrito
2022-2023, que traen la clasificación final (n = 13.966; 39,1 % severos: 5.348 con signos de alarma
y 106 dengue grave). Intervalos de Wilson al 95 % en `fig06_severidad_por_grupo.png`.

| Grupo | n | % severos (IC 95 %) |
|---|---|---|
| 65 años o más | 698 | **50,9 %** (47,2-54,6) |
| 15-29 años | 3.937 | 42,7 % (41,2-44,3) |
| 0-4 años | 1.295 | 31,7 % (29,3-34,3) |
| Régimen subsidiado | 2.957 | **48,8 %** (47,0-50,6) |
| No asegurado | 90 | 45,6 % (35,7-55,8) |
| Régimen contributivo | 10.522 | 36,3 % (35,4-37,3) |

| Prueba chi-cuadrado (severidad frente a…) | n | p | V de Cramér |
|---|---|---|---|
| Afiliación | 13.949 | < 0,001 | 0,105 |
| Grupo de edad | 13.966 | < 0,001 | 0,092 |
| Estrato | 13.412 | < 0,001 | 0,041 |
| Sexo | 13.966 | 0,67 | 0,004 |

Las diferencias son estadísticamente significativas para edad, afiliación y estrato, pero el
tamaño del efecto es **débil** (V ≤ 0,10): con casi 14.000 casos cualquier diferencia pequeña da
p < 0,05. Ningún rasgo sociodemográfico separa por sí solo a los pacientes que se agravan, y eso
anticipa que el modelo de riesgo tendrá una discriminación modesta. El sexo no se asocia con la
severidad.

## 6. Oportunidad de consulta y notificación

- Mediana de 3 días entre el inicio de síntomas y la consulta en 2022-2025 (4 días en 2020 y 2021). La consulta tardía (más de 3 días) bajó de 58,8 % en 2020 a 43,6 % en 2025.
- La notificación es oportuna: mediana de 1 día después de la consulta; el 90 % se notifica en 8 días o menos.
- **Los casos severos consultan más tarde:** mediana de 4 días frente a 3 (Mann-Whitney, p < 0,001; n = 13.966; `fig07_oportunidad_consulta.png`). Ver la sección 10 sobre cómo interpretarlo.

## 7. Registros inconsistentes y lecturas anómalas

| Hallazgo | Registros | % |
|---|---|---|
| Casos con más de 30 días entre dos de sus fechas | 1.437 de 71.586 | 2,0 % |
| Casos con fechas en orden imposible (síntomas después de la consulta) | 0 | 0 % |
| Semana reportada que no coincide con el inicio de síntomas | 32 | 0,04 % |
| Casos sin estrato | 3.954 | 5,5 % |
| Casos del Distrito sin comuna | 486 de 13.968 | 3,5 % |
| Casos del Distrito con código de comuna no válido | 153 de 13.968 | 1,1 % |
| Lecturas del IDEAM fuera de rango (lluvia 0-60 mm por lectura, temperatura 5-45 °C) | 4 de 551.038 | < 0,01 % |
| Lecturas del IDEAM repetidas (misma estación, hora y variable) | 70.145 de 551.038 | 12,7 % |
| Cambio de sistema de medición del IDEAM (segunda mitad de 2024) | 3 estaciones urbanas | La temperatura registrada bajó entre 1 y 4 °C con el cambio; se corrige con anomalías por estación y sistema. |

Los registros con intervalos atípicos se excluyen de los análisis de oportunidad; los demás se
conservan con su marca.

## 8. Comunas

- Incidencia 2022-2023 en las 22 comunas urbanas: de **177 por 100.000 en la comuna 1 a 1.188 en la comuna 4** (6,7 veces más). Le siguen las comunas 22 (1.004), 21 (946) y 7 (934) (`fig08_incidencia_comunas.png`).
- Las comunas de estrato medio más alto tienen pacientes mayores (Spearman 0,77) y menos afiliación subsidiada (−0,97). La incidencia se correlaciona con el porcentaje de casos severos (0,48) y de hospitalizados (0,49) (`fig09_correlacion_comunas.png`).
- Los corregimientos tienen tasas muy inestables (por ejemplo, Navarro: 31 casos, 2.773 por 100.000) porque su población es pequeña; por eso el agrupamiento usa solo las 22 comunas urbanas.
- El agrupamiento K-means confirma tres perfiles de comunas (ver `reports/model_card.md`).

## 9. Clima

Con la descarga completa de la API del IDEAM hay lluvia en 159 de las 314 semanas y temperatura en
121; 2021 no tiene temperatura y 2023 casi no tiene lecturas (`fig10_clima_casos.png`).

- El clima de la misma semana no se relaciona con el número de casos (lluvia: rho = 0,02, n = 157;
  temperatura: rho = −0,01, n = 119).
- Para buscar un efecto anticipado se comparó el clima de hace k semanas (k = 0 a 12) con el
  crecimiento de los casos en las 4 semanas siguientes. La temperatura media parece anticiparlo
  (rho de 0,22 a 0,33, p < 0,05), pero es un **artefacto de medición**: en la segunda mitad de 2024
  el IDEAM cambió el sistema con que publica los sensores y la temperatura registrada bajó entre 1 y
  4 °C, justo cuando terminaba la epidemia. Con la anomalía (diferencia con la media de cada
  estación y sistema), la relación desaparece (rho entre −0,11 y 0,18; ningún p < 0,05).
- La lluvia de hace 9 semanas muestra una relación débil y negativa (rho = −0,24, p = 0,005,
  n = 143), la única significativa entre 13 rezagos: puede ser casualidad por comparaciones
  múltiples.

Conclusión: con los datos públicos del IDEAM no aparece una relación robusta entre el clima y los
casos de Cali. La temperatura se probó en el pronóstico y no lo mejora (ver `model_card.md`).

## 10. Lo que NO podemos afirmar

- **Asociación no es causalidad.** Que los pacientes subsidiados tengan más casos severos no prueba que la afiliación cause la gravedad: puede reflejar acceso tardío, comorbilidades o diferencias en cómo cada red de IPS clasifica y hospitaliza.
- **Consultar tarde y agravarse están relacionados, pero la dirección no es clara.** Los signos de alarma suelen aparecer entre el día 3 y el 7 de la enfermedad, así que muchos pacientes consultan cuando aparecen; no se puede afirmar que consultar antes evitaría la gravedad.
- **Correlaciones entre comunas no aplican a personas** (falacia ecológica): una comuna de estrato alto con incidencia alta no dice que las personas de estrato alto se enfermen más.
- **Clima:** que no aparezca relación no prueba que el clima no influya en el dengue (la literatura muestra que sí). Las series públicas del IDEAM tienen la mitad de las semanas vacías y un cambio de sistema de medición en 2024; con esos datos el efecto no se puede medir con confianza.
- **Severidad y comunas solo para 2022-2023:** los microdatos del INS no traen síntomas, clasificación final ni comuna, así que esos análisis no cubren las epidemias de 2020 y 2024.
- **Una sola muerte** en los datos: no se puede estimar la letalidad.
- **Subregistro:** todos los conteos son casos notificados; los casos que no consultan no aparecen.
