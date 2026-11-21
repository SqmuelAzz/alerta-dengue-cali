# Ficha de los modelos — Alerta Dengue Cali

**Autores:** Samuel Acevedo Mambuscay y Luis Carlos Tez — Universidad Libre, seccional Cali
**Organización usuaria:** Secretaría de Salud Pública Distrital de Santiago de Cali (vigilancia epidemiológica y control vectorial)
**Versión:** v1 · **Código:** `flows/05_ml_models.py` · **Metadatos completos:** `models/*.metadata.json` (se generan con `make train`)

El producto tiene cuatro modelos. Todos se reentrenan con `make train` y guardan, junto al
`.joblib`, un archivo de metadatos con la huella SHA-256 de los datos, los parámetros, las
métricas y las versiones de Python, pandas y scikit-learn.

---

## 1. Pronóstico de casos semanales y alerta de brote (`dengue_pronostico_semanal_v1`)

**Propósito.** Estimar los casos de dengue de Cali en las próximas 1 a 4 semanas epidemiológicas,
con un intervalo del 80 %, y ubicar cada semana pronosticada en el canal endémico (semáforo). Si
el pronóstico supera el cuartil 3 del canal, se emite la alerta de brote.

**Uso previsto.** Comité de vigilancia epidemiológica: preparar brigadas, camas e insumos con
1 a 4 semanas de anticipación. **No recomendado:** pronosticar por comuna, más allá de 4 semanas o
como cifra oficial de casos.

**Datos.** Serie semanal de Silver (`serie_semanal.parquet`), 314 semanas de 2020 a 2025 por
semana de inicio de síntomas; 9 semanas interpoladas que no se usan para entrenar ni evaluar.
Fuentes: microdatos del INS (2020, 2021, 2024, 2025) y portal de Datos Abiertos de Cali
(2022-2023); temperatura de las estaciones urbanas del IDEAM (2020-2025) como variable candidata.

**Modelo.** Por horizonte, el de menor error en la validación 2023; si otros quedan a menos del 1 %,
el más simple:

| Horizonte | Modelo | Qué hace |
|---|---|---|
| 1 semana | Ingenuo | Repite los casos de la última semana |
| 2 y 3 semanas | Tendencia amortiguada al 25 % | Extrapola el 25 % de la tendencia de las últimas 4 semanas |
| 4 semanas | Tendencia amortiguada al 50 % | Extrapola el 50 % de esa tendencia |

Se probaron también Ridge, Random Forest, gradient boosting, un ensamble Random Forest + ingenuo y
la tendencia con la anomalía de temperatura de las últimas 4 semanas. Las versiones con
temperatura empataron con la tendencia sin clima (diferencias de 0,01 a 0,06 casos en la
validación) y no se eligieron.

Intervalo del 80 %: cuantiles 10 y 90 de los errores de validación (escala logarítmica).
Canal endémico operativo: cuartiles con los años base 2021, 2022 y 2025 (±2 semanas).

**Métricas en la prueba (2024-2025, 101 a 104 semanas, una sola evaluación):**

| Horizonte | MAE (casos) | MAE ingenuo | Mejora | RMSE | MAPE | Cobertura del IC 80 % | Sensibilidad de la alerta | Precisión de la alerta |
|---|---|---|---|---|---|---|---|---|
| 1 | 33,4 | 33,4 | 0,0 % | 58,6 | 14,4 % | 60,6 % | 96,2 % | 96,2 % |
| 2 | 46,1 | 46,9 | 1,7 % | 79,5 | 18,8 % | 67,0 % | 98,1 % | 96,2 % |
| 3 | 59,6 | 62,5 | 4,7 % | 100,3 | 21,6 % | 60,8 % | 98,0 % | 98,0 % |
| 4 | 72,0 | 76,8 | 6,3 % | 120,3 | 25,5 % | 61,4 % | 98,0 % | 98,0 % |

La alerta se evaluó con un canal construido solo con 2021 y 2022 (años anteriores a la prueba).

**Limitaciones.**
- La mejora frente al ingenuo es pequeña: la serie es muy persistente (autocorrelación 0,99).
- El intervalo cubre 61-67 % de las semanas, no el 80 %: subestima la incertidumbre en epidemias grandes. Debe leerse como un rango mínimo.
- La alerta acierta casi siempre en 2024-2025 porque esas semanas estaban muy lejos del umbral (2024 todo en brote, 2025 casi todo debajo); en años de transición su desempeño puede ser menor.
- Los modelos de aprendizaje automático (Random Forest, gradient boosting, Ridge y el ensamble con el ingenuo) fueron peores que el ingenuo en la validación; se volverán a comparar cuando la serie tenga más años de historia.
- El clima no entra en el pronóstico: la temperatura del IDEAM no aportó información (en la prueba habría cambiado el error en menos de 1 caso por semana). Las series públicas tienen la mitad de las semanas vacías; la variable queda lista en el pipeline para evaluarla de nuevo con datos más completos.
- Depende de que los casos de la última semana estén completos: una semana con notificación incompleta baja el pronóstico.

---

## 2. Riesgo de signos de alarma o dengue grave (`dengue_riesgo_alarma_v1`)

**Propósito.** Estimar, con la información de la primera consulta, la probabilidad de que un caso
de dengue tenga signos de alarma o sea dengue grave, y asignarle un nivel de riesgo (Bajo, Medio,
Alto).

**Uso previsto.** Priorizar el seguimiento telefónico o domiciliario de casos ambulatorios por
parte de EPS, IPS y la Secretaría. **No recomendado:** diagnosticar, decidir hospitalización o
alta, negar atención, ni usarlo fuera de Cali o con casos sin edad, sexo y fecha de inicio de
síntomas.

**Datos.** 13.966 casos del Distrito (2022-2023) con clasificación final; 39,1 % severos (5.348
con signos de alarma y 106 graves). Entrenamiento de la evaluación: 11.133 casos notificados hasta
el 24 de octubre de 2023; prueba: 2.833 casos desde el 25 de octubre de 2023. El modelo definitivo
usa los 13.966.

**Variables usadas.** Edad, días desde el inicio de síntomas (0 a 30), sexo, estrato, comuna,
afiliación, migrante, desplazamiento en los últimos 15 días, fiebre, cefalea, dolor retroocular,
mialgias, artralgia y erupción.
**Variables excluidas.** Los 15 signos de alarma y de gravedad (fuga de datos: con ellos el AUC
es 1,000), hospitalización, conducta, tipo de caso, condición final y clasificación.

**Modelo.** `Pipeline` de scikit-learn: imputación por mediana y escalado de numéricas, imputación
por moda de binarias, one-hot de categóricas (las categorías con menos de 20 casos se agrupan) y
Random Forest (400 árboles, profundidad máxima 10, hojas de al menos 20 casos, 20 % de las
variables por nodo, semilla 42). Umbral de Youden y cortes de los niveles por tercios, calculados
con predicciones fuera de muestra.

**Métricas en la prueba (n = 2.833, prevalencia 36,1 %):**

| Métrica | Valor |
|---|---|
| ROC-AUC | **0,616** (IC 95 % por remuestreo: 0,594-0,637) |
| AUC en validación cruzada (entrenamiento) | 0,623 ± 0,010 |
| Sensibilidad / especificidad (umbral de Youden) | 40,5 % / 74,2 % |
| Valor predictivo positivo / negativo | 46,9 % / 68,8 % |
| Brier | 0,223 |
| % severos en el nivel Bajo / Medio / Alto | 27,0 % / 36,8 % / 47,8 % |

**Desempeño por subgrupo (prueba, grupos con al menos 50 casos):**

| Grupo | n | AUC | Sensibilidad | Especificidad |
|---|---|---|---|---|
| Mujeres | 1.429 | 0,620 | 38,6 % | 75,2 % |
| Hombres | 1.404 | 0,612 | 42,7 % | 73,2 % |
| 0-4 años | 246 | 0,522 | **2,4 %** | 96,9 % |
| 5-14 años | 902 | 0,591 | 34,2 % | 77,0 % |
| 15-29 años | 792 | 0,612 | 50,8 % | 65,4 % |
| 65 años o más | 155 | 0,584 | 72,2 % | **43,4 %** |
| Contributivo | 2.206 | 0,596 | 26,8 % | 82,6 % |
| Subsidiado | 535 | 0,558 | 79,4 % | **27,4 %** |

Variables más importantes (caída del AUC al permutarlas en la prueba): afiliación, días desde el
inicio de síntomas y edad. Sexo, migrante y desplazamiento no aportan.

**Limitaciones y riesgos.**
- Discriminación modesta: sirve para ordenar una lista, no para decidir sobre una persona.
- Usa la afiliación y la edad como atajos: casi no alerta en niños de 0 a 4 años y alerta casi siempre en mayores de 65 y en el régimen subsidiado. Para usarlo en un grupo concreto se debe revisar primero su fila en esta tabla.
- Entrenado con casos de 2022-2023 de la red del Distrito, la única fuente con clasificación clínica y síntomas (los microdatos del INS de 2020, 2021, 2024 y 2025 no los traen); la forma de clasificar puede cambiar entre años e IPS.
- Los síntomas generales (fiebre, cefalea, dolores, erupción) casi no cambian el resultado: los tiene casi cualquier paciente con dengue. Los signos de alarma no son variables porque definen lo que se predice.

---

## 3. Probabilidad de hospitalización (`dengue_hospitalizacion_v1`)

**Propósito y uso previsto.** Estimar la probabilidad de que un caso sea hospitalizado, para que
las IPS anticipen la demanda de camas. **No recomendado:** decidir la hospitalización de un
paciente.

**Datos.** La hospitalización viene en todas las fuentes, así que este modelo usa los **71.586
casos de 2020 a 2025** (INS y Distrito), 35,5 % hospitalizados. Entrenamiento de la evaluación:
32.924 casos notificados en 2020-2023; prueba: 38.662 casos de 2024-2025, dos años que el modelo
no vio. El modelo definitivo usa los 71.586.

**Variables usadas.** Solo las que traen las dos fuentes: edad, días desde el inicio de síntomas,
sexo, estrato, afiliación y condición de migrante. La comuna, los síntomas y el desplazamiento
solo están en los casos del Distrito y no se usan.

**Modelo.** El mismo pipeline del modelo de riesgo con un Random Forest de 400 árboles,
profundidad máxima 10, hojas de al menos 20 casos y 50 % de las variables por nodo
(`HOSPITALIZACION_PARAMS`, de la búsqueda en malla con 2020-2023). Gradient boosting quedó
empatado en la validación cruzada (0,654 frente a 0,650).

| Métrica (prueba 2024-2025, n = 38.662, prevalencia 33,6 %) | Valor |
|---|---|
| ROC-AUC | **0,650** (IC 95 %: 0,644-0,656) |
| AUC en validación cruzada (2020-2023) | 0,650 ± 0,005 |
| Sensibilidad / especificidad (umbral de Youden) | 53,8 % / 67,5 % |
| Valor predictivo positivo | 45,6 % |
| % hospitalizados en el nivel Bajo / Medio / Alto | 22,7 % / 36,5 % / 48,6 % |

**Desempeño por subgrupo (prueba):**

| Grupo | n | AUC | Sensibilidad | Especificidad |
|---|---|---|---|---|
| Mujeres | 19.916 | 0,644 | 52,8 % | 67,6 % |
| Hombres | 18.746 | 0,656 | 54,8 % | 67,4 % |
| 0-4 años | 2.732 | 0,650 | 67,2 % | 57,6 % |
| 30-44 años | 6.310 | 0,653 | 21,8 % | 89,6 % |
| 65 años o más | 2.534 | 0,563 | 82,7 % | **23,9 %** |
| Contributivo | 30.187 | 0,628 | 38,2 % | 77,0 % |
| Subsidiado | 7.261 | 0,571 | 92,5 % | **12,9 %** |

**Más años no mejoran la discriminación.** El mismo modelo entrenado solo con 2022-2023 obtiene
un AUC de 0,655 en 2024-2025, y los dos están igual de calibrados por edad (en los mayores de 65
años predicen 46-47 % y se hospitalizó el 43 %). El límite lo pone la poca información de la
primera consulta. Lo que se gana es una evaluación con dos años completos que el modelo no vio
(antes eran 2.833 casos de 2023, con un intervalo tres veces más ancho) y un modelo final que
aprende de los seis años.

**Limitación principal.** En el régimen subsidiado el modelo marca a casi todos (especificidad
13 %), porque la mitad de esos pacientes se hospitalizó; con los mayores de 65 años pasa algo
parecido. La hospitalización refleja la oferta de camas y la práctica de cada red, no solo la
gravedad.

---

### Explicación de cada resultado (modelos 2 y 3)

La aplicación y la API muestran qué características de un caso suben o bajan su probabilidad.
Para cada grupo de variables (edad, días, sexo, estrato, comuna, afiliación, migrante o viaje y
síntomas generales) se reemplaza el valor del caso por el de 200 casos reales al azar
(`muestra_de_fondo`) y se promedia la probabilidad; la diferencia, en puntos porcentuales, es el
peso de esa característica frente a lo habitual (`contribuciones`). También se muestra el
promedio de los casos (39,1 % con signos de alarma y 35,5 % hospitalizados) para que la
probabilidad se lea en contexto: un paciente de 100 años con todos los síntomas generales obtiene
43 %, 1,1 veces el promedio, porque los síntomas generales casi no cambian el riesgo y la edad
suma unos 9 puntos. Los mayores de 90 años son muy pocos, así que el modelo los trata como a los
de 80-90.

---

## 4. Perfiles de comunas (`dengue_perfiles_comunas_v1`)

**Propósito.** Agrupar las 22 comunas urbanas según su situación epidemiológica y
socioeconómica para diseñar estrategias de intervención diferentes.

**Datos.** Tabla Gold `indicadores_comunas.parquet` (2022-2023): incidencia por 100.000, % severos,
% hospitalizados, edad mediana, estrato medio, % régimen subsidiado y hospitalizaciones de menores
de 5 años por arbovirosis por 100.000 (centinela 2010-2020). La población de cada comuna es la
proyección oficial del DANE para la cabecera, repartida según la participación de la comuna en la
proyección de Planeación Distrital. Los corregimientos no se incluyen
(pocos casos y tasas inestables).

**Modelo.** Estandarización + K-means (k = 3, 50 inicios, semilla 42). Silueta: k=2 0,241; **k=3
0,307**; k=4 0,297; k=5 0,286; k=6 0,279.

| Grupo | Comunas | Incidencia | % severos | Estrato medio | % subsidiado | Hosp. infantiles por arbovirosis |
|---|---|---|---|---|---|---|
| **A. Mayor incidencia y severidad** | 3, 4, 6, 7, 8, 9, 11 | 850 | 42,4 | 2,5 | 19,4 | 83,3 |
| **B. Estratos medios-altos, pacientes mayores** | 2, 5, 10, 17, 19, 22 | 713 | 37,5 | 3,4 | 8,6 | 26,5 |
| **C. Estratos bajos, población joven** | 1, 12, 13, 14, 15, 16, 18, 20, 21 | 428 | 37,4 | 2,0 | 31,7 | 50,0 |

**Limitaciones.** Silueta moderada: algunas comunas están cerca de la frontera entre grupos. Los
indicadores son de 2022-2023 y la población es una proyección. La incidencia baja del grupo C
puede reflejar menor consulta y no solo menor transmisión.

---

## Cómo reproducir

```bash
make install       # dependencias
make pipeline      # Bronze -> Silver -> Gold -> modelos (lee la carpeta Dataset_Dengue)
make test          # pruebas automáticas
```

Versiones con las que se obtuvieron estas cifras: Python 3.13, pandas 3.0, scikit-learn 1.9;
`RANDOM_STATE=42`. Cada `.metadata.json` guarda la huella SHA-256 del archivo de datos con el que
se entrenó el modelo y la del propio modelo. Si los datos de origen cambian, las métricas pueden
cambiar, y `make train` avisa si la validación ya no elige los mismos modelos de pronóstico.
