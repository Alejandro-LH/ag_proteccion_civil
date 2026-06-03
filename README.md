# Proyecto de Investigación

## AG — Asignación de Recursos de Protección Civil

## Región Occidente de México (Jalisco, Colima, Michoacán, Nayarit)

Algoritmo Genético aplicado al problema de asignación de presupuesto de
protección civil usando datos del CENAPRED.

## Estructura del repositorio

```

ag\_proteccion\_civil/ 
│
├── data/ 
│   └── municipios\_occidente\_riesgo\_CENAPRED.csv 
│
├── src/ 
│   ├── 01\_exploracion.py 
│   └── 02\_algoritmo\_genetico.py 
│
├── outputs/ 
│    ├── exploracion\_resumen.xlsx 
│    ├── convergencia.png 
│    ├── mejor\_solucion.xlsx 
│    └── tabla\_comparativa.xlsx 
│
├── requirements.txt
│
└── README.md
```

## Instalación

```
git clone https://github.com/Alejandro-LH/ag_proteccion_civil.git
cd ag_proteccion_civil
pip install -r requirements.txt
```

## Ejecución

```bash
# 1. Exploración del dataset
python src/01_exploracion.py

# 2. Algoritmo Genético
python src/02_algoritmo_genetico.py
```

## Problema

Dado un presupuesto de **$10,000,000 MXN**, seleccionar qué municipios
de la región occidente de México reciben intervención de protección civil,
maximizando la cobertura ponderada de riesgo.


| Parámetro                  | Valor              |
| --------------------------- | ------------------ |
| Municipios                  | 267                |
| Presupuesto                 | $10,000,000 MXN    |
| Cromosoma                   | Binario (267 bits) |
| Generaciones                | 300                |
| Población                  | 200 individuos     |
| Selección                  | Torneo (k=5)       |
| Cruzamiento                 | Un punto (p=0.85)  |
| Mutación comparada         | 0.01 vs 0.05       |
| Corridas por configuración | 5                  |

## Función de fitness

```
f(x) = Σᵢ [xᵢ · bᵢ] − 500 · max(0, (Σᵢ xᵢ·cᵢ − P) / P)²

bᵢ = 0.35·score\_sísmica + 0.30·score\_inundación + 0.25·score\_vulnerabilidad + 0.10·sin\_ARM
```

## Resultados


| Tasa mutación | Mejor fitness | Media     | SD        | Municipios |
| -------------- | ------------- | --------- | --------- | ---------- |
| **0.01**       | **86.01**     | **85.98** | **0.024** | **134**    |
| 0.05           | 83.29         | 82.81     | 0.247     | 134        |

p_mut = 0.01 obtiene mayor fitness y es 10x más estable entre corridas.
