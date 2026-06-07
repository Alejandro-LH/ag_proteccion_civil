# Proyecto de Investigación

## AG — Asignación de Recursos de Protección Civil

## Región Occidente de México (Jalisco, Colima, Michoacán, Nayarit)

Algoritmo Genético aplicado al problema de asignación de presupuesto de protección civil usando datos del CENAPRED.

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
│    ├── exploracion.csv 
│    ├── convergencia.png 
│    ├── mejor\_solucion.csv 
│    └── tabla\_comparativa.csv 
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

Dado un presupuesto de **\$10,000,000 MXN**, seleccionar qué municipios de la región occidente de México reciben intervención de protección civil, maximizando la cobertura ponderada de riesgo.


| Parámetro                  | Valor              |
| --------------------------- | ------------------ |
| Municipios                  | 267                |
| Presupuesto                 | \$10,000,000 MXN   |
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


| Tasa mutación | Mejor fitness | Media     | SD        | Municipios | Presupuesto usado | Factible |
| -------------- | ------------- | --------- | --------- | ---------- | ----------------- | -------- |
| **0.01**       | **82.77**     | **82.61** | **0.118** | **125**    | **\$10,000,000**  | **Sí**  |
| 0.05           | 79.05         | 78.50     | 0.362     | 128        | \$10,000,000      | Sí      |

p\_mut = 0.01 obtiene mayor fitness y es 3× más estable entre corridas. Ambas configuraciones respetan el presupuesto.
