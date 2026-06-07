"""
01_exploracion.py
─────────────────
Exploración del dataset CENAPRED — Municipios Occidente de México
Genera: outputs/exploracion.csv  (top-10 municipios por beneficio)

Ejecución:
    python src/01_exploracion.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent.parent
DATA   = ROOT / "data" / "municipios_occidente_riesgo_CENAPRED.csv"

# ── Carga ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA, encoding="utf-8")
print(f"✓ Dataset cargado: {len(df)} municipios, {df.shape[1]} columnas")
print(f"  Columnas: {list(df.columns)}\n")

# ── 1. Distribución por estado ─────────────────────────────────────────────────
tab_estado = df["estado"].value_counts().rename_axis("Estado").reset_index()
tab_estado.columns = ["Estado", "N municipios"]
tab_estado["% del total"] = (tab_estado["N municipios"] / len(df) * 100).round(1)
print("── Municipios por estado ──")
print(tab_estado.to_string(index=False))

# ── 2. Zona sísmica ────────────────────────────────────────────────────────────
orden_zona = ["A", "B", "C", "D"]
tab_zona = (df["zona_sismica_CFE"]
            .value_counts()
            .reindex(orden_zona, fill_value=0)
            .rename_axis("Zona CFE")
            .reset_index())
tab_zona.columns = ["Zona CFE", "N municipios"]
tab_zona["% del total"] = (tab_zona["N municipios"] / len(df) * 100).round(1)
print("\n── Zona sísmica CFE ──")
print(tab_zona.to_string(index=False))

# ── 3. Peligro de inundación ───────────────────────────────────────────────────
orden_grado = ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]
tab_inund = (df["grado_peligro_inundacion_CENAPRED"]
             .value_counts()
             .reindex(orden_grado, fill_value=0)
             .rename_axis("Grado inundación")
             .reset_index())
tab_inund.columns = ["Grado inundación", "N municipios"]
tab_inund["% del total"] = (tab_inund["N municipios"] / len(df) * 100).round(1)
print("\n── Peligro inundación ──")
print(tab_inund.to_string(index=False))

# ── 4. Vulnerabilidad social ───────────────────────────────────────────────────
tab_vuln = (df["grado_vulnerabilidad_social_CENAPRED"]
            .value_counts()
            .reindex(orden_grado, fill_value=0)
            .rename_axis("Vulnerabilidad social")
            .reset_index())
tab_vuln.columns = ["Vulnerabilidad social", "N municipios"]
tab_vuln["% del total"] = (tab_vuln["N municipios"] / len(df) * 100).round(1)
print("\n── Vulnerabilidad social ──")
print(tab_vuln.to_string(index=False))

# ── 5. Atlas de Riesgo Municipal ───────────────────────────────────────────────
tab_arm = df["tiene_ARM"].value_counts().rename_axis("Tiene ARM").reset_index()
tab_arm.columns = ["Tiene ARM", "N municipios"]
tab_arm["% del total"] = (tab_arm["N municipios"] / len(df) * 100).round(1)
print("\n── Atlas de Riesgos Municipal (ARM) ──")
print(tab_arm.to_string(index=False))

tab_arm_estado = (df.groupby(["estado", "tiene_ARM"])
                  .size()
                  .unstack(fill_value=0)
                  .reset_index())
tab_arm_estado.columns.name = None
print("\n── ARM por estado ──")
print(tab_arm_estado.to_string(index=False))

# ── 6. Cruce zona sísmica × inundación ────────────────────────────────────────
cruce = pd.crosstab(df["zona_sismica_CFE"],
                    df["grado_peligro_inundacion_CENAPRED"],
                    margins=True, margins_name="Total")
cruce = cruce.reindex(columns=orden_grado + ["Total"])
print("\n── Cruce: zona sísmica × peligro inundación ──")
print(cruce)

# ── 7. Scores y costos (base para el AG) ──────────────────────────────────────
MAP_ZONA  = {"A": 1, "B": 2, "C": 3, "D": 4}
MAP_GRADO = {"Muy bajo": 1, "Bajo": 2, "Medio": 3, "Alto": 4, "Muy alto": 5}

df["z_sis"]   = df["zona_sismica_CFE"].map(MAP_ZONA)
df["z_inund"] = df["grado_peligro_inundacion_CENAPRED"].map(MAP_GRADO)
df["z_vuln"]  = df["grado_vulnerabilidad_social_CENAPRED"].map(MAP_GRADO)
df["sin_arm"] = (df["tiene_ARM"] == "No").astype(int)

df["s_sis"]   = (df["z_sis"]   - 1) / 3
df["s_inund"] = (df["z_inund"] - 1) / 4
df["s_vuln"]  = (df["z_vuln"]  - 1) / 4

W_SIS, W_INUND, W_VULN, W_ARM = 0.35, 0.30, 0.25, 0.10
df["beneficio"] = (W_SIS   * df["s_sis"]
                 + W_INUND * df["s_inund"]
                 + W_VULN  * df["s_vuln"]
                 + W_ARM   * df["sin_arm"])

COSTO_BASE, COSTO_SIN_ARM, COSTO_ZONA_D = 50_000, 30_000, 20_000
df["costo"] = (COSTO_BASE
             + df["sin_arm"] * COSTO_SIN_ARM
             + (df["zona_sismica_CFE"] == "D").astype(int) * COSTO_ZONA_D)

PRESUPUESTO = 10_000_000
print(f"\n── Análisis de costo / presupuesto ──")
print(f"  Costo total si se atienden TODOS: ${df['costo'].sum():>12,.0f}")
print(f"  Presupuesto disponible:           ${PRESUPUESTO:>12,.0f}")
print(f"  Costo medio por municipio:        ${df['costo'].mean():>12,.0f}")
print(f"  Municipios que caben (aprox.):    {PRESUPUESTO // df['costo'].mean():.0f} de {len(df)}")

print("\n── Top 10 municipios por beneficio compuesto ──")
top10 = (df[["estado","municipio","zona_sismica_CFE",
             "grado_peligro_inundacion_CENAPRED",
             "grado_vulnerabilidad_social_CENAPRED",
             "tiene_ARM","beneficio","costo"]]
         .sort_values("beneficio", ascending=False)
         .head(10))
print(top10.to_string(index=False))

# ── 8. Exportar a CSV ─────────────────────────────────────────────────────────
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

top10.to_csv(OUT_DIR / "exploracion.csv", index=False, encoding="utf-8")
print(f"\n✓ exploracion.csv guardado en: {OUT_DIR}")