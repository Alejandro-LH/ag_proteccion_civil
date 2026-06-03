"""
02_algoritmo_genetico.py
Algoritmo Genético — Asignación de recursos de Protección Civil
Región Occidente de México — datos CENAPRED

Ejecución:
    python src/02_algoritmo_genetico.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS
# ══════════════════════════════════════════════════════════════════════════════

PRESUPUESTO   = 10_000_000
COSTO_BASE    =     50_000
COSTO_SIN_ARM =     30_000
COSTO_ZONA_D  =     20_000

W_SIS, W_INUND, W_VULN, W_ARM = 0.35, 0.30, 0.25, 0.10

POP_SIZE   = 200
N_GEN      = 300
K_TORNEO   = 5
P_CRUCE    = 0.85
LAMBDA     = 500.0
N_RUNS     = 5
MUT_RATES  = [0.01, 0.05]
SEED_BASE  = 42

MAP_ZONA  = {"A": 1, "B": 2, "C": 3, "D": 4}
MAP_GRADO = {"Muy bajo": 1, "Bajo": 2, "Medio": 3, "Alto": 4, "Muy alto": 5}

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "municipios_occidente_riesgo_CENAPRED.csv"
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def cargar_datos(path):
    df = pd.read_csv(path, encoding="utf-8")

    df["s_sis"]   = (df["zona_sismica_CFE"].map(MAP_ZONA) - 1) / 3
    df["s_inund"] = (df["grado_peligro_inundacion_CENAPRED"].map(MAP_GRADO) - 1) / 4
    df["s_vuln"]  = (df["grado_vulnerabilidad_social_CENAPRED"].map(MAP_GRADO) - 1) / 4
    df["sin_arm"] = (df["tiene_ARM"] == "No").astype(int)

    df["beneficio"] = (W_SIS   * df["s_sis"]
                     + W_INUND * df["s_inund"]
                     + W_VULN  * df["s_vuln"]
                     + W_ARM   * df["sin_arm"])

    df["costo"] = (COSTO_BASE
                 + df["sin_arm"] * COSTO_SIN_ARM
                 + (df["zona_sismica_CFE"] == "D").astype(int) * COSTO_ZONA_D)

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. FITNESS
# ══════════════════════════════════════════════════════════════════════════════

def fitness_fn(cromosoma, beneficios, costos):
    sel    = cromosoma.astype(bool)
    b_tot  = beneficios[sel].sum()
    c_tot  = costos[sel].sum()
    exceso = max(0.0, (c_tot - PRESUPUESTO) / PRESUPUESTO)
    return float(b_tot - LAMBDA * exceso**2)


# ══════════════════════════════════════════════════════════════════════════════
# 3. OPERADORES
# ══════════════════════════════════════════════════════════════════════════════

def init_poblacion(pop_size, n, rng):
    return rng.integers(0, 2, size=(pop_size, n), dtype=np.int8)

def torneo(pop, fits, k, rng):
    idx = rng.choice(len(pop), size=k, replace=False)
    return int(idx[np.argmax(fits[idx])])

def cruzamiento(p1, p2, rng):
    if rng.random() < P_CRUCE:
        pt = rng.integers(1, len(p1))
        h1 = np.concatenate([p1[:pt], p2[pt:]])
        h2 = np.concatenate([p2[:pt], p1[pt:]])
        return h1, h2
    return p1.copy(), p2.copy()

def mutacion(cromosoma, p_mut, rng):
    mascara = rng.random(len(cromosoma)) < p_mut
    return np.where(mascara, 1 - cromosoma, cromosoma).astype(np.int8)


# ══════════════════════════════════════════════════════════════════════════════
# 4. BUCLE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def ejecutar_ag(beneficios, costos, p_mut, seed):
    rng  = np.random.default_rng(seed)
    n    = len(beneficios)
    pop  = init_poblacion(POP_SIZE, n, rng)
    fits = np.array([fitness_fn(ind, beneficios, costos) for ind in pop])

    mejor_ind = pop[np.argmax(fits)].copy()
    mejor_fit = float(fits.max())
    hist_mejor, hist_prom = [], []

    for _ in range(N_GEN):
        nueva_pop = []
        while len(nueva_pop) < POP_SIZE:
            i1 = torneo(pop, fits, K_TORNEO, rng)
            i2 = torneo(pop, fits, K_TORNEO, rng)
            h1, h2 = cruzamiento(pop[i1], pop[i2], rng)
            h1 = mutacion(h1, p_mut, rng)
            h2 = mutacion(h2, p_mut, rng)
            nueva_pop.extend([h1, h2])

        pop  = np.array(nueva_pop[:POP_SIZE], dtype=np.int8)
        fits = np.array([fitness_fn(ind, beneficios, costos) for ind in pop])

        if fits.max() >= mejor_fit:
            mejor_fit = float(fits.max())
            mejor_ind = pop[np.argmax(fits)].copy()
        else:
            pop[np.argmin(fits)]  = mejor_ind
            fits[np.argmin(fits)] = mejor_fit

        hist_mejor.append(mejor_fit)
        hist_prom.append(float(fits.mean()))

    return mejor_ind, mejor_fit, hist_mejor, hist_prom


# ══════════════════════════════════════════════════════════════════════════════
# 5. EXPERIMENTOS
# ══════════════════════════════════════════════════════════════════════════════

def correr_experimentos(df):
    beneficios = df["beneficio"].values
    costos     = df["costo"].values
    resultados = {}

    for p_mut in MUT_RATES:
        print(f"\n{'─'*50}")
        print(f"  Tasa de mutación: {p_mut}")
        print(f"{'─'*50}")
        runs = []
        for run in range(N_RUNS):
            ind, fit, hm, hp = ejecutar_ag(beneficios, costos, p_mut, SEED_BASE + run * 100)
            c_used = costos[ind.astype(bool)].sum()
            runs.append({"ind": ind, "fit": fit, "hm": hm, "hp": hp, "costo": c_used, "n_sel": int(ind.sum())})
            print(f"  Run {run+1}: fitness={fit:.4f} | municipios={ind.sum():3d} | costo=${c_used:>12,.0f}")

        mejor_run = max(range(N_RUNS), key=lambda i: runs[i]["fit"])
        resultados[p_mut] = {"runs": runs, "mejor_run": mejor_run}

    return resultados


# ══════════════════════════════════════════════════════════════════════════════
# 6. GRÁFICA DE CONVERGENCIA
# ══════════════════════════════════════════════════════════════════════════════

def graficar(resultados):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Convergencia del AG — Protección Civil Occidente de México",
                 fontsize=12, fontweight="bold")
    palette = {0.01: "#1a6faf", 0.05: "#c0392b"}

    for ax, p_mut in zip(axes, MUT_RATES):
        runs = resultados[p_mut]["runs"]
        arr_mejor = np.array([r["hm"] for r in runs])
        arr_prom  = np.array([r["hp"] for r in runs])

        for r in runs:
            ax.plot(r["hm"], alpha=0.25, linewidth=0.9, color=palette[p_mut])

        ax.plot(arr_mejor.mean(axis=0), linewidth=2.2, color=palette[p_mut],
                label=f"Mejor (prom. {N_RUNS} corridas)")
        ax.plot(arr_prom.mean(axis=0), linewidth=1.5, color="gray",
                linestyle="--", alpha=0.7, label="Fitness promedio población")
        ax.fill_between(range(N_GEN), arr_mejor.min(axis=0), arr_mejor.max(axis=0),
                        alpha=0.12, color=palette[p_mut], label="Rango min-max")

        ax.set_title(f"Tasa de mutación = {p_mut}", fontsize=11)
        ax.set_xlabel("Generación")
        ax.set_ylabel("Fitness")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "convergencia.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ convergencia.png guardado")


# ══════════════════════════════════════════════════════════════════════════════
# 7. EXPORTAR RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

def exportar_resultados(df, resultados):
    costos = df["costo"].values

    # Resumen por tasa (2 filas)
    resumen = []
    for p_mut, data in resultados.items():
        fits = [r["fit"] for r in data["runs"]]
        mr   = data["mejor_run"]
        resumen.append({
            "Tasa mutación":     p_mut,
            "Mejor fitness":     round(max(fits), 4),
            "Media fitness":     round(float(np.mean(fits)), 4),
            "SD fitness":        round(float(np.std(fits)), 4),
            "Municipios":        data["runs"][mr]["n_sel"],
            "Presupuesto usado": data["runs"][mr]["costo"],
            "Factible":          "Sí" if data["runs"][mr]["costo"] <= PRESUPUESTO else "No",
        })
    df_resumen = pd.DataFrame(resumen)
    print("\n── Resumen por tasa de mutación ──")
    print(df_resumen.to_string(index=False))

    # Mejor solución global
    mejor_pmut = max(resultados, key=lambda k: resultados[k]["runs"][resultados[k]["mejor_run"]]["fit"])
    mr_idx     = resultados[mejor_pmut]["mejor_run"]
    ind_mejor  = resultados[mejor_pmut]["runs"][mr_idx]["ind"]
    sel        = ind_mejor.astype(bool)

    df_sol = df[sel][["estado","municipio","zona_sismica_CFE",
                       "grado_peligro_inundacion_CENAPRED",
                       "grado_vulnerabilidad_social_CENAPRED",
                       "tiene_ARM","beneficio","costo"]].copy()
    df_sol = df_sol.sort_values("beneficio", ascending=False).reset_index(drop=True)
    df_sol.index += 1

    print(f"\n── Mejor solución (p_mut={mejor_pmut}) ──")
    print(f"  Municipios: {sel.sum()} de {len(df)}")
    print(f"  Presupuesto usado: ${costos[sel].sum():,.0f} / ${PRESUPUESTO:,.0f}")
    print(f"  Fitness: {resultados[mejor_pmut]['runs'][mr_idx]['fit']:.4f}")
    print(f"\n  Por estado:")
    print(df[sel]["estado"].value_counts().to_string())

    # Guardar CSV
    df_resumen.to_csv(OUT / "tabla_comparativa.csv", index=False, encoding="utf-8")
    df_sol.to_csv(OUT / "mejor_solucion.csv", index=False, encoding="utf-8")
    print(f"\n✓ tabla_comparativa.csv guardado")
    print(f"✓ mejor_solucion.csv guardado")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 50)
    print("  AG — Protección Civil Occidente de México")
    print("═" * 50)

    df = cargar_datos(DATA)
    print(f"\n✓ Dataset: {len(df)} municipios cargados")

    resultados = correr_experimentos(df)
    graficar(resultados)
    exportar_resultados(df, resultados)

    print("\n✓ Listo.")