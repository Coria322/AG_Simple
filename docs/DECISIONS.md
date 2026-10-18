# DECISIONS.md — Resolución de ambigüedades del ERS

> Fecha: 2026-10-18 · Revisadas una por una con el usuario.
> En el código, cada decisión se referencia con un comentario `# AMB-XXX`.

| ID | Requerimiento | Decisión confirmada |
|----|---------------|---------------------|
| AMB-001 | RF-04 | Por defecto `NEGATE_PLUS_C`: aptitud = −f + C, con C = máximo f(x) de la generación actual. `INVERSE_SHIFT` (1/(1+f−f_min)) queda como estrategia alternativa intercambiable. |
| AMB-002 | RF-05/06 | Bits por dimensión en [1, 64]. Estrategias `SATURATION` (recorte al límite) y `REJECTION_RESAMPLE` (regenerar con tope de reintentos y evento en bitácora). |
| AMB-003 | RF-11 | GUI con 4 opciones (Binaria, Gray, Real, Permutación). En JSON, Gray se serializa como `chromosome_type=BINARY` + `encoding=GRAY`. |
| AMB-004 | RF-15 | En binario/Gray: *Uniforme* = flip de cada bit con prob. Pm; *No uniforme* = Pm efectiva decreciente con t/Gmax (exponente b). |
| AMB-005 | RF-16 | Un esquema principal (Generacional / Estado estable / Crowding) + elitismo E opcional combinable. `sampling` del JSON se ignora si existe `selection.method`. |
| AMB-006 | RF-11 | Permutación de tamaño N; fenotipo = valores de un vector de dominio ordenado reasignados según la permutación; se evalúa con la misma f(x). Sin benchmark combinatorio nuevo. |
| AMB-007 | RF-20/23 | **El CSV guarda todos los parámetros de todos los individuos**: genotipo, fenotipo, f(x) crudo, aptitud transformada e `is_elite`. *(Detalle de columnas a confirmar en TASK-043.)* |
| AMB-008 | RF-22 | Campo *Semilla* opcional en la GUI; vacío = aleatoria. Se exporta en `genetic_parameters.seed`. |
| AMB-009 | RF-09/18 | Gmax siempre obligatorio; el estancamiento es criterio adicional. |
| AMB-010 | RNF-03/RF-13 | Individuos penalizados (±∞) reciben probabilidad de selección 0. Si toda la población está penalizada: selección uniforme + `ERROR` en bitácora. |
| AMB-011 | RF-05 | N ≥ 1 sin tope fijo (tabla de dominios con scroll). |
| AMB-012 | RF-13/14/16 | Ranking lineal con presión s ∈ [1, 2] (default 1.5). Cruce Plano = BLX-α con α = 0. Crowding: el descendiente reemplaza al más similar solo si es mejor. |
