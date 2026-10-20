"""Servicio de generación de reportes en PDF con ReportLab (RF-24, TASK-044).

Estructura del reporte técnico (4 secciones obligatorias):
  1. Ficha técnica: parámetros del problema, hiperparámetros, operadores y parada.
  2. Gráficas de evolución cartesiana: curva de convergencia renderizada a 300 DPI.
  3. Cuadro de la mejor solución encontrada: genotipo, fenotipo, fitness crudo.
  4. Extracto de la generación final: muestra de los mejores individuos.

Maneja excepciones lanzando PersistenceError ante problemas de escritura o permisos.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Backend no interactivo sin ventanas
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from plataforma_ag.domain.config import ExperimentConfig
from plataforma_ag.domain.enums import ObjectiveSense
from plataforma_ag.domain.errors import PersistenceError, ValidationError
from plataforma_ag.services.history import RunHistoryStore


def _generate_convergence_chart_image(
    history: RunHistoryStore, dpi: int = 300
) -> io.BytesIO:
    """Genera la figura de convergencia y la devuelve como BytesIO en PNG a 300 DPI."""
    series = history.get_convergence_series()
    gens = np.asarray(series.generations)
    best_f = np.asarray(series.best_fitness)
    mean_f = np.asarray(series.mean_fitness)
    elite_f = np.asarray(series.cumulative_elite)

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=dpi)

    # Mascaras de valores finitos
    finite_best = np.isfinite(best_f)
    finite_mean = np.isfinite(mean_f)
    finite_elite = np.isfinite(elite_f)

    if np.any(finite_best):
        ax.plot(
            gens[finite_best],
            best_f[finite_best],
            label="Mejor local",
            color="#2563eb",  # Azul corporativo
            linewidth=1.5,
            alpha=0.8,
        )

    if np.any(finite_mean):
        ax.plot(
            gens[finite_mean],
            mean_f[finite_mean],
            label="Promedio población",
            color="#059669",  # Verde esmeralda
            linewidth=1.2,
            linestyle="--",
            alpha=0.7,
        )

    if np.any(finite_elite):
        ax.plot(
            gens[finite_elite],
            elite_f[finite_elite],
            label="Élite histórico acumulado",
            color="#dc2626",  # Rojo oscuro
            linewidth=2.0,
        )

    ax.set_title(
        "Curvas de Convergencia del Algoritmo Genético", fontsize=11, fontweight="bold", pad=8
    )
    ax.set_xlabel("Generaciones", fontsize=9)
    ax.set_ylabel(f"Fitness ({history.sense.value})", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


class PdfReportService:
    """Generador de reportes técnicos de optimización en formato PDF."""

    @staticmethod
    def generate_report(
        config: ExperimentConfig,
        history: RunHistoryStore,
        output_path: Path | str,
        max_final_sample: int = 15,
    ) -> Path:
        """Crea el documento PDF con las cuatro secciones estipuladas por RF-24.

        Raises:
            ValidationError: Si el historial no contiene generaciones.
            PersistenceError: Si falla la creación o escritura del archivo.
        """
        if history.total_generations == 0:
            raise ValidationError(
                "No hay registros de generaciones en el historial para generar el reporte"
            )

        path = Path(output_path)

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            doc = SimpleDocTemplate(
                str(path),
                pagesize=letter,
                rightMargin=36,
                leftMargin=36,
                topMargin=36,
                bottomMargin=36,
            )

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                "DocTitle",
                parent=styles["Heading1"],
                fontSize=18,
                leading=22,
                textColor=colors.HexColor("#0f172a"),
                spaceAfter=4,
            )
            section_style = ParagraphStyle(
                "SectionHeading",
                parent=styles["Heading2"],
                fontSize=13,
                leading=16,
                textColor=colors.HexColor("#1e3a8a"),
                spaceBefore=10,
                spaceAfter=4,
            )
            body_style = ParagraphStyle(
                "BodyTextSmall",
                parent=styles["Normal"],
                fontSize=9,
                leading=12,
                textColor=colors.HexColor("#334155"),
            )

            story: list[Any] = []

            # --- Encabezado del documento ---
            story.append(
                Paragraph(
                    "Reporte de Simulación y Optimización con Algoritmos Genéticos", title_style
                )
            )
            story.append(
                Paragraph(
                    f"Problema: <b>{config.problem.benchmark.value}</b> | "
                    f"Sentido: <b>{config.problem.objective.value}</b> | "
                    f"Generaciones ejecutadas: <b>{history.total_generations}</b>",
                    body_style,
                )
            )
            story.append(Spacer(1, 6))
            story.append(
                HRFlowable(
                    width="100%", thickness=1.5, color=colors.HexColor("#cbd5e1"), spaceAfter=8
                )
            )

            # --- SECCIÓN 1: Ficha Técnica ---
            story.append(Paragraph("1. Ficha Técnica del Experimento", section_style))

            bounds_str = ", ".join(
                f"{b.variable}∈[{b.min}, {b.max}]" for b in config.problem.bounds[:4]
            )
            if len(config.problem.bounds) > 4:
                bounds_str += f"... (+{len(config.problem.bounds) - 4} más)"

            tech_data = [
                ["Parámetro", "Configuración", "Parámetro", "Configuración"],
                [
                    "Expresión / Benchmark",
                    config.problem.expression or config.problem.benchmark.value,
                    "Tamaño población (Np)",
                    str(config.genetic.population_size),
                ],
                [
                    "Dimensiones (N)",
                    str(config.problem.dimensions),
                    "Tipo de cromosoma",
                    config.genetic.chromosome_type.value,
                ],
                [
                    "Límites de variables",
                    bounds_str,
                    "Prob. cruce (Pc) / mutación (Pm)",
                    f"{config.genetic.p_crossover} / {config.genetic.p_mutation}",
                ],
                [
                    "Método de selección",
                    config.operators.selection_method.value,
                    "Método de cruce",
                    config.operators.crossover_method.value,
                ],
                [
                    "Método de mutación",
                    config.operators.mutation_method.value,
                    "Esquema de reemplazo",
                    (
                        f"{config.operators.replacement_method.value} "
                        f"(E: {config.operators.elitism_count or 0})"
                    ),
                ],
                [
                    "Máximo generaciones (Gmax)",
                    str(config.stopping.max_generations),
                    "Criterio estancamiento",
                    "Habilitado" if config.stopping.stagnation_enabled else "Deshabilitado",
                ],
                [
                    "Semilla PRNG",
                    str(config.genetic.seed if config.genetic.seed is not None else "Aleatoria"),
                    "Estrategia espacio red.",
                    config.problem.redundant_space_strategy.value,
                ],
            ]

            t_tech = Table(tech_data, colWidths=[1.8 * inch, 2.0 * inch, 1.8 * inch, 1.8 * inch])
            t_tech.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                        ("FONTNAME", (2, 1), (2, -1), "Helvetica-Bold"),
                    ]
                )
            )
            story.append(t_tech)
            story.append(Spacer(1, 10))

            # --- SECCIÓN 2: Gráficas de Evolución Cartesiana (300 DPI) ---
            story.append(Paragraph("2. Curvas de Evolución Cartesiana (300 DPI)", section_style))
            chart_stream = _generate_convergence_chart_image(history, dpi=300)
            story.append(Image(chart_stream, width=6.8 * inch, height=3.4 * inch))
            story.append(Spacer(1, 10))

            # --- SECCIÓN 3: Cuadro de la Mejor Solución Encontrada ---
            story.append(Paragraph("3. Mejor Solución Global Encontrada", section_style))
            best_ind = history.global_best
            if best_ind is not None:
                pheno_str = "[" + ", ".join(f"{float(x):.6f}" for x in best_ind.phenotype) + "]"
                geno_preview = str(best_ind.genotype)
                if len(geno_preview) > 60:
                    geno_preview = geno_preview[:57] + "..."

                best_data = [
                    ["Atributo", "Valor"],
                    ["ID del individuo", str(best_ind.individual_id)],
                    ["Generación en que apareció", str(history.global_best_generation)],
                    [
                        "Fitness crudo f(x)",
                        f"{best_ind.fitness:.8f}" if best_ind.fitness is not None else "nan",
                    ],
                    ["Fenotipo (coordenadas x1..xN)", pheno_str],
                    ["Genotipo", geno_preview],
                ]
            else:
                best_data = [["Atributo", "Valor"], ["Estado", "No se encontró solución válida"]]

            t_best = Table(best_data, colWidths=[2.2 * inch, 5.2 * inch])
            t_best.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1e1b4b")),
                        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(t_best)
            story.append(Spacer(1, 10))

            # --- SECCIÓN 4: Extracto de la Generación Final ---
            last_gen = history.available_generations[-1]
            story.append(
                Paragraph(f"4. Extracto de la Generación Final (Gen {last_gen})", section_style)
            )

            rec_final = history.get_generation(last_gen)
            if rec_final is not None and rec_final.individuals:
                # Ordenar individuos de la última generación según el sentido de optimización
                sorted_inds = sorted(
                    rec_final.individuals,
                    key=lambda ind: (
                        ind.fitness
                        if ind.fitness is not None and math.isfinite(ind.fitness)
                        else (
                            float("inf")
                            if history.sense is ObjectiveSense.MINIMIZE
                            else float("-inf")
                        )
                    ),
                    reverse=(history.sense is ObjectiveSense.MAXIMIZE),
                )
                sample = sorted_inds[:max_final_sample]

                final_table_data = [["ID", "Fenotipo (x1..xN)", "Fitness f(x)", "¿Es Élite?"]]
                for ind in sample:
                    p_str = "[" + ", ".join(f"{float(x):.4g}" for x in ind.phenotype[:4])
                    if len(ind.phenotype) > 4:
                        p_str += ", ..."
                    p_str += "]"
                    fit_val = f"{ind.fitness:.6g}" if ind.fitness is not None else "nan"
                    elite_val = "★ Sí" if ind.is_elite else "No"
                    final_table_data.append(
                        [str(ind.individual_id), p_str, fit_val, elite_val]
                    )

                t_final = Table(
                    final_table_data, colWidths=[0.8 * inch, 4.4 * inch, 1.4 * inch, 0.8 * inch]
                )
                t_final.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 8),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                            ("TOPPADDING", (0, 0), (-1, -1), 2),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                            ("ALIGN", (0, 0), (0, -1), "CENTER"),
                            ("ALIGN", (2, 0), (-1, -1), "CENTER"),
                        ]
                    )
                )
                story.append(t_final)
            else:
                story.append(Paragraph("Sin datos en la última generación", body_style))

            # Construir el documento PDF
            doc.build(story)
            return path

        except OSError as err:
            raise PersistenceError(
                f"No se pudo escribir el reporte PDF en '{path}': {err}"
            ) from err
