"""Motor de evolución del Algoritmo Genético (RF-17, RNF-01, RF-07..RF-16).

Ejecuta el ciclo generacional como un generador paso a paso:
  1. Selección de padres según la aptitud interna.
  2. Cruce probabilístico (Pc).
  3. Mutación probabilística (Pm) con contexto (t, Gmax, límites).
  4. Decodificación a fenotipos y evaluación de aptitud de la descendencia.
  5. Esquema de reemplazo y preservación de élite.
  6. Cálculo de estadísticas y evaluación de criterios de parada.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field

import numpy as np

from plataforma_ag.core.encoding.codecs import ChromosomeCodec, Genotype, create_codec
from plataforma_ag.core.engine.population import initialize_population
from plataforma_ag.core.engine.stopping import StoppingEvaluator
from plataforma_ag.core.operators import (
    CrossoverOperator,
    MutationContext,
    MutationOperator,
    Population,
    ReplacementContext,
    ReplacementScheme,
    SelectionOperator,
    operator_registry,
    validate_operator_config,
)
from plataforma_ag.core.operators.base import resolve_params
from plataforma_ag.core.problem.expression import compile_expression
from plataforma_ag.core.problem.objective import ObjectiveFunction
from plataforma_ag.domain.bounds import bounds_to_arrays
from plataforma_ag.domain.config import ExperimentConfig
from plataforma_ag.domain.enums import OperatorKind, StopReason
from plataforma_ag.domain.errors import ValidationError
from plataforma_ag.domain.run import GenerationRecord, Individual


@dataclass(slots=True)
class GAEngine:
    """Motor del algoritmo genético desacoplado de hilos y de la interfaz gráfica."""

    config: ExperimentConfig
    objective: ObjectiveFunction
    codec: ChromosomeCodec
    selection: SelectionOperator
    crossover: CrossoverOperator
    mutation: MutationOperator
    replacement: ReplacementScheme
    stopping: StoppingEvaluator
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    # Estado mutable interno durante la corrida
    current_generation: int = 0
    cumulative_elite_fitness: float = float("nan")
    best_fitness_history: list[float] = field(default_factory=list)

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        objective: ObjectiveFunction | None = None,
        rng: np.random.Generator | None = None,
    ) -> GAEngine:
        """Construye un GAEngine completo a partir de un ExperimentConfig."""
        generator = (
            rng
            if rng is not None
            else (
                np.random.default_rng(config.genetic.seed)
                if config.genetic.seed is not None
                else np.random.default_rng()
            )
        )

        # 1. Validar compatibilidad de operadores
        validate_operator_config(config.genetic.chromosome_type, config.operators)

        # 2. Objetivo si no se proveyó
        if objective is None:
            expr_text = config.problem.expression or "x1^2 + x2^2"
            evaluator = compile_expression(expr_text, config.problem.dimensions)
            objective = ObjectiveFunction(
                evaluator=evaluator,
                sense=config.problem.objective,
                transform=config.problem.fitness_transform,
            )

        # 3. Codec
        codec = create_codec(
            config.genetic.chromosome_type,
            config.problem.bounds,
            config.problem.redundant_space_strategy,
        )

        # 4. Operadores desde el registro
        sel_op = operator_registry.get(
            OperatorKind.SELECTION, config.operators.selection_method.value
        )
        cross_op = operator_registry.get(
            OperatorKind.CROSSOVER, config.operators.crossover_method.value
        )
        mut_op = operator_registry.get(
            OperatorKind.MUTATION, config.operators.mutation_method.value
        )
        rep_op = operator_registry.get(
            OperatorKind.REPLACEMENT, config.operators.replacement_method.value
        )

        if not isinstance(sel_op, SelectionOperator):
            raise ValidationError(f"Operador de selección inválido: {sel_op}")
        if not isinstance(cross_op, CrossoverOperator):
            raise ValidationError(f"Operador de cruce inválido: {cross_op}")
        if not isinstance(mut_op, MutationOperator):
            raise ValidationError(f"Operador de mutación inválido: {mut_op}")
        if not isinstance(rep_op, ReplacementScheme):
            raise ValidationError(f"Esquema de reemplazo inválido: {rep_op}")

        # 5. Criterios de parada
        stopping = StoppingEvaluator(config.stopping)

        return cls(
            config=config,
            objective=objective,
            codec=codec,
            selection=sel_op,
            crossover=cross_op,
            mutation=mut_op,
            replacement=rep_op,
            stopping=stopping,
            rng=generator,
        )

    def run(
        self,
    ) -> Generator[tuple[GenerationRecord, tuple[StopReason, ...] | None], None, None]:
        """Generador que produce (instantánea, razones_de_parada) en cada generación."""
        pop_size = self.config.genetic.population_size
        lower, upper, _ = bounds_to_arrays(self.codec.bounds)
        p_cross = self.config.genetic.p_crossover
        max_gen = self.config.stopping.max_generations

        # Resolver parámetros de operadores con esquemas
        sel_params = resolve_params(
            self.selection.params_schema, self.config.operators.selection_params
        )
        cross_params = resolve_params(
            self.crossover.params_schema, self.config.operators.crossover_params
        )
        mut_params = resolve_params(
            self.mutation.params_schema, self.config.operators.mutation_params
        )
        rep_params_dict: dict[str, int | float] = {}
        if self.config.operators.elitism_count is not None:
            rep_params_dict["elitism_count"] = self.config.operators.elitism_count
        if self.config.operators.n_replace is not None:
            rep_params_dict["n_replace"] = self.config.operators.n_replace
        if self.config.operators.crowding_factor is not None:
            rep_params_dict["crowding_factor"] = self.config.operators.crowding_factor
        rep_params = resolve_params(self.replacement.params_schema, rep_params_dict)

        # --- Generación 0: Inicialización factible ---
        genotypes = initialize_population(self.codec, pop_size, self.rng)
        phenotypes = self.codec.decode(genotypes)
        raw_fitness, aptitude = self.objective.evaluate(phenotypes)

        record, stop_reasons = self._record_generation(
            0, genotypes, phenotypes, raw_fitness, aptitude
        )
        yield record, stop_reasons
        if stop_reasons is not None:
            return

        # --- Ciclo generacional t >= 1 ---
        for t in range(1, max_gen + 1):
            self.current_generation = t

            # 1. Selección de Np padres
            parent_indices = self.selection.select(aptitude, n=pop_size, rng=self.rng, **sel_params)
            parents = genotypes[parent_indices]

            # 2. Cruce probabilístico por parejas
            offspring_list: list[Genotype] = []
            for i in range(0, pop_size, 2):
                p1 = parents[i]
                if i + 1 < pop_size:
                    p2 = parents[i + 1]
                    if self.rng.random() < p_cross:
                        c1, c2 = self.crossover.cross(p1, p2, self.rng, **cross_params)
                    else:
                        c1, c2 = p1.copy(), p2.copy()
                    offspring_list.extend([c1, c2])
                else:
                    # Descendiente impar
                    offspring_list.append(p1.copy())

            offspring_genotypes = np.array(offspring_list[:pop_size])

            # 3. Mutación probabilística con contexto
            mut_ctx = MutationContext(
                generation=t,
                max_generations=max_gen,
                p_mutation=self.config.genetic.p_mutation,
                lower=lower,
                upper=upper,
            )
            mutated_genotypes = np.empty_like(offspring_genotypes)
            for i in range(pop_size):
                mutated_genotypes[i] = self.mutation.mutate(
                    offspring_genotypes[i], ctx=mut_ctx, rng=self.rng, **mut_params
                )

            # 4. Decodificar y evaluar descendencia
            off_phenotypes = self.codec.decode(mutated_genotypes)
            _off_raw, off_aptitude = self.objective.evaluate(off_phenotypes)

            # 5. Esquema de reemplazo
            rep_ctx = ReplacementContext(generation=t, rng=self.rng, params=rep_params)
            current_pop = Population(genotypes=genotypes, aptitude=aptitude)
            off_pop = Population(genotypes=mutated_genotypes, aptitude=off_aptitude)

            next_population = self.replacement.apply(current_pop, off_pop, rep_ctx)
            genotypes = next_population.genotypes

            # 6. Actualizar fenotipos y aptitud de la población resultante
            phenotypes = self.codec.decode(genotypes)
            raw_fitness, aptitude = self.objective.evaluate(phenotypes)

            # 7. Registrar estadísticas y verificar criterios de parada
            record, stop_reasons = self._record_generation(
                t, genotypes, phenotypes, raw_fitness, aptitude
            )
            yield record, stop_reasons
            if stop_reasons is not None:
                return

    def _record_generation(
        self,
        generation: int,
        genotypes: Genotype,
        phenotypes: np.ndarray,
        raw_fitness: np.ndarray,
        aptitude: np.ndarray,
    ) -> tuple[GenerationRecord, tuple[StopReason, ...] | None]:
        """Calcula estadísticas, crea objetos Individual y evalúa parada."""
        ranked_indices = self.objective.rank_indices(raw_fitness)
        best_idx = ranked_indices[0]
        best_f = float(raw_fitness[best_idx])

        # Actualizar élite histórico acumulado
        if not np.isfinite(self.cumulative_elite_fitness) or self.objective.is_better(
            best_f, self.cumulative_elite_fitness
        ):
            self.cumulative_elite_fitness = best_f

        # Media de valores finitos
        finite_mask = np.isfinite(raw_fitness)
        mean_f = float(np.mean(raw_fitness[finite_mask])) if np.any(finite_mask) else float("nan")

        self.best_fitness_history.append(best_f)

        # Crear lista de objetos Individual
        individuals: list[Individual] = []
        for i in range(len(genotypes)):
            is_elite = i == best_idx
            # Tuple de genes y fenotipo
            geno_tuple = tuple(genotypes[i].tolist())
            pheno_tuple = tuple(phenotypes[i].tolist())
            ind = Individual(
                individual_id=i,
                genotype=geno_tuple,
                phenotype=pheno_tuple,
                fitness=float(raw_fitness[i]),
                aptitude=float(aptitude[i]),
                is_elite=is_elite,
            )
            individuals.append(ind)

        record = GenerationRecord(
            generation=generation,
            best_fitness=best_f,
            mean_fitness=mean_f,
            elite_fitness_cumulative=self.cumulative_elite_fitness,
            individuals=tuple(individuals),
        )

        # Evaluar criterios de parada
        should_stop, reasons = self.stopping.evaluate(generation, self.best_fitness_history)
        stop_reasons = reasons if should_stop else None

        return record, stop_reasons
