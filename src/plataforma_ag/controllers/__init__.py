"""Capa de controladores MVC (coordinación entre UI y Core/Services)."""

from plataforma_ag.controllers.params_controller import ParamsController
from plataforma_ag.controllers.persistence_controller import PersistenceController
from plataforma_ag.controllers.problem_controller import ProblemController
from plataforma_ag.controllers.run_controller import RunController
from plataforma_ag.controllers.telemetry_controller import TelemetryController

__all__ = [
    "ParamsController",
    "PersistenceController",
    "ProblemController",
    "RunController",
    "TelemetryController",
]
