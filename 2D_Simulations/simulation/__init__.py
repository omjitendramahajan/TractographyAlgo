# Simulation package for 2D and 3D tractography experiments

# 2D simulation (original)
from .Agent import HybridAgent
from .DataGenerator import GroundTruth
from .ComparisonSimulation import ComparisonSimulation, FODOnlyAgent
from .SeedSimulation import SeedToTargetSimulation
from .CrossingFiberDemo import run_crossing_fiber_comparison

# 3D simulation (new)
from .GroundTruth3D import GroundTruth3D
from .SimulatedData import (
    SimulatedBedpostxData,
    SimulatedPSOCTData,
    create_simulation_environment
)
