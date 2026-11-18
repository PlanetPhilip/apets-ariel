from .cpg import RevolveCPG
from .neighborhood_cpg import NeighborhoodCPG
from .ABCpg import ABCpg, SymmetricalABCPG
from .na_cpg import NaCPGController
from .real_na_cpg import RealNaCPGController

__all__ = [
    RevolveCPG,
    NeighborhoodCPG,
    ABCpg, SymmetricalABCPG,
    NaCPGController, RealNaCPGController,
]

try:
    from .mlp_tensor import MLPTensorBrain
    __all__ += [MLPTensorBrain]
except ModuleNotFoundError:
    pass


__dict__ = {
    **{c.__name__.lower(): c for c in __all__},
    **{c.name(): c for c in __all__}
}


def get(name):
    if (controller := __dict__.get(name.lower())) is None:
        raise ValueError(f"No known controller of type {name}")
    return controller
