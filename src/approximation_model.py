
from numpy.typing import ArrayLike, NDArray
from costum_typing import  GradModel,DrawAndLogP, Seed, VectorType
from typing import Iterator, Optional
class ApproximationModel():


    def dims(self) -> int:
        """number of parameters"""
        ...  # pragma: no cover

    def log_density(self, params_unc: VectorType) -> float:
            """unnormalized log density"""
            ...  # pragma: no cover

    def sample( self, n: int,
                seed: Optional[Seed] = None) -> VectorType[DrawAndLogP]:
        """sample from the approximation"""
        ...  # pragma: no cover