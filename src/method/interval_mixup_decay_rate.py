from enum import Enum
import math

class MixupEpsilonDecayRateType(Enum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    LOG = "log"
    COS = "cos"


class MixupEpsilonDecayRate:
    """
    MixupEpsilonDecayRate provides various decay rate functions for Interval MixUp epsilon scheduling.
    This class allows selection of different decay rate strategies (linear, quadratic, logarithmic, cosine)
    to compute a decay value based on the Interval MixUp interpolation parameter `alpha`. The decay rate type is
    specified as a string during initialization.

    Args:
        decay_rate_type (str): The type of decay rate to use. Must be one of: "linear", "quadratic", "log", "cos".

    Attributes:
        decay_rate_type (MixupEpsilonDecayRateType): The selected decay rate type as an enum.
        decay_rate_fnc (dict): Mapping from decay rate type enum to the corresponding function.

    Raises:
        ValueError: If an invalid decay_rate_type is provided.

    Methods:
        linear(alpha: float) -> float:
            Computes the linear decay rate for a given alpha.
        quadratic(alpha: float) -> float:
            Computes the quadratic decay rate for a given alpha.
        log(alpha: float) -> float:
            Computes the logarithmic decay rate for a given alpha.
        cos(alpha: float) -> float:
            Computes the cosine decay rate for a given alpha.
        __call__(alpha: float) -> float:
            Computes the decay rate for the given alpha using the selected decay rate function.
    """

    def __init__(self, decay_rate_type: str) -> None:
        """
        Initializes the decay rate handler with the specified decay rate type.
        Args:
            decay_rate_type (str): The type of decay rate to use. Supported types are mapped internally.
        Raises:
            ValueError: If the provided decay_rate_type is not supported.
        Attributes:
            decay_rate_type: The mapped decay rate type.
            decay_rate_fnc (dict): A dictionary mapping decay rate types to their corresponding functions.
        """

        super().__init__()

        self.decay_rate_type = self._map_to_decay_rate_type(decay_rate_type)

        self.decay_rate_fnc = {
            MixupEpsilonDecayRateType.LINEAR: self.linear,
            MixupEpsilonDecayRateType.QUADRATIC: self.quadratic,
            MixupEpsilonDecayRateType.LOG: self.log,
            MixupEpsilonDecayRateType.COS: self.cos
        }

    def _map_to_decay_rate_type(self, decay_rate_type: str) -> MixupEpsilonDecayRateType:
        """
        Maps a string identifier of the decay rate type to the corresponding MixupEpsilonDecayRateType enum value.
        Args:
            decay_rate_type (str): The type of decay rate as a string. 
                Supported values are "linear", "quadratic", "log", and "cos".
        Returns:
            MixupEpsilonDecayRateType: The corresponding enum value for the specified decay rate type.
        Raises:
            ValueError: If the provided decay_rate_type is not one of the supported types.
        """
        
        if decay_rate_type == "linear":
            return MixupEpsilonDecayRateType.LINEAR
        elif decay_rate_type == "quadratic":
            return MixupEpsilonDecayRateType.QUADRATIC
        elif decay_rate_type == "log":
            return MixupEpsilonDecayRateType.LOG
        elif decay_rate_type == "cos":
            return MixupEpsilonDecayRateType.COS
        else:
            raise ValueError("Invalid Interval MixUp epsilon decay.")
        
    def linear(self, alpha: float) -> float:
        """
        The linear decay is defined as abs(2 * alpha - 1.0), which produces a V-shaped curve
        with a minimum at alpha = 0.5 and maximum at alpha = 0 or 1.
        Args:
            alpha (float): The interpolation parameter, typically in the range [0, 1].
        Returns:
            float: The computed linear decay rate.
        """
        return abs(2*alpha-1.0)
    
    def quadratic(self, alpha: float) -> float:
        """
        The quadratic decay is defined as 4 * (alpha - 0.5) ** 2, which produces a parabola
        with a minimum at alpha = 0.5 and maximum at alpha = 0 or 1.
        Args:
            alpha (float): The interpolation parameter, typically in the range [0, 1].
        Returns:float: The computed quadratic decay rate.
        """
        return 4*(alpha-0.5)**2
    
    def log(self, alpha: float) -> float:
        """
        The logarithmic decay is defined as a scaled logarithm, symmetric around alpha = 0.5:
        - For alpha <= 0.5: const * log(alpha + 0.5),
        - For alpha > 0.5:  const * log(3/2 - alpha),
        where const = 1 / log(0.5), ensuring the output is normalized.
        Args:
            alpha (float): The interpolation parameter, typically in the range [0, 1].
        
        Returns:
            float: The computed logarithmic decay rate.
        """
        const = 1 / math.log(0.5)

        if alpha <= 0.5:
            return const * math.log(alpha+0.5) 
        elif alpha > 0.5:
            return const * math.log(3/2-alpha)
        
    def cos(self, alpha: float) -> float:
        """
        The cosine decay is defined as abs(cos(pi * alpha)), which produces a symmetric curve
        with maximum at alpha = 0 and 1, and minimum at alpha = 0.5.
        Args:
            alpha (float): The interpolation parameter, typically in the range [0, 1].
        Returns:
            float: The computed cosine decay rate.
        """
        return abs(math.cos(math.pi * alpha))

    def __call__(self, alpha: float) -> float:
        
        loss_fn = self.decay_rate_fnc[self.decay_rate_type]
        return loss_fn(alpha)
