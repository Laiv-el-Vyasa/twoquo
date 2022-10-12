class TooQuoException(Exception):
    """
    Base class for exceptions.
    """
    pass


class EmptyGraphError(TooQuoException):
    """Raised when the input Graph consists of zero nodes when it shouldn't."""
    pass


class BadProblemParametersError(TooQuoException):
    """Raised when the input parameters to a problem are invalid.

    For example a LongestPath problem with a 4-node graph where the terminal
    node is node #12.
    """
    pass