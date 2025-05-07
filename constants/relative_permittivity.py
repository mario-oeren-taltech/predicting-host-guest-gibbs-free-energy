from enum import Enum


class RelativePermittivity(Enum):
    """
    The relative permittivity (also known as dielectric constant) is the permittivity of a material expressed as a
    ratio with the electric permittivity of a vacuum.
    """

    ACETONE = 20.493  # https://gaussian.com/scrf/
    ACETONITRILE = 35.6880  # https://gaussian.com/scrf/
    CHLOROFORM = 4.7113  # https://gaussian.com/scrf/
    DICHLOROMETHANE = 8.9300  # https://gaussian.com/scrf/
    DIMETHYLSULFOXIDE = 46.8260  # https://gaussian.com/scrf/
    ETHANOL = 24.852  # https://gaussian.com/scrf/
    ETHYL_ACETATE = 5.9867  # https://gaussian.com/scrf/
    METHANOL = 32.6130  # https://gaussian.com/scrf/
    TOLUENE = 2.3741  # https://gaussian.com/scrf/
    WATER = 78.3553  # https://gaussian.com/scrf/
