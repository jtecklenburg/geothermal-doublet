import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expi  # Exponentialintegral


def theis_W(u: float) -> float:
    """
    Theis well function W(u) computed with SciPy:
    W(u) = ∫_u^∞ e^{-t}/t dt = -Ei(-u)
    """
    return -expi(-u)


# --- Injection-well pressure -----------------------------------

def pressure_injection_theis(
    t_array,
    Q,
    T,
    S,
    rw,
    rho,
    g,
    p_res,
    skin=0.0,
    scale_loss_grad=0.0,
    a_doublet=None,
):
    """
    Time-dependent bottom-hole pressure p_w(t) in a confined aquifer
    (Theis solution with optional Darcy skin, linear scaling loss, and doublet correction).

    Single-well form:

        p_w(t) = p_res + rho*g*[-Q/(4*pi*T) * W(u_w)]

        u_w = rw^2 * S / (4*T*t),   W(u) = -Ei(-u)

    Optional doublet correction (if a_doublet is provided):

        p_w,doublet(t) = p_res
                        + rho*g*[-Q/(4*pi*T) * (W(u_w) - W(u_extr))]

        u_extr = (2*a_doublet)^2 * S / (4*T*t)

    The term W(u_extr) subtracts the pressure perturbation of the extraction
    well at center-to-center distance 2*a_doublet.

    Parameters
    ---------
    t_array : 1D-Array
        Times [s] since start of injection.
    Q : float
        Volumetric well rate [m^3/s].
        Sign convention: Q < 0 for injection, Q > 0 for pumping.
    T : float
        Transmissivity [m^2/s].
    S : float
        Storativity [-].
    rw : float
        Well radius [m].
    rho : float
        Density of injected fluid [kg/m^3].
    g : float
        Gravitational acceleration [m/s^2].
    p_res : float
        Undisturbed reservoir pressure [Pa].
    skin : float, optional
        Dimensionless skin factor (Darcy skin, >0 = additional loss).
    scale_loss_grad : float, optional
        Additional linear scaling-loss coefficient [Pa /(m^3/s)].
        p_scale = scale_loss_grad * Q.
        Use scale_loss_grad > 0 for physically meaningful additional losses;
        for injection (Q < 0) this increases the required well pressure.
    a_doublet : float or None, optional
        Half the distance between injection and extraction wells [m].
        If given, the doublet superposition is applied: the pressure
        perturbation of the extraction well (at distance 2*a_doublet)
        is subtracted from the injection-well pressure.  For t -> inf
        the result converges to the Thiem doublet steady-state pressure.
        Default is None (single-well, no doublet correction).

    Returns
    --------
    p_w : 1D-Array
        Bottom-hole pressure [Pa] at times t_array.
    """
    t_array = np.asarray(t_array, dtype=float)
    t_safe = np.maximum(t_array, 1e-6)  # avoid t=0
    r = rw

    # Theis parameter u
    u = (r**2 * S) / (4.0 * T * t_safe)

    # Theis well function W(u) — vectorized
    W_vals = -expi(-u)

    # Drawdown s(r,t) for pumping; with Q < 0 this yields pressure buildup for injection
    coeff = Q / (4.0 * np.pi * T)     # [m] per W(u)
    s_pump = coeff * W_vals            # drawdown for extraction

    # For injection: pressure buildup head = -s_pump
    delta_h = -s_pump

    # Darcy skin as additional local head loss
    s_skin = coeff * skin             # head loss [m]
    delta_h_eff = delta_h - s_skin    # effective buildup head

    # Convert head to pressure
    delta_p = rho * g * delta_h_eff   # [Pa]

    # Additional linear scaling pressure loss
    p_scale = scale_loss_grad * Q     # [Pa]

    # Doublet correction: subtract pressure perturbation of extraction well at r = 2*a_doublet
    if a_doublet is not None:
        u_extr = (2.0 * a_doublet) ** 2 * S / (4.0 * T * t_safe)
        W_extr = -expi(-u_extr)
        delta_p_extr = rho * g * (-coeff * W_extr)   # same sign logic as above
        delta_p = delta_p - delta_p_extr

    p_w = p_res + delta_p - p_scale
    return p_w


def transmissivity(b, k, rho, g, mu):
    """
    Compute hydraulic transmissivity T of a confined aquifer.

    Parameters
    ---------
    b : float
        Aquifer thickness [m].
    k : float
        Intrinsic permeability [m^2].
    rho : float
        Fluid density [kg/m^3].
    g : float
        Gravitational acceleration [m/s^2].
    mu : float
        Dynamic viscosity [Pa s].

    Returns
    --------
    float
        Transmissivity T [m^2/s].
    """
    return k * rho * g / mu * b


def thiem_doublet_pressure(
    Q: float,
    k: float,
    M: float,
    mu: float,
    a: float,
    r_w: float,
    p_res: float = 0.0,
) -> float:
    """
    Steady-state injection-well pressure for a doublet (Thiem solution,
    potential theory, S = 0).

    Assumes two wells (injection + extraction) at distance 2*a apart in an
    infinite, homogeneous, confined aquifer.  Regional groundwater flow is
    neglected.  The result is time-independent (steady state):

        p_w = p_res + Q * mu / (2 * pi * k * M) * ln(2*a / r_w)

    Why 2*a appears:
    the center-to-center spacing between injection and extraction well is 2*a.
    By superposition of source and sink in potential theory:

        Delta p_inj = Q*mu/(2*pi*k*M) * ln(R_inf/r_w)
        Delta p_ext = Q*mu/(2*pi*k*M) * ln(R_inf/(2*a))

        Delta p = Delta p_inj - Delta p_ext
                = Q*mu/(2*pi*k*M) * ln((2*a)/r_w)

    The far-field radius R_inf cancels, leaving only the geometric ratio
    (2*a)/r_w.

    Parameters
    ----------
    Q : float
        Volumetric injection rate [m^3/s].  Positive for injection.
    k : float
        Intrinsic permeability [m^2].
    M : float
        Aquifer thickness [m].
    mu : float
        Dynamic viscosity of the fluid [Pa·s].
    a : float
        Half the distance between the two wells [m].
    r_w : float
        Wellbore radius [m].
    p_res : float, optional
        Undisturbed reservoir pressure [Pa].  Default is 0.

    Returns
    -------
    float
        Steady-state injection wellbore pressure [Pa].

    References
    ----------
    Charbeneau, R.J. (2000). Groundwater Hydraulics and Pollutant Transport.
        Prentice Hall. (Thiem equation for doublet, Chapter 3.)
    Schulz, R. (1987). Analytical model calculations for heat exchange in a
        confined aquifer. Journal of Geophysics, 61, 12-20.
    """
    return p_res + Q * mu / (2.0 * np.pi * k * M) * np.log(2.0 * a / r_w)


import numpy as np
from scipy.special import exp1  # W(u) = -Ei(-u) = exp1(u)


def radial_composite_pressure(
    t: np.ndarray,
    Q: float,
    M: float,
    k: float,
    phi: float,
    ct: float,
    mu1: float,
    mu2: float,
    R: float,
    r_w: float,
    p0: float = 0.0,
    a_doublet: float | None = None,
) -> np.ndarray:
    """
    Compute the injection well pressure for a radial composite model with a
    moving thermal front.

    The model assumes two radial zones separated by a sharp thermal front at
    radius r_f(t):

    - Zone 1  (r_w <= r < r_f):  cold injected water,  viscosity mu1
    - Zone 2  (r_f <= r < inf):  warm formation water, viscosity mu2

    The thermal front advances as a piston displacement:

        r_f(t) = sqrt(Q * t / (pi * M * phi * R))

    Zone 2 is described by the Theis solution at the wellbore radius r_w.
    Zone 1 is approximated as quasi-stationary between r_w and r_f(t),
    yielding a logarithmic viscosity-contrast correction.

    The resulting wellbore pressure is:

        p_w(t) = p0
                 + Q / (4 * pi * M * k)
                   * [ mu2 * W(r_w^2 / (4 * eta2 * t))
                       + (mu1 - mu2) * ln(r_f(t) / r_w) ]

    with

        r_f(t) = sqrt(Q*t / (pi*M*phi*R)),   eta2 = k / (phi*mu2*ct)

    Optional doublet correction (if a_doublet is provided):

        p_w,doublet(t) = p0
                        + Q / (4 * pi * M * k)
                          * [ mu2 * W(r_w^2 / (4 * eta2 * t))
                              + (mu1 - mu2) * ln(r_f(t) / r_w)
                              - mu2 * W((2*a_doublet)^2 / (4 * eta2 * t)) ]

    For mu1 == mu2, the viscosity-contrast term vanishes and the model reduces
    to the Theis-based form with (optional) extraction-well correction.

    Parameters
    ----------
    t : np.ndarray
        Time values [s].  Must be strictly positive.
    Q : float
        Volumetric injection rate [m^3/s].
    M : float
        Aquifer thickness [m].
    k : float
        Permeability [m^2].
    phi : float
        Porosity [-].
    ct : float
        Total compressibility of the fluid-saturated porous medium [1/Pa].
    mu1 : float
        Dynamic viscosity of the injected (cold) water [Pa·s].
    mu2 : float
        Dynamic viscosity of the formation (warm) water [Pa·s].
    R : float
        Thermal retardation factor [-]:
            R = (rho_A * c_A) / (rho_F * c_F)
        where rho_A*c_A is the volumetric heat capacity of the saturated
        aquifer and rho_F*c_F that of the fluid.
    r_w : float
        Wellbore radius [m].
    p0 : float, optional
        Initial (undisturbed) reservoir pressure [Pa].  Default is 0.
    a_doublet : float or None, optional
        Half the distance between injection and extraction wells [m].
        If given, the pressure contribution of the extraction well
        (evaluated at r = 2*a_doublet with Zone-2 diffusivity) is
        subtracted to obtain the correct doublet wellbore pressure.
        Default is None (single-well, no doublet correction).

    Returns
    -------
    p_w : np.ndarray
        Wellbore pressure [Pa] at each time in *t*.

    References
    ----------
    Benson, S.M. & Bodvarsson, G.S. (1986). Nonisothermal effects during
        injection and falloff tests. Water Resources Research, 22(5), 702-712.
    Tsang, C.F. & Tsang, Y.W. (1978). A study of the effects of thermal
        loading on the permeability of saturated rock. Lawrence Berkeley
        Laboratory Report LBL-7216.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.logspace(4, 9, 200)          # 10^4 to 10^9 seconds
    >>> p = radial_composite_pressure(
    ...     t=t, Q=0.03, M=30.0, k=1e-12,
    ...     phi=0.2, ct=1e-9, mu1=5e-4,
    ...     mu2=8e-4, R=1.5, r_w=0.1
    ... )
    """
    t = np.asarray(t, dtype=float)

    # Hydraulic diffusivity of Zone 2 (warm formation water)
    eta2 = k / (phi * mu2 * ct)

    # Zone 2: Theis solution evaluated at wellbore with Zone-2 diffusivity (time-dependent)
    u_w2 = r_w**2 / (4.0 * eta2 * t)
    theis_zone2 = mu2 * exp1(u_w2)

    # Moving thermal front radius [m] (piston-displacement approximation)
    r_f = np.sqrt(Q * t / (np.pi * M * phi * R))

    # Zone-1 viscosity-contrast correction (quasi-steady inner zone)
    viscosity_correction = (mu1 - mu2) * np.log(r_f / r_w)

    prefactor = Q / (4.0 * np.pi * M * k)

    # Doublet correction: subtract Zone-2 Theis term of extraction well at r = 2*a_doublet.
    # The extraction well is always in the undisturbed formation (Zone 2), so mu2 is used.
    doublet_correction = 0.0
    if a_doublet is not None:
        u_extr = (2.0 * a_doublet) ** 2 / (4.0 * eta2 * t)
        doublet_correction = mu2 * exp1(u_extr)

    p_w = p0 + prefactor * (theis_zone2 + viscosity_correction - doublet_correction)

    return p_w


# --- Example plot ----------------------------------------------

if __name__ == "__main__":
    # Example parameters (pumping scenario: Q > 0)
    Q   = 0.01       # m^3/s  (positive = pumping)
    T   = 1e-2       # m^2/s
    S   = 1e-4       # -
    rw  = 0.1        # m
    rho = 1000.0     # kg/m^3
    g   = 9.81       # m/s^2
    p_res = 1.0e6    # Pa (~10 bar)
    skin = 5.0       # dimensionless skin
    scale_loss_grad = 1e6  # Pa /(m^3/s), example scaling-loss value

    # Times from 10 s to ~10 days
    t = np.logspace(1, 6, 100)  # [s]

    p_w = pressure_injection_theis(
        t, Q, T, S, rw, rho, g, p_res,
        skin=skin,
        scale_loss_grad=scale_loss_grad,
    )

    plt.figure(figsize=(7, 4))
    plt.semilogx(t / 3600.0, p_w / 1e5)
    plt.xlabel("Time [h]")
    plt.ylabel("Bottom-hole pressure $p_w$ [bar]")
    plt.title("Time-dependent injection-well pressure\n(Theis + Skin + Scaling)")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()
