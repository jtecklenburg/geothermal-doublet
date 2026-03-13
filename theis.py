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
):
    """
    Time-dependent bottom-hole pressure p_w(t) in a confined aquifer
    (Theis solution + Darcy skin + linear scaling loss).

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
