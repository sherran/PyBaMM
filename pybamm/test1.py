#
# Basic Doyle-Fuller-Newman (DFN) Model
#
import pybamm
import sys
import numpy as np
sys.path.insert(0,"../..")
print(sys.path)
from pybamm.models.full_battery_models.lithium_ion.base_lithium_ion_model import BaseModel


class BasicDFNTest(BaseModel):
    """Doyle-Fuller-Newman (DFN) model of a lithium-ion battery, from [2]_.

    This class differs from the :class:`pybamm.lithium_ion.DFN` model class in that it
    shows the whole model in a single class. This comes at the cost of flexibility in
    comparing different physical effects, and in general the main DFN class should be
    used instead.

    Parameters
    ----------
    name : str, optional
        The name of the model.

    References
    ----------
    .. [2] SG Marquis, V Sulzer, R Timms, CP Please and SJ Chapman. “An asymptotic
           derivation of a single particle model with electrolyte”. Journal of The
           Electrochemical Society, 166(15):A3693–A3706, 2019

    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(self, name="Doyle-Fuller-Newman model"):
        super().__init__({}, name)
        pybamm.citations.register("Marquis2019")
        # `param` is a class containing all the relevant parameters and functions for
        # this model. These are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.
        param = self.param

        ######################
        # Variables
        ######################
        # Variables that depend on time only are created without a domain
        Q = pybamm.Variable("Discharge capacity [A.h]")
        # Variables that vary spatially are created with a domain
        c_e_n = pybamm.Variable(
            "Negative electrolyte concentration", domain="negative electrode"
        )
        c_e_s = pybamm.Variable(
            "Separator electrolyte concentration", domain="separator"
        )
        c_e_p = pybamm.Variable(
            "Positive electrolyte concentration", domain="positive electrode"
        )
        # Concatenations combine several variables into a single variable, to simplify
        # implementing equations that hold over several domains
        c_e = pybamm.concatenation(c_e_n, c_e_s, c_e_p)

        # Electrolyte potential
        phi_e_n = pybamm.Variable(
            "Negative electrolyte potential", domain="negative electrode"
        )
        phi_e_s = pybamm.Variable("Separator electrolyte potential", domain="separator")
        phi_e_p = pybamm.Variable(
            "Positive electrolyte potential", domain="positive electrode"
        )
        phi_e = pybamm.concatenation(phi_e_n, phi_e_s, phi_e_p)

        # Electrode potential
        phi_s_n = pybamm.Variable(
            "Negative electrode potential", domain="negative electrode"
        )
        phi_s_p = pybamm.Variable(
            "Positive electrode potential", domain="positive electrode"
        )
        # Particle concentrations are variables on the particle domain, but also vary in
        # the x-direction (electrode domain) and so must be provided with auxiliary
        # domains
        c_s_n = pybamm.Variable(
            "Negative particle concentration",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        c_s_p = pybamm.Variable(
            "Positive particle concentration",
            domain="positive particle",
            auxiliary_domains={"secondary": "positive electrode"},
        )

        j_n = pybamm.Variable("Negative electrode current density")
        j0_n = pybamm.Variable("Negative electrode exchange current density")
        j_p = pybamm.Variable("Positive electrode current density")
        j0_p = pybamm.Variable("Positive electrode exchange current density")


        # Constant temperature
        T = param.T_init

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = param.current_with_time

        # Porosity
        # Primary broadcasts are used to broadcast scalar quantities across a domain
        # into a vector of the right shape, for multiplying with other vectors
        eps_n = pybamm.PrimaryBroadcast(
            pybamm.Parameter("Negative electrode porosity"), "negative electrode"
        )
        eps_s = pybamm.PrimaryBroadcast(
            pybamm.Parameter("Separator porosity"), "separator"
        )
        eps_p = pybamm.PrimaryBroadcast(
            pybamm.Parameter("Positive electrode porosity"), "positive electrode"
        )
        eps = pybamm.concatenation(eps_n, eps_s, eps_p)

        # Active material volume fraction (eps + eps_s + eps_inactive = 1)
        eps_s_n = pybamm.Parameter("Negative electrode active material volume fraction")
        eps_s_p = pybamm.Parameter("Positive electrode active material volume fraction")

        # transport_efficiency
        tor = pybamm.concatenation(
            eps_n**param.n.b_e, eps_s**param.s.b_e, eps_p**param.p.b_e
        )

        # Interfacial reactions
        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
                # SEI parameter

        U_inner_dimensional = pybamm.Parameter("Inner SEI open-circuit potential [V]")
        kappa_inner_dimensional = pybamm.Parameter("Inner SEI electron conductivity [S.m-1]")

        L_inner_0_dim = pybamm.Parameter("Initial inner SEI thickness [m]")
        L_outer_0_dim = pybamm.Parameter("Initial outer SEI thickness [m]")
        V_bar_inner_dimensional = pybamm.Parameter("Inner SEI partial molar volume [m3.mol-1]")
        z_sei = pybamm.Parameter("Ratio of lithium moles to SEI moles")
        j_scale = param.i_typ

        L_sei_0_dim = L_inner_0_dim + L_outer_0_dim


        C_sei_electron = (
            j_scale
            * param.F
            * L_sei_0_dim
            / (kappa_inner_dimensional * param.R * param.T_ref)
        )
        
        U_inner_electron = (
                param.F * U_inner_dimensional / param.R / param.T_ref
            )
        
        L_sei_inner = pybamm.Variable("Inner SEI thickness", domain="negative electrode")
        j_inner = pybamm.Variable("Inner SEI interfacial current density")
        
        U_inner = U_inner_electron
        C_sei = C_sei_electron
        j_sei = (phi_s_n - U_inner) / (C_sei * L_sei_inner)

        Gamma_SEI = (
                V_bar_inner_dimensional * j_scale * param.timescale
            ) / (param.F * z_sei * L_sei_0_dim)
        
                
        # All SEI growth mechanisms assumed to have Arrhenius dependence
        E_sei_dimensional = pybamm.Parameter("SEI growth activation energy [J.mol-1]")
        E_over_RT_sei = E_sei_dimensional / param.R / param.T_ref
        inner_sei_proportion = pybamm.Parameter("Inner SEI reaction proportion")
        
        Arrhenius = pybamm.exp(E_over_RT_sei)

        j_inner = inner_sei_proportion * Arrhenius * j_sei

        L_inner_0 = L_inner_0_dim / L_sei_0_dim


        self.rhs[L_sei_inner] = -Gamma_SEI * j_inner
        self.initial_conditions[L_sei_inner] = L_inner_0

        Q_sei = param.A_cc * j_inner / param.F * param.timescale / 3600

        c_s_surf_n = pybamm.surf(c_s_n)
        j0_n = param.n.prim.j0(c_e_n, c_s_surf_n, T)
        j_n = (
            2
            * j0_n
            * pybamm.sinh(
                param.n.prim.ne
                / 2
                * (phi_s_n - phi_e_n - param.n.prim.U(c_s_surf_n, T) - U_inner)
            )
        )
        c_s_surf_p = pybamm.surf(c_s_p)
        j0_p = param.p.prim.j0(c_e_p, c_s_surf_p, T)
        j_s = pybamm.PrimaryBroadcast(0, "separator")
        j_p = (
            2
            * j0_p
            * pybamm.sinh(
                param.p.prim.ne
                / 2
                * (phi_s_p - phi_e_p - param.p.prim.U(c_s_surf_p, T))
            )
        )
        j = pybamm.concatenation(j_n, j_s, j_p)

        # L_sei_inner = pybamm.Variable("Inner SEI thickness")

        # U_inner = param.U_inner_electron
        # C_sei = param.C_sei_electron
        # j_sei = (phi_s_n - U_inner) / (C_sei * L_sei_inner)

        ######################
        # State of Charge
        ######################
        I = param.dimensional_current_with_time
        # The `rhs` dictionary contains differential equations, with the key being the
        # variable in the d/dt
        self.rhs[Q] = I * param.timescale / 3600
        # Initial conditions must be provided for the ODEs
        self.initial_conditions[Q] = pybamm.Scalar(0)

        Q_neg = Q - Q_sei


        ######################
        # Particles
        ######################

        # The div and grad operators will be converted to the appropriate matrix
        # multiplication at the discretisation stage
        N_s_n = -param.n.prim.D(c_s_n, T) * pybamm.grad(c_s_n)
        N_s_p = -param.p.prim.D(c_s_p, T) * pybamm.grad(c_s_p)
        self.rhs[c_s_n] = -(1 / param.n.prim.C_diff) * pybamm.div(N_s_n)
        self.rhs[c_s_p] = -(1 / param.p.prim.C_diff) * pybamm.div(N_s_p)
        # Boundary conditions must be provided for equations with spatial derivatives
        self.boundary_conditions[c_s_n] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -param.n.prim.C_diff
                * j_n
                / param.n.prim.a_R
                / param.n.prim.gamma
                / param.n.prim.D(c_s_surf_n, T),
                "Neumann",
            ),
        }
        self.boundary_conditions[c_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -param.p.prim.C_diff
                * j_p
                / param.p.prim.a_R
                / param.p.prim.gamma
                / param.p.prim.D(c_s_surf_p, T),
                "Neumann",
            ),
        }
        self.initial_conditions[c_s_n] = param.n.prim.c_init
        self.initial_conditions[c_s_p] = param.p.prim.c_init
        ######################
        # Current in the solid
        ######################
        sigma_eff_n = param.n.sigma(T) * eps_s_n**param.n.b_s
        i_s_n = -sigma_eff_n * pybamm.grad(phi_s_n)
        sigma_eff_p = param.p.sigma(T) * eps_s_p**param.p.b_s
        i_s_p = -sigma_eff_p * pybamm.grad(phi_s_p)
        # The `algebraic` dictionary contains differential equations, with the key being
        # the main scalar variable of interest in the equation
        self.algebraic[phi_s_n] = pybamm.div(i_s_n) + j_n
        self.algebraic[phi_s_p] = pybamm.div(i_s_p) + j_p
        self.boundary_conditions[phi_s_n] = {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.boundary_conditions[phi_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (i_cell / pybamm.boundary_value(-sigma_eff_p, "right"), "Neumann"),
        }
        # Initial conditions must also be provided for algebraic equations, as an
        # initial guess for a root-finding algorithm which calculates consistent initial
        # conditions
        # We evaluate c_n_init at r=0, x=0 and c_p_init at r=0, x=1
        # (this is just an initial guess so actual value is not too important)
        self.initial_conditions[phi_s_n] = pybamm.Scalar(0)
        self.initial_conditions[phi_s_p] = param.ocv_init

        ######################
        # Current in the electrolyte
        ######################
        i_e = (param.kappa_e(c_e, T) * tor * param.gamma_e / param.C_e) * (
            param.chi(c_e, T)/c_e * pybamm.grad(c_e) - pybamm.grad(phi_e)
        )
        self.algebraic[phi_e] = pybamm.div(i_e) - j
        self.boundary_conditions[phi_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[phi_e] = -param.n.prim.U_init

        ######################
        # Electrolyte concentration
        ######################
        N_e = -tor * param.D_e(c_e, T) * pybamm.grad(c_e)
        self.rhs[c_e] = (1 / eps) * (
            -pybamm.div(N_e) / param.C_e
            + (1 - param.t_plus(c_e, T)) * j / param.gamma_e
        )
        self.boundary_conditions[c_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[c_e] = param.c_e_init

        ######################
        # (Some) variables
        ######################
        voltage = pybamm.boundary_value(phi_s_p, "right")
        pot_scale = param.potential_scale
        U_ref = param.ocv_ref
        voltage_dim = U_ref + voltage * pot_scale
        c_s_n_rav = pybamm.r_average(c_s_n)
        c_s_n_vol_av = pybamm.x_average(eps_s_n * c_s_n_rav)
        c_s_p_rav = pybamm.r_average(c_s_p)
        c_s_p_vol_av = pybamm.x_average(eps_s_p * c_s_p_rav)
        c_e_total = pybamm.x_average(eps * c_e)
        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        self.variables = {
            "Negative particle surface concentration": c_s_surf_n,
            "Negative particle concentration": c_s_n,
            "Positive particle concentration": c_s_p,
            "Electrolyte concentration": c_e,
            "Positive particle surface concentration": c_s_surf_p,
            "Current [A]": I,
            "Current [mA]": I*1000,
            "Negative electrode potential": phi_s_n,
            "Electrolyte potential": phi_e,
            "Positive electrode potential": phi_s_p,
            "Positive electrolyte concentration": c_e_p,
            "Negative electrolyte concentration": c_e_n,
            "Separator electrolyte concentration": c_e_s,
            "Positive electrolyte potential": phi_e_p,
            "Negative electrolyte potential": phi_e_n,
            "Separator electrolyte potential": phi_e_s,
            "Terminal voltage": voltage,
            "Terminal voltage [V]": voltage_dim,
            "Time [s]": pybamm.t * self.param.timescale,
            "Time [h]": pybamm.t * self.param.timescale / 3600,
            "Negative electrode current density": j_n,
            "Negative electrode exchange current density": j0_n,
            "Positive electrode current density": j_p,
            "Positive electrode exchange current density": j0_p,
            "Discharge capacity [A.h]": Q,
            "Inner SEI thickness": L_sei_inner,
            "Throughput capacity [A.h]": I*voltage_dim*pybamm.t * self.param.timescale / 3600,
            "Loss of capacity to SEI [A.h]": Q_sei,
            "Negative electrode capacity [A.h]": j_n*param.A_cc*pybamm.t * self.param.timescale / 3600,
            "Positive electrode capacity [A.h]": j_p*param.A_cc*pybamm.t * self.param.timescale / 3600,
            "Total lithium in particles [mol]": c_s_p_vol_av*param.c_max*param.p.L*param.A_cc + c_s_n_vol_av
            * param.c_max
            * param.n.L
            * param.A_cc,
            "Total lithium in positive electrode": c_s_p_vol_av,
            "Total lithium in positive electrode [mol]": c_s_p_vol_av
            * param.c_max
            * param.p.L
            * param.A_cc,
            "Total lithium in negative electrode": c_s_n_vol_av,
            "Total lithium in negative electrode [mol]": c_s_n_vol_av
            * param.c_max
            * param.n.L
            * param.A_cc,
        }
        #Events specify points at which a solution should terminate
        self.events += [
            pybamm.Event("Minimum voltage", voltage - param.voltage_low_cut),
            pybamm.Event("Maximum voltage", param.voltage_high_cut - voltage),
        ]

        self.summary_variables = [
            "Time [s]",
            "Time [h]",
            "Throughput capacity [A.h]",
            # "Throughput energy [W.h]",
            # # LAM, LLI
            # "Loss of lithium inventory [%]",
            # "Loss of lithium inventory, including electrolyte [%]",
            # # Total lithium
            # "Total lithium [mol]",
            # "Total lithium in electrolyte [mol]",
            # "Total lithium in particles [mol]",
            # # Lithium lost
            # "Total lithium lost [mol]",
            # "Total lithium lost from particles [mol]",
            # "Total lithium lost from electrolyte [mol]",
            # "Loss of lithium to SEI [mol]",
            "Loss of capacity to SEI [A.h]",
            # "Total lithium lost to side reactions [mol]",
            # "Total capacity lost to side reactions [A.h]",
            # # Resistance
            # "Local ECM resistance [Ohm]",
        ]

        # self.summary_variables += [
        #         "Negative electrode capacity [A.h]",
        #         "Loss of active material in negative electrode [%]",
        #         "Total lithium in negative electrode [mol]",
        #         "Loss of lithium to lithium plating [mol]",
        #         "Loss of capacity to lithium plating [A.h]",
        #         "Loss of lithium to SEI on cracks [mol]",
        #         "Loss of capacity to SEI on cracks [A.h]",
        #     ]
        
        # self.summary_variables += [
        #         "Positive electrode capacity [A.h]",
        #         "Loss of active material in positive electrode [%]",
        #         "Total lithium in positive electrode [mol]",
        #     ]


model = BasicDFNTest()

def graphite_mcmb2528_diffusivity_Dualfoil1998(sto, T):
    """
    Graphite MCMB 2528 diffusivity as a function of stochiometry, in this case the
    diffusivity is taken to be a constant. The value is taken from Dualfoil [1].

    References
    ----------
    .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_ref = 3.9 * 10 ** (-14)
    E_D_s = 42770
    arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def graphite_mcmb2528_ocp_Dualfoil1998(sto):
    """
    Graphite MCMB 2528 Open Circuit Potential (OCP) as a function of the
    stochiometry. The fit is taken from Dualfoil [1]. Dualfoil states that the data
    was measured by Chris Bogatu at Telcordia and PolyStor materials, 2000. However,
    we could not find any other records of this measurment.

    References
    ----------
    .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html
    """

    u_eq = (
        0.194
        + 1.5 * pybamm.exp(-120.0 * sto)
        + 0.0351 * pybamm.tanh((sto - 0.286) / 0.083)
        - 0.0045 * pybamm.tanh((sto - 0.849) / 0.119)
        - 0.035 * pybamm.tanh((sto - 0.9233) / 0.05)
        - 0.0147 * pybamm.tanh((sto - 0.5) / 0.034)
        - 0.102 * pybamm.tanh((sto - 0.194) / 0.142)
        - 0.022 * pybamm.tanh((sto - 0.9) / 0.0164)
        - 0.011 * pybamm.tanh((sto - 0.124) / 0.0226)
        + 0.0155 * pybamm.tanh((sto - 0.105) / 0.029)
    )

    return u_eq


def graphite_electrolyte_exchange_current_density_Dualfoil1998(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref = 2 * 10 ** (-5)  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 37480
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )


def graphite_entropic_change_Moura2016(sto, c_s_max):
    """
    Graphite entropic change in open circuit potential (OCP) at a temperature of
    298.15K as a function of the stochiometry taken from Scott Moura's FastDFN code
    [1].

    References
    ----------
    .. [1] https://github.com/scott-moura/fastDFN

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Stochiometry of material (li-fraction)

    """
    du_dT = (
        -1.5 * (120.0 / c_s_max) * pybamm.exp(-120 * sto)
        + (0.0351 / (0.083 * c_s_max)) * ((pybamm.cosh((sto - 0.286) / 0.083)) ** (-2))
        - (0.0045 / (0.119 * c_s_max)) * ((pybamm.cosh((sto - 0.849) / 0.119)) ** (-2))
        - (0.035 / (0.05 * c_s_max)) * ((pybamm.cosh((sto - 0.9233) / 0.05)) ** (-2))
        - (0.0147 / (0.034 * c_s_max)) * ((pybamm.cosh((sto - 0.5) / 0.034)) ** (-2))
        - (0.102 / (0.142 * c_s_max)) * ((pybamm.cosh((sto - 0.194) / 0.142)) ** (-2))
        - (0.022 / (0.0164 * c_s_max)) * ((pybamm.cosh((sto - 0.9) / 0.0164)) ** (-2))
        - (0.011 / (0.0226 * c_s_max)) * ((pybamm.cosh((sto - 0.124) / 0.0226)) ** (-2))
        + (0.0155 / (0.029 * c_s_max)) * ((pybamm.cosh((sto - 0.105) / 0.029)) ** (-2))
    )

    return du_dT


def lico2_diffusivity_Dualfoil1998(sto, T):
    """
    LiCo2 diffusivity as a function of stochiometry, in this case the
    diffusivity is taken to be a constant. The value is taken from Dualfoil [1].

    References
    ----------
    .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """
    D_ref = 1 * 10 ** (-13)
    E_D_s = 18550
    arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def lico2_ocp_Dualfoil1998(sto):
    """
    Lithium Cobalt Oxide (LiCO2) Open Circuit Potential (OCP) as a a function of the
    stochiometry. The fit is taken from Dualfoil [1]. Dualfoil states that the data
    was measured by Oscar Garcia 2001 using Quallion electrodes for 0.5 < sto < 0.99
    and by Marc Doyle for sto<0.4 (for unstated electrodes). We could not find any
    other records of the Garcia measurements. Doyles fits can be found in his
    thesis [2] but we could not find any other record of his measurments.

    References
    ----------
    .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html
    .. [2] CM Doyle. Design and simulation of lithium rechargeable batteries,
           1995.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """

    stretch = 1.062
    sto = stretch * sto

    u_eq = (
        2.16216
        + 0.07645 * pybamm.tanh(30.834 - 54.4806 * sto)
        + 2.1581 * pybamm.tanh(52.294 - 50.294 * sto)
        - 0.14169 * pybamm.tanh(11.0923 - 19.8543 * sto)
        + 0.2051 * pybamm.tanh(1.4684 - 5.4888 * sto)
        + 0.2531 * pybamm.tanh((-sto + 0.56478) / 0.1316)
        - 0.02167 * pybamm.tanh((sto - 0.525) / 0.006)
    )

    return u_eq


def lico2_electrolyte_exchange_current_density_Dualfoil1998(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between lico2 and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref = 6 * 10 ** (-7)  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 39570
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )


def lico2_entropic_change_Moura2016(sto, c_s_max):
    """
    Lithium Cobalt Oxide (LiCO2) entropic change in open circuit potential (OCP) at
    a temperature of 298.15K as a function of the stochiometry. The fit is taken
    from Scott Moura's FastDFN code [1].

    References
    ----------
    .. [1] https://github.com/scott-moura/fastDFN

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Stochiometry of material (li-fraction)
    """
    # Since the equation for LiCo2 from this ref. has the stretch factor,
    # should this too? If not, the "bumps" in the OCV don't line up.
    stretch = 1.062
    sto = stretch * sto

    du_dT = (
        0.07645
        * (-54.4806 / c_s_max)
        * ((1.0 / pybamm.cosh(30.834 - 54.4806 * sto)) ** 2)
        + 2.1581 * (-50.294 / c_s_max) * ((pybamm.cosh(52.294 - 50.294 * sto)) ** (-2))
        + 0.14169
        * (19.854 / c_s_max)
        * ((pybamm.cosh(11.0923 - 19.8543 * sto)) ** (-2))
        - 0.2051 * (5.4888 / c_s_max) * ((pybamm.cosh(1.4684 - 5.4888 * sto)) ** (-2))
        - (0.2531 / 0.1316 / c_s_max)
        * ((pybamm.cosh((-sto + 0.56478) / 0.1316)) ** (-2))
        - (0.02167 / 0.006 / c_s_max) * ((pybamm.cosh((sto - 0.525) / 0.006)) ** (-2))
    )

    return du_dT


def electrolyte_diffusivity_Capiglia1999(c_e, T):
    """
    Diffusivity of LiPF6 in EC:DMC as a function of ion concentration. The original data
    is from [1]. The fit from Dualfoil [2].

    References
    ----------
    .. [1] C Capiglia et al. 7Li and 19F diffusion coefficients and thermal
    properties of non-aqueous electrolyte solutions for rechargeable lithium batteries.
    Journal of power sources 81 (1999): 859-862.
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature


    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_c_e = 5.34e-10 * pybamm.exp(-0.65 * c_e / 1000)
    E_D_e = 37040
    arrhenius = pybamm.exp(E_D_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_c_e * arrhenius


def electrolyte_conductivity_Capiglia1999(c_e, T):
    """
    Conductivity of LiPF6 in EC:DMC as a function of ion concentration. The original
    data is from [1]. The fit is from Dualfoil [2].

    References
    ----------
    .. [1] C Capiglia et al. 7Li and 19F diffusion coefficients and thermal
    properties of non-aqueous electrolyte solutions for rechargeable lithium batteries.
    Journal of power sources 81 (1999): 859-862.
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature


    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    sigma_e = (
        0.0911
        + 1.9101 * (c_e / 1000)
        - 1.052 * (c_e / 1000) ** 2
        + 0.1554 * (c_e / 1000) ** 3
    )

    E_k_e = 34700
    arrhenius = pybamm.exp(E_k_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return sigma_e * arrhenius

geometry = model.default_geometry

param = model.default_parameter_values

param['Ratio of lithium moles to SEI moles'] = 2.0
param['Inner SEI reaction proportion'] = 0.5
param['Inner SEI partial molar volume [m3.mol-1]'] = 9.585e-05
param['Outer SEI partial molar volume [m3.mol-1]'] = 9.585e-05
param['SEI reaction exchange current density [A.m-2]'] = 1.5e-07
param['SEI resistivity [Ohm.m]'] = 200000.0
param['Outer SEI solvent diffusivity [m2.s-1]'] = 2.5000000000000002e-22
param['Bulk solvent concentration [mol.m-3]'] = 2636.0
param['Inner SEI open-circuit potential [V]'] = 0.1
param['Outer SEI open-circuit potential [V]'] = 0.8
param['Inner SEI electron conductivity [S.m-1]'] = 8.95e-14
param['Inner SEI lithium interstitial diffusivity [m2.s-1]'] = 1e-20
param['Lithium interstitial reference concentration [mol.m-3]'] = 15.0
param['Initial inner SEI thickness [m]'] = 2.5e-09
param['Initial outer SEI thickness [m]'] = 2.5e-09
param['EC initial concentration in electrolyte [mol.m-3]'] = 4541.0
param['EC diffusivity [m2.s-1]'] = 2e-18
param['SEI kinetic rate constant [m.s-1]'] = 1e-12
param['SEI open-circuit potential [V]'] = 0.4
param['SEI growth activation energy [J.mol-1]'] = 0.0
param['Negative electrode reaction-driven LAM factor [m3.mol-1]'] = 0.0
param['Positive electrode reaction-driven LAM factor [m3.mol-1]'] = 0.0
#cell
param['Negative current collector thickness [m]'] = 2.5e-05
param['Negative electrode thickness [m]'] = 0.0001
param['Separator thickness [m]'] = 2.5e-05
param['Positive electrode thickness [m]'] = 0.0001
param['Positive current collector thickness [m]'] = 2.5e-05
param['Electrode height [m]'] = 0.337
param['Electrode width [m]'] = 0.207
param['Negative tab width [m]'] = 0.04
param['Negative tab centre y-coordinate [m]'] = 0.06
param['Negative tab centre z-coordinate [m]'] = 0.137
param['Positive tab width [m]'] = 0.04
param['Positive tab centre y-coordinate [m]'] = 0.147
param['Positive tab centre z-coordinate [m]'] = 0.137
param['Cell cooling surface area [m2]'] = 0.0569
param['Cell volume [m3]'] = 0.0018
param['Negative current collector conductivity [S.m-1]'] = 59600000.0
param['Positive current collector conductivity [S.m-1]'] = 35500000.0
param['Negative current collector density [kg.m-3]'] = 8954.0
param['Positive current collector density [kg.m-3]'] = 2707.0
param['Negative current collector specific heat capacity [J.kg-1.K-1]'] = 385.0
param['Positive current collector specific heat capacity [J.kg-1.K-1]'] = 897.0
param['Negative current collector thermal conductivity [W.m-1.K-1]'] = 401.0
param['Positive current collector thermal conductivity [W.m-1.K-1]'] = 237.0
param['Nominal cell capacity [A.h]'] = 0.6
# param['Typical current [A]'] = 0.680616
# param['Current function [A]'] = current_interpolant
#negative electrode
param['Negative electrode conductivity [S.m-1]'] = 100.0
param['Maximum concentration in negative electrode [mol.m-3]'] = 24983.2619938437
param['Negative electrode diffusivity [m2.s-1]'] = graphite_mcmb2528_diffusivity_Dualfoil1998
param['Negative electrode OCP [V]'] = graphite_mcmb2528_ocp_Dualfoil1998
param['Negative electrode porosity'] = 0.3
param['Negative electrode active material volume fraction'] = 0.6
param['Negative particle radius [m]'] = 1e-05
param['Negative electrode Bruggeman coefficient (electrolyte)'] = 1.5
param['Negative electrode Bruggeman coefficient (electrode)'] = 1.5
param['Negative electrode cation signed stoichiometry'] = -1.0
param['Negative electrode electrons in reaction'] = 1.0
param['Negative electrode charge transfer coefficient'] = 0.5
param['Negative electrode double-layer capacity [F.m-2]'] = 0.2
param['Negative electrode exchange-current density [A.m-2]'] = graphite_electrolyte_exchange_current_density_Dualfoil1998
param['Negative electrode density [kg.m-3]'] = 1657.0
param['Negative electrode specific heat capacity [J.kg-1.K-1]'] = 700.0
param['Negative electrode thermal conductivity [W.m-1.K-1]'] = 1.7
param['Negative electrode OCP entropic change [V.K-1]'] = graphite_entropic_change_Moura2016
#positive electrode
param['Positive electrode conductivity [S.m-1]'] = 100.0
param['Maximum concentration in positive electrode [mol.m-3]'] = 24983.2619938437
param['Positive electrode diffusivity [m2.s-1]'] = graphite_mcmb2528_diffusivity_Dualfoil1998 
param['Positive electrode OCP [V]'] = graphite_mcmb2528_ocp_Dualfoil1998
param['Positive electrode porosity'] = 0.3
param['Positive electrode active material volume fraction'] = 0.6
param['Positive particle radius [m]'] = 1e-05
param['Positive electrode Bruggeman coefficient (electrolyte)'] = 1.5
param['Positive electrode Bruggeman coefficient (electrode)'] = 1.5
param['Positive electrode cation signed stoichiometry'] = -1.0
param['Positive electrode electrons in reaction'] = 1.0
param['Positive electrode charge transfer coefficient'] = 0.5
param['Positive electrode double-layer capacity [F.m-2]'] = 0.2
param['Positive electrode exchange-current density [A.m-2]'] = graphite_electrolyte_exchange_current_density_Dualfoil1998
param['Positive electrode density [kg.m-3]'] = 1657.0
param['Positive electrode specific heat capacity [J.kg-1.K-1]'] = 700.0
param['Positive electrode thermal conductivity [W.m-1.K-1]'] = 1.7
param['Positive electrode OCP entropic change [V.K-1]'] = graphite_entropic_change_Moura2016
#separator
param['Separator porosity'] = 1.0
param['Separator Bruggeman coefficient (electrolyte)'] = 1.5
param['Separator density [kg.m-3]'] = 397.0
param['Separator specific heat capacity [J.kg-1.K-1]'] = 700.0
param['Separator thermal conductivity [W.m-1.K-1]'] = 0.16
#electrolyte
param['Typical electrolyte concentration [mol.m-3]'] = 1000.0
param['Initial concentration in electrolyte [mol.m-3]'] = 1000.0
param['Cation transference number'] = 0.4
param['1 + dlnf/dlnc'] = 1.0
param['Electrolyte diffusivity [m2.s-1]'] = electrolyte_diffusivity_Capiglia1999
param['Electrolyte conductivity [S.m-1]'] = 0.1
#experiment
param['Reference temperature [K]'] = 298.15
param['Ambient temperature [K]'] = 298.15
param['Negative current collector surface heat transfer coefficient [W.m-2.K-1]'] = 0.0
param['Positive current collector surface heat transfer coefficient [W.m-2.K-1]'] = 0.0
param['Negative tab heat transfer coefficient [W.m-2.K-1]'] = 10.0
param['Positive tab heat transfer coefficient [W.m-2.K-1]'] = 10.0
param['Edge heat transfer coefficient [W.m-2.K-1]'] = 0.3
param['Total heat transfer coefficient [W.m-2.K-1]'] = 10.0
param['Number of electrodes connected in parallel to make a cell'] = 1.0
param['Number of cells connected in series to make a battery'] = 1.0
param['Lower voltage cut-off [V]'] = 0
param['Upper voltage cut-off [V]'] = 10
# param['Initial concentration in negative electrode [mol.m-3]'] = 19986.609595075
# param['Initial concentration in positive electrode [mol.m-3]'] = 30730.7554385565
param['Initial temperature [K]'] = 298.15
param["Current function [A]"] = 5e-4
param["Typical current [A]"] = 5e-4
param["Initial concentration in negative electrode [mol.m-3]"] = 24983
param["Initial concentration in positive electrode [mol.m-3]"] = 1 
param["Upper voltage cut-off [V]"] = 5




param.process_geometry(geometry)
param.process_model(model)

#experiment (cycling regime)
experiment = pybamm.Experiment(
    [
        (#"Rest for 8 hours",
         "Discharge at 0.5 mA until 0.01 V",)
         #"Rest for 0.5 hours",)
         #"Charge at 0.5 mA until 1.0 V",
        #"Rest for 30 minutes"),
    ] 
)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 3, var.x_s: 3, var.x_p: 3, var.r_n: 3, var.r_p: 3}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model, inplace=False) #inplace=False


# solve model
# t_eval = np.linspace(0, 7200, 100)
solver = pybamm.CasadiSolver(mode="safe", atol=1e-3, rtol=1e-3)
# solution = solver.solve(model, t_eval)

sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param, geometry=geometry, var_pts=var_pts)
solution = sim.solve()

plot = pybamm.QuickPlot(
    solution,
    [
        # 'Time [s]',
         'Positive particle surface concentration',
         'Negative particle surface concentration',
         
         'Electrolyte concentration',

         'Current [mA]',

         'Electrolyte potential',

         'Terminal voltage [V]',

         "Discharge capacity [A.h]",

    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()
# solution.solve()
# solution.plot(output_variables=[            
#             "Negative particle surface concentration",
#             "Electrolyte concentration",
#             "Positive particle surface concentration",
#             "Current [A]",
#             "Negative electrode potential",
#             "Electrolyte potential",
#             "Positive electrode potential",
#             "Terminal voltage",
#             "Battery voltage [V]",
#             "Time [s]",
#             "Negative electrode current density",
#             "Negative electrode exchange current density",
#             "Positive electrode current density",
#             "Positive electrode exchange current density",
#             "Discharge capacity [A.h]",
#             ])