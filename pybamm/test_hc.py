#
# Basic Doyle-Fuller-Newman (DFN) Model
#
import pybamm
import sys
import numpy as np
sys.path.insert(0,"../..")
print(sys.path)
from pybamm.models.full_battery_models.lithium_ion.base_lithium_ion_model import BaseModel

class HalfCell(BaseModel):
    """Doyle-Fuller-Newman (DFN) model of a lithium-ion battery with lithium counter
    electrode, adapted from [2]_.

    This class differs from the :class:`pybamm.lithium_ion.BasicDFN` model class in
    that it is for a cell with a lithium counter electrode (half cell). This is a
    feature under development (for example, it cannot be used with the Experiment class
    for the moment) and in the future it will be incorporated as a standard model with
    the full functionality.

    The electrode labeled "positive electrode" is the working electrode, and the
    electrode labeled "negative electrode" is the counter electrode. If the "negative
    electrode" is the working electrode, then the parameters for the "negative
    electrode" are used to define the "positive electrode".
    This facilitates compatibility with the full-cell models.

    """

    def __init__(self, options=None, name="Doyle-Fuller-Newman half cell model"):
        super().__init__(options, name)
        if self.options["working electrode"] not in ["positive"]:
            raise ValueError(
                "The option 'working electrode' should be 'positive'"
            )
        pybamm.citations.register("Marquis2019")
        # `param` is a class containing all the relevant parameters and functions for
        # this model. These are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.
        param = self.param

        R_w_typ = param.p.prim.R_typ

        # Set default length scales
        self._length_scales = {
            "separator": param.L_x,
            "positive electrode": param.L_x,
            "positive particle": R_w_typ,
            "current collector y": param.L_z,
            "current collector z": param.L_z,
        }

        ######################
        # Variables
        ######################
        # Variables that depend on time only are created without a domain
        Q = pybamm.Variable("Discharge capacity [A.h]")

        # Define some useful scalings
        pot_scale = param.potential_scale
        i_typ = param.current_scale

        # Variables that vary spatially are created with a domain.
        c_e_s = pybamm.Variable(
            "Separator electrolyte concentration", domain="separator"
        )
        c_e_w = pybamm.Variable(
            "Positive electrolyte concentration", domain="positive electrode"
        )
        c_e = pybamm.concatenation(c_e_s, c_e_w)
        c_s_w = pybamm.Variable(
            "Positive particle concentration",
            domain="positive particle",
            auxiliary_domains={"secondary": "positive electrode"},
        )
        phi_s_w = pybamm.Variable(
            "Positive electrode potential", domain="positive electrode"
        )
        phi_e_s = pybamm.Variable("Separator electrolyte potential", domain="separator")
        phi_e_w = pybamm.Variable(
            "Positive electrolyte potential", domain="positive electrode"
        )
        phi_e = pybamm.concatenation(phi_e_s, phi_e_w)

        # Constant temperature
        T = param.T_init

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = param.current_with_time

        # Define particle surface concentration
        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        c_s_surf_w = pybamm.surf(c_s_w)

        # Define parameters. We need to assemble them differently depending on the
        # working electrode

        # Porosity and Transport_efficiency
        eps_s = pybamm.PrimaryBroadcast(
            pybamm.Parameter("Separator porosity"), "separator"
        )
        eps_w = pybamm.PrimaryBroadcast(
            pybamm.Parameter("Positive electrode porosity"), "positive electrode"
        )
        b_e_s = param.s.b_e
        b_e_w = param.p.b_e

        # Interfacial reactions
        j0_w = param.p.prim.j0(c_e_w, c_s_surf_w, T)
        U_w = param.p.prim.U
        ne_w = param.p.prim.ne

        # Particle diffusion parameters
        D_w = param.p.prim.D
        C_w = param.p.prim.cap_init
        a_R_w = param.p.prim.a_R
        gamma_e = param.c_e_typ / param.p.prim.c_max
        c_w_init = param.p.prim.c_init

        # Electrode equation parameters
        eps_s_w = pybamm.Parameter("Positive electrode active material volume fraction")
        b_s_w = param.p.b_s
        sigma_w = param.p.sigma

        # Other parameters (for outputs)
        c_w_max = param.p.prim.c_max
        U_w_ref = param.p.U_ref
        U_Li_ref = param.n.U_ref
        L_w = param.p.L

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
        
        L_sei_inner = pybamm.Variable("Inner SEI thickness", domain="positive electrode")
        j_inner = pybamm.Variable("Inner SEI interfacial current density")
        
        U_inner = U_inner_electron
        j_sei = (phi_s_w - U_inner) / (C_sei_electron * L_sei_inner)

        L_sei_bound = pybamm.BoundaryValue(L_sei_inner * L_inner_0_dim, "left")

        C_sei = 3.9*6*14.24*(0.035e30)*(L_sei_bound)/(6.02e23)

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

        # gamma_w is always 1 because we choose the timescale based on the working
        # electrode
        gamma_w = pybamm.Scalar(1)

        eps = pybamm.concatenation(eps_s, eps_w)
        tor = pybamm.concatenation(eps_s**b_e_s, eps_w**b_e_w)

        j_w = (
            2 * j0_w * pybamm.sinh(ne_w / 2 * (phi_s_w - phi_e_w - U_w(c_s_surf_w, T)))
        )
        j_s = pybamm.PrimaryBroadcast(0, "separator")
        j = pybamm.concatenation(j_s, j_w)

        ######################
        # State of Charge
        ######################
        I = param.dimensional_current_with_time
        # The `rhs` dictionary contains differential equations, with the key being the
        # variable in the d/dt
        self.rhs[Q] = (I * self.timescale / 3600)
        # Initial conditions must be provided for the ODEs
        self.initial_conditions[Q] = pybamm.Scalar(0)

        ######################
        # Particles
        ######################
        # The div and grad operators will be converted to the appropriate matrix
        # multiplication at the discretisation stage
        N_s_w = -D_w(c_s_w, T) * pybamm.grad(c_s_w)
        self.rhs[c_s_w] = -(1 / C_w) * pybamm.div(N_s_w)

        # Boundary conditions must be provided for equations with spatial
        # derivatives
        self.boundary_conditions[c_s_w] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -C_w * j_w / a_R_w / gamma_w / D_w(c_s_surf_w, T),
                "Neumann",
            ),
        }
        self.initial_conditions[c_s_w] = c_w_init

        # Events specify points at which a solution should terminate
        self.events += [
            pybamm.Event(
                "Minimum positive particle surface concentration",
                pybamm.min(c_s_surf_w) - 0.0001,
            ),
            pybamm.Event(
                "Maximum positive particle surface concentration",
                (1 - 0.0001) - pybamm.max(c_s_surf_w),
            ),
        ]

        ######################
        # Current in the solid
        ######################
        sigma_eff_w = sigma_w(T) * eps_s_w**b_s_w
        i_s_w = -sigma_eff_w * pybamm.grad(phi_s_w)
        self.boundary_conditions[phi_s_w] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                i_cell / pybamm.boundary_value(-sigma_eff_w, "right"),
                "Neumann",
            ),
        }
        self.algebraic[phi_s_w] = pybamm.div(i_s_w) + j_w
        # Initial conditions must also be provided for algebraic equations, as an
        # initial guess for a root-finding algorithm which calculates consistent
        # initial conditions
        self.initial_conditions[phi_s_w] = param.p.prim.U_init

        ######################
        # Electrolyte concentration
        ######################
        N_e = -tor * param.D_e(c_e, T) * pybamm.grad(c_e)
        self.rhs[c_e] = (1 / eps) * (
            -pybamm.div(N_e) / param.C_e + (1 - param.t_plus(c_e, T)) * j / gamma_e
        )
        dce_dx = (
            -(1 - param.t_plus(c_e, T))
            * i_cell
            * param.C_e
            / (tor * gamma_e * param.D_e(c_e, T))
        )

        self.boundary_conditions[c_e] = {
            "left": (pybamm.boundary_value(dce_dx, "left"), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }

        self.initial_conditions[c_e] = param.c_e_init
        self.events.append(
            pybamm.Event(
                "Zero electrolyte concentration cut-off", pybamm.min(c_e) - 0.002
            )
        )

        ######################
        # Current in the electrolyte
        ######################
        i_e = (param.kappa_e(c_e, T) * tor * gamma_e / param.C_e) * (
            param.chi(c_e, T)/c_e * pybamm.grad(c_e) - pybamm.grad(phi_e)
        )
        self.algebraic[phi_e] = pybamm.div(i_e) - j

        # dimensionless reference potential so that dimensional reference potential
        # is zero (phi_dim = U_n_ref + pot_scale * phi)
        l_Li = param.p.l
        sigma_Li = param.p.sigma
        j_Li = param.j0_plating(pybamm.boundary_value(c_e, "left"), 1, T)
        eta_Li = 2 * (1 + param.Theta * T) * pybamm.arcsinh(i_cell / (2 * j_Li))

        phi_s_cn = 0
        delta_phi = eta_Li + U_Li_ref
        delta_phis_Li = l_Li * i_cell / sigma_Li(T)
        ref_potential = phi_s_cn - delta_phis_Li - delta_phi

        self.boundary_conditions[phi_e] = {
            "left": (ref_potential, "Dirichlet"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }

        self.initial_conditions[phi_e] = param.n.U_ref / pot_scale

        ######################
        # (Some) variables
        ######################
        vdrop_cell = pybamm.boundary_value(phi_s_w, "right") - ref_potential
        vdrop_Li = -eta_Li - delta_phis_Li
        voltage = vdrop_cell + vdrop_Li
        voltage_dim = U_w_ref - U_Li_ref + pot_scale * voltage
        c_e_total = pybamm.x_average(eps * c_e)
        c_s_surf_w_av = pybamm.x_average(c_s_surf_w)

        c_s_rav = pybamm.r_average(c_s_w)
        c_s_vol_av = pybamm.x_average(eps_s_w * c_s_rav)

        # Cut-off voltage
        self.events.append(
            pybamm.Event(
                "Minimum voltage",
                voltage_dim - self.param.voltage_low_cut_dimensional,
                pybamm.EventType.TERMINATION,
            )
        )
        self.events.append(
            pybamm.Event(
                "Maximum voltage",
                self.param.voltage_high_cut_dimensional - voltage_dim,
                pybamm.EventType.TERMINATION,
            )
        )

        # Cut-off open-circuit voltage (for event switch with casadi 'fast with events'
        # mode)
        tol = 0.1
        self.events.append(
            pybamm.Event(
                "Minimum voltage switch",
                voltage_dim - (self.param.voltage_low_cut_dimensional - tol),
                pybamm.EventType.SWITCH,
            )
        )
        self.events.append(
            pybamm.Event(
                "Maximum voltage switch",
                voltage_dim - (self.param.voltage_high_cut_dimensional + tol),
                pybamm.EventType.SWITCH,
            )
        )

        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        self.variables = {
            "Time [s]": self.timescale * pybamm.t,
            "Positive particle surface concentration": c_s_surf_w,
            "X-averaged positive particle surface concentration": c_s_surf_w_av,
            "Positive particle concentration": c_s_w,
            "Negative particle surface concentration [mol.m-3]": c_w_max * c_s_surf_w,
            "X-averaged positive particle surface concentration "
            "[mol.m-3]": c_w_max * c_s_surf_w_av,
            "Positive particle concentration [mol.m-3]": c_w_max * c_s_w,
            "Total lithium in positive electrode": c_s_vol_av,
            "Total lithium in positive electrode [mol]": c_s_vol_av
            * c_w_max
            * L_w
            * param.A_cc,
            "Electrolyte concentration": c_e,
            "Electrolyte concentration [mol.m-3]": param.c_e_typ * c_e,
            "Positive electrolyte concentration": c_e,
            "Negative electrolyte concentration": c_e,
            "Separator electrolyte concentration": c_e,
            "Positive electrolyte potential": phi_e,
            "Negative electrolyte potential": phi_e,
            "Separator electrolyte potential": phi_e,
            "Total lithium in electrolyte": c_e_total,
            "Total lithium in electrolyte [mol]": c_e_total
            * param.c_e_typ
            * param.L_x
            * param.A_cc,
            "Current [A]": I,
            "Current [mA]": I*1000,
            "Current density [A.m-2]": i_cell * i_typ,
            "Positive electrode potential": phi_s_w,
            "Positive electrode potential [V]": U_w_ref
            - U_Li_ref
            + pot_scale * phi_s_w,
            "Positive electrode open circuit potential": U_w(c_s_surf_w, T),
            "Positive electrode open circuit potential [V]": U_w_ref
            + pot_scale * U_w(c_s_surf_w, T),
            "Electrolyte potential": phi_e,
            "Electrolyte potential [V]": -U_Li_ref + pot_scale * phi_e,
            "Voltage drop in the cell": vdrop_cell,
            "Voltage drop in the cell [V]": U_w_ref - U_Li_ref + pot_scale * vdrop_cell,
            "Negative electrode exchange current density": j_Li,
            "Negative electrode reaction overpotential": eta_Li,
            "Negative electrode reaction overpotential [V]": pot_scale * eta_Li,
            "Negative electrode potential drop": delta_phis_Li,
            "Negative electrode potential drop [V]": pot_scale * delta_phis_Li,
            "Terminal voltage": voltage,
            "Terminal voltage [V]": voltage_dim,
            "Instantaneous power [W.m-2]": i_cell * i_typ * voltage_dim,
            "Pore-wall flux [mol.m-2.s-1]": j_w,
            "Area [cm2]": param.A_cc*10000,
            "Inner SEI thickness": L_sei_inner * L_inner_0_dim,
            "SEI voltage": U_inner_electron * pot_scale,
            "U_w_ref":U_w_ref,
            #"U_Li":U_Li_ref,
            #"U_w":U_w,
            "C_sei":(I * self.timescale / 3600)*(C_sei) ,
        }

    def new_copy(self, build=False):
        new_model = self.__class__(name=self.name, options=self.options)
        new_model.use_jacobian = self.use_jacobian
        new_model.convert_to_format = self.convert_to_format
        new_model._timescale = self.timescale
        new_model._length_scales = self.length_scales
        return new_model
    
model = HalfCell(options={"working electrode":"positive"})                    # here: positive is anode and negative is lithium metal

geometry = model.default_geometry

param = model.default_parameter_values

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


def graphite_sr_ocp_Dualfoil1998(sto):
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
        0.65
        + 1.6 * pybamm.exp(-550.0 * sto)
        + 0.0351 * pybamm.tanh((sto - 0.286) / 0.083)
        - 0.0045 * pybamm.tanh((sto - 0.849) / 0.119)
        - 0.035 * pybamm.tanh((sto - 0.9233) / 0.05)
        - 0.0147 * pybamm.tanh((sto - 0.5) / 0.034)
        - 0.102 * pybamm.tanh((sto - 0.194) / 0.142)
        - 0.022 * pybamm.tanh((sto - 0.9) / 0.0164)
        - 0.011 * pybamm.tanh((sto - 0.124) / 0.0226)
        + 0.0155 * pybamm.tanh((sto - 0.105) / 0.029)
        - 0.6 * pybamm.exp(115 * sto - 16)
    )


    return u_eq

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

def graphite_LGM50_ocp_Chen2020(sto):
    """
    LG M50 Graphite open-circuit potential as a function of stochiometry, fit taken
    from [1].

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open circuit potential
    """

    u_eq = (
        1.9793 * pybamm.exp(-39.3631 * sto)
        + 0.2482
        - 0.0909 * pybamm.tanh(29.8538 * (sto - 0.1234))
        - 0.04478 * pybamm.tanh(14.9159 * (sto - 0.2769))
        - 0.0205 * pybamm.tanh(30.4444 * (sto - 0.6103))
    )

    return u_eq

def graphite_LGM50_electrolyte_exchange_current_density_Chen2020(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

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
    m_ref = 6.48e-7  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 35000
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )

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
param['Inner SEI reaction proportion'] = 1
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
param['Initial inner SEI thickness [m]'] = 1e-10
param['Initial outer SEI thickness [m]'] = 0
param['EC initial concentration in electrolyte [mol.m-3]'] = 4541.0
param['EC diffusivity [m2.s-1]'] = 2e-18
param['SEI kinetic rate constant [m.s-1]'] = 1e-12
param['SEI open-circuit potential [V]'] = 0.8
param['SEI growth activation energy [J.mol-1]'] = 0.5
param['Negative electrode reaction-driven LAM factor [m3.mol-1]'] = 0.0
param['Positive electrode reaction-driven LAM factor [m3.mol-1]'] = 0.0
#cell
param['Negative electrode thickness [m]'] = 3.1e-4
param['Separator thickness [m]'] = 2.75e-04
param['Positive electrode thickness [m]'] = 4.9e-5
param['Electrode height [m]'] = 0.008*np.pi
param['Electrode width [m]'] = 0.008
param['Nominal cell capacity [A.h]'] = 5e-3

#graphite-siox electrode
param['Positive electrode conductivity [S.m-1]'] = 100.0
param['Maximum concentration in positive electrode [mol.m-3]'] = 29583
param['Positive electrode diffusivity [m2.s-1]'] = 1.74e-15 
param['Positive electrode OCP [V]'] = graphite_LGM50_ocp_Chen2020
param['Positive electrode porosity'] = 0.25
param['Positive electrode active material volume fraction'] = 0.75
param['Positive particle radius [m]'] = 5.86e-06
param['Positive electrode Bruggeman coefficient (electrolyte)'] = 1.5
param['Positive electrode Bruggeman coefficient (electrode)'] = 1.5
param['Positive electrode cation signed stoichiometry'] = -1.0
param['Positive electrode electrons in reaction'] = 1.0
param['Positive electrode charge transfer coefficient'] = 0.5
param['Positive electrode exchange-current density [A.m-2]'] = graphite_LGM50_electrolyte_exchange_current_density_Chen2020
param['Positive electrode OCP entropic change [V.K-1]'] = graphite_entropic_change_Moura2016

#lithium electrode
param['Exchange-current density for plating [A.m-2]'] = 12.6

#separator
param['Separator porosity'] = 1.0
param['Separator Bruggeman coefficient (electrolyte)'] = 1.5
#electrolyte
param['Typical electrolyte concentration [mol.m-3]'] = 1000.0
param['Initial concentration in electrolyte [mol.m-3]'] = 800.0
param['Cation transference number'] = 0.4
param['1 + dlnf/dlnc'] = 1.0
param['Electrolyte diffusivity [m2.s-1]'] = electrolyte_diffusivity_Capiglia1999
param['Electrolyte conductivity [S.m-1]'] = 0.1
#experiment
param['Reference temperature [K]'] = 298.15
param['Ambient temperature [K]'] = 298.15
param['Number of electrodes connected in parallel to make a cell'] = 1.0
param['Number of cells connected in series to make a battery'] = 1.0
param['Lower voltage cut-off [V]'] = 0
param['Upper voltage cut-off [V]'] = 10
param['Initial temperature [K]'] = 298.15
param["Current function [A]"] = 0.25e-3
param["Typical current [A]"] = 0.25e-3
param["Initial concentration in positive electrode [mol.m-3]"] = 10
param["Upper voltage cut-off [V]"] = 5



param.process_geometry(geometry)
param.process_model(model)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 5, var.r_p: 5}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 18000, 400)
solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-3)
solution = solver.solve(model, t_eval)

# sim = pybamm.Simulation(model, experiment=experiment)
# solution = sim.solve()

plot = pybamm.QuickPlot(
    solution,
    [
        # 'Time [s]',
         "Negative particle surface concentration [mol.m-3]",
        # 'Negative particle surface concentration',
        # 'X-averaged positive particle surface concentration',
        # 'Positive particle concentration',
        # 'Positive particle surface concentration [mol.m-3]',
        # 'X-averaged positive particle surface concentration [mol.m-3]',
        # 'Positive particle concentration [mol.m-3]',
        # 'Total lithium in positive electrode',
        # 'Total lithium in positive electrode [mol]',
         'Electrolyte concentration',
        # 'Electrolyte concentration [mol.m-3]',
        # 'Total lithium in electrolyte',
        # 'Total lithium in electrolyte [mol]',
         'Current [mA]',
        # 'Current density [A.m-2]',
        # 'Positive electrode potential',
        # 'Positive electrode potential [V]',
         'Positive electrode open circuit potential',
         'Positive electrode open circuit potential [V]',
        # 'Electrolyte potential',
         'Electrolyte potential',
        # 'Voltage drop in the cell',
        # 'Voltage drop in the cell [V]',
        # 'Negative electrode exchange current density',
        # 'Negative electrode reaction overpotential',
        # 'Negative electrode reaction overpotential [V]',
        # 'Negative electrode potential drop',
        # 'Negative electrode potential drop [V]',
        # 'Terminal voltage',
         'Terminal voltage [V]',
        # 'Instantaneous power [W.m-2]',
        # 'Pore-wall flux [mol.m-2.s-1]'
        "Discharge capacity [A.h]",
        #"Area [cm2]",
        #"Inner SEI thickness",
        #"SEI voltage",
        #"U_w_ref",
        #"U_Li",
        #"U_w",
        #"C_sei"
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()

print(solution["Terminal voltage [V]"].entries)
solution.save_data("1mah.csv", ["Time [s]","Terminal voltage [V]"], to_format="csv")