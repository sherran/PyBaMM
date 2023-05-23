import pybamm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

### 1. Initialize model -----------------------------------------------------------------------------------------------------------------
model = pybamm.BaseModel()

### 2. Define Variables and Parameters --------------------------------------------------------------------------------------------------

## Physical constants

R = pybamm.constants.R          # Gas constant
F = pybamm.constants.F          # Faraday's constant
k_b = pybamm.constants.k_b      # Boltzmann constant
q_e = pybamm.constants.q_e      # Electron volt

## Thermal Parameters                                                           # ToDo: Check dependancies

T_ref = pybamm.Parameter("Reference temperature [K]")                           # Ref temp (298.15)
Delta_T = pybamm.Scalar(1)                                                      # Typical temperature rise
T_init_dim = pybamm.Parameter("Initial temperature [K]")                        # Initial temperature (298.15)
T_init = (T_init_dim - T_ref) / Delta_T                                         # Dimensionless initial temp (0)
Theta = Delta_T / T_ref
T = T_init

## Geometrical Parameters

L_n = pybamm.Parameter("Negative electrode thickness [m]")
L_e = pybamm.Parameter("Distance between two electrodes [m]")
L_p = pybamm.Parameter("Positive electrode thickness [m]")
L_y = pybamm.Parameter("Electrode width [m]")
L_z = pybamm.Parameter("Electrode height [m]")
L_n_cc = pybamm.Parameter("Negative current collector thickness [m]")
L_p_cc = pybamm.Parameter("Positive current collector thickness [m]")

# Dimensional Parameter
L_x = L_n + L_e + L_p       # Total distance between current collectors
L = L_n_cc + L_x + L_p_cc   # Total cell thickness
A_cc = L_y * L_z            # Current collector cross sectional area    

# Dimensionless Parameter
l_x = L_x / L_x             # P.U. distance between current collectors
l_n = L_n / L_x             # End of Negative electrode
l_e = L_e / L_x             
l_n_l_e = l_n + l_e         # Start of Positive electrode
l_p = L_p / L_x
l_y = L_y / L_z
l_z = L_z / L_z
a_cc = l_y * l_z            # P.U. Cross sectional area

# i_boundary_cc = pybamm.Variable(
#     "Current collector current density", domain="current collector"
# )

x_n = (
    pybamm.SpatialVariable(
        "x_n",
        domain=["negative electrode"],
        auxiliary_domains={"secondary": "current collector"},
        coord_sys="cartesian",
    ) * L_x
)

x_s = (
    pybamm.SpatialVariable(
        "x_s",
        domain=["separator"],
        coord_sys="cartesian",
    ) * L_x
)
x_p = (
    pybamm.SpatialVariable(
        "x_p",
        domain=["positive electrode"],
        auxiliary_domains={"secondary": "current collector"},
        coord_sys="cartesian",
    ) * L_x
)


dmn = ["negative","positive"]


def R_dimensional(domain):
    if domain == "negative":
        x = pybamm.standard_spatial_vars.x_n
    elif domain == "positive":
        x = pybamm.standard_spatial_vars.x_p

    inputs = {"Through-cell distance (x) [m]": x * L_x}
    Domain = domain.capitalize()
    return pybamm.FunctionParameter(
        f"{Domain} particle radius [m]", inputs
    )

for dn in dmn:
    if dn == "negative":
        R_n = R_dimensional(dn)
    elif dn == "positive":
        R_p = R_dimensional(dn)

R_typ_n = pybamm.xyz_average(R_n)
R_typ_p = pybamm.xyz_average(R_p)

r_n = (
    pybamm.SpatialVariable(
        "r_n",
        domain=["negative particle"],
        auxiliary_domains={
            "secondary": "negative electrode",
            "tertiary": "current collector",
        },
        coord_sys="spherical polar",
    ) * R_typ_n

)
r_p = (
    pybamm.SpatialVariable(
        "r_p",
        domain=["positive particle"],
        auxiliary_domains={
            "secondary": "positive electrode",
            "tertiary": "current collector",
        },
        coord_sys="spherical polar",
    ) * R_typ_p
)



length_scales = {
            "negative electrode": L_x,
            "separator": L_x,
            "positive electrode": L_x,
            "current collector y": L_z,
            "current collector z": L_z,
            "negative particle": R_typ_n,
            "positive particle": R_typ_p,
        }


c_max = pybamm.Parameter(
    "Initial concentration in electrolyte [mol.m-3]"                                            # eventually have to change
)

## Electrical Parameters

I_typ = pybamm.Parameter("Typical current [A]")

i_typ = pybamm.Function(
            np.abs, I_typ / A_cc
        )

tau_discharge = F * c_max * L_x / i_typ
timescale = tau_discharge
model.timescale = timescale

I = pybamm.FunctionParameter("Current Function [A]",{"Time [s]": pybamm.t * timescale} )


voltage_low_cut_dimensional = pybamm.Parameter("Lower voltage cut-off [V]")
voltage_high_cut_dimensional = pybamm.Parameter("Upper voltage cut-off [V]")

i = I / A_cc        # Current Density (A/m2)

i_cell = I / I_typ * pybamm.sign(I_typ)




## charge current function

t_rep = []
i_rep = []
t = 0
step = 0
while t <= 1800:
    if step*60 <= t < step*60 + 40:
        t_rep.append(t)
        i_rep.append(-0.98)
    elif t == step*60 + 40:
        t_rep.append(t)
        i_rep.append(-0.98)
        t_rep.append(t + 0.0000001)
        i_rep.append(-0.08)
    elif step*60 + 40 < t < step*60 + 60:
        t_rep.append(t)
        i_rep.append(-0.08)
    elif t == step*60 + 60:
        t_rep.append(t)
        i_rep.append(-0.08)
        t_rep.append(t + 0.0000001)
        i_rep.append(-0.98)
        step += 1
    t += 1   
i_func = np.asarray(i_rep)
t_func = np.asarray(t_rep)
current_func = np.column_stack((t_func, i_func))


# plt.plot(t_func, i_func)
# plt.xlabel('Time')
# plt.ylabel('Current [A]')
# plt.show()


current_interpolant = pybamm.Interpolant(current_func[:, 0], current_func[:, 1], pybamm.t * timescale)
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
eps_mass_n = pybamm.FunctionParameter(
            f"Negative electrode active material volume fraction",
            {"Through-cell distance (x) [m]": x_n},
        )
eps_mass_p = pybamm.FunctionParameter(
            f"Positive electrode active material volume fraction",
            {"Through-cell distance (x) [m]": x_p},
        )

eps_s_n = pybamm.Parameter("Negative electrode active material volume fraction")
eps_s_p = pybamm.Parameter("Positive electrode active material volume fraction")


# transport_efficiency
tor = pybamm.concatenation(
    eps_n**1.5, eps_s**1.5, eps_p**1.5
)

# Variables that vary spatially are created with a domain
c_e_n_Li = pybamm.Variable(
    "Negative electrolyte Lithium ion concentration", domain="negative electrode"
)
c_e_s_Li= pybamm.Variable(
     "Separator electrolyte Lithium ion concentration", domain="separator"
)
c_e_p_Li = pybamm.Variable(
    "Positive electrolyte Lithium ion concentration", domain="positive electrode"
)
# Concatenations combine several variables into a single variable, to simplify
# implementing equations that hold over several domains
c_e_Li = pybamm.concatenation(c_e_n_Li, c_e_s_Li, c_e_p_Li)

c_e_n_Cl = pybamm.Variable(
    "Negative electrolyte Chloride ion concentration", domain="negative electrode"
)
c_e_s_Cl = pybamm.Variable(
     "Separator electrolyte Chloride ion concentration", domain="separator"
 )
c_e_p_Cl = pybamm.Variable(
    "Positive electrolyte Chloride ion concentration", domain="positive electrode"
)
# Concatenations combine several variables into a single variable, to simplify
# implementing equations that hold over several domains
c_e_Cl = pybamm.concatenation(c_e_n_Cl, c_e_s_Cl, c_e_p_Cl)

c_max = pybamm.Parameter(
    "Initial concentration in electrolyte [mol.m-3]"                                            # eventually have to change
)

tau_discharge = F * c_max * L_x / i_typ
timescale = tau_discharge

c_e_typ = pybamm.Parameter("Typical electrolyte concentration [mol.m-3]")
c_e_init_dim = pybamm.Parameter("Initial concentration in electrolyte [mol.m-3]")

c_e_init = c_e_init_dim / c_e_typ

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

#Lithium ion in solid
c_s_n_Li = pybamm.Variable(
    "Negative particle Lithium concentration",
    domain="negative particle",
    auxiliary_domains={"secondary": "negative electrode"},
)

c_s_n_Cl = pybamm.Variable(
    "Negative particle Chloride concentration",
    domain="negative particle",
    auxiliary_domains={"secondary": "negative electrode"},
)

c_max_n = pybamm.Parameter(
    "Maximum concentration in Negative electrode [mol.m-3]"
)



#Chloride ion in solid
c_s_p_Li = pybamm.Variable(
    "Positive particle Lithium concentration",
    domain="positive particle",
    auxiliary_domains={"secondary": "positive electrode"},
)

c_s_p_Cl = pybamm.Variable(
    "Positive particle Chloride concentration",
    domain="positive particle",
    auxiliary_domains={"secondary": "positive electrode"},
)

c_max_p = pybamm.Parameter(
    "Maximum concentration in Positive electrode [mol.m-3]"
)

c_init_dim_n = pybamm.FunctionParameter(
            "Initial concentration in Negative electrode [mol.m-3]",
            {
                "Radial distance (r) [m]": r_n,
                "Through-cell distance (x) [m]": pybamm.PrimaryBroadcast(
                    x_n, "negative particle"
                ),
            },
        )

c_init_dim_p = pybamm.FunctionParameter(
            "Initial concentration in Positive electrode [mol.m-3]",
            {
                "Radial distance (r) [m]": r_p,
                "Through-cell distance (x) [m]": pybamm.PrimaryBroadcast(
                    x_p, "positive particle"
                ),
            },
        )

c_s_surf_n_Li = pybamm.surf(c_s_n_Li)
c_s_surf_n_Cl = pybamm.surf(c_s_n_Cl)
c_s_surf_p_Li = pybamm.surf(c_s_p_Li)
c_s_surf_p_Cl = pybamm.surf(c_s_p_Cl)


c_init_n = c_init_dim_n / c_max_n
c_init_p = c_init_dim_p / c_max_p

c_init_av_n = pybamm.xyz_average(pybamm.r_average(c_init_n))
c_init_av_p = pybamm.xyz_average(pybamm.r_average(c_init_p))

# Specific interfacial surface area
a_s_n = 3*eps_s_n/R_n                    
a_s_p = 3*eps_s_p/R_p
a_s_s = pybamm.PrimaryBroadcast(0, "separator")

a_typ_n = 3*pybamm.xyz_average(eps_mass_n)/R_typ_n
a_typ_p = 3*pybamm.xyz_average(eps_mass_p)/R_typ_p

a_R_n = a_typ_n/R_typ_n
a_R_p = a_typ_p/R_typ_p

potential_scale = R * T_ref / F
j_scale_n = i_typ / a_typ_n / L_x
j_scale_p = i_typ / a_typ_p / L_x


gamma_n = (tau_discharge / timescale) * c_max_n / c_max_n
gamma_p = (tau_discharge / timescale) * c_max_p / c_max_p

# Electrode conductivity
sigma_n = pybamm.Parameter("Negative electrode conductivity [S.m-1]") * potential_scale / i_typ / L_x
sigma_p = pybamm.Parameter("Positive electrode conductivity [S.m-1]") * potential_scale / i_typ / L_x

# Parameters for current density in Positive electrode
K_0_cl = pybamm.Parameter("Equilibrium constant for adsorbtion and desorption of chloride anions")
k_a = pybamm.Parameter("Rate constant for Tafel step")
a_cl = pybamm.Parameter("Activity of chloride anions")
#eta_a = pybamm.Variable("Anode Overpotential", domain="positive electrode")
#eta = pybamm.Variable("Overpotential")
r_Cl = pybamm.Parameter("Reaction rate of Chloride at Positive Electrode")

# Transference Numbers
t_plus = pybamm.Parameter('Cation transference number')
t_minus = pybamm.Parameter('Anion transference number')

# Mean molar activity coefficient
f_plus_minus_n = pybamm.PrimaryBroadcast(
    pybamm.Parameter("Mean molar activity coefficient"), "negative electrode"
)
f_plus_minus_s = pybamm.PrimaryBroadcast(
    pybamm.Parameter("Mean molar activity coefficient"), "separator"
)
f_plus_minus_p = pybamm.PrimaryBroadcast(
    pybamm.Parameter("Mean molar activity coefficient"), "positive electrode"
)
f_plus_minus = pybamm.concatenation(f_plus_minus_n, f_plus_minus_s, f_plus_minus_p)

f_plus_n = pybamm.PrimaryBroadcast(
    pybamm.Parameter("Cation molar activity coefficient"), "negative electrode"
)
f_plus_s = pybamm.PrimaryBroadcast(
    pybamm.Parameter("Cation molar activity coefficient"), "separator"
)
f_plus_p = pybamm.PrimaryBroadcast(
    pybamm.Parameter("Cation molar activity coefficient"), "positive electrode"
)
f_plus = pybamm.concatenation(f_plus_n, f_plus_s, f_plus_p)

f_minus_n = pybamm.PrimaryBroadcast(
    pybamm.Parameter("Anion molar activity coefficient"), "negative electrode"
)
f_minus_s = pybamm.PrimaryBroadcast(
    pybamm.Parameter("Anion molar activity coefficient"), "separator"
)
f_minus_p = pybamm.PrimaryBroadcast(
    pybamm.Parameter("Anion molar activity coefficient"), "positive electrode"
)
f_minus = pybamm.concatenation(f_minus_n, f_minus_s, f_minus_p)

# Functions

def D_e_dimensional_n(c_e, T):
    """Dimensional diffusivity in electrolyte"""
    tol = pybamm.settings.tolerances["D_e__c_e"]
    c_e = pybamm.maximum(c_e, tol)
    inputs = {"Electrolyte Lithium concentration [mol.m-3]": c_e, "Temperature [K]": T}
    return pybamm.FunctionParameter("Electrolyte Lithium diffusivity [m2.s-1]", inputs)

def D_e_dimensional_p(c_e, T):
    """Dimensional diffusivity in electrolyte"""
    tol = pybamm.settings.tolerances["D_e__c_e"]
    c_e = pybamm.maximum(c_e, tol)
    inputs = {"Electrolyte Chloride concentration [mol.m-3]": c_e, "Temperature [K]": T}
    return pybamm.FunctionParameter("Electrolyte Chloride diffusivity [m2.s-1]", inputs)

D_e_typ = D_e_dimensional_n(c_e = c_e_typ, T=T_ref)

def D_e_Li(c_e, T):
    """Dimensionless electrolyte diffusivity"""
    c_e_dimensional = c_e * c_e_typ
    T_dim = Delta_T * T + T_ref
    return D_e_dimensional_n(c_e_dimensional, T_dim) / D_e_typ

def D_e_Cl(c_e, T):
    """Dimensionless electrolyte diffusivity"""
    c_e_dimensional = c_e * c_e_typ
    T_dim = Delta_T * T + T_ref
    return D_e_dimensional_p(c_e_dimensional, T_dim) / D_e_typ


tau_diffusion_e = L_x**2 / D_e_typ

gamma_e = (tau_discharge / timescale) * c_e_typ / c_max

C_e = tau_diffusion_e / timescale

kappa_scale = F**2 * D_e_typ * c_e_typ / (R * T_ref)

def kappa_e_dimensional(c_e_Li, c_e_Cl, T):
    """Dimensional electrolyte conductivity"""
    tol = pybamm.settings.tolerances["D_e__c_e"]
    c_e_Li = pybamm.maximum(c_e_Li, tol)
    c_e_Cl = pybamm.maximum(c_e_Cl, tol)
    inputs = {"Electrolyte Lithium concentration [mol.m-3]": c_e_Li, "Electrolyte Chloride concentration [mol.m-3]": c_e_Cl, "Temperature [K]": T}
    return pybamm.FunctionParameter("Electrolyte conductivity [S.m-1]", inputs)

def kappa_e(c_e_Li, c_e_Cl, T):
    """Dimensionless electrolyte conductivity"""
    c_e_dimensional_Li = c_e_Li * c_e_typ
    c_e_dimensional_Cl = c_e_Cl * c_e_typ
    T_dim = Delta_T * T + T_ref
    return kappa_e_dimensional(c_e_dimensional_Li, c_e_dimensional_Cl, T_dim) / kappa_scale


def chi_dimensional(c_e, T):
    """
    Thermodynamic factor:
        (1-2*t_plus) is for Nernst-Planck,
        2*(1-t_plus) for Stefan-Maxwell,
    see Bizeray et al (2016) "Resolving a discrepancy ...".
    """
    return ((t_plus - 1))


def chi(c_e, T):
    """
    Thermodynamic factor:
        (1-2*t_plus) is for Nernst-Planck,
        2*(1-t_plus) for Stefan-Maxwell,
    see Bizeray et al (2016) "Resolving a discrepancy ...".
    """
    c_e_dimensional = c_e * c_e_typ
    T_dim = Delta_T * T + T_ref
    return chi_dimensional(c_e_dimensional, T_dim)


def chiRT_over_Fc(c_e, T):
    """
    chi * (1 + Theta * T) / c,
    as it appears in the electrolyte potential equation
    """
    tol = pybamm.settings.tolerances["chi__c_e"]
    c_e = pybamm.maximum(c_e, tol)
    return chi(c_e, T) * (1 + Theta * T) / c_e

def D_dimensional_n(sto, T):
    """Dimensional diffusivity in particle. Note this is defined as a
    function of stochiometry"""
    inputs = {
        "Negative particle stoichiometry": sto,
        "Temperature [K]": T,
    }
    return pybamm.FunctionParameter(
        "Negative electrode diffusivity [m2.s-1]",
        inputs,
    )

def D_dimensional_p(sto, T):
    """Dimensional diffusivity in particle. Note this is defined as a
    function of stochiometry"""
    inputs = {
        "Positive particle stoichiometry": sto,
        "Temperature [K]": T,
    }
    return pybamm.FunctionParameter(
        "Positive electrode diffusivity [m2.s-1]",
        inputs,
    )

D_typ_dim_n = D_dimensional_n(pybamm.Scalar(1), T_ref)
D_typ_dim_p = D_dimensional_p(pybamm.Scalar(1), T_ref)

tau_diffusion_n = R_typ_n**2 / D_typ_dim_n
tau_diffusion_p = R_typ_p**2 / D_typ_dim_p

C_diff_n = tau_diffusion_n / timescale
C_diff_p = tau_diffusion_p / timescale

def D_n(c_s, T):
    """Dimensionless particle diffusivity"""
    sto = c_s
    T_dim = Delta_T * T + T_ref
    return D_dimensional_n(sto, T_dim) / D_typ_dim_n

def D_p(c_s, T):
    """Dimensionless particle diffusivity"""
    sto = c_s
    T_dim = Delta_T * T + T_ref
    return D_dimensional_p(sto, T_dim) / D_typ_dim_p

def U_dimensional_n(sto, T):                                                 # ToDo: To be replaced by Voltage function
    """Dimensional open-circuit potential [V]"""
    # bound stoichiometry between tol and 1-tol. Adding 1/sto + 1/(sto-1) later
    # will ensure that ocp goes to +- infinity if sto goes into that region
    # anyway
    tol = pybamm.settings.tolerances["U__c_s"]
    sto = pybamm.maximum(pybamm.minimum(sto, 1 - tol), tol)
    inputs = {"Negative particle stoichiometry": sto}
    u_ref = pybamm.FunctionParameter(
        "Negative electrode OCP [V]", inputs
    )
    # add a term to ensure that the OCP goes to infinity at 0 and -infinity at 1
    # this will not affect the OCP for most values of sto
    # see #1435
    u_ref = u_ref 

    return u_ref 

def U_dimensional_p(sto, T):                                                 # ToDo: To be replaced by Voltage function
    """Dimensional open-circuit potential [V]"""
    # bound stoichiometry between tol and 1-tol. Adding 1/sto + 1/(sto-1) later
    # will ensure that ocp goes to +- infinity if sto goes into that region
    # anyway
    tol = pybamm.settings.tolerances["U__c_s"]
    sto = pybamm.maximum(pybamm.minimum(sto, 1 - tol), tol)
    inputs = {"Positive particle stoichiometry": sto}
    u_ref = pybamm.FunctionParameter(
        "Positive electrode OCP [V]", inputs
    )
    # add a term to ensure that the OCP goes to infinity at 0 and -infinity at 1
    # this will not affect the OCP for most values of sto
    # see #1435
    u_ref = u_ref #+ 1e-6 * (1 / sto + 1 / (sto - 1))

    return u_ref


U_init_dim_n = U_dimensional_n(c_init_av_n, T_init_dim)
U_init_dim_p = U_dimensional_p(c_init_av_p, T_init_dim)

U_ref_n = U_dimensional_n(c_init_av_n, T_ref)
U_ref_p = U_dimensional_p(c_init_av_p, T_ref)

ocv_ref = U_ref_p - U_ref_n

ocv_init_dim = U_init_dim_p - U_init_dim_n

ocv_init = (ocv_init_dim - ocv_ref) / potential_scale

U_init_n = (U_init_dim_n - U_ref_n) / potential_scale
U_init_p = (U_init_dim_p - U_ref_p) / potential_scale
U_init_e = (U_dimensional_n(c_e_init,T_ref) - U_dimensional_n(c_e_typ,T_ref))/potential_scale

def U_n(c_s, T):
    """Dimensionless open-circuit potential in the electrode"""
    sto = c_s
    T_dim = Delta_T * T + T_ref
    return (
        U_dimensional_n(sto, T_dim) - U_ref_n
    ) / potential_scale 

def U_p(c_s, T):
    """Dimensionless open-circuit potential in the electrode"""
    sto = c_s
    T_dim = Delta_T * T + T_ref
    return (
        U_dimensional_p(sto, T_dim) - U_ref_p
    ) / potential_scale 

# Current densities

def i0_dimensional_n(c_e, c_s_surf, T):                                                              # ToDo: User defined negative current density function
    """Dimensional exchange-current density [A.m-2] at negative electrode"""
    inputs = {
        "Electrolyte concentration [mol.m-3]": c_e,
        "Negative particle surface concentration [mol.m-3]": c_s_surf,
        "Maximum negative particle surface concentration [mol.m-3]": c_max_n,
        "Temperature [K]": T,
    }
    return pybamm.FunctionParameter(
        "Negative electrode exchange-current density [A.m-2]",
        inputs,
    )

def i0_dimensional_p(c_e, c_s_surf, T):                                                              # ToDo: User defined negative current density function
    """Dimensional exchange-current density [A.m-2] at negative electrode"""
    inputs = {
        "Electrolyte concentration [mol.m-3]": c_e,
        "Positive particle surface concentration [mol.m-3]": c_s_surf,
        "Maximum positive particle surface concentration [mol.m-3]": c_max_p,
        "Temperature [K]": T,
    }
    return pybamm.FunctionParameter(
        "Positive electrode exchange-current density [A.m-2]",
        inputs,
    )

def i0_ne(c_e, c_s_surf, T):
    """Dimensionless exchange-current density"""
    tol = pybamm.settings.tolerances["j0__c_e"]
    c_e = pybamm.maximum(c_e, tol)
    tol = pybamm.settings.tolerances["j0__c_s"]
    c_s_surf = pybamm.maximum(pybamm.minimum(c_s_surf, 1 - tol), tol)
    c_e_dim = c_e * c_e_typ
    c_s_surf_dim = c_s_surf * c_max_n
    T_dim = Delta_T * T + T_ref

    return i0_dimensional_n(c_e_dim, c_s_surf_dim, T_dim) / j_scale_n

def i0_po(c_e, c_s_surf, T):
    """Dimensionless exchange-current density"""
    tol = pybamm.settings.tolerances["j0__c_e"]
    c_e = pybamm.maximum(c_e, tol)
    tol = pybamm.settings.tolerances["j0__c_s"]
    c_s_surf = pybamm.maximum(pybamm.minimum(c_s_surf, 1 - tol), tol)
    c_e_dim = c_e * c_e_typ
    c_s_surf_dim = c_s_surf * c_max_p
    T_dim = Delta_T * T + T_ref

    return i0_dimensional_p(c_e_dim, c_s_surf_dim, T_dim) / j_scale_p

i0_n = i0_ne(c_e_n_Li, c_s_surf_n_Li, T)
i0_p = i0_po(c_e_p_Cl, c_s_surf_p_Cl, T)

i_n = (
    2
    * i0_n
    * pybamm.sinh(
        0.5 * (phi_s_n - phi_e_n - (U_n(c_s_surf_n_Li, T)))                       # ToDo: subtract F*R_sei*j_sei
    )
)

i_s = pybamm.PrimaryBroadcast(0, "separator")

#i_p = 2*F*k_a*pybamm.exp(2*F*eta/(R*T_ref))*(1 - pybamm.exp(-2*F*eta_a/(R*T_ref)))*(K_0_cl*a_cl/(1 + K_0_cl*a_cl*pybamm.exp(F*eta_a/(R*T_ref))))**2     # ToDo: Check sign of function. Eventually replace with standard current density function 

i_p = (
    2
    * i0_p
    * pybamm.sinh(
        0.5 * (phi_s_p - phi_e_p - U_p(c_s_surf_p_Cl, T))                      
    )
)

i_electrolyte = pybamm.concatenation(i_n, i_s, i_p)


# Exchange Current Density

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

# OCP

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


# Electrolyte conductivity

def electrolyte_conductivity_Capiglia1999(c_e_Li, c_e_Cl, T):
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
        + 1.9101 * (c_e_Cl + c_e_Li / 2000)
        - 1.052 * (c_e_Cl + c_e_Li / 2000) ** 2
        + 0.1554 * (c_e_Cl + c_e_Li / 2000) ** 3
    )

    E_k_e = 34700
    arrhenius = pybamm.exp(E_k_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return sigma_e * arrhenius


# Diffusivity

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
    sto = sto
    D_ref = 3.9 * 10 ** (-14)
    E_D_s = 42770
    arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius

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
    sto = sto
    D_ref = 1 * 10 ** (-13)
    E_D_s = 18550
    arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


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

i_s_Li = pybamm.PrimaryBroadcast(0, "separator")
i_s_Cl = pybamm.PrimaryBroadcast(0, "separator")
i_p_Li = pybamm.PrimaryBroadcast(0, "positive electrode")
i_n_Cl = pybamm.PrimaryBroadcast(0, "negative electrode")

i_e_Li = pybamm.concatenation(i_n,i_s_Li,i_p_Li)
i_e_Cl = pybamm.concatenation(i_n_Cl,i_s_Cl,i_p)



### 3. State governing equations -----------------------------------------------------------------------------------------

######################
# Current in the solid
######################
sigma_eff_n = sigma_n * (eps_s_n**1.5) 
i_s_n = -sigma_eff_n * pybamm.grad(phi_s_n)

sigma_eff_p = sigma_p * (eps_s_p**1.5)
i_s_p = -sigma_eff_p * pybamm.grad(phi_s_p)

model.algebraic[phi_s_n] = pybamm.div(i_s_n) + i_n                             #
model.algebraic[phi_s_p] = pybamm.div(i_s_p) + i_p

######################
# Mass in the solid (Lithium)                                                                                  
######################
N_s_n_Li = -D_n(c_s_n_Li,T)*pybamm.grad(c_s_n_Li)

model.rhs[c_s_n_Li] = -(1/C_diff_n)*pybamm.div(N_s_n_Li)

N_s_p_Li = -D_p(c_s_p_Li,T)*pybamm.grad(c_s_p_Li)

model.rhs[c_s_p_Li] = -(1/C_diff_p)*pybamm.div(N_s_p_Li)*0 


######################
# Mass in the electrolyte (Lithium ion)
######################

N_e_Li = -tor*D_e_Li(c_e_Li, T)*pybamm.grad(c_e_Li)

model.rhs[c_e_Li] = (1/eps)*(-pybamm.div(N_e_Li)/C_e + (1-t_plus)*(i_e_Li)/gamma_e)                  


######################
# Mass in the solid (Chloride)
######################

N_s_n_Cl = -D_n(c_s_n_Cl,T)*pybamm.grad(c_s_n_Cl)

model.rhs[c_s_n_Cl] = -(1/C_diff_n)*pybamm.div(N_s_n_Cl)*0

N_s_p_Cl = -D_p(c_s_p_Cl,T)*pybamm.grad(c_s_p_Cl)

model.rhs[c_s_p_Cl] = -(1/C_diff_p)*pybamm.div(N_s_p_Cl)

######################
# Mass in the electrolyte (Chloride ion)
######################

N_e_Cl = -tor*D_e_Cl(c_e_Cl, T)*pybamm.grad(c_e_Cl)

model.rhs[c_e_Cl] = (1/eps)*(-pybamm.div(N_e_Cl)/C_e + (1-t_minus)*(i_e_Cl)/gamma_e)   


######################
# Current in the electrolyte                                                                                               
######################

i_e = -(kappa_e(c_e_Li,c_e_Cl,T)*tor*gamma_e/C_e)*(pybamm.grad(phi_e)+ chiRT_over_Fc(c_e_Li,T)*pybamm.grad(c_e_Li) + chiRT_over_Fc(c_e_Cl,T)*pybamm.grad(c_e_Cl) + ((t_plus-1)/f_plus)*pybamm.grad(f_plus) + ((t_plus-1)/f_minus)*pybamm.grad(f_minus)) 
model.algebraic[phi_e] = pybamm.div(i_e) - i_electrolyte




### 4. State boundary conditions ------------------------------------------------------------------------------------------

model.boundary_conditions = {
    ##
    phi_s_n: {"left": (i_cell /pybamm.BoundaryValue(-sigma_eff_n, "right"), "Neumann"),"right": (pybamm.Scalar(0), "Neumann")},
    phi_s_p: {"left": (pybamm.Scalar(0), "Neumann"),"right": (i_cell /pybamm.BoundaryValue(-sigma_eff_p, "right"), "Neumann")},
    
    ##
    c_s_n_Li: {"left": (pybamm.Scalar(0), "Neumann"),"right": (-C_diff_n*i_n/a_R_n/gamma_n/D_n(c_s_surf_n_Li,T), "Neumann")},
    c_s_p_Li: {"left": (pybamm.Scalar(0), "Neumann"),"right": (pybamm.Scalar(0), "Neumann")},

    ##
    c_e_Li: {"left": (pybamm.Scalar(0), "Neumann"), "right": (pybamm.Scalar(0), "Neumann")},
    
    ##
    c_s_n_Cl: {"left": (pybamm.Scalar(0), "Neumann"),"right": (pybamm.Scalar(0), "Neumann")},
    c_s_p_Cl: {"left": (pybamm.Scalar(0), "Neumann"),"right": (-C_diff_p*i_p/a_R_p/gamma_p/D_p(c_s_surf_p_Cl,T), "Neumann")},
    
    ##
    c_e_Cl: {"left": (pybamm.Scalar(0), "Neumann"), "right": (pybamm.Scalar(0), "Neumann")},
    
    ##
    phi_e: {"left": (pybamm.Scalar(0), "Neumann"),"right": (pybamm.Scalar(0), "Neumann")},
    }

### 5. State initial conditions --------------------------------------------------------------------------------------------

model.initial_conditions = {
    phi_s_n: pybamm.Scalar(0), 
    phi_s_p: pybamm.Scalar(0),
    phi_e: -U_init_e,
    c_s_n_Li: pybamm.Scalar(0),
    c_s_n_Cl: pybamm.Scalar(0),
    c_s_p_Li: pybamm.Scalar(0),
    c_s_p_Cl: pybamm.Scalar(0),
    c_e_Li: c_e_init,
    c_e_Cl: c_e_init,
    }




### 6. State output variables ---------------------------------------------------------------------------------------------

model.variables = {
    "Negative electrolyte Lithium ion concentration": c_e_n_Li,
    "Positive electrolyte Lithium ion concentration": c_e_p_Li,
    "Negative electrolyte Chloride ion concentration": c_e_n_Cl,
    "Positive electrolyte Chloride ion concentration": c_e_p_Cl,
    "Negative particle Lithium concentration": c_s_n_Li,
    "Current [A]": I
}





### 7. Set up electrode/separator/electrode geometry--------------------------------------------------------------------------

geometry = {
    "negative electrode": {"x_n": {"min": pybamm.Scalar(0), "max": l_n}},
    "separator": {"x_s": {"min": l_n, "max": l_n_l_e}},
    "positive electrode": {"x_p": {"min": l_n_l_e, "max": pybamm.Scalar(1)}},
    "negative particle": {"r_n": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}},
    "positive particle": {"r_p": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}},
    "current collector": {"z": {"position": 1}}
}

### 8. Setting parameter values--------------------------------------------------------------------------------------------------

param = pybamm.ParameterValues(
    {
        #cell dimensions
        "Negative electrode thickness [m]": 0.0001,
        "Distance between two electrodes [m]": 2.5e-05,
        "Positive electrode thickness [m]": 0.0001,
        "Electrode width [m]": 0.207,
        "Electrode height [m]": 0.137,
        "Negative current collector thickness [m]": 2.5e-05,
        "Positive current collector thickness [m]": 2.5e-05,
        "Negative particle radius [m]":1e-05,
        "Positive particle radius [m]":1e-05,
        #current
        "Current Function [A]": current_interpolant,
        "Typical current [A]":1,
        #voltage
        "Lower voltage cut-off [V]": 10,
        "Upper voltage cut-off [V]": 0,
        #negative electrode
        "Negative electrode active material volume fraction": 0.7,
        "Negative electrode conductivity [S.m-1]": 100,
        "Maximum concentration in Negative electrode [mol.m-3]":24983.2619938437,
        'Initial concentration in Negative electrode [mol.m-3]':0,
        "Negative electrode porosity":0.3,
        "Negative electrode exchange-current density [A.m-2]":graphite_electrolyte_exchange_current_density_Dualfoil1998,
        "Negative electrode OCP [V]":graphite_mcmb2528_ocp_Dualfoil1998,
        "Negative electrode OCP entropic change [V.K-1]":0,
        "Negative electrode diffusivity [m2.s-1]":graphite_mcmb2528_diffusivity_Dualfoil1998,
        #positive electrode
        "Positive electrode active material volume fraction": 0.7,
        "Positive electrode conductivity [S.m-1]": 70,
        "Maximum concentration in Positive electrode [mol.m-3]":51217.9257309275,
        'Initial concentration in Positive electrode [mol.m-3]':0,
        "Positive electrode porosity":0.3,
        "Equilibrium constant for adsorbtion and desorption of chloride anions":8.51e-07,
        "Rate constant for Tafel step":1.239e4,
        "Activity of chloride anions":0.5,
        "Reaction rate of Chloride at Positive Electrode":0.5,
        "Positive electrode OCP [V]":lico2_ocp_Dualfoil1998,
        "Positive electrode OCP entropic change [V.K-1]":0,
        "Positive electrode diffusivity [m2.s-1]":lico2_diffusivity_Dualfoil1998,
        "Positive electrode exchange-current density [A.m-2]": lico2_electrolyte_exchange_current_density_Dualfoil1998,
        #electrolyte
        "Initial concentration in electrolyte [mol.m-3]": 1000,
        "Separator porosity":1,
        "Typical electrolyte concentration [mol.m-3]":1000,
        'Cation transference number':0.4,
        'Anion transference number':0.6,
        "Mean molar activity coefficient":1,
        "Cation molar activity coefficient":1,
        "Anion molar activity coefficient":1,
        "Electrolyte conductivity [S.m-1]":0.1, #electrolyte_conductivity_Capiglia1999,
        "Electrolyte Lithium diffusivity [m2.s-1]":electrolyte_diffusivity_Capiglia1999,
        "Electrolyte Chloride diffusivity [m2.s-1]":electrolyte_diffusivity_Capiglia1999,
        #temperature
        "Reference temperature [K]": 298.15,
        "Initial temperature [K]": 298.15,
    }
)

### 9. Process model and geometry --------------------------------------------------------------------------------------------------

param.process_model(model)
param.process_geometry(geometry)

### 10. Mesh and discretize --------------------------------------------------------------------------------------------------------

submesh_types = {
    "negative electrode": pybamm.Uniform1DSubMesh,
    "separator": pybamm.Uniform1DSubMesh,
    "positive electrode": pybamm.Uniform1DSubMesh,
    "negative particle": pybamm.Uniform1DSubMesh,
    "positive particle": pybamm.Uniform1DSubMesh,
    "current collector": pybamm.SubMesh0D,
}

var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 5, var.r_p: 5}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)



spatial_methods = {
    "macroscale": pybamm.FiniteVolume(),
    "negative particle": pybamm.FiniteVolume(),
    "positive particle": pybamm.FiniteVolume(),
    "current collector": pybamm.ZeroDimensionalSpatialMethod()
}
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

### 11. Solve ----------------------------------------------------------------------------------------------------------------------------
t_eval = np.linspace(0, 1800, 100)
#solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-3)
# load solvers
safe_solver = pybamm.CasadiSolver(atol=1, rtol=1, mode="safe")
fast_solver = pybamm.CasadiSolver(atol=1e-3, rtol=1e-3, mode="fast")
#solution = solver.solve(model, t_eval)
sim = pybamm.Simulation(model, solver=safe_solver)
sim.solve(t_eval)


### 12. Plot -----------------------------------------------------------------------------------------------------------------------------

plot = pybamm.QuickPlot(
    sim,
    [
        "Negative electrolyte Lithium ion concentration",
        "Positive electrolyte Lithium ion concentration",
        # "Negative electrolyte Chloride ion concentration",
        # "Positive electrolyte Chloride ion concentration",
        "Negative electrode Lithium concentration [mol.m-3]",
        "Current [A]"
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()