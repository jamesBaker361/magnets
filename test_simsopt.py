from simsopt.field import ToroidalField
from simsopt.geo import CurveXYZFourier
from simsopt.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
from simsopt.field import trace_particles_starting_on_curve,trace_particles
# NOTE: Most of the functions and classes implemented in the tracing
# NOTE: submodule can be imported directly from the field module
import numpy as np
from simsopt.geo import CurveXYZFourier
from simsopt.field import Current, Coil
from simsopt.field import BiotSavart
import matplotlib.pyplot as plt 

#make some coils
curve = CurveXYZFourier(1000, 1)  # 100 = Number of quadrature points, 1 = max Fourier mode number

curve.x = [0, 0, 1., 0., 1., 0., 0., 0., 0.]  # Set Fourier amplitudes

coil = Coil(curve, Current(1.0e4))  # 10 kAmpere-turns
ax=coil.plot()
#plt.savefig("coil.png")

curve_b = CurveXYZFourier(1000, 1)  # 100 = Number of quadrature points, 1 = max Fourier mode number
curve_b.x = [0, 0, 1., 0., 1., 0., 1., 0., 0.]  # Set Fourier amplitudes
#curve_b.y=[1., 1., 1., 0., 1., 0., 0., 0., 0.]
coil_b = Coil(curve_b, Current(1.0e4))  # 10 kAmpere-turns

coil_b.plot(ax=ax)
plt.savefig("coil.png")



#find their field

field = BiotSavart([coil])  # Multiple coils can be included in the list


nparticles = 2
m = PROTON_MASS
q = ELEMENTARY_CHARGE
tmax = 1e-5
Ekin = 10*ONE_EV
tol=1e-6

res_tys,res_phi_hits=trace_particles(field,np.array([[0,0,0.1],[0,0.1,0.1]]),[0.1,0.15],tmax=tmax,mass=m,charge=ELEMENTARY_CHARGE,Ekin=Ekin,tol=tol,mode="full",forget_exact_path=True)

print(res_tys)
print(res_phi_hits)