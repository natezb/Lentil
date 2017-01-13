import matplotlib.pyplot as plt
from lentil import BeamParam, Lens
from lentil.plotting import plot_profile

# Indices of refraction
n0, nc = 1, 2.18

# Create cavity elements
cavity_elems = [Lens(z='4cm', f='20mm')]

# Find tangential and sagittal cavity modes
q = BeamParam(wavlen='852nm', z0='2cm', zR='1cm')

# Beam profile inside the cavity
plot_profile(q, q, cavity_elems, z_start='0cm', z_end='10cm', cyclical=True)
plt.legend()
plt.show()
