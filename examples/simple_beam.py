from lentil import plotting as plt, Space, BeamParam
from lentil.plotting import plot_profile
# Indices of refraction
n0, nc = 1, 2.18

# Create cavity elements
cavity_elems = [Space('10cm')]

# Find tangential and sagittal cavity modes
q = BeamParam(wavlen='852nm', z0='2cm', zR='1cm')

# Beam profile inside the cavity
plot_profile(q, q, cavity_elems, cyclical=True)
plt.legend()
plt.show()
