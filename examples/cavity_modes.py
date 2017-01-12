from lentil import plotting as plt, Space, Mirror, Interface, find_cavity_modes
from lentil.plotting import plot_profile
# Indices of refraction
n0, nc = 1, 2.18

# Create cavity elements
cavity_elems = [Mirror(R='50mm', aoi='18deg'), Space('1.7cm'),
                Interface(n0, nc), Space('5cm', nc),
                Interface(nc, n0), Space('1.7cm'),
                Mirror(R='50mm', aoi='18deg'), Space('6.86cm'),
                Mirror(), Space('2.7cm'),
                Mirror(), Space('6.86cm')]

# Find tangential and sagittal cavity modes
qt_r, qs_r = find_cavity_modes(cavity_elems)

# Beam profile inside the cavity
plot_profile(qt_r, qs_r, '1064nm', cavity_elems, cyclical=True)
plt.legend()
plt.show()
