import os

import numpy as np
from renderer import MGLRenderer
from pyrr import matrix44

datdir = os.path.expanduser('~/data/ax3d/tmp/lowpoly_models_final')
objdir = os.path.join(datdir, 'schoolbus/schoolbus01_49')
objfn = os.path.join(objdir, 'schoolbus01_49.obj')
texfn = os.path.join(objdir, 'SchoolBus01.png')

rdr = MGLRenderer()
rdr.loadobj(objfn, texfn)
# mat = matrix44.create_from_y_rotation(np.pi / 6)
mat = matrix44.create_from_z_rotation(np.pi / 4)
# mat = matrix44.create_identity()
img = rdr.render(mat)
os.makedirs('tmp/get', mode=0o755, exist_ok=True)
img.save('tmp/get/hello.png')
