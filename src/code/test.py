import sys
import os

sys.path.append(r"C:\Users\Vinay\Code\NN")

from lib import draw_dot, MLP

o = MLP(2, [4, 4, 1])
out = o([2, 3])
diagram = draw_dot(out)
diagram.render('MLP2[4,4,1]', format='png')