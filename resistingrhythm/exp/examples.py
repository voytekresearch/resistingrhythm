#%%
from bokeh.plotting import figure
from bokeh.io import show, output_notebook
from bokeh.layouts import column, row
from bokeh.models import Range1d
output_notebook()

from resistingrhythm.util import poisson_impulse

#%%
time = 1
ns, ts = poisson_impulse(time)

p = figure(plot_width=500, plot_height=300)
p.circle(ts, ns, color="grey")
p.xaxis.axis_label = 'Time (s)'
p.yaxis.axis_label = 'N'
p.x_range = Range1d(0, time)
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
show(p)

#%%
print(1)