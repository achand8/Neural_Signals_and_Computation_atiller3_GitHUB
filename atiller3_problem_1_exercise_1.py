import numpy as np
import imageio.v2 as io
import plotly as ploy

ploy.io.renderers.default = "json"

plo = ploy.graph_objs
# Load the video via imageio
mov = "TEST_MOVIE_00001-small-motion.tif"
mov_frames = io.imread(mov) # shape: t x width x height

# create a scrollable visualizer for the video via plotly
fig = plo.Figure(
    data=[plo.Heatmap(z=mov_frames[0,:,:])],
    layout=plo.Layout(
        title="Frame 0",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="View",
                          method="animate",
                          args=[None])])]
    ),
    frames=[plo.Frame(data=[plo.Heatmap(z=mov_frames[i,:,:])],
            layout=plo.Layout(title_text=f"Frame {i}"))
            for i in range(1, mov_frames.shape[0])]
)

fig.show()


# # returns V=(X,Y)~N(m, Sigma)
# correls=[-0.95, -0.85, -0.75, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.95]
#
# m=[0., 0.]
# stdev=[1., 1.]
# V=bivariate_N()
# x, y=pdf_bivariate_N(m, stdev,  V)[:2]
# my_columns=[Column(x, 'x'), Column(y, 'y')]
# zvmax=[]
# for k, rho in enumerate(correls):
#     V = bivariate_N(rho = rho)
#     z = pdf_bivariate_N(m, stdev, V)[2]
#     zvmax.append(np.max(z))
#     my_columns.append(Column(z, 'z{}'.format(k + 1)))
# grid = Grid(my_columns)
# py.grid_ops.upload(grid, 'norm-bivariate1'+str(time.time()), auto_open=False)
#
# tdata=[dict(type='heatmap',
#            xsrc=grid.get_column_reference('x'),
#            ysrc=grid.get_column_reference('y'),
#            zsrc=grid.get_column_reference('z1'),
#            zmin=0,
#            zmax=zvmax[6],
#            zsmooth='best',
#            colorscale=colorscale,
#            colorbar=dict(thickness=20, ticklen=4))]
#
# title='Contour plot for bivariate normal distribution'+\
# '<br> N(m=[0,0], sigma=[1,1], rho in (-1, 1))'
#
# layout = dict(title=title,
#               autosize=False,
#               height=600,
#               width=600,
#               hovermode='closest',
#               xaxis=dict(range=[-3, 3], autorange=False),
#               yaxis=dict(range=[-3, 3], autorange=False),
#               showlegend=False,
#               updatemenus=[dict(type='buttons', showactive=False,
#                                 y=1, x=-0.05, xanchor='right',
#                                 yanchor='top', pad=dict(t=0, r=10),
#                                 buttons=[dict(label='Play',
#                                               method='animate',
#                                               args=[None,
#                                                     dict(frame=dict(duration=100,
#                                                                     redraw=True),
#                                                     transition=dict(duration=0),
#                                                     fromcurrent=True,
#                                                     mode='immediate')])])])
#
# frames=[dict(data=[dict(zsrc=grid.get_column_reference('z{}'.format(k + 1)),
#                         zmax=zvmax[k])],
#                         traces=[0],
#                         name='frame{}'.format(k),
#                         ) for k in range(len(correls))]
#
#
# fig=dict(data=data, layout=layout, frames=frames)
# py.icreate_animations(fig, filename='animheatmap'+str(time.time()))
#
# # Create and add slider
# steps = []
# for i in range(len(fig.data)):
#     step = dict(
#         method="update",
#         args=[{"visible": [False] * len(fig.data)},
#               {"title": "Slider switched to step: " + str(i)}],  # layout attribute
#     )
#     step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
#     steps.append(step)
#
# sliders = [dict(
#     active=10,
#     currentvalue={"prefix": "Frequency: "},
#     pad={"t": 50},
#     steps=steps
# )]
#
# fig.update_layout(
#     sliders=sliders
# )
#
# fig.show()