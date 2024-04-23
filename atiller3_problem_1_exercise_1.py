"""
Exercise 1, Part 1
"""

# Import tools
import numpy as np
import plotly.express as px


def load_tif(mov_name):  # Define function to load the .tif
    """
    :param mov_name: string, name of multi-page tif to load
    :return: movie, ndarray of multipage tif of dims t x width_pixels x height_pixels
    """
    import imageio.v2 as io
    mov_frames = io.imread(mov_name) # shape: t x width_pixels x height_pixels
    movie = np.array(mov_frames) # M for movie
    return movie


# Let's load the movie and observe the first few frames for the wiggle
mov = "TEST_MOVIE_00001-small-motion.tif"  # movie should be in same directory
M = load_tif(mov)
N = 30
fig = px.imshow(M[0:N,:,:], animation_frame=0, binary_string=True)

# Format the video to show frame title
for i in np.arange(N):
    fig["frames"][i]["layout"]["title"] = f"Frame {str(i)}"
fig.update_layout(
    minreducedwidth=300, minreducedheight=300,
    autosize=False,width=550,height=450,margin=dict(l=0,r=0,b=0,t=0,pad=0,
    ),
)
fig.show(renderer='colab') # Press "Autoscale" (maybe a few times) if GUI breaks