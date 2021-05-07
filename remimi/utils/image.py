import matplotlib.pyplot as plt

import streamlit as st

def show_image_ui(image, **imshow_arg):
    """Show image in UI.
    
    It is instead of cv2.imshow to wait for user to check a image.
    Usually it blocked, but this doesn't block and just draw in streamtlit 
    when call in streamlit thread.
    """
    is_streamlit_thread = st._is_running_with_streamlit
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect='auto', **imshow_arg)

    if is_streamlit_thread:
        st.pyplot(fig)
    else:
        plt.show()