def initialize_pycharm_gui():
    try:
        import IPython
        IPython.get_ipython().run_line_magic("gui", "qt")  # qt for interactive figures in pycharm
    except:
        pass

    try:
        import matplotlib as mpl
        mpl.use('Qt5Agg')  # qt for interactive figures in pycharm
    except (ModuleNotFoundError, ValueError):
        pass


initialize_pycharm_gui()


import matplotlib.pyplot as plt

import pyqtgraph as pg

import pyvista as pv
import pyvistaqt as pvq


def plot_embedding_3d(points, labels=None, plot_data_item=None, chart=dict(size=(0.4, 0.3), loc=(0.05, 0.05)), **kwargs): # noqa
    points = pv.PolyData(points)
    if labels is not None:
        points['colors'] = labels

    plotter = pvq.BackgroundPlotter()
    plotter.add_points(points, scalars='colors' if labels is not None else None, pickable=True, **kwargs)

    if plot_data_item is not None:
        fig, axs = plt.subplots(tight_layout=True)

        chart = pv.ChartMPL(fig, **chart)
        chart.background_color = (1.0, 1.0, 1.0, 0.5)
        plotter.add_chart(chart)

        def update_waveform_inset(picked_point):
            index = points.find_closest_point(picked_point)
            plot_data_item(fig, index)
            chart.render()

        plotter.enable_point_picking(callback=update_waveform_inset, picker='point', left_clicking=True)

    return plotter
