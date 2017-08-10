import os
from bokeh.plotting import figure, output_file, save


def plot_progress(losses, file_name):
    if os.path.exists(file_name):
        os.rename(file_name, file_name + ".old")
    output_file(file_name)
    f = figure(title="Learning progress", x_axis_label="Epoch", y_axis_label="Loss")
    f.width = 1280
    f.height = 640
    f.line(range(len(losses)), losses, legend="Loss", line_width=2)
    save(f)
