import os
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column, row


def plot_progress(losses, ratios, ratios_train, file_name):
    if os.path.exists(file_name):
        os.rename(file_name, file_name + ".old")
    output_file(file_name)
    f = figure(title="Learning progress", x_axis_label="Epoch", y_axis_label="Loss")
    f.width = 800
    f.height = 500
    f.line(range(len(losses)), losses, legend="Loss", line_width=2)

    f2 = figure(title="Test ratios of valid tokens", x_axis_label='Epoch', y_axis_label="Ratio")
    f2.width = 800
    f2.height = 500
    f2.line(range(len(losses)), ratios, legend="Ratio test", line_width=2, color='red')
    f2.line(range(len(losses)), ratios_train, legend="Ratio train", line_width=2, color='green')
    f2.legend.location = 'top_left'

    save(row(f, f2))
