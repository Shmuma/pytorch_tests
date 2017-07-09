from bokeh.plotting import figure
from bokeh.io import output_file, save


def plot_data(loss_pairs, file_name):
    output_file(file_name)
    f = figure()
    f.title.text = "Loss"
    f.plot_width = 1200
    f.plot_height = 600

    x, y = zip(*loss_pairs)
    f.line(x, y, color='blue')
    f.xaxis.axis_label = "Epoch"
    f.yaxis.axis_label = "LogLoss"
    save(f)
