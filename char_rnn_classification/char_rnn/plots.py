from bokeh.plotting import figure
from bokeh.io import output_file, save


def plot_data(train_loss, test_loss, file_name):
    output_file(file_name)
    f = figure()
    f.title.text = "Loss"
    f.plot_width = 1200
    f.plot_height = 600

    x, y = zip(*train_loss)
    f.line(x, y, color='blue', legend="Train loss", line_width=2)
    x, y = zip(*test_loss)
    f.line(x, y, color='red', legend="Test loss", line_width=2)
    f.xaxis.axis_label = "Epoch"
    f.yaxis.axis_label = "LogLoss"
    save(f)
