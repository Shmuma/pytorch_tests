from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.io import output_file, save
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn, NumberFormatter


def plot_data(train_loss, test_loss, lr_values, file_name):
    output_file(file_name)
    f = figure()
    f.title.text = "Loss"
    f.plot_width = 1200
    f.plot_height = 600

    epoches, train_losses = zip(*train_loss)
    _, test_losses = zip(*test_loss)
    _, learning_rates = zip(*lr_values)

    f.line(epoches, train_losses, color='blue', legend="Train loss", line_width=2)
    f.line(epoches, test_losses, color='red', legend="Test loss", line_width=2)
    f.xaxis.axis_label = "Epoch"
    f.yaxis.axis_label = "LogLoss"

    f_lr = figure()
    f_lr.title.text = "Learning rate"
    f_lr.plot_width = 1200
    f_lr.plot_height = 600
    f_lr.line(epoches, learning_rates, color='blue', legend="LR", line_width=2)
    f_lr.xaxis.axis_label = "Epoch"


    # table with data
    source = ColumnDataSource({
        'epoch':      epoches,
        'train_loss':   train_losses,
        'test_loss':    test_losses,
        'lr':           learning_rates
    })

    columns = [
        TableColumn(field='epoch', title='Epoch'),
        TableColumn(field='train_loss', title='Train loss', formatter=NumberFormatter(format="0.0000")),
        TableColumn(field='test_loss', title='Test loss', formatter=NumberFormatter(format="0.0000")),
        TableColumn(field='lr', title='LR', formatter=NumberFormatter(format="0.0000")),
    ]

    data_table = DataTable(source=source, columns=columns, width=1200, height=600)

    save(column(data_table, f, f_lr))
