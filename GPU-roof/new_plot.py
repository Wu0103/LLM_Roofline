import csv
import sys
import numpy
import itertools
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

START = -1
STOP = 3
N = abs(STOP - START + 1)

def plot(info, Data):
    Atten_Flops = Data.step1_flops + Data.step2_flops + Data.step3_flops + Data.step4_flops
    Atten_Read = Data.step1_read + Data.step2_read + Data.step3_read + Data.step4_read
    Atten_Write = Data.step1_write + Data.step2_write + Data.step3_write + Data.step4_write

    FFN_Flops = Data.step5_flops + Data.step6_flops
    FFN_Read = Data.step5_read + Data.step6_read
    FFN_Write = Data.step5_write + Data.step6_write

    attendata = [Atten_Flops, Atten_Read, Atten_Write]
    csv_file_path = "./files/attendata.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(attendata)

    ffndata = [FFN_Flops, FFN_Read, FFN_Write]
    csv_file_path = "./files/ffndata.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(ffndata)

    Perf = info.get('hardware').get('peak performance')
    BW = info.get('hardware').get('Memory BW')

    Atten_Latency = max(Atten_Flops / (Perf * 1024 * 1024 * 1024), 4 * (Atten_Read + Atten_Write) / (BW * 1024 * 1024 * 1024))
    FFN_Latency = max(FFN_Flops / (Perf * 1024 * 1024 * 1024), 4 * (FFN_Read + FFN_Write) / (BW * 1024 * 1024 * 1024))

    Atten_AI = Atten_Flops / (4 * (Atten_Read + Atten_Write))
    FFN_AI = FFN_Flops / (4 * (FFN_Read + FFN_Write))
    Overall_AI = (Atten_Flops + FFN_Flops) / (4 * (Atten_Read + FFN_Read + Atten_Write + FFN_Write))

    print("Atten_Flops/Atten_Latency:", Atten_Flops / Atten_Latency)
    print("FFN_Flops/FFN_Latency:", FFN_Flops / FFN_Latency)
    print("Atten_AI:", Atten_AI)
    print("FFN_AI:", FFN_AI)
    print("Overall_AI:", Overall_AI)

    data_for_plot = ["Atten", Atten_Flops / (Atten_Latency * 1024 * 1024 * 1024), Atten_AI]
    csv_file_path = "./files/plot.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(data_for_plot)
    data_for_plot = ["FFN", FFN_Flops / (FFN_Latency * 1024 * 1024 * 1024), FFN_AI]
    with open(csv_file_path, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(data_for_plot)
    data_for_plot = ["Decoder", (Atten_Flops + FFN_Flops) / (1024 * 1024 * 1024 * (Atten_Latency + FFN_Latency)), Overall_AI]
    with open(csv_file_path, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(data_for_plot)

    hw_platforms = list()
    hw_platforms.append("GPU")
    hw_platforms.append(Perf)
    hw_platforms.append(BW)

    print(hw_platforms)

    apps = read_file("./files/plot.csv", 3)
    process(hw_platforms, apps)

    sys.exit(0)

def roofline(num_platforms, peak_performance, peak_bandwidth, intensity):
    assert isinstance(num_platforms, int) and num_platforms > 0
    assert isinstance(peak_performance, numpy.ndarray)
    assert isinstance(peak_bandwidth, numpy.ndarray)
    assert isinstance(intensity, numpy.ndarray)

    achievable_performance = numpy.zeros((1, len(intensity)))
    achievable_performance = numpy.minimum(peak_performance, peak_bandwidth * intensity)
    return achievable_performance

def process(hw_platforms, sw_apps):
    assert isinstance(hw_platforms, list)
    assert isinstance(sw_apps, list)

    x_intensity = numpy.logspace(START, STOP, num=100, base=10)
    y_performance = numpy.logspace(START, STOP, num=100, base=10)

    platforms = hw_platforms[0]

    roofs = list()
    F_I = roofline(1, numpy.array(hw_platforms[1]), numpy.array(hw_platforms[2]), x_intensity)
    roofs.append(F_I)

    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    plt.setp(axis, xticks=x_intensity, yticks=y_performance)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    axis.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=11)
    axis.set_ylabel("Theoretical Performance (GFLOPs)", fontsize=11)

    if sw_apps != []:
        apps = [a[0] for a in sw_apps]
        Performance = numpy.array([a[1] for a in sw_apps])
        apps_intensity = numpy.array([a[2] for a in sw_apps])

    axis.set_xscale('log', basex=10)
    axis.set_yscale('log', basey=10)

    axis.plot(x_intensity, roofs[0], label=platforms, color='#D2691E')

    x_point = x_intensity[70]
    y_point = roofs[0][70]
    axis.annotate('GPU Perf: 7 TFLOPS', xy=(x_point, y_point), xytext=(x_point, y_point + 1500),
                  horizontalalignment='center', verticalalignment='top', fontsize=10)

    x_point = x_intensity[40]
    y_point = roofs[0][40]
    axis.text(x_point, y_point, 'Memory BW: 900 GB/s', fontsize=10, rotation=65,
              verticalalignment='top', horizontalalignment='right')

    color = matplotlib.pyplot.cm.Plasma(numpy.linspace(0, 1, len(apps)))
    marker = itertools.cycle(('o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd'))
    if sw_apps != []:
        for idx, val in enumerate(apps):
            axis.plot(apps_intensity[idx], Performance[idx], label=val, linestyle='-.', marker=next(marker), color=color[idx])

    fig.tight_layout()
    axis.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, prop={'size': 9})
    plt.show()
    plt.savefig('newplot_roofline.png', dpi=500)

def read_file(filename, row_len):
    assert isinstance(row_len, int)
    elements = list()
    try:
        in_file = open(filename, 'r') if filename is not None else sys.stdin
        reader = csv.reader(in_file, dialect='excel')
        for row in reader:
            if len(row) != row_len:
                sys.exit(1)
            element = tuple([row[0]] + [float(r) for r in row[1:]])
            elements.append(element)
        if filename is not None:
            in_file.close()
    except IOError as ex:
        print(ex, file=sys.stderr)
        sys.exit(1)
    return elements
