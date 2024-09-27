import csv
import sys
import numpy
import itertools
import matplotlib.pyplot
import matplotlib
import numpy as np

START = -1
STOP = 3
N = abs(STOP - START + 1)
Ele_size = 2
def plot(info,Data):
    Perf = info.get('hardware').get('peak performance')
    GPUPerf = info.get('hardware').get('GPU performance')
    BW = info.get('hardware').get('bw')
    SysBW = info.get('hardware').get('System BW')
    TP = info.get('deployment').get('tensor parallelism')

    Atten_Flops = Data.step1_flops + Data.step2_flops + Data.step3_flops + Data.step4_flops
    Atten_Flops_ = Data.step1_flops_ + Data.step2_flops_ + Data.step3_flops_ + Data.step4_flops_
    FFN_Flops = Data.step5_flops + Data.step6_flops
    FFN_Flops_ = Data.step5_flops_ + Data.step6_flops_

    Atten_Read = Data.step1_read + Data.step2_read + Data.step3_read + Data.step4_read
    Atten_Write = Data.step1_write + Data.step2_write + Data.step3_write + Data.step4_write
    FFN_Read = Data.step5_read + Data.step6_read
    FFN_Write = Data.step5_write + Data.step6_write

    Atten_Latency = Atten_Flops_/(GPUPerf*1024*1024*1024)+(Atten_Flops/(Perf*1024*1024*1024/TP)+(Ele_size*(Atten_Read+Atten_Write)/(SysBW*1024*1024*1024/TP)))
    FFN_Latency = FFN_Flops_/(GPUPerf*1024*1024*1024)+(FFN_Flops/(Perf*1024*1024*1024/TP)+(Ele_size*(FFN_Read+FFN_Write)/(SysBW*1024*1024*1024/TP)))

    attendata = [Atten_Flops,Atten_Read,Atten_Write,Atten_Flops_]
    csv_file_path = "./files/attendata.csv"
    # 使用 "a" 模式打开文件，如果文件不存在则创建
    with open(csv_file_path, mode='w', newline='') as file:
        # 创建 CSV writer 对象
        csv_writer = csv.writer(file)
        # 写入数据到 CSV 文件
        csv_writer.writerow(attendata)

    ffndata = [FFN_Flops,FFN_Read,FFN_Write,FFN_Flops_]
    csv_file_path = "./files/ffndata.csv"
    # 使用 "a" 模式打开文件，如果文件不存在则创建
    with open(csv_file_path, mode='w', newline='') as file:
        # 创建 CSV writer 对象
        csv_writer = csv.writer(file)
        # 写入数据到 CSV 文件
        csv_writer.writerow(ffndata)

    Atten_AI=Atten_Flops/(Ele_size*(Atten_Read+Atten_Write))
    FFN_AI=FFN_Flops/(Ele_size*(FFN_Read+FFN_Write))
    Overall_AI = (Atten_Flops+FFN_Flops)/(Ele_size*(Atten_Read+FFN_Read+Atten_Write+FFN_Write))


    print("Atten_Flops/Atten_Latency:", Atten_Flops / Atten_Latency)
    print("FFN_Flops/FFN_Latency:", FFN_Flops / FFN_Latency)
    print("Atten_AI:", Atten_AI)
    print("FFN_AI:", FFN_AI)
    print("Overall_AI:", Overall_AI)

    data_for_plot=["Attention",Atten_Flops/(Atten_Latency*1024*1024*1024),Atten_AI]
    csv_file_path = "./files/plot.csv"
    # 使用 "a" 模式打开文件，如果文件不存在则创建
    with open(csv_file_path, mode='w', newline='') as file:
        # 创建 CSV writer 对象
        csv_writer = csv.writer(file)
        # 写入数据到 CSV 文件
        csv_writer.writerow(data_for_plot)
    data_for_plot=["FFN",FFN_Flops/(FFN_Latency*1024*1024*1024),FFN_AI]
    csv_file_path = "./files/plot.csv"
    # 使用 "a" 模式打开文件，如果文件不存在则创建
    with open(csv_file_path, mode='a', newline='') as file:
        # 创建 CSV writer 对象
        csv_writer = csv.writer(file)
        # 写入数据到 CSV 文件
        csv_writer.writerow(data_for_plot)
    data_for_plot=["Decoder",(Atten_Flops+FFN_Flops)/(1024*1024*1024*(Atten_Latency+FFN_Latency)),Overall_AI]
    csv_file_path = "./files/plot.csv"
    # 使用 "a" 模式打开文件，如果文件不存在则创建
    with open(csv_file_path, mode='a', newline='') as file:
        # 创建 CSV writer 对象
        csv_writer = csv.writer(file)
        # 写入数据到 CSV 文件
        csv_writer.writerow(data_for_plot)

    hw_platforms = list()
    hw_platforms.append("PIM")
    hw_platforms.append(Perf)
    hw_platforms.append(SysBW)

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
    achievable_performance = numpy.minimum(peak_performance,
                                                   peak_bandwidth * intensity)
    return achievable_performance

def process(hw_platforms, sw_apps):
    assert isinstance(hw_platforms, list)
    assert isinstance(sw_apps, list)

    x_intensity = numpy.logspace(START, STOP, num=1000, base=10)
    y_performance = numpy.logspace(START, STOP, num=1000, base=10)

    platforms = hw_platforms[0]

    roofs = list()
    F_I = roofline(1,
            numpy.array(hw_platforms[1]),
            numpy.array(hw_platforms[2]),
            x_intensity)
    roofs.append(F_I)

    fig, axis = matplotlib.pyplot.subplots(nrows=1, ncols=1,figsize=(4.1, 3.5))
    matplotlib.pyplot.setp(axis, xticks=x_intensity,yticks=y_performance)
    matplotlib.pyplot.yticks(fontsize=14)
    matplotlib.pyplot.xticks(fontsize=14)
    axis.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=14)
    axis.set_ylabel("Performance (GFLOPs)", fontsize=14)

    #axis.grid(True, which="both", ls="--", linewidth=0.5) # 添加网格线

    # Apps
    if sw_apps != []:
        apps = [a[0] for a in sw_apps]
        Performance = numpy.array([a[1] for a in sw_apps])
        apps_intensity = numpy.array([a[2] for a in sw_apps])
    

    axis.set_xscale('log', base=10)
    axis.set_yscale('log', base=10)

    axis.plot(x_intensity, roofs[0],color='#D2691E')
        # 要注释的点
    
    x_point = x_intensity[700]
    y_point = roofs[0][700]

    # 在线上添加文字
    axis.annotate('PIM Perf: 9.6 TFLOPS', xy=(x_point, y_point), xytext=(x_point, y_point-1000),
                horizontalalignment='center', verticalalignment='top',fontsize=12) 


    x_point = x_intensity[200]
    y_point = roofs[0][450]
    axis.text(x_point, y_point, 'Memory BW: 1024 GB/s', fontsize=12, rotation=55, 
        verticalalignment='top', horizontalalignment='center')

    marker = itertools.cycle(('o', '*', 's', '<', '>', 's', 'p', 'v', 'h', 'H', 'D', 'd'))
    if sw_apps != []:
        for idx, val in enumerate(apps):
            axis.plot(apps_intensity[idx], Performance[idx], label=val, linestyle='-.', marker=next(marker), color="orange")#color[idx])

    
    axis.legend(loc='upper center', bbox_to_anchor=(0.41, 1.2), ncol=3, prop={'size': 12}, frameon=False)
    fig.tight_layout()
    matplotlib.pyplot.show()
    matplotlib.pyplot.savefig('PIM_Roofline.png', dpi=2000)


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

