import csv
import sys
import numpy
import matplotlib.pyplot
import matplotlib
import numpy as np

START = -3
STOP = 7
N = abs(STOP - START + 1)
def plot(PP,Data,Latency):
    atten_flops = Data.step1_flops + Data.step2_flops + Data.step3_flops + Data.step4_flops
    atten_bank = Data.step1_bank + Data.step2_bank + Data.step3_bank + Data.step4_bank
    atten_channel = Data.step1_channel + Data.step2_channel + Data.step3_channel + Data.step4_channel
    atten_stack = Data.step1_stack + Data.step2_stack + Data.step3_stack + Data.step4_stack

    ffn_flops = Data.step5_flops + Data.step6_flops
    ffn_bank = Data.step5_bank + Data.step6_bank
    ffn_channel = Data.step5_channel + Data.step6_channel
    ffn_stack = Data.step5_stack + Data.step6_stack

    
    atten_flops_latency = Latency.step1_flops_latency + Latency.step2_flops_latency + Latency.step3_flops_latency + Latency.step4_flops_latency
    atten_bank_latency = Latency.step1_bank_latency + Latency.step2_bank_latency + Latency.step3_bank_latency + Latency.step4_bank_latency
    atten_channel_latency = Latency.step1_channel_latency + Latency.step2_channel_latency + Latency.step3_channel_latency + Latency.step4_channel_latency
    atten_stack_latency = Latency.step1_stack_latency + Latency.step2_stack_latency + Latency.step3_stack_latency + Latency.step4_stack_latency
    
    ffn_flops_latency = Latency.step5_flops_latency + Latency.step6_flops_latency
    ffn_bank_latency = Latency.step5_bank_latency + Latency.step6_bank_latency
    ffn_channel_latency = Latency.step5_channel_latency + Latency.step6_channel_latency
    ffn_stack_latency = Latency.step5_stack_latency + Latency.step6_stack_latency
    

    attendata = [atten_flops,atten_bank,atten_channel,atten_stack]
    csv_file_path = "./files/attendata.csv"
    # 使用 "a" 模式打开文件，如果文件不存在则创建
    with open(csv_file_path, mode='a', newline='') as file:
        # 创建 CSV writer 对象
        csv_writer = csv.writer(file)
        # 写入数据到 CSV 文件
        csv_writer.writerow(attendata)

    ffndata = [ffn_flops,ffn_bank,ffn_channel,ffn_stack]
    csv_file_path = "./files/ffndata.csv"
    # 使用 "a" 模式打开文件，如果文件不存在则创建
    with open(csv_file_path, mode='a', newline='') as file:
        # 创建 CSV writer 对象
        csv_writer = csv.writer(file)
        # 写入数据到 CSV 文件
        csv_writer.writerow(ffndata)

    attenlatency = [atten_flops_latency*1000*PP,atten_bank_latency*1000*PP,atten_channel_latency*1000*PP,atten_stack_latency*1000*PP]
    csv_file_path = "./files/attenlatency.csv"
    # 使用 "a" 模式打开文件，如果文件不存在则创建
    with open(csv_file_path, mode='a', newline='') as file:
        # 创建 CSV writer 对象
        csv_writer = csv.writer(file)
        # 写入数据到 CSV 文件
        csv_writer.writerow(attenlatency)

    ffnlatency = [ffn_flops_latency*1000*PP,ffn_bank_latency*1000*PP,ffn_channel_latency*1000*PP,ffn_stack_latency*1000*PP]
    csv_file_path = "./files/ffnlatency.csv"
    # 使用 "a" 模式打开文件，如果文件不存在则创建
    with open(csv_file_path, mode='a', newline='') as file:
        # 创建 CSV writer 对象
        csv_writer = csv.writer(file)
        # 写入数据到 CSV 文件
        csv_writer.writerow(ffnlatency)
    '''
    atten_ici=atten_flops/(4*atten_intra)
    atten_eci=atten_flops/(4*atten_inter)
    ffn_ici=ffn_flops/(4*ffn_intra)
    ffn_eci=ffn_flops/(4*ffn_inter)
    data_for_plot=["PIM(atten)",atten_flops/(1024*1024*1024),atten_ici,atten_eci]
    csv_file_path = "./files/plot.csv"
    # 使用 "a" 模式打开文件，如果文件不存在则创建
    with open(csv_file_path, mode='w', newline='') as file:
        # 创建 CSV writer 对象
        csv_writer = csv.writer(file)
        # 写入数据到 CSV 文件
        csv_writer.writerow(data_for_plot)
    data_for_plot=["PIM(ffn)",ffn_flops/(1024*1024*1024),ffn_ici,ffn_eci]
    csv_file_path = "./files/plot.csv"
    # 使用 "a" 模式打开文件，如果文件不存在则创建
    with open(csv_file_path, mode='a', newline='') as file:
        # 创建 CSV writer 对象
        csv_writer = csv.writer(file)
        # 写入数据到 CSV 文件
        csv_writer.writerow(data_for_plot)

        



    hw_platforms = list()
    hw_platforms.append("PIM")
    hw_platforms.append(args.Computation_power*1024)
    hw_platforms.append(args.Intra_bandwidth)
    hw_platforms.append(args.Inter_bandwidth)

    

    
    hw_platforms = read_file("HW.csv", 4)
    apps = read_file("./files/plot.csv", 4)
    process(hw_platforms, apps)
    '''
    sys.exit(0)

def roofline(num_platforms, peak_performance, peak_bandwidth, intensity):
    assert isinstance(num_platforms, int) and num_platforms > 0
    assert isinstance(peak_performance, numpy.ndarray)
    assert isinstance(peak_bandwidth, numpy.ndarray)
    assert isinstance(intensity, numpy.ndarray)
    assert (num_platforms == peak_performance.shape[0] and
            num_platforms == peak_bandwidth.shape[0])

    achievable_performance = numpy.zeros((num_platforms, len(intensity)))
    for i in range(num_platforms):
        achievable_performance[i:] = numpy.minimum(peak_performance[i],
                                                   peak_bandwidth[i] * intensity)
    return achievable_performance


def process(hw_platforms, sw_apps):
    assert isinstance(hw_platforms, list)
    assert isinstance(sw_apps, list)

    x_intensity = numpy.logspace(START, STOP, num=100, base=10)
    y_performance = numpy.logspace(START+2, STOP, num=100, base=10)

    platforms = [p[0] for p in hw_platforms]

    roofs = list()
    F_I = roofline(len(platforms),
            numpy.array([p[1] for p in hw_platforms]),
            numpy.array([p[2] for p in hw_platforms]),
            x_intensity)
    roofs.append(F_I)
    F_E = roofline(len(platforms),
            numpy.array([p[1] for p in hw_platforms]),
            numpy.array([p[3] for p in hw_platforms]),
            x_intensity)
    roofs.append(F_E)
    I_E = roofline(len(platforms),
            numpy.array([p[2] for p in hw_platforms]),
            numpy.array([p[3] for p in hw_platforms]),
            x_intensity)
    #roofs.append(I_E)

    #fig, axis = matplotlib.pyplot.subplots(nrows=1, ncols=3,figsize=(15, 6))
    fig, axis = matplotlib.pyplot.subplots(nrows=1, ncols=2,figsize=(15, 6))
    matplotlib.pyplot.setp(axis, xticks=x_intensity,yticks=y_performance)
    matplotlib.pyplot.yticks(fontsize=12)
    matplotlib.pyplot.xticks(fontsize=12)
    axis[0].set_xlabel('Internal communication Intensity or Arithmetic Intensity (FLOP/byte)', fontsize=14)
    axis[0].set_ylabel("Theoretical Performance (GFLOP/s)", fontsize=14)
    axis[1].set_xlabel('External communication Intensity (FLOP/byte)', fontsize=14)
    axis[1].set_ylabel("Theoretical Performance (GFLOP/s)", fontsize=14)
    #axis[2].set_xlabel('External communication Intensity (FLOP/byte)', fontsize=14)
    #axis[2].set_ylabel("Internal communication Intensity or Arithmetic Intensity (FLOP/byte)", fontsize=14)

    # Apps
    if sw_apps != []:
        apps = [a[0] for a in sw_apps]
        apps_intensity = numpy.array([a[1] for a in sw_apps])
        internel = numpy.array([a[2] for a in sw_apps])
        externel = numpy.array([a[3] for a in sw_apps])
    
    
    #for index in range(0,3):
    for index in range(0,2):
        axis[index].set_xscale('log', basex=10)
        axis[index].set_yscale('log', basey=10)

        for idx, val in enumerate(platforms):
            axis[index].plot(x_intensity, roofs[index][idx, 0:],label=val)

    
    if sw_apps != []:
        color = matplotlib.pyplot.cm.rainbow(numpy.linspace(0, 1, len(apps)))
        for idx, val in enumerate(apps):
            axis[0].plot(internel[idx],apps_intensity[idx], label=val,linestyle='-.', marker='o', color=color[idx])
            axis[1].plot(externel[idx],apps_intensity[idx], label=val,linestyle='-.', marker='o', color=color[idx])
            #axis[2].plot(externel[idx], internel[idx], label=val,linestyle='-.', marker='o', color=color[idx])
    
    
    fig.tight_layout()
    axis[0].legend(loc='upper left', prop={'size': 9})
    matplotlib.pyplot.show()
    matplotlib.pyplot.savefig('plot_roofline.png', dpi=500 )


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

