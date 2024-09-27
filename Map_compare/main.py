########
# this one is based on a UPMEM-like PIM
# but the PIM performance data is from HBM-PIM


import json
import sys
from record import Data
from model import Modelparameter,Step1,Step2,Step3,Step4,Step5,Step6
from Count import count

def read_info(file_path):
    try:
        with open(file_path, 'r') as file:
            info = json.load(file)
        return info
    except FileNotFoundError:
        print("Error: The file was not found.")
        return None
    except json.JSONDecodeError:
        print("Error: File is not a valid JSON.")
        return None

def show_info(info):
    print(f"Info: {info}")

# 读取和模拟硬件信息

def simulate(batch,maxLength,column):
    info = read_info('info.json')
    
    #show_info(info)

    data = Data()

    model = Modelparameter(info)
    step1 = Step1(info)
    step2 = Step2(info)
    step3 = Step3(info)
    step4 = Step4(info)
    step5 = Step5(info)
    step6 = Step6(info)
    

    generated = 0
    for generated in range(maxLength):
        step1.count(batch,generated,column)
        step2.count(batch,generated,column)
        step3.count(batch,generated,column)
        step4.count(batch,generated,column)
        step5.count(batch,generated,column)
        step6.count(batch,generated,column)
        generated += 1

    data.step1_flops = step1.Flops
    data.step2_flops = step2.Flops
    data.step3_flops = step3.Flops
    data.step4_flops = step4.Flops
    data.step5_flops = step5.Flops
    data.step6_flops = step6.Flops

    data.step1_read = step1.Read
    data.step2_read = step2.Read
    data.step3_read = step3.Read
    data.step4_read = step4.Read
    data.step5_read = step5.Read
    data.step6_read = step6.Read

    data.step1_write = step1.Write
    data.step2_write = step2.Write
    data.step3_write = step3.Write
    data.step4_write = step4.Write
    data.step5_write = step5.Write
    data.step6_write = step6.Write

    data.step1_flops_ = step1.Flops_
    data.step2_flops_ = step2.Flops_
    data.step3_flops_ = step3.Flops_
    data.step4_flops_ = step4.Flops_
    data.step5_flops_ = step5.Flops_
    data.step6_flops_ = step6.Flops_


    data.iterate_layer(model.n_layer)
    count(info,data,batch,maxLength,column)

if __name__ == "__main__":
    Batch = [1,4,16]
    Length = [64,256,1024]
    Column = [True,False]
    for column in Column:
        for batch in Batch:
            for maxLength in Length:
                simulate(batch,maxLength,column)