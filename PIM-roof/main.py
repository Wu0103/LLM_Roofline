import json
import sys
from record import Data
from model import Modelparameter,Step1,Step2,Step3,Step4,Step5,Step6
from Plot import plot

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

def simulate():
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
    for generated in range(model.output_tokens):
        step1.count(generated)
        step2.count(generated)
        step3.count(generated)
        step4.count(generated)
        step5.count(generated)
        step6.count(generated)
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


    data.iterate_layer(model.n_layer)


    plot(info,data)

if __name__ == "__main__":
    simulate()