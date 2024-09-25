import json
import sys
from record import Data,Latency
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
    latency = Latency()
    model = Modelparameter(info)
    step1 = Step1(info)
    step2 = Step2(info)
    step3 = Step3(info)
    step4 = Step4(info)
    step5 = Step5(info)
    step6 = Step6(info)

    Mem_Usage = 4*(model.n_layer/model.PP)*(model.n_embd*model.n_embd*4+model.n_hidden*model.n_embd*2)   #Model Size after parallel
    Mem_Usage += 4*(model.n_layer/model.PP)*(model.n_embd*(model.input_tokens+model.output_tokens)/model.TP)    #K/V Cache

    Mem_Capacity = info.get("hardware").get("memory capacity")
    Mem_Capacity *= (1024*1024*1024) 


    

    generated = 0
    for generated in range(model.output_tokens):
        step1.count(generated)
        step2.count(generated)
        step3.count(generated)
        step4.count(generated)
        step5.count(generated)
        step6.count(generated)
        generated += 1

    step1.update()
    step2.update()
    step3.update()
    step4.update()
    step5.update()
    step6.update()

    data.step1_flops = step1.flopsb+step1.flopsa 
    data.step2_flops = step2.flopsb+step2.flopsa
    data.step3_flops = step3.flopsb+step3.flopsa
    data.step4_flops = step4.flopsb+step4.flopsa
    data.step5_flops = step5.flopsb+step5.flopsa
    data.step6_flops = step6.flopsb+step6.flopsa

    data.step1_bank = step1.inter_bankb+step1.inter_banka 
    data.step2_bank = step2.inter_bankb+step2.inter_banka
    data.step3_bank = step3.inter_bankb+step3.inter_banka
    data.step4_bank = step4.inter_bankb+step4.inter_banka
    data.step5_bank = step5.inter_bankb+step5.inter_banka
    data.step6_bank = step6.inter_bankb+step6.inter_banka

    data.step1_channel = step1.inter_channelb+step1.inter_channela 
    data.step2_channel = step2.inter_channelb+step2.inter_channela
    data.step3_channel = step3.inter_channelb+step3.inter_channela
    data.step4_channel = step4.inter_channelb+step4.inter_channela
    data.step5_channel = step5.inter_channelb+step5.inter_channela
    data.step6_channel = step6.inter_channelb+step6.inter_channela

    data.step1_stack = step1.inter_stack
    data.step2_stack = step2.inter_stack
    data.step3_stack = step3.inter_stack
    data.step4_stack = step4.inter_stack
    data.step5_stack = step5.inter_stack
    data.step6_stack = step6.inter_stack

    latency.step1_flops_latency = step1.flops_latency
    latency.step2_flops_latency = step2.flops_latency
    latency.step3_flops_latency = step3.flops_latency
    latency.step4_flops_latency = step4.flops_latency
    latency.step5_flops_latency = step5.flops_latency
    latency.step6_flops_latency = step6.flops_latency

    latency.step1_bank_latency = step1.inter_bank_latency
    latency.step2_bank_latency = step2.inter_bank_latency
    latency.step3_bank_latency = step3.inter_bank_latency
    latency.step4_bank_latency = step4.inter_bank_latency
    latency.step5_bank_latency = step5.inter_bank_latency
    latency.step6_bank_latency = step6.inter_bank_latency

    latency.step1_channel_latency = step1.inter_channel_latency
    latency.step2_channel_latency = step2.inter_channel_latency
    latency.step3_channel_latency = step3.inter_channel_latency
    latency.step4_channel_latency = step4.inter_channel_latency
    latency.step5_channel_latency = step5.inter_channel_latency
    latency.step6_channel_latency = step6.inter_channel_latency

    latency.step1_stack_latency = step1.inter_stack_latency
    latency.step2_stack_latency = step2.inter_stack_latency
    latency.step3_stack_latency = step3.inter_stack_latency
    latency.step4_stack_latency = step4.inter_stack_latency
    latency.step5_stack_latency = step5.inter_stack_latency
    latency.step6_stack_latency = step6.inter_stack_latency



    data.iterate_layer(model.n_layer/model.PP)
    latency.iterate_layer(model.n_layer/model.PP)

    plot(model.PP,data,latency)

if __name__ == "__main__":
    simulate()