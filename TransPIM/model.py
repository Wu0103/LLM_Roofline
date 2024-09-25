import math
class Modelparameter(object):
    def __init__(self,info):
        self.n_embd = info.get('model').get('n_embd')
        self.n_hidden = info.get('model').get('n_hidden')
        self.n_layer = info.get('model').get('n_layer')
        self.n_head = info.get('model').get('n_head')
        self.each_head = self.n_embd/self.n_head
        self.HPC = info.get("deployment").get("head per channel")
        self.CPS = info.get("hardware").get("channel per stack")
        self.BPC = info.get("hardware").get("bank per channel")
        self.Peakperf = info.get("hardware").get("peak performance")
        self.bankperf = math.ceil(self.Peakperf/(self.CPS*self.BPC))*1024*1024*1024 #FLOPs
        self.bankbw = info.get("hardware").get("bank level bw")
        self.channelbw = info.get("hardware").get("channel level bw")
        self.stackbw = info.get("hardware").get("stack level bw")
        self.batch = info.get('input').get("batch size")
        self.input_tokens = info.get('input').get('input tokens')
        self.output_tokens = info.get('input').get('output tokens')
        self.TP = info.get("deployment").get("tensor parallelism")
        self.PP = info.get("deployment").get("pipeline parallelism")

def compute(compute,range,bw,size,pattern,algorithm):
    latency = 0
    reduce_size = 0
    a = 10/500000000
    size = 4*size
    bw = float(bw)
    bw = bw*1024*1024*1024
    if pattern == "All Gather":
        if algorithm == "Ring":
            latency = (range-1)*(size/bw) + (range-1)*a
        elif algorithm == "Recursive_Doubling": #range is power of 2
            latency = (range-1)*(size/bw) + round(math.log(range,2))*a
        elif algorithm == "Bruck": #range can be non power of 2
            latency = (range-1)*(size/bw) + round(math.log(range,2))*a
        else:
            print("Unsupport algorithm for All Gather")
            exit()
    elif pattern == "Broadcast":
        if algorithm == "Tree":
            latency = round(math.log(range,2))*(a+(size/bw))
        elif algorithm == "Vand":
            latency = (round(math.log(range,2)) + range -1)*a + 2*((range-1)/range)*(size/bw)
        elif algorithm == "Ring":
            latency = (size/bw)*range
        elif algorithm == "bank":
            latency = (size/bw)
        else:
            print("Unsupport algorithm for Broadcast")
            exit()
    elif pattern == "Reduce Scatter":
        if algorithm == "Old":
            latency = (round(math.log(range,2)) + range -1)*a + (round(math.log(range,2))+(range-1)/range)*(size/bw)
            reduce_size = (size)*round(math.log(range,2))
        elif algorithm == "Recursive_halving": #range is power of 2
            latency = round(math.log(range,2))*a + ((range-1)/range)*(size/bw)
            reduce_size = ((range-1)/range)*(size)
        else:
            print("Unsupport algorithm for Reduce Scatter")
            exit()
    elif pattern ==  "Reduce":
        if algorithm == "Tree":
            latency = round(math.log(range,2))*(a+(size/bw))
            reduce_size += round(math.log(range,2))*(size)
        elif algorithm == "Raben":
            latency = 2*round(math.log(range,2))*a + 2*((range-1)/range)*(size/bw)
            reduce_size += ((range-1)/range)*(size)
        else:
            print("Unsupport algorithm for Reduce")
            exit()
    elif pattern == "Scatter":
        if algorithm == "Ring":
            latency = ((range-1)/range)*(size/bw) + (range-1)*a#might not be true
        else:
            print("Unsupport algorithm for Scatter")
            exit()
    elif pattern =="Compute":
        size = size/4
        latency = size/compute
    else:
        print("No such communicate pattern")
        exit()
    return latency,reduce_size



class Step1(Modelparameter):    #communicate latency counted under hierarchy case
    def __init__(self,info):
        super().__init__(info)
        self.cross_channel = False
        self.CPH = math.ceil(1/self.HPC)
        self.flopsb = 0
        self.inter_bankb = 0
        self.inter_channelb = 0
        self.inter_stack = 0
        self.flopsa = 0
        self.inter_banka = 0
        self.inter_channela = 0
        self.flops_latency = 0
        self.inter_bank_latency = 0
        self.inter_channel_latency = 0
        self.inter_stack_latency = 0

    def count(self,generated):

        self.flopsb += 6*self.batch*self.n_embd*self.each_head
        self.inter_bankb += self.batch*self.each_head # Broadcast

        return
    
    def update(self):
        a,b=compute(self.bankperf,0,0,self.flopsb,"Compute",0)
        self.flops_latency += a

        a,b=compute(0,self.BPC,self.bankbw,self.inter_bankb,"Broadcast","Ring")
        self.inter_bank_latency += a
        return
    
class Step2(Modelparameter):    #communicate latency counted under hierarchy
    def __init__(self,info):
        super().__init__(info)
        self.cross_channel = False
        self.CPH = math.ceil(1/self.HPC)
        self.flopsb = 0
        self.inter_bankb = 0
        self.inter_channelb = 0
        self.inter_stack = 0
        self.flopsa = 0
        self.inter_banka = 0
        self.inter_channela = 0
        self.flops_latency = 0
        self.inter_bank_latency = 0
        self.inter_channel_latency = 0
        self.inter_stack_latency = 0

    def count(self,generated):

        self.flopsb += 2*self.batch*self.each_head*(generated+self.input_tokens/self.BPC)

        return
    
    def update(self):
        a,b=compute(self.bankperf,0,0,self.flopsb,"Compute",0)
        self.flops_latency += a
        return
    
class Step3(Modelparameter):    
    def __init__(self,info):
        super().__init__(info)
        self.cross_channel = False
        self.CPH = math.ceil(1/self.HPC)
        self.flopsb = 0
        self.inter_bankb = 0
        self.inter_channelb = 0
        self.inter_stack = 0
        self.flopsa = 0
        self.inter_banka = 0
        self.inter_channela = 0
        self.flops_latency = 0
        self.inter_bank_latency = 0
        self.inter_channel_latency = 0
        self.inter_stack_latency = 0

    def count(self,generated):
        self.flopsb += 2*self.batch*(generated+self.input_tokens/self.BPC)*self.each_head
        self.inter_bankb += self.batch*self.each_head  # Reduce Scatter
        return
    
    def update(self):
        a,b=compute(self.bankperf,0,0,self.flopsb,"Compute",0)
        self.flops_latency += a
        a,b=compute(0,self.BPC,self.bankbw,self.inter_bankb,"Reduce Scatter","Old")
        self.inter_bank_latency += a
        self.flopsa += b
        a,b=compute(self.bankperf,0,0,self.flopsa,"Compute",0) 
        self.flops_latency += a
        return
    
class Step4(Modelparameter):    #communicate latency counted under hierarchy case
    def __init__(self,info):
        super().__init__(info)
        self.cross_channel = False
        self.CPH = math.ceil(1/self.HPC)
        self.flopsb = 0
        self.inter_bankb = 0
        self.inter_channelb = 0
        self.inter_stack = 0
        self.flopsa = 0
        self.inter_banka = 0
        self.inter_channela = 0
        self.flops_latency = 0
        self.inter_bank_latency = 0
        self.inter_channel_latency = 0
        self.inter_stack_latency = 0

    def count(self,generated):
        self.flopsb += 2*self.batch*self.each_head*self.n_embd
        self.inter_bankb += self.batch*self.each_head   # Broadcast
        self.inter_channelb += self.batch*self.each_head 
        self.inter_channela += self.batch*self.n_embd
        self.inter_banka += self.batch*(self.n_embd-self.each_head) #Broadcast


    def update(self):
        a,b=compute(self.bankperf,0,0,self.flopsb,"Compute",0)
        self.flops_latency += a

        a,b=compute(0,self.BPC,self.bankbw,self.inter_bankb,"Broadcast","Ring")
        self.inter_bank_latency += a

        a,b=compute(0,self.CPS,self.channelbw,self.inter_channelb+self.inter_channela,"Broadcast","bank")
        self.inter_channel_latency += a

        a,b=compute(0,self.BPC,self.bankbw,self.inter_banka,"Broadcast","Ring")
        self.inter_bank_latency += a


            
class Step5(Modelparameter):    #communicate latency counted under hierarchy case
    def __init__(self,info):
        super().__init__(info)
        self.cross_channel = False
        self.CPH = math.ceil(1/self.HPC)
        self.flopsb = 0
        self.inter_bankb = 0
        self.inter_channelb = 0
        self.inter_stack = 0
        self.flopsa = 0
        self.inter_banka = 0
        self.inter_channela = 0
        self.flops_latency = 0
        self.inter_bank_latency = 0
        self.inter_channel_latency = 0
        self.inter_stack_latency = 0

    def count(self,generated):
        self.flopsb += 2*self.batch*self.each_head*self.n_embd*4
        self.inter_bankb += self.batch*self.each_head*4   # Broadcast
        self.inter_channelb += self.batch*self.each_head*4 
        self.inter_channela += self.batch*self.n_embd*4
        self.inter_banka += self.batch*(self.n_embd-self.each_head)*4 #Broadcast


    def update(self):
        a,b=compute(self.bankperf,0,0,self.flopsb,"Compute",0)
        self.flops_latency += a

        a,b=compute(0,self.BPC,self.bankbw,self.inter_bankb,"Broadcast","Ring")
        self.inter_bank_latency += a

        a,b=compute(0,self.CPS,self.channelbw,self.inter_channelb+self.inter_channela,"Broadcast","bank")
        self.inter_channel_latency += a

        a,b=compute(0,self.BPC,self.bankbw,self.inter_banka,"Broadcast","Ring")
        self.inter_bank_latency += a
            
class Step6(Modelparameter):    #communicate latency counted under hierarchy case
    def __init__(self,info):
        super().__init__(info)
        self.cross_channel = False
        self.CPH = math.ceil(1/self.HPC)
        self.flopsb = 0
        self.inter_bankb = 0
        self.inter_channelb = 0
        self.inter_stack = 0
        self.flopsa = 0
        self.inter_banka = 0
        self.inter_channela = 0
        self.flops_latency = 0
        self.inter_bank_latency = 0
        self.inter_channel_latency = 0
        self.inter_stack_latency = 0

    def count(self,generated):
        self.flopsb += 2*self.batch*self.each_head*self.n_embd
        self.inter_bankb += self.batch*self.each_head   # Broadcast
        self.inter_channelb += self.batch*self.each_head 
        self.inter_channela += self.batch*self.n_embd
        self.inter_banka += self.batch*(self.n_embd-self.each_head) #Broadcast


    def update(self):
        a,b=compute(self.bankperf,0,0,self.flopsb,"Compute",0)
        self.flops_latency += a

        a,b=compute(0,self.BPC,self.bankbw,self.inter_bankb,"Broadcast","Ring")
        self.inter_bank_latency += a

        a,b=compute(0,self.CPS,self.channelbw,self.inter_channelb+self.inter_channela,"Broadcast","bank")
        self.inter_channel_latency += a

        a,b=compute(0,self.BPC,self.bankbw,self.inter_banka,"Broadcast","Ring")
        self.inter_bank_latency += a
  