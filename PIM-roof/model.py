import math
class Modelparameter(object):
    def __init__(self,info):
        self.n_embd = info.get('model').get('n_embd')
        self.n_hidden = info.get('model').get('n_hidden')
        self.n_layer = info.get('model').get('n_layer')
        self.n_head = info.get('model').get('n_head')
        self.batch = info.get('input').get("batch size")
        self.input_tokens = info.get('input').get('input tokens')
        self.output_tokens = info.get('input').get('output tokens')
        self.column = info.get('deployment').get('column partition')
        self.TP = info.get('deployment').get('tensor parallelism')
        self.BPC = info.get('hardware').get('Unit per channel')
        self.CPS = info.get('hardware').get('channel per stack')
        self.HPS = self.n_head/self.TP
        self.CPH = self.CPS/self.HPS


class Step1(Modelparameter):    #communicate latency counted under hierarchy case
    def __init__(self,info):
        super().__init__(info)
        self.Flops = 0
        self.Read = 0
        self.Write = 0

    def count(self,generated):
        if self.column == "True":
            self.Flops += 3*2*self.batch*self.n_embd*self.n_embd/self.TP
            self.Read += self.batch*(self.n_embd/self.n_head)*self.HPS
            self.Write += self.batch*(self.n_embd/self.n_head)*self.BPC*self.CPS
            #self.Read += self.batch*(self.n_embd/self.CPS)/self.TP
            #self.Write += 3*self.batch*(self.n_embd/self.n_head)*self.BPC
        else:
            self.Flops += 3*2*self.batch*self.n_embd*self.n_embd/self.TP
            self.Read += 3*self.batch*(self.n_embd/self.n_head)*self.BPC*self.CPS
            self.Write += 3*self.batch*(self.n_embd/self.n_head)*self.HPS
            #self.Read += 3*self.batch*(self.n_embd/self.n_head)*self.BPC
            #self.Write += 3*self.batch*(self.n_embd/self.n_head)
        return

class Step2(Modelparameter):    #communicate latency counted under hierarchy
    def __init__(self,info):
        super().__init__(info)
        self.Flops = 0
        self.Read = 0
        self.Write = 0

    def count(self,generated):
        if self.column == "True":
            self.Flops += 2*self.batch*self.n_embd*(self.input_tokens+generated)/self.TP
            self.Read += (self.batch*(self.input_tokens+generated))*self.HPS
            self.Write += self.batch*(self.input_tokens+generated)*self.BPC*self.CPS
            #self.Read += (self.batch*(self.input_tokens+generated)/self.CPS)/self.TP
            #self.Write += self.batch*(self.input_tokens+generated)*self.BPC
        else:    
            self.Flops += 2*self.batch*self.n_embd*(self.input_tokens+generated)/self.TP
            self.Read += self.batch*(self.input_tokens+generated)*self.BPC*self.CPS
            self.Write += self.batch*(self.input_tokens+generated)*self.HPS
            #self.Read += self.batch*(self.input_tokens+generated)*self.BPC
            #self.Write += self.batch*(self.input_tokens+generated)/self.CPS
        
 
class Step3(Modelparameter):    
    def __init__(self,info):
        super().__init__(info)
        self.Flops = 0
        self.Read = 0
        self.Write = 0

    def count(self,generated):
        if self.column == "True":
            self.Flops += 2*self.batch*(self.input_tokens+generated)*self.n_embd/self.TP
            self.Read += self.batch*(self.n_embd/self.TP)
            self.Write += self.batch*self.n_embd*self.BPC*self.CPS
            #self.Read += (self.batch*self.n_embd/self.CPS)/self.TP
            #self.Write += self.batch*self.n_embd*self.BPC
        else:
            self.Flops += 2*self.batch*(self.input_tokens+generated)*self.n_embd/self.TP
            self.Read += self.batch*(self.n_embd/self.n_head)*self.BPC*self.CPS
            self.Write += self.batch*(self.n_embd/self.TP)
            #self.Read += self.batch*self.n_embd*self.BPC
            #self.Write += self.batch*self.n_embd/self.CPS
        
 
class Step4(Modelparameter):    #communicate latency counted under hierarchy case
    def __init__(self,info):
        super().__init__(info)
        self.Flops = 0
        self.Read = 0
        self.Write = 0

    def count(self,generated):
        if self.column == "True":
            self.Flops += 2*self.batch*self.n_embd*self.n_embd/self.TP
            self.Read += self.batch*(self.n_embd/self.TP)
            self.Write += self.batch*self.n_embd*self.BPC*self.CPS
            #self.Read += (self.batch*self.n_embd/self.CPS )/self.TP
            #self.Write += self.batch*self.n_embd*self.BPC
        else:
            self.Flops += 2*self.batch*self.n_embd*self.n_embd/self.TP
            self.Read += self.batch*self.n_embd*self.BPC*self.CPS
            self.Write += self.batch*(self.n_embd/self.TP)
            #self.Read += self.batch*self.n_embd*self.BPC
            #self.Write += self.batch*self.n_embd/self.CPS        

           
class Step5(Modelparameter):    #communicate latency counted under hierarchy case
    def __init__(self,info):
        super().__init__(info)
        self.Flops = 0
        self.Read = 0
        self.Write = 0

    def count(self,generated):
        if self.column == "True":
            self.Flops += 2*self.batch*self.n_embd*self.n_hidden/self.TP
            self.Read += self.batch*(self.n_hidden/self.TP)
            self.Write += self.batch*self.n_hidden*self.BPC*self.CPS
            #self.Read += (self.batch*self.n_hidden/self.CPS)/self.TP
            #self.Write += self.batch*self.n_hidden*self.BPC
        else:
            self.Flops += 2*self.batch*self.n_embd*self.n_hidden/self.TP
            self.Read += self.batch*self.n_hidden*self.BPC*self.CPS
            self.Write += self.batch*self.n_hidden/self.TP

       
class Step6(Modelparameter):    #communicate latency counted under hierarchy case
    def __init__(self,info):
        super().__init__(info)
        self.Flops = 0
        self.Read = 0
        self.Write = 0

    def count(self,generated):
        if self.column == "True":
            self.Flops += 2*self.batch*self.n_embd*self.n_hidden/self.TP
            self.Read += (self.batch*self.n_embd/self.TP)
            self.Write += self.batch*self.n_embd*self.BPC*self.CPS
        else:
            self.Flops += 2*self.batch*self.n_embd*self.n_hidden/self.TP
            self.Read += self.batch*self.n_embd*self.BPC*self.CPS
            self.Write += self.batch*self.n_embd/self.TP

  