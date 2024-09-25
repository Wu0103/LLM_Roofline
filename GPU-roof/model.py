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




class Step1(Modelparameter):    #communicate latency counted under hierarchy case
    def __init__(self,info):
        super().__init__(info)
        self.Flops = 0
        self.Read = 0
        self.Write = 0

    def count(self,generated):
        self.Flops += 3*2*self.batch*self.n_embd*self.n_embd
        self.Read += 3*self.n_embd*self.n_embd + self.batch*self.n_embd
        self.Write += 3*self.batch*self.n_embd
        return

class Step2(Modelparameter):    #communicate latency counted under hierarchy
    def __init__(self,info):
        super().__init__(info)
        self.Flops = 0
        self.Read = 0
        self.Write = 0

    def count(self,generated):
        self.Flops += 2*self.batch*self.n_embd*(self.input_tokens+generated)
        self.Read += self.batch*self.n_embd*(self.input_tokens+generated) + self.batch*self.n_embd
        self.Write += self.batch*(self.input_tokens+generated)
        
 
class Step3(Modelparameter):    
    def __init__(self,info):
        super().__init__(info)
        self.Flops = 0
        self.Read = 0
        self.Write = 0

    def count(self,generated):
        self.Flops += 2*self.batch*(self.input_tokens+generated)*self.n_embd
        self.Read += self.batch*(self.input_tokens+generated)*self.n_embd + self.batch*(self.input_tokens+generated)
        self.Write += self.batch*self.n_embd
        
 
class Step4(Modelparameter):    #communicate latency counted under hierarchy case
    def __init__(self,info):
        super().__init__(info)
        self.Flops = 0
        self.Read = 0
        self.Write = 0

    def count(self,generated):
        self.Flops += 2*self.batch*self.n_embd*self.n_embd
        self.Read += self.n_embd*self.n_embd + self.batch*self.n_embd
        self.Write += self.batch*self.n_embd        

           
class Step5(Modelparameter):    #communicate latency counted under hierarchy case
    def __init__(self,info):
        super().__init__(info)
        self.Flops = 0
        self.Read = 0
        self.Write = 0

    def count(self,generated):
        self.Flops += 2*self.batch*self.n_embd*self.n_hidden
        self.Read += self.n_embd*self.n_hidden + self.batch*self.n_embd
        self.Write += self.batch*self.n_hidden

       
class Step6(Modelparameter):    #communicate latency counted under hierarchy case
    def __init__(self,info):
        super().__init__(info)
        self.Flops = 0
        self.Read = 0
        self.Write = 0

    def count(self,generated):
        self.Flops += self.batch*self.n_embd*self.n_hidden
        self.Read += self.n_hidden*self.n_embd + self.batch*self.n_hidden
        self.Write += self.n_embd

  