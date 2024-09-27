import math
class Modelparameter(object):
    def __init__(self,info):
        self.n_embd = info.get('model').get('n_embd')
        self.n_hidden = info.get('model').get('n_hidden')
        self.n_layer = info.get('model').get('n_layer')
        self.n_head = info.get('model').get('n_head')
        self.input_tokens = info.get('input').get('input tokens')
        #self.column = info.get('deployment').get('column partition')
        self.TP = info.get('deployment').get('tensor parallelism')
        self.BPC = info.get('hardware').get('Unit per channel')
        self.CPS = info.get('hardware').get('channel per stack')
        self.HPS = self.n_head/self.TP
        self.CPH = self.CPS/self.HPS

#### everything is count per stack, except flops_

class Step1(Modelparameter):    # Calculate Q/K/V
    def __init__(self,info):
        super().__init__(info)
        self.Flops = 0
        self.Read = 0
        self.Write = 0
        self.Flops_ = 0 # reduction

    def count(self,batch,generated,column):
        if column == True:
            self.Write += batch*self.n_embd*self.BPC*self.CPS # write input, consider no broadcast
            self.Flops += 3*2*batch*self.n_embd*self.n_embd/self.TP # FLOPS counted per Stack
            self.Read += 3*batch*(self.n_embd/self.n_head)*self.HPS # Read Q/K/V out
        else:
            self.Write += batch*self.n_embd/self.TP # write input, partitioned
            self.Flops += 3*2*batch*self.n_embd*self.n_embd/self.TP
            self.Read += 3*batch*(self.n_embd/self.n_head)*self.BPC*self.CPH*self.HPS
            self.Flops_ += 3*batch*(self.n_embd/self.n_head)*self.BPC*self.CPH*self.HPS*self.TP
        return

class Step2(Modelparameter):    #Calculate Q*K
    def __init__(self,info):
        super().__init__(info)
        self.Flops = 0
        self.Read = 0
        self.Write = 0
        self.Flops_ = 0 # reduction

    def count(self,batch,generated,column):
        if column == True:
            self.Write += batch*(self.n_embd/self.n_head)*self.HPS*self.BPC*self.CPH # write Q to all PU
            self.Write += batch*(self.n_embd/self.n_head)*self.HPS # write K in
            self.Flops += 2*batch*self.n_embd*(self.input_tokens+generated)/self.TP
            self.Read += (batch*(self.input_tokens+generated))*self.HPS # Read Q*K out
        else:   
            self.Write += batch*self.n_embd/self.TP # write Q in
            self.Write += batch*self.n_embd/self.TP # write K in
            self.Flops += 2*batch*self.n_embd*(self.input_tokens+generated)/self.TP
            self.Read += batch*(self.input_tokens+generated)*self.BPC*self.CPS
            self.Flops_ += batch*(self.input_tokens+generated)*self.BPC*self.CPS*self.TP
        
 
class Step3(Modelparameter):    # Calculate Q*K*V
    def __init__(self,info):
        super().__init__(info)
        self.Flops = 0
        self.Read = 0
        self.Write = 0
        self.Flops_ = 0 # reduction

    def count(self,batch,generated,column):
        if column == True:
            self.Write += (batch*(self.input_tokens+generated))*self.HPS*self.BPC*self.CPH # write Q*K to all PU
            self.Write += batch*(self.input_tokens+generated)/self.TP # write V in
            self.Flops += 2*batch*(self.input_tokens+generated)*self.n_embd/self.TP
            self.Read += batch*(self.n_embd/self.TP) # Read Q*K*V out
        else:
            self.Write += batch*(self.input_tokens+generated)*self.HPS # write QK in
            self.Write += batch*(self.n_embd/self.n_head)*self.HPS # write V in
            self.Flops += 2*batch*(self.input_tokens+generated)*self.n_embd/self.TP
            self.Read += batch*(self.n_embd/self.n_head)*self.BPC*self.CPS
            self.Flops_ += batch*(self.n_embd/self.n_head)*self.BPC*self.CPS*self.TP
        
 
class Step4(Modelparameter):    #Calculate O
    def __init__(self,info):
        super().__init__(info)
        self.Flops = 0
        self.Read = 0
        self.Write = 0
        self.Flops_ = 0 # reduction

    def count(self,batch,generated,column):
        if column == True:
            self.Write += batch*self.n_embd*self.BPC*self.CPS # Write Q*K*V to all PU
            self.Flops += 2*batch*self.n_embd*self.n_embd/self.TP
            self.Read += batch*(self.n_embd/self.TP)
        else:
            self.Write += batch*self.n_embd/self.TP
            self.Flops += 2*batch*self.n_embd*self.n_embd/self.TP
            self.Read += batch*self.n_embd*self.BPC*self.CPS
            self.Flops_ += batch*self.n_embd*self.BPC*self.CPS*self.TP      

           
class Step5(Modelparameter):    # Calculte FC1
    def __init__(self,info):
        super().__init__(info)
        self.Flops = 0
        self.Read = 0
        self.Write = 0
        self.Flops_ = 0 # reduction

    def count(self,batch,generated,column):
        if column == True:
            self.Write += batch*self.n_embd*self.BPC*self.CPS # Write O to all PU
            self.Flops += 2*batch*self.n_embd*self.n_hidden/self.TP
            self.Read += batch*(self.n_hidden/self.TP)
        else:
            self.Write += batch*self.n_embd/self.TP
            self.Flops += 2*batch*self.n_embd*self.n_hidden/self.TP
            self.Read += batch*self.n_hidden*self.BPC*self.CPS
            self.Flops_ += batch*self.n_hidden*self.BPC*self.CPS*self.TP

       
class Step6(Modelparameter):    #Calculate FC2
    def __init__(self,info):
        super().__init__(info)
        self.Flops = 0
        self.Read = 0
        self.Write = 0
        self.Flops_ = 0 # reduction

    def count(self,batch,generated,column):
        if column == True:
            self.Write += batch*self.n_hidden*self.BPC*self.CPS
            self.Flops += 2*batch*self.n_embd*self.n_hidden/self.TP
            self.Read += (batch*self.n_embd/self.TP)
        else:
            self.Write += batch*self.n_hidden/self.TP
            self.Flops += 2*batch*self.n_embd*self.n_hidden/self.TP
            self.Read += batch*self.n_embd*self.BPC*self.CPS
            self.Flops_ += batch*self.n_embd*self.BPC*self.CPS*self.TP

  