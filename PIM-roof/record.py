class Data(object):
    def __init__(self):
        self.step1_flops=0
        self.step1_read=0
        self.step1_write=0    
        self.step1_stack=0
        self.step2_flops=0
        self.step2_read=0
        self.step2_write=0
        self.step2_stack=0
        self.step3_flops=0
        self.step3_read=0
        self.step3_write=0
        self.step3_stack=0
        self.step4_flops=0
        self.step4_read=0
        self.step4_write=0
        self.step4_stack=0
        self.step5_flops=0
        self.step5_read=0
        self.step5_write=0
        self.step5_stack=0
        self.step6_flops=0
        self.step6_read=0
        self.step6_write=0
        self.step6_stack=0

    def iterate_layer(self,layers):
        for key, value in self.__dict__.items():
            setattr(self, key, value*layers)
