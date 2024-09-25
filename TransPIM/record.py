class Data(object):
    def __init__(self):
        self.step1_flops=0
        self.step1_bank=0
        self.step1_channel=0    
        self.step1_stack=0
        self.step2_flops=0
        self.step2_bank=0
        self.step2_channel=0
        self.step2_stack=0
        self.step3_flops=0
        self.step3_bank=0
        self.step3_channel=0
        self.step3_stack=0
        self.step4_flops=0
        self.step4_bank=0
        self.step4_channel=0
        self.step4_stack=0
        self.step5_flops=0
        self.step5_bank=0
        self.step5_channel=0
        self.step5_stack=0
        self.step6_flops=0
        self.step6_bank=0
        self.step6_channel=0
        self.step6_stack=0

    def iterate_layer(self,layers):
        for key, value in self.__dict__.items():
            setattr(self, key, value*layers)


class Latency(object):
    def __init__(
            self,
            step1_flops_latency=0,
            step1_bank_latency=0,
            step1_channel_latency=0,
            step1_stack_latency=0,
            step2_flops_latency=0,
            step2_bank_latency=0,
            step2_channel_latency=0,
            step2_stack_latency=0,
            step3_flops_latency=0,
            step3_bank_latency=0,
            step3_channel_latency=0,
            step3_stack_latency=0,
            step4_flops_latency=0,
            step4_bank_latency=0,
            step4_channel_latency=0,
            step4_stack_latency=0,
            step5_flops_latency=0,
            step5_bank_latency=0,
            step5_channel_latency=0,
            step5_stack_latency=0,
            step6_flops_latency=0,
            step6_bank_latency=0,
            step6_channel_latency=0,
            step6_stack_latency=0,
    ):
        self.step1_flops_latency = step1_flops_latency
        self.step1_bank_latency = step1_bank_latency
        self.step1_channel_latency = step1_channel_latency
        self.step1_stack_latency = step1_stack_latency
        self.step2_flops_latency = step2_flops_latency
        self.step2_bank_latency = step2_bank_latency
        self.step2_channel_latency = step2_channel_latency
        self.step2_stack_latency = step2_stack_latency
        self.step3_flops_latency = step3_flops_latency
        self.step3_bank_latency = step3_bank_latency
        self.step3_channel_latency = step3_channel_latency
        self.step3_stack_latency = step3_stack_latency
        self.step4_flops_latency = step4_flops_latency
        self.step4_bank_latency = step4_bank_latency
        self.step4_channel_latency = step4_channel_latency
        self.step4_stack_latency = step4_stack_latency
        self.step5_flops_latency = step5_flops_latency
        self.step5_bank_latency = step5_bank_latency
        self.step5_channel_latency = step5_channel_latency
        self.step5_stack_latency = step5_stack_latency
        self.step6_flops_latency = step6_flops_latency
        self.step6_bank_latency = step6_bank_latency
        self.step6_channel_latency = step6_channel_latency
        self.step6_stack_latency = step6_stack_latency

    def iterate_layer(self,layers):
        for key, value in self.__dict__.items():
            setattr(self, key, value*layers)

