import math

def warmup_linear(x, warmup=0.002):
    if warmup == 0.0:
        return 1.0
    elif x < warmup:
        return x/warmup
    return 1.0 - x


def warmup_linear_decay_exp(global_step, decay_rate, decay_steps, total_steps, warmup=0.002):
    x = global_step/total_steps
    warmup_end = warmup * total_steps
    if warmup == 0.0:
        return 1.0
    elif x < warmup:
        return x/warmup
    return decay_rate**((global_step-warmup_end)/decay_steps)

class LinearWarmupExponentialSchedule():
    def __init__(self, warmup=0.002, t_total=-1, initial_lr = 2e-5, final_lr=5e-6, decay_rate=0.99):
        self.warmup = warmup
        self.total_steps = t_total
        self.decay_rate = decay_rate
        self.warmup_end = self.warmup * t_total

        # Calculate the decay Steps
        self.decay_steps = int(math.ceil((math.log(self.decay_rate)/ math.log(final_lr/initial_lr)) * (1.0 - warmup) * t_total))

    def get_lr(self, global_step):
        x = global_step/self.total_steps
        if self.warmup == 0.0:
            return 1.0
        elif x < self.warmup:
            return x/self.warmup
        return self.decay_rate**((global_step-self.warmup_end)/self.decay_steps)
