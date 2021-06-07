""" 
   * Source: libFER.utils.py
   * License: PBR License (Dual License)
   * Modified by ByungOk Han <byungok.han@etri.re.kr>
   * Date: 13 Mar 2021, ETRI

"""


class AverageMeter(object):

    """AverageMeter class

    Note:   class for statistics at training process
            
    """

    # Computes and stores the average and current value
    def __init__(self):

        """__init__ function

        Note: function for __init__

        """

        self.reset()

    def reset(self):

        """reset function

        Note: function for reset

        """

        self.val    = 0
        self.avg    = 0
        self.sum    = 0
        self.count  = 0
    
    def update(self, val, n=1):

        """update function

        Note: function for update

        """

        self.val   = val
        self.sum   += val * n
        self.count +=n
        self.avg    = self.sum / self.count


class Perimagestandardization_transform(object):

    """Perimagestandardization_transform class

    Note:   class for Perimagestandardization_transform
            
    """

    def __call__(self, sample):

        """__call__ function

        Note: function for __call__

        """

        return transforms.Normalize([sample[0].mean(), sample[1].mean(), sample[2].mean()],
                                    [sample[0].std(), sample[1].std(), sample[2].std()])(sample)



def rate_scheduler(epoch, boundary_epochs, rate_values):
 
    """rate_scheduler function

    Note: function for rate_scheduler

    """    

    for i, boundary_epoch in enumerate(boundary_epochs):
        if epoch < boundary_epoch:
            return rate_values[i]
    
    return rate_values[-1]
