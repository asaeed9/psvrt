class Feeder(object):
    """
    An object for feeding data
    """
    def __init__(self, raw_input_size, batch_size=1):
        self.raw_input_size = raw_input_size
        self.batch_size = batch_size

    def initialize_vars(self, **optional_args):
        raise NotImplementedError

    def single_batch(self, **optional_args):
        raise NotImplementedError
