

class NHomogeneousBatchLoss:
    def __init__(self, loss, *args, **kwargs):
        """
        Apply loss on list of Non-Homogeneous tensors
        """
        self.loss = loss(**kwargs)

    def __call__(self, output, target):
        """
        output: must be a list of torch tensors [1 x ? x spatial]
        target: must be a list of torch tensors or a single tensor, shape depends on loss function
        """
        assert isinstance(output, list), "output must be a list of torch tensors"

        l, it = 0, 0
        for it, o in enumerate(output):
            l = l + self.loss(o, target[it])

        return l / (it + 1)
