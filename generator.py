class KspaceGenerator:
    def __init__(self, full_kspace, max_iteration: int = 200):
        self.kspace = full_kspace.copy()
        self.full_kspace = full_kspace.copy()
        self.max_iteration = max_iteration
        self.mask = np.ones(full_kspace.shape[1:])
        self.iteration = -1

    def __len__(self):
        return self.max_iteration

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        return self.kspace, self.mask

    def __next__(self):
        if self.iteration < self.max_iteration:
            self.iteration += 1
            return self.__getitem__(self.iteration)
        else:
            raise StopIteration

    @property
    def shape(self):
        return self.kspace.shape

    @property
    def dtype(self):
        return self.kspace.dtype


class KspaceColumnGenerator(KspaceGenerator):
    def __init__(self, full_kspace, mask_cols, max_iteration: int = 200):
        super().__init__(full_kspace, max_iteration=max_iteration)
        self.mask = np.zeros(full_kspace.shape[1:])
        self.mask[:, mask_cols] = 1
        self.full_mask = self.mask.copy()
        self.full_kspace.flags.writeable = False

    def __getitem__(self, idx):
        return self.full_kspace, self.full_mask


class KspaceOnlineColumnGenerator(KspaceGenerator):
    """A Mask generator, adding new sampling column at each iteration
    Parameters
    ----------
    full_kspace: the fully sampled kspace for every coils
    mask_cols: the final mask to be sampled, composed of columns
     the 2D dimension of the k-space to be sample
    max_iteration: the number of steps to be use.
     If max_iteration = -1, use the number of columns
    from_center: if True, the column are added into the mask starting
    from the center and alternating left/right.
    """

    def __init__(self, full_kspace, mask_cols, max_iteration: int = -1, from_center: bool = True):
        super().__init__(full_kspace)
        self.full_kspace = full_kspace.copy()
        kspace_dim = full_kspace.shape[1:]
        self.full_mask = np.zeros(kspace_dim, dtype="int")
        self.full_mask[:, mask_cols] = 1
        self.mask = np.zeros(kspace_dim, dtype="int")
        self.kspace = np.zeros_like(full_kspace)
        self.max_iteration = max_iteration if max_iteration >= 0 else len(mask_cols)
        # reorder the column sampling by starting in the center
        # and alternating left/right expansion
        if from_center:
            center_pos = np.argmin(np.abs(mask_cols - kspace_dim[1] // 2))
            mask_cols = list(mask_cols)
            left = mask_cols[center_pos::-1]
            right = mask_cols[center_pos + 1:]
            new_cols = []
            while left or right:
                if left:
                    new_cols.append(left.pop(0))
                if right:
                    new_cols.append(right.pop(0))
            self.cols = np.array(new_cols)
        else:
            self.cols = mask_cols

    def __getitem__(self, idx):
        idx = min(idx, len(self.cols))
        self.mask[:, self.cols[:idx]] = 1
        self.kspace.flags.writeable = True
        self.kspace = self.full_kspace * self.mask[np.newaxis, ...]
        self.kspace.flags.writeable = False
        return self.kspace, self.mask
