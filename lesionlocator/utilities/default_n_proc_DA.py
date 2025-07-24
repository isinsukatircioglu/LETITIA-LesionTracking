import os


def get_allowed_n_proc_DA():
    if 'nnUNet_n_proc_DA' in os.environ.keys():
        use_this = int(os.environ['nnUNet_n_proc_DA'])
    else:
        use_this = 12  # default value

    use_this = min(use_this, os.cpu_count())
    return use_this
