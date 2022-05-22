import os
import torch


def getAndPrintDeviceData_CUDAorCPU():
    '''
    This function is supposed to be used only once during the STDL object's init phase.
    it returns a cuda object if it exists, and if not, returns a cpu device.
    also, the function prints information on the current gpu if it exists.
    '''
    #
    print(f'\nentered `getAndPrintDeviceData_CUDAorCPU`')
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    print(f'cuda debugging allowed')
    #
    print(f'cuda device count: {torch.cuda.device_count()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    # Additional Info when using cuda
    if device.type == 'cuda':
        print(f'device name: {torch.cuda.get_device_name(0)}')
        print(f'torch.cuda.device(0): {torch.cuda.device(0)}')
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
    # NOTE: important !!!!!!
    # clearing out the cache before beginning
    torch.cuda.empty_cache()
    print(f'\nfinished: getAndPrintDeviceData_CUDAorCPU')
    return device
