import numba.cuda as cuda

"""Because vim and ipynb does not get along too well,, I will print every function that I used in this .py file for reference."""
# Get GPU information
print(cuda.detect())

# Get MULTIPROCESSOR_COUNT
device = cuda.get_current_device()
print(device.MULTIPROCESSOR_COUNT)

print(
    (device.MAX_THREADS_PER_MULTI_PROCESSOR / device.WARP_SIZE)
    * (device.MAX_THREADS_PER_MULTI_PROCESSOR / device.MAX_THREADS_PER_BLOCK)
    * device.MULTIPROCESSOR_COUNT
)

print(cuda.current_context().get_memory_info())
