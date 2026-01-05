import cutlass
import cutlass.cute as cute
import numpy as np
from cutlass.cute.runtime import from_dlpack

@cute.jit
def print_tensor_basic(x : cute.Tensor):
  print("Basic Output:")
  cute.print_tensor(x)

@cute.jit
def print_tensor_verbose(x : cute.Tensor):
  print("Verbose Output:")
  cute.print_tensor(x, verbose=True)

def print_tensor_slice(x : cute.Tensor, coord : tuple):
  sliced_data = cute.slice_(x, coord)
  y = cute.make_rmem_tensor(sliced_data.layout, sliced_data.element_type)
  y.store(sliced_data.load())
  cute.print_tensor(y)

def tensor_print_example1():
  shape = (4,3,2)
  data = np.arange(24, dtype=np.float32).reshape(*shape)
  print_tensor_basic(from_dlpack(data))

def tensor_print_example2():
  shape = (4,3,2)
  data = np.arange(24, dtype=np.float32).reshape(*shape)
  print_tensor_verbose(from_dlpack(data))

def tensor_print_example2():
  shape = (4,3,2)
  data = np.arange(24, dtype=np.float32).reshape(*shape)
  print_tensor_slice(from_dlpack(data), )
    
print("Running basic example (verbose = off)")
tensor_print_example1()
print("Running basic example (verbose = ON)")
tensor_print_example2()
print("Running print sliced tensor (verbose = OFF)")
