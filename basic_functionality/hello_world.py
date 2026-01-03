''' 
Reference: https://chrischoy.org/posts/cutedsl-basics/
'''

import cutlass
import cutlass.cute as cute

@cute.kernel
def hello_kernel():
    tidx, tidy, tidz = cute.arch.thread_idx()
    cute.printf(tidx, tidy, tidz)


@cute.jit
def hello_world():
    cutlass.cuda.initialize_cuda_context()
    '''
    The kernel is launching 1 block of size 2,2,2 where
    tid[x,y,z] range from [0-1]. 
    If you're new to cuda programming, then we're launching
    one thread-block in this program containing 8 threads, 
    which can be addressed in 3D co-ordinate space
    '''
    hello_kernel().launch(grid=(1,1,1), block=(2,2,2))

compiled = cute.compile(hello_world)
compiled()
