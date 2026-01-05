import cutlass
import cutlass.cute as cute
import numpy

# CRUX, use cute.printf("") to print dynamic values

@cute.jit
def print_example(is_dynamic : bool, a : cutlass.Int32, b : cutlass.Constexpr[int]):
  '''
  Static prints: "b" is a static value known at compile time & pythons
  default print function can only print static values

  Dynamic prints: "a" is a dynamic value known at runtime and cute.printf()
  can print both static (a) and dynamic (b) at runtime.

  >>> denotes static prints
  ?>> denotes dynamic prints
  '''
  layout = cute.make_layout((a, b))
  if is_dynamic == False:
    print(">>> a", a)
    print(">>> b", b)
    print(f"{layout}")

  if is_dynamic == True:
    cute.printf("?>> {}", a)
    cute.printf("?>> {}", b)
    cute.printf("?>> {}", layout)

def print_static():
  print_example(False, cutlass.Int32(2), 8)

def print_dynamic():
  print("compiling function")
  print_example_compiled = cute.compile(print_example, False, cutlass.Int32(2), 8)
  print("running compiled function")
  print_example_compiled(True, cutlass.Int32(2))


# static function
print("Running static function")
print_static()
print("Running dynamic function")
print_dynamic()
