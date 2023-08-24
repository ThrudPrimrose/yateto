from ..common import BatchedOperationsAux

# Optional modules
import importlib.util
gf_spec = importlib.util.find_spec('gemmforge')
try:
  if gf_spec:
    gf = gf_spec.loader.load_module()
except:
  raise ('Cannot load gemmforge.')

class GemmforgeProduct(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr

  def generate(self, cpp, routineCache):
    """Generates a tensor equation of a form: B = beta * B + alpha * A
    Args:
      cpp (IO): a file stream
      routineCache:

    Returns:

    """
    if (gf_spec):
      d = self._descr  # type: product description
      print(d)
      loopRanges = list()
      for i in range(len(d.result.indices)):
          m = d.loopRanges[d.result.indices[i]]
          loopRanges.append(m)
      #m = d.loopRanges[d.result.indices[0]]
      #n = d.loopRanges[d.result.indices[1]]
      print(loopRanges)
      alpha = d.alpha

      aux = BatchedOperationsAux(self._arch.typename)
      tensor_a = gf.YatetoInterface.produce_dense_tensor(loopRanges,
                                                         d.leftTerm.memoryLayout.bbox(),
                                                         addressing=aux.deduce_addresing(d.leftTerm),
                                                         transpose=False)

      tensor_b = gf.YatetoInterface.produce_dense_tensor(loopRanges,
                                                         d.rightTerm.memoryLayout.bbox(),
                                                         addressing=aux.deduce_addresing(d.rightTerm),
                                                         transpose=False)

      tensor_c = gf.YatetoInterface.produce_dense_tensor(loopRanges,
                                                         d.result.memoryLayout.bbox(),
                                                         addressing=aux.deduce_addresing(d.result),
                                                         transpose=False)

      try:
        vm = gf.vm_factory(self._arch.name, self._arch.backend, fp_type=self._arch.typename)
        forge_generator = gf.ProductGenerator(vm)
        forge_generator.set(tensor_a, tensor_b, tensor_c, alpha, d.beta)
        routine_name = forge_generator.get_base_name()

        args = [str(alpha),
                aux.deduce_arg(d.leftTerm),
                aux.deduce_arg(d.rightTerm),
                aux.deduce_arg(d.result),
                BatchedOperationsAux.NUM_ELEMENTS_NAME,
                BatchedOperationsAux.FLAGS_NAME,
                BatchedOperationsAux.STREAM_PTR_NAME]
        cpp("{}({});".format(routine_name, ', '.join(args)))

        routineCache.addRoutine(routine_name, GemmForgeWriter(forge_generator, vm.get_headers()))

      except gf.GenerationError as err:
        print("ERROR: {}".format(err))
        raise err

      #return m.size() * n.size()

    else:
      raise RuntimeError('gemmforge module is not found. You can install it with pip3. '
                         'e.g., pip3 install gemmforge')
