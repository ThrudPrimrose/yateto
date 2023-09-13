import hashlib
import subprocess
import tempfile

from ..cache import RoutineGenerator, GpuRoutineGenerator
from ...gemm_configuration import GemmForge
from ..common import BatchedOperationsAux
import importlib.util
from .gemmgen import ExecuteGemmGen

# Optional modules
gf_spec = importlib.util.find_spec('gemmforge')
try:
  if gf_spec:
    gf = gf_spec.loader.load_module()
except:
  raise ('Cannot load gemmforge.')


class GemmforgeGemmGen(object):
  gemmforge_descriptions = list()

  def __init__(self, arch, descr, gemm_cfg):
    self._arch = arch
    self._descr = descr
    self._gemm_cfg = gemm_cfg
    self._mode = gemm_cfg.operation_name
    self._generate_code = True

    assert(isinstance(self._gemm_cfg, GemmForge))

  def set_code_generation_on(self):
    self._generate_code = True

  def set_code_generation_off(self):
    self._generate_code = False
      
  def _is_special(self, value, specials):
    result = 'generic'
    try:
      candidate = int(value)
      if candidate in specials:
        result = candidate
    except:
      pass
    return result

  def _alpha(self, alpha):
    return self._is_special(alpha, {1})

  def _beta(self, beta):
    return self._is_special(beta, {0,1})
  
  def get_last_description(self):
    assert(len(GemmforgeGemmGen.gemmforge_descriptions) > 0)
    return GemmforgeGemmGen.gemmforge_descriptions[:-1]

  def generateRoutineName(self, gemm, spp):
    name = self._gemm_cfg.operation_name
    if spp is not None:
      sha = hashlib.md5()
      sha.update(str(spp).encode())
      name += 'sparse_' + sha.hexdigest()
    return '{name}_m{M}_n{N}_k{K}_ldA{LDA}_ldB{LDB}_ldC{LDC}_alpha{alphaSubs}_beta{betaSubs}_alignedA{alignedA}_alignedC{alignedC}_transA{transA}_transB{transB}_{prefetch}'.format(
      name=name,
      alphaSubs=self._alpha(gemm['alpha']),
      betaSubs=self._beta(gemm['beta']),
      **gemm
    )
  
  def _pointer(self, term, offset2, transpose):
    if transpose:
      # swaps elements of tuple if transpose
      offset2 = offset2[::-1]
    o = term.memoryLayout.subtensorOffset(topLeftEntry=offset2)
    if o > 0:
      return '{} + {}'.format(term.name, o)
    return term.name
    
  def generate(self, cpp, routineCache):
    d = self._descr
    m, n, k = d.mnk()
    ldA = 0 if d.isACsc else d.leftTerm.memoryLayout.stridei(1)
    ldB = 0 if d.isBCsc else d.rightTerm.memoryLayout.stridei(1)
    ldC = d.result.memoryLayout.stridei(1)
    
    assert (d.transA and (k,m) in d.leftTerm.memoryLayout) or (not d.transA and (m,k) in d.leftTerm.memoryLayout)
    assert (d.transB and (n,k) in d.rightTerm.memoryLayout) or (not d.transB and (k,n) in d.rightTerm.memoryLayout)
    assert (m,n) in d.result.memoryLayout

    spp = None
    sppRows = None
    flops = 0
    if d.isACsc:
      spp = d.leftTerm.memoryLayout.entries(m, k)
      sppRows = d.leftTerm.memoryLayout.shape()[0]
      flops = 2 * len(spp) * n.size()
    elif d.isBCsc:
      spp = d.rightTerm.memoryLayout.entries(k, n)
      sppRows = d.rightTerm.memoryLayout.shape()[0]
      flops = 2 * m.size() * len(spp)
    else:
      flops = 2 * m.size() * n.size() * k.size()
    
    if gf_spec:
        aux = BatchedOperationsAux(self._arch.typename)

        matrix_a = gf.YatetoInterface.produce_dense_matrix((m, k),
                                                           d.leftTerm.memoryLayout.bbox(),
                                                           addressing=aux.deduce_addresing(d.leftTerm),
                                                           transpose=d.transA,
                                                           leading_dimension=ldA)

        matrix_b = gf.YatetoInterface.produce_dense_matrix((k, n),
                                                           d.rightTerm.memoryLayout.bbox(),
                                                           addressing=aux.deduce_addresing(d.rightTerm),
                                                           transpose=d.transB,
                                                           leading_dimension=ldB)

        matrix_c = gf.YatetoInterface.produce_dense_matrix((m, n),
                                                           d.result.memoryLayout.bbox(),
                                                           addressing=aux.deduce_addresing(d.result),
                                                           transpose=False,
                                                           leading_dimension=ldC)
        args = [
                aux.deduce_arg(d.leftTerm),
                aux.deduce_arg(d.rightTerm),
                aux.deduce_arg(d.result),
                BatchedOperationsAux.NUM_ELEMENTS_NAME,
                BatchedOperationsAux.FLAGS_NAME,
                BatchedOperationsAux.STREAM_PTR_NAME
                ]
        complete_operation_description = {"descr": d,
                                          "matrix_a": matrix_a, "matrix_b": matrix_b, "matrix_c":matrix_c, 
                                          "args":args}
        GemmforgeGemmGen.gemmforge_descriptions.append(complete_operation_description)


        if self._generate_code:
          try:
            for complete_descr in GemmforgeGemmGen.gemmforge_descriptions:
              args = complete_descr[-1]
              d = args["descr"]
              args_str = ', '.join(args)

              vm = gf.vm_factory(self._arch.name, self._arch.backend, fp_type=self._arch.typename)
              forge_generator = gf.GemmGenerator(vm)

              forge_generator.set(d.transA, d.transB, matrix_a, matrix_b, matrix_c, d.alpha, d.beta)
              routine_name = forge_generator.get_base_name()

              if not isinstance(d.alpha, float):
                args_str = f'{d.alpha}, {args_str}'

              cpp("{}({});".format(routine_name, ', '.join(args)))
              routineCache.addRoutine(routine_name, GemmForgeWriter(forge_generator, vm.get_headers()))
            GemmforgeGemmGen.gemmforge_descriptions.clear()
          except gf.GenerationError as err:
            print(f'ERROR from GemmForge: {err}')
            raise err
    else:
        raise RuntimeError('gemmforge module is not found. You can install it with pip3. '
                           'e.g., pip3 install gemmforge')
    return flops

class GemmForgeWriter(GpuRoutineGenerator):
  def __init__(self, forge_generator, headers):
    self._generator = forge_generator
    self._basename = forge_generator.get_base_name()
    self._headers = headers

  def __eq__(self, other):
    if isinstance(other, GemmForgeWriter):
      return self._basename == other._basename
    else:
      return False

  def header(self, cpp):
    cpp.includes(self._headers)

  def __call__(self, routineName, fileName):
    self._generator.generate()
    declaration = self._generator.get_launcher_header()
    launcher = self._generator.get_launcher()
    kernel = self._generator.get_kernel()

    with open(fileName, "a") as file:
      file.write(kernel)
      file.write(launcher)

    return declaration
