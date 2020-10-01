import hashlib
import subprocess
import tempfile

from ..cache import RoutineGenerator, GpuRoutineGenerator
from ...gemm_configuration import BLASlike, CodeGenerator, GemmForge

from gemmforge import DenseMatrix, GemmGenerator, GenerationError
from gemmforge import arch as GemmForgeArch
from ...ast.indices import BoundingBox
import re


class GemmGen(object):
  NUM_ELEMENTS_NAME = 'NumElements'
  EXTRA_OFFSET_NAME = 'ExtraOffset'

  def __init__(self, arch, descr, gemm_cfg):
    self._arch = arch
    self._descr = descr
    self._gemm_cfg = gemm_cfg
    self._mode = gemm_cfg.operation_name

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

  def generateRoutineName(self, gemm, spp):
    name = self._gemm_cfg.operation_name
    if spp is not None:
      sha = hashlib.md5()
      sha.update(str(spp).encode())
      name += 'sparse_' + sha.hexdigest()
    return '{name}_m{M}_n{N}_k{K}_ldA{LDA}_ldB{LDB}_ldC{LDC}_alpha{alphaSubs}_beta{betaSubs}_alignedA{alignedA}_alignedC{alignedC}_{prefetch}'.format(
      name=name,
      alphaSubs=self._alpha(gemm['alpha']),
      betaSubs=self._beta(gemm['beta']),
      **gemm
    )
  
  def _pointer(self, term, offset2, transpose):
    if transpose:
      offset2 = offset2[::-1]
    o = term.memoryLayout.subtensorOffset(offset2)
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
    
    if isinstance(self._gemm_cfg, BLASlike):
      cpp(  self._gemm_cfg.call(d.transA,
                                d.transB,
                                m.size(), n.size(), k.size(),
                                d.alpha, self._pointer(d.leftTerm, (m.start, k.start), d.transA), ldA,
                                self._pointer(d.rightTerm, (k.start, n.start), d.transB), ldB,
                                d.beta, self._pointer(d.result, (m.start, n.start), False), ldC))

    elif isinstance(self._gemm_cfg, GemmForge):

      def deduce_addresing(term):
        if term.is_compute_constant:
          return 'none'
        temp_variable_name = re.compile(r'_tmp*')
        if temp_variable_name.match(term.name):
          return 'strided'
        else:
          return 'pointer_based'

      def deduce_bbox(rows_range, cols_range, is_trans, ml_bbox):
        if is_trans:
          bbox = [cols_range.start - ml_bbox[0].start,
                  rows_range.start - ml_bbox[1].start,
                  cols_range.stop - ml_bbox[0].start - 1,
                  rows_range.stop - ml_bbox[1].start - 1]
        else:
          bbox = [rows_range.start - ml_bbox[0].start,
                  cols_range.start - ml_bbox[1].start,
                  rows_range.stop - ml_bbox[0].start - 1,
                  cols_range.stop - ml_bbox[1].start - 1]
        return bbox


      matrix_a = DenseMatrix(num_rows=d.leftTerm.memoryLayout._bbox[0].stop,
                             num_cols=d.leftTerm.memoryLayout._bbox[1].stop,
                             addressing=deduce_addresing(d.leftTerm),
                             bbox=deduce_bbox(m, k, d.transA, d.leftTerm.memoryLayout._bbox),
                             transpose=d.transA)

      matrix_b = DenseMatrix(num_rows=d.rightTerm.memoryLayout._bbox[0].stop,
                             num_cols=d.rightTerm.memoryLayout._bbox[1].stop,
                             addressing=deduce_addresing(d.rightTerm),
                             bbox=deduce_bbox(k, n, d.transB, d.rightTerm.memoryLayout._bbox),
                             transpose=d.transB)

      matrix_c = DenseMatrix(num_rows=d.result.memoryLayout._bbox[0].stop,
                             num_cols=d.result.memoryLayout._bbox[1].stop,
                             addressing=deduce_addresing(d.result),
                             bbox=deduce_bbox(m, n, False, d.result.memoryLayout._bbox),
                             transpose=False)
      try:
        forge_generator = GemmGenerator(GemmForgeArch.produce(self._arch.name, self._arch.sub_name),
                                        self._arch.typename)
        forge_generator.generate(matrix_a, matrix_b, matrix_c, d.alpha, d.beta)
        routine_name = forge_generator.get_base_name()

        def deduce_arg(term, as_const=False):
          temp_variable_name = re.compile(r'_tmp*')
          if term.is_compute_constant or temp_variable_name.match(term.name):
            extra_offset = '0'
          else:
            extra_offset = f'{self.EXTRA_OFFSET_NAME}_{term.name}'

          if as_const:
            addressing = deduce_addresing(term)
            ptr = '**' if addressing == 'pointer_based' else '*'
            const_ptr_type = f'const {self._arch.typename} {ptr}'
            return f'const_cast<{const_ptr_type}>({term.name}), {extra_offset}'
          else:
            return f'{term.name}, {extra_offset}'

        args = [deduce_arg(d.leftTerm, as_const=True),
                deduce_arg(d.rightTerm, as_const=True),
                deduce_arg(d.result, as_const=False),
                self.NUM_ELEMENTS_NAME]
        args_str = ', '.join(args)

        if not isinstance(d.alpha, float):
          args_str = f'{d.alpha}, {args_str}'

        cpp("{}({});".format(routine_name, args_str))

        routineCache.addRoutine(routine_name, GemmForgeWriter(forge_generator))

      except GenerationError as err:
        print(f'ERROR from GemmForge: {err}')
        raise err


    else:
      assert not (d.transA or d.transB)

      gemm = {
        'M':            m.size(),
        'N':            n.size(),
        'K':            k.size(),
        'LDA':          ldA,
        'LDB':          ldB,
        'LDC':          ldC,
        'alpha':        self._alpha(d.alpha),
        'beta':         self._beta(d.beta),
        'alignedA':     int(d.alignedA),
        'alignedC':     int(d.alignedC),
        'prefetch':     'BL2viaC' if self._arch.enablePrefetch and d.prefetchName is not None else 'pfsigonly'
      }

      routineName = self.generateRoutineName(gemm, spp)

      if self._mode == 'pspamm':
        cpp( '{}({}, {}, {}, {}, {}, {});'.format(
          routineName,
          self._pointer(d.leftTerm, (m.start, k.start), d.transA),
          self._pointer(d.rightTerm, (k.start, n.start), d.transB),
          self._pointer(d.result, (m.start, n.start), False),
          str(d.alpha),
          str(d.beta),
          d.prefetchName if d.prefetchName is not None else 'nullptr'
        ))
      else:
        cpp( '{}({}, {}, {}, nullptr, {}, nullptr);'.format(
          routineName,
          self._pointer(d.leftTerm, (m.start, k.start), d.transA),
          self._pointer(d.rightTerm, (k.start, n.start), d.transB),
          self._pointer(d.result, (m.start, n.start), False),
          d.prefetchName if d.prefetchName is not None else 'nullptr'
        ))

      routineCache.addRoutine(routineName, ExecuteGemmGen(self._arch, gemm, spp, sppRows, self._gemm_cfg))
    
    return flops

class ExecuteGemmGen(RoutineGenerator):  
  def __init__(self, arch, gemmDescr, spp, sppRows, gemm_cfg):
    self._arch = arch
    self._gemmDescr = gemmDescr
    self._spp = spp
    self._sppRows = sppRows
    self._mode = gemm_cfg.operation_name
    self._cmd = gemm_cfg.cmd
    self._blockSize = gemm_cfg.blockSize(gemmDescr['M'], gemmDescr['N'], gemmDescr['K']) if hasattr(gemm_cfg, 'blockSize') else dict()
  
  def __eq__(self, other):
    return self._arch == other._arch and \
           self._gemmDescr == other._gemmDescr and \
           self._spp == other._spp
  
  def header(self, cpp):
    with cpp.PPIfndef('NDEBUG'):
      cpp('extern long long libxsmm_num_total_flops;')
      cpp('extern long long pspamm_num_total_flops;')
    with cpp.PPIf('defined( __SSE3__) || defined(__MIC__)'):
      cpp.includeSys('immintrin.h')

  def _callGenerator(self, argList):
    try:
      subprocess.call([str(arg) for arg in argList])
    except OSError:
      raise RuntimeError('GEMM code generator executable "{}" not found. (Make sure to add the folder containing the executable to your PATH.)'.format(self._cmd))
  
  def __call__(self, routineName, fileName):
    cpu_arch = self._arch.host_name if self._arch.host_name else self._arch.name

    if self._mode == 'pspamm':
      argList = [
        self._cmd,
        self._gemmDescr['M'],
        self._gemmDescr['N'],
        self._gemmDescr['K'],
        self._gemmDescr['LDA'],
        self._gemmDescr['LDB'],
        self._gemmDescr['LDC'],
        self._gemmDescr['alpha'],
        self._gemmDescr['beta'],
        '--arch',
        cpu_arch,
        '--prefetching',
        self._gemmDescr['prefetch'],
        '--output_funcname',
        routineName,
        '--output_filename',
        fileName,
        '--precision',
        self._arch.precision
      ]
      for key, val in self._blockSize.items():
        argList.extend(['--' + key, val])
    else:
      argList = [
        self._cmd,
        'dense',
        fileName,
        routineName,
        self._gemmDescr['M'],
        self._gemmDescr['N'],
        self._gemmDescr['K'],
        self._gemmDescr['LDA'],
        self._gemmDescr['LDB'],
        self._gemmDescr['LDC'],
        self._gemmDescr['alpha'],
        self._gemmDescr['beta'],
        self._gemmDescr['alignedA'],
        self._gemmDescr['alignedC'],
        cpu_arch,
        self._gemmDescr['prefetch'],
        self._arch.precision + 'P'
      ]
    if self._spp is not None:
      cols = self._gemmDescr['K'] if self._gemmDescr['LDA'] == 0 else self._gemmDescr['N']
      rows = self._gemmDescr['M'] if self._gemmDescr['LDA'] == 0 else self._gemmDescr['K']
      if self._mode == 'pspamm':
        rows = self._sppRows
      shape = (rows, cols)      
      with tempfile.NamedTemporaryFile() as temp:
        temp.write('%%MatrixMarket matrix coordinate real general\n'.encode())
        temp.write('%\n'.encode())
        temp.write('{} {} {}\n'.format(shape[0], shape[1], len(self._spp)).encode())
        for r,c in self._spp:
          temp.write('{} {} 1.0\n'.format(r+1,c+1).encode())
        temp.flush()
        if self._mode == 'libxsmm':
          argList[1] = 'sparse'
        if self._mode == 'pspamm':
          argList.append('--mtx_filename')
        argList.append(temp.name)
        self._callGenerator(argList)
    else:
      self._callGenerator(argList)

    if self._mode == 'pspamm':
      return 'void {name}(const {type}* A, const {type}* B, {type}* C, {type} alpha, {type} beta, const {type}* prefetch);'.format(name=routineName, type=self._arch.typename)
    return 'void {name}(const {type}* A, const {type}* B, {type}* C, const {type}* A_prefetch, const {type}* B_prefetch, const {type}* C_prefetch);'.format(name=routineName, type=self._arch.typename)
  

class GemmForgeWriter(GpuRoutineGenerator):
  def __init__(self, forge_generator):
    self._basename = forge_generator.get_base_name()
    self._declaration = forge_generator.get_launcher_header()
    self._launcher = forge_generator.get_launcher()
    self._kernel = forge_generator.get_kernel()

  def __eq__(self, other):
    if isinstance(other, GemmForgeWriter):
      return self._basename == other._basename
    else:
      return False

  def header(self, cpp):
    cpp.include('gemmgen_aux.h')

  def __call__(self, routineName, fileName):
    with open(fileName, "a") as file:
      file.write(self._kernel)
      file.write(self._launcher)

    return self._declaration
