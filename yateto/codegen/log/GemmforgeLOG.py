from copy import deepcopy
import importlib
import sys
from ...ast.indices import Indices
from ..common import *
from .. import gemm
from ...memory import DenseMemoryLayout
from ...gemm_configuration import GemmForge

from yateto.codegen.gemm import GemmforgeGemmGen

from yateto import codegen

gf_spec = importlib.util.find_spec('gemmforge')
try:
  if gf_spec:
    gf = gf_spec.loader.load_module()
except:
  raise ('Cannot load gemmforge.')

class GemmforgeLOG(object):
  gemmforge_log_descriptions = list()

  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr
    self._target = "gpu"

    self._generate_code = False

    # For GPUs we may only generate one function for the whole kernel, for CPU
    # Kernel is split into multiple smaller functions
    # e.g., A = B dot C dot D
    # This results with 2 binary ops in 2 functions
    # tmp0 = B dot C
    # A = tmp0 dot D
    # in GPUs we should only generate one function for both of them
    # The complete kernel has not yet been completed
    if (self._descr.result.is_temporary):
      self.set_code_generation_off()
    else:
      self.set_code_generation_on()

  def set_code_generation_on(self):
    self._generate_code = True

  def set_code_generation_off(self):
    self._generate_code = False

  def _pointer(self, cpp, targetName, baseName, term, loopIndices, const=True):
    indices = term.indices & loopIndices
    addressStr = term.memoryLayout.addressString(term.indices, indices) if len(indices) > 0 else ''
    if len(addressStr) > 0:
      addressStr = ' + ' + addressStr
    cpp('{} {}* {} = {}{};'.format(self._arch.typename, 'const' if const else '', targetName, baseName, addressStr))
    print("[_POINTER]", '{} {}* {} = {}{};'.format(self._arch.typename, 'const' if const else '', targetName, baseName, addressStr))
    return (self._arch.typename, 'const' if const else '', targetName, baseName, addressStr)

  def _alignedStart(self, term, loopIndices):
    if len(loopIndices) == 0:
      return True
    return term.memoryLayout.isAlignedAddressString(term.indices, term.indices & loopIndices)
    
  def _memLayout(self, term, I, J):
    if len(I) == 0 and len(J) == 0:
      return DenseMemoryLayout((1,1))
    elif len(I) == 0:
      ml = term.memoryLayout.vec(term.indices, J)
      return ml.withDummyDimension()
    elif len(J) == 0:
      ml = term.memoryLayout.vec(term.indices, I)
      return ml.withDummyDimension()
    elif len(term.indices) == 2:
      return term.memoryLayout
    return term.memoryLayout.unfold(term.indices, I, J)

  def _reduce(self, term, subset, memLayout):
    return reduceSpp(term.eqspp, term.indices, subset).reshape(memLayout.shape())
  
  def _defuse(self, fusedRange, term, I):
    if len(I) == 1:
      return  {next(iter(I)): fusedRange}
    return term.memoryLayout.defuse(fusedRange, term.indices, I)

  def generate(self, cpp, routineCache, gemm_cfg):
    d = self._descr
    
    A = d.leftTerm.indices - d.loopIndices
    B = d.rightTerm.indices - d.loopIndices
    C = d.result.indices - d.loopIndices
    Im = set(A) & set(C)
    In = set(B) & set(C)
    Ik = set(A) & set(B)
    
    hasOuterLoops = len(d.outerLoopIndices) > 0

    #Bring back old changes...
    #if hasOuterLoops and self._target == 'gpu':
    #  raise RuntimeError("Loop over GEMM with the outer loop hasn't been implemented yet "
    #                     "for the GPU-like architectures")

    outerAname = '_A' if hasOuterLoops else d.leftTerm.name
    outerBname = '_B' if hasOuterLoops else d.rightTerm.name
    outerCname = '_C' if hasOuterLoops else d.result.name
    outerPrefetchName = '_Cprefetch' if hasOuterLoops and d.prefetchName is not None else d.prefetchName
    
    hasInnerLoops = len(d.innerLoopIndices) > 0
    innerAname = '_Ain' if hasInnerLoops else outerAname
    innerBname = '_Bin' if hasInnerLoops else outerBname
    innerCname = '_Cin' if hasInnerLoops else outerCname
    innerPrefetchName = '_Cprefetchin' if hasInnerLoops and outerPrefetchName is not None else outerPrefetchName

    alignedStartA = not hasOuterLoops or self._alignedStart(d.leftTerm, d.outerLoopIndices)
    
    AmemLayout = self._memLayout(d.leftTerm, Im, Ik)
    BmemLayout = self._memLayout(d.rightTerm, Ik, In)
    CmemLayout = self._memLayout(d.result, Im, In)
    print(AmemLayout)

    Aeqspp = self._reduce(d.leftTerm, A, AmemLayout)
    Beqspp = self._reduce(d.rightTerm, B, BmemLayout)
    Ceqspp = self._reduce(d.result, C, CmemLayout)
    print(Aeqspp)

    gemmDescr = gemm.Description(
      leftTerm = TensorDescription(d.leftTerm.name, AmemLayout, Aeqspp, d.leftTerm.is_compute_constant, d.leftTerm.is_temporary),
      rightTerm = TensorDescription(d.rightTerm.name, BmemLayout, Beqspp, d.rightTerm.is_compute_constant, d.rightTerm.is_temporary),
      result = TensorDescription(d.result.name, CmemLayout, Ceqspp, d.result.is_compute_constant, d.result.is_temporary),
      transA = d.transA,
      transB = d.transB,
      alpha = d.alpha,
      beta = 1.0 if d.add else 0.0,
      arch = self._arch,
      alignedStartA = self._alignedStart(d.leftTerm, d.outerLoopIndices) and self._alignedStart(d.leftTerm, d.innerLoopIndices),
      alignedStartC = self._alignedStart(d.result, d.outerLoopIndices) and self._alignedStart(d.result, d.innerLoopIndices),
      prefetchName = innerPrefetchName
    )
    #print("A BEGIN:\n", gemmDescr, "A END\n")
    
    if not d.add:
      lr = dict()
      m, n, k = gemmDescr.mnk()
      lr.update(d.loopRanges)
      lr.update( self._defuse(m, d.leftTerm, Im) )
      lr.update( self._defuse(n, d.rightTerm, In) )
      writeBB = boundingBoxFromLoopRanges(d.result.indices, lr)
      initializeWithZero(cpp, self._arch, d.result, writeBB)

    class LoGBody(object):
      def __call__(s):
        if hasInnerLoops:
          print("LOGBODY")
          lg_for_loop_descr = ("OuterLoopBody", dict())
          (float_type, const_identifier, lhs, baseName, addressStr) = self._pointer(cpp, innerAname, outerAname, d.leftTerm, d.innerLoopIndices)
          lg_for_loop_descr[1]["lhs"] = {"float_type": float_type, "const_identifier": const_identifier, "lhs": lhs, "rhs": baseName, "offset": addressStr}
          (float_type, const_identifier, lhs, baseName, addressStr) = self._pointer(cpp, innerBname, outerBname, d.rightTerm, d.innerLoopIndices)
          lg_for_loop_descr[1]["rhs"] = {"float_type": float_type, "const_identifier": const_identifier, "lhs": lhs, "rhs": baseName, "offset": addressStr}
          (float_type, const_identifier, lhs, baseName, addressStr) = self._pointer(cpp, innerCname, outerCname, d.result, d.innerLoopIndices, const=False)
          lg_for_loop_descr[1]["result"] = {"float_type": float_type, "const_identifier": const_identifier, "lhs": lhs, "rhs": baseName, "offset": addressStr}
          GemmforgeLOG.gemmforge_log_descriptions.append(lg_for_loop_descr)
          print("APPEND: ", lg_for_loop_descr)

          if outerPrefetchName is not None:
            self._pointer(cpp, innerPrefetchName, outerPrefetchName, d.result, d.innerLoopIndices)
        generator = gemm.generator(self._arch, gemmDescr, gemm_cfg, self._target)
        #compare_generator = GemmforgeGemmGen(self._arch, gemmDescr, GemmForge)
        #if compared against "GemmforgeGemmGen" the type is taken as a module and does not work
        if not isinstance(generator, codegen.gemm.GemmforgeGemmGen.GemmforgeGemmGen):
          #sys.stderr.write(str(type(generator))+ ", " + str(codegen.gemm.GemmforgeGemmGen.GemmforgeGemmGen) + ", " + str(isinstance(generator, codegen.gemm.GemmforgeGemmGen.GemmforgeGemmGen)) + "\n")
          GemmforgeLOG.gemmforge_log_descriptions.clear()
          codegen.gemm.GemmforgeGemmGen.GemmforgeGemmGen.gemmforge_descriptions.clear()
          raise Exception("Non-unit stride GEMM required in both dimensions, it is not supported by Gemmforge yet")

        generator.set_code_generation_off()
        flops = generator.generate(cpp, routineCache)
        GemmforgeLOG.gemmforge_log_descriptions.append(generator.get_last_description())
        print("APPEND: ", generator.get_last_description())
        generator.set_code_generation_on()
        return flops

    class InnerLoopBody(object):
      def __call__(s):
        print("INNERLOOPBODY")
        flops = 0
        if hasOuterLoops:
          print("INNTERLOOPBODY HASOUTERLOOPS")
          print(d.outerLoopIndices)
          ilg_for_loop_descr = ("InnerLoopBody", dict())
          (float_type, const_identifier, lhs, baseName, addressStr) = self._pointer(cpp, outerAname, d.leftTerm.name, d.leftTerm, d.outerLoopIndices)
          ilg_for_loop_descr[1]["lhs"] = {"float_type": float_type, "const_identifier": const_identifier, "lhs": lhs, "rhs": baseName, "offset": addressStr}
          (float_type, const_identifier, lhs, baseName, addressStr) =  self._pointer(cpp, outerBname, d.rightTerm.name, d.rightTerm, d.outerLoopIndices)
          ilg_for_loop_descr[1]["rhs"] = {"float_type": float_type, "const_identifier": const_identifier, "lhs": lhs, "rhs": baseName, "offset": addressStr}
          (float_type, const_identifier, lhs, baseName, addressStr) =  self._pointer(cpp, outerCname, d.result.name, d.result, d.outerLoopIndices, const=False)
          ilg_for_loop_descr[1]["result"] = {"float_type": float_type, "const_identifier": const_identifier, "lhs": lhs, "rhs": baseName, "offset": addressStr}
          GemmforgeLOG.gemmforge_log_descriptions.append(ilg_for_loop_descr)
          print("APPEND: ", ilg_for_loop_descr)

          if d.prefetchName is not None:
            self._pointer(cpp, outerPrefetchName, d.prefetchName, d.result, d.outerLoopIndices)
        if d.assignLoopRanges is not None:
          gemmDescr.setBeta(0.0)
          flops += forLoopsAppendDescriptions(cpp, GemmforgeLOG.gemmforge_log_descriptions, d.innerLoopIndices, d.assignLoopRanges, LoGBody(), pragmaSimd=False)
        if d.addLoopRanges is not None:
          gemmDescr.setBeta(1.0)
          flops += forLoopsAppendDescriptions(cpp, GemmforgeLOG.gemmforge_log_descriptions, d.innerLoopIndices, d.addLoopRanges, LoGBody(), pragmaSimd=False)
        return flops

    flops = forLoopsAppendDescriptions(cpp, GemmforgeLOG.gemmforge_log_descriptions, d.outerLoopIndices, d.loopRanges, InnerLoopBody(), pragmaSimd=False)

    # TODO Always call LOG generator of gemmforge now on
    #generator = gemm.generator(self._arch, gemmDescr, gemm_cfg, self._target)
    #generator.set_code_generation_on()
    #generator.generate_buffered_descriptions(cpp, routineCache)
    if self._generate_code:
      try:
        vm = gf.vm_factory(self._arch.name, self._arch.backend, fp_type=self._arch.typename)
        forge_generator = gf.LoopOverGemmGenerator(vm)
        forge_generator.set(deepcopy(GemmforgeLOG.gemmforge_log_descriptions))
        routine_name = forge_generator.get_base_name()

        aux = BatchedOperationsAux(self._arch.typename)
        args = list()
        print(GemmforgeLOG.gemmforge_log_descriptions)
        for gemmforge_descr in GemmforgeLOG.gemmforge_log_descriptions:
          print(gemmforge_descr)
          if gemmforge_descr[0] == "gemm":
            gemm_args = gemmforge_descr[1]["args"]
            args += gemm_args
        args.append(BatchedOperationsAux.NUM_ELEMENTS_NAME)
        args.append(BatchedOperationsAux.FLAGS_NAME)
        args.append(BatchedOperationsAux.STREAM_PTR_NAME)

        if not isinstance(d.alpha, float):
          args_str = f'{d.alpha}, {args_str}'

        cpp("{}({});".format(routine_name, ', '.join(args)))
        routineCache.addRoutine(routine_name, GemmforgeGemmGen.GemmForgeWriter(forge_generator, vm.get_headers()))
        # May be cleared because the list was deepcopied
        GemmforgeLOG.gemmforge_log_descriptions.clear()
        #GemmforgeGemmGen.gemmforge_descriptions.clear()
      except gf.GenerationError as err:
        print("ERROR: {}".format(err))
        GemmforgeLOG.gemmforge_log_descriptions.clear()
        codegen.gemm.GemmforgeGemmGen.GemmforgeGemmGen.gemmforge_descriptions.clear()
        #GemmforgeGemmGen.gemmforge_descriptions.clear()
        raise err

    return flops