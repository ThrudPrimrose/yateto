from ...ast.indices import Indices
from ..common import *
from .. import gemm
from ...memory import DenseMemoryLayout

class Generic(object):
  def __init__(self, arch, descr, target):
    self._arch = arch
    self._descr = descr
    self._target = target
  
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
      leftTerm = TensorDescription(innerAname, AmemLayout, Aeqspp, d.leftTerm.is_compute_constant, d.leftTerm.is_temporary),
      rightTerm = TensorDescription(innerBname, BmemLayout, Beqspp, d.rightTerm.is_compute_constant, d.rightTerm.is_temporary),
      result = TensorDescription(innerCname, CmemLayout, Ceqspp, d.result.is_compute_constant, d.result.is_temporary),
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
          (float_type, const_identifier, lhs, baseName, addressStr) = self._pointer(cpp, innerAname, outerAname, d.leftTerm, d.innerLoopIndices)
          cpp._tokens.append(("_POINTER", [("float type", float_type), ("const_identifier" ,const_identifier), 
                            ("lhs", lhs), ("rhs", baseName), ("offset", addressStr)]))
          (float_type, const_identifier, lhs, baseName, addressStr) = self._pointer(cpp, innerBname, outerBname, d.rightTerm, d.innerLoopIndices)
          cpp._tokens.append(("_POINTER", [("float type", float_type), ("const_identifier" ,const_identifier), 
                            ("lhs", lhs), ("rhs", baseName), ("offset", addressStr)]))
          (float_type, const_identifier, lhs, baseName, addressStr) = self._pointer(cpp, innerCname, outerCname, d.result, d.innerLoopIndices, const=False)
          cpp._tokens.append(("_POINTER", [("float type", float_type), ("const_identifier" ,const_identifier), 
                            ("lhs", lhs), ("rhs", baseName), ("offset", addressStr)]))
          if outerPrefetchName is not None:
            self._pointer(cpp, innerPrefetchName, outerPrefetchName, d.result, d.innerLoopIndices)
        generator = gemm.generator(self._arch, gemmDescr, gemm_cfg, self._target)
        if hasOuterLoops and self._target == 'gpu':
          generator._set_generate_code(False)
        return generator.generate(cpp, routineCache)

    class InnerLoopBody(object):
      def __call__(s):
        print("INNERLOOPBODY")
        flops = 0
        if hasOuterLoops:
          print("INNTERLOOPBODY HASOUTERLOOPS")
          print(d.outerLoopIndices)
          (float_type, const_identifier, lhs, baseName, addressStr) = self._pointer(cpp, outerAname, d.leftTerm.name, d.leftTerm, d.outerLoopIndices)
          cpp._tokens.append(("_POINTER", [("float type", float_type), ("const_identifier" ,const_identifier), 
                            ("lhs", lhs), ("rhs", baseName), ("offset", addressStr)]))
          (float_type, const_identifier, lhs, baseName, addressStr) =  self._pointer(cpp, outerBname, d.rightTerm.name, d.rightTerm, d.outerLoopIndices)
          cpp._tokens.append(("_POINTER", [("float type", float_type), ("const_identifier" ,const_identifier), 
                            ("lhs", lhs), ("rhs", baseName), ("offset", addressStr)]))
          (float_type, const_identifier, lhs, baseName, addressStr) =  self._pointer(cpp, outerCname, d.result.name, d.result, d.outerLoopIndices, const=False)
          cpp._tokens.append(("_POINTER", [("float type", float_type), ("const_identifier" ,const_identifier), 
                            ("lhs", lhs), ("rhs", baseName), ("offset", addressStr)]))
          if d.prefetchName is not None:
            self._pointer(cpp, outerPrefetchName, d.prefetchName, d.result, d.outerLoopIndices)
        if d.assignLoopRanges is not None:
          gemmDescr.setBeta(0.0)
          flops += forLoops(cpp, d.innerLoopIndices, d.assignLoopRanges, LoGBody(), pragmaSimd=False)
        if d.addLoopRanges is not None:
          gemmDescr.setBeta(1.0)
          flops += forLoops(cpp, d.innerLoopIndices, d.addLoopRanges, LoGBody(), pragmaSimd=False)
        return flops

    flops = forLoops(cpp, d.outerLoopIndices, d.loopRanges, InnerLoopBody(), pragmaSimd=False)
    if hasOuterLoops and self._target == 'gpu':
      generator = gemm.generator(self._arch, gemmDescr, gemm_cfg, self._target)
      generator._set_generate_code(True)
      generator.generate(cpp, routineCache)
  
    return flops