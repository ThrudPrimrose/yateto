from ...ast.indices import Indices
from ..common import *
from .. import gemm

class Generic(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr
  
  def _pointer(self, cpp, targetName, baseName, term, loopIndices):
    addressStr = term.memoryLayout.addressString(term.indices, term.indices & loopIndices)
    if len(addressStr) > 0:
      addressStr = ' + ' + addressStr
    cpp('{}* {} = {}{};'.format(self._arch.typename, targetName, baseName, addressStr))
    
  def _memLayout(self, term, I, J):
    if len(term.indices) > 2:
      return term.memoryLayout.fusedSlice(term.indices, I, J)
    return term.memoryLayout

  def _reduce(self, term, subset, memLayout):
    if len(term.indices) > 2:
      return reduceSpp(term.eqspp, term.indices, subset).reshape(memLayout.shape(), order='F')
    return term.eqspp
  
  def _defuse(self, fusedRange, term, I):
    if len(I) == 1:
      return  {next(iter(I)): fusedRange}
    return term.memoryLayout.defuse(fusedRange, term.indices, I)

  def generate(self, cpp, routineCache):
    d = self._descr
    
    A = d.leftTerm.indices - d.loopIndices
    B = d.rightTerm.indices - d.loopIndices
    C = d.result.indices - d.loopIndices
    Im = set(A) & set(C)
    In = set(B) & set(C)
    Ik = set(A) & set(B)
    
    hasOuterLoops = len(d.outerLoopIndices) > 0
    outerAname = '_A' if hasOuterLoops else d.leftTerm.name
    outerBname = '_B' if hasOuterLoops else d.rightTerm.name
    outerCname = '_C' if hasOuterLoops else d.result.name
    
    hasInnerLoops = len(d.innerLoopIndices) > 0
    innerAname = '_Ain' if hasInnerLoops else outerAname
    innerBname = '_Bin' if hasInnerLoops else outerBname
    innerCname = '_Cin' if hasInnerLoops else outerCname
    
    AmemLayout = self._memLayout(d.leftTerm, Im, Ik)
    BmemLayout = self._memLayout(d.rightTerm, Ik, In)
    CmemLayout = self._memLayout(d.result, Im, In)

    Aeqspp = self._reduce(d.leftTerm, A, AmemLayout)
    Beqspp = self._reduce(d.rightTerm, B, BmemLayout)
    Ceqspp = self._reduce(d.result, C, CmemLayout)

    gemmDescr = gemm.Description(
      leftTerm = TensorDescription(innerAname, AmemLayout, Aeqspp),
      rightTerm = TensorDescription(innerBname, BmemLayout, Beqspp),
      result = TensorDescription(innerCname, CmemLayout, Ceqspp),
      transA = d.transA,
      transB = d.transB,
      alpha = 1.0,
      beta = 1.0 if d.add else 0.0,
      arch = self._arch
    )
    
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
          self._pointer(cpp, innerAname, outerAname, d.leftTerm, d.innerLoopIndices)
          self._pointer(cpp, innerBname, outerBname, d.rightTerm, d.innerLoopIndices)
          self._pointer(cpp, innerCname, outerCname, d.result, d.innerLoopIndices)
        generator = gemm.generator(self._arch, gemmDescr)
        return generator.generate(cpp, routineCache)

    class InnerLoopBody(object):
      def __call__(s):
        flops = 0
        if hasOuterLoops:
          self._pointer(cpp, outerAname, d.leftTerm.name, d.leftTerm, d.outerLoopIndices)
          self._pointer(cpp, outerBname, d.rightTerm.name, d.rightTerm, d.outerLoopIndices)
          self._pointer(cpp, outerCname, d.result.name, d.result, d.outerLoopIndices)
        if d.assignLoopRanges is not None:
          gemmDescr.setBeta(0.0)
          flops += forLoops(cpp, d.innerLoopIndices, d.assignLoopRanges, LoGBody())
        if d.addLoopRanges is not None:
          gemmDescr.setBeta(1.0)
          flops += forLoops(cpp, d.innerLoopIndices, d.addLoopRanges, LoGBody())
        return flops

    return forLoops(cpp, d.outerLoopIndices, d.loopRanges, InnerLoopBody())
