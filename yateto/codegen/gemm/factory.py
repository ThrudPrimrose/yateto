from ...ast.indices import BoundingBox, Range
from ...memory import CSCMemoryLayout
from ..common import TensorDescription
from .generic import Generic
from .gemmgen import GemmGen
from ...gemm_configuration import GemmForge
from .GemmforgeGemmGen import GemmforgeGemmGen

class Description(object):
  def __init__(self,
               result: TensorDescription,
               leftTerm: TensorDescription,
               rightTerm: TensorDescription,
               transA,
               transB,
               alpha,
               beta,
               arch,
               alignedStartA,
               alignedStartC,
               prefetchName = None):

    self.result = result
    self.leftTerm = leftTerm
    self.rightTerm = rightTerm
    self.transA = transA
    self.transB = transB
    self.alpha = alpha
    self.beta = beta
    self.prefetchName = prefetchName
    
    self.isACsc = isinstance(self.leftTerm.memoryLayout, CSCMemoryLayout)
    self.isBCsc = isinstance(self.rightTerm.memoryLayout, CSCMemoryLayout)
    
    if self.isACsc and self.isBCsc:
      raise RuntimeError('GEMM: sparse x sparse is currently not supported.')
    
    bbA = BoundingBox.fromSpp(self.leftTerm.eqspp)
    bbB = BoundingBox.fromSpp(self.rightTerm.eqspp)
    bbC = BoundingBox.fromSpp(self.result.eqspp)
    
    kA = 1 if not transA else 0
    kB = 0 if not transB else 1
    
    k = bbA[kA] & bbB[kB]
    m = bbA[1-kA]
    n = bbB[1-kB]

    assert m in bbC[0]
    assert n in bbC[1]

    self.alignedA = alignedStartA and not transA and self.leftTerm.memoryLayout.alignedStride()
    self.alignedC = alignedStartC and self.result.memoryLayout.alignedStride()
    
    if self.alignedA and self.alignedC:
      m = m.aligned(arch)
    else:
      mStartAligned = arch.checkAlignment(m.start)
      self.alignedA = self.alignedA & mStartAligned
      self.alignedC = self.alignedC & mStartAligned
    
    self._mnk = (m, n, k)

  def mnk(self):
    return self._mnk
  
  def setBeta(self, beta):
    self.beta = beta

  def __str__(self):
      return ("Description("
              f"  result={self.result},\t"
              f"  leftTerm={self.leftTerm},\t"
              f"  rightTerm={self.rightTerm},\t"
              f"  transA={self.transA},\t"
              f"  transB={self.transB},\t"
              f"  alpha={self.alpha},\t"
              f"  beta={self.beta},\t"
              f"  prefetchName={self.prefetchName},\t"
              f"  isACsc={self.isACsc},\t"
              f"  isBCsc={self.isBCsc},\t"
              f"  alignedA={self.alignedA},\t"
              f"  alignedC={self.alignedC},\t"
              f"  mnk={self._mnk}"
              ")")

  def __repr__(self):
    return self.__str__()

def generator(arch, descr, gemm_cfg, target):
  AOk = descr.isACsc or descr.leftTerm.memoryLayout.stridei(0) == 1
  BOk = descr.isBCsc or descr.rightTerm.memoryLayout.stridei(0) == 1
  strideOneC = descr.result.memoryLayout.stridei(0) == 1
  memLayoutOk = AOk and BOk and strideOneC
  if memLayoutOk:
    m, n, k = descr.mnk()
    gemmTool = gemm_cfg.getGemmTool(m.size(),
                                    n.size(),
                                    k.size(),
                                    descr.isACsc,
                                    descr.isBCsc,
                                    descr.transA,
                                    descr.transB,
                                    descr.alpha,
                                    descr.beta,
                                    descr.alignedA,
                                    descr.alignedC,
                                    target)

    if isinstance(gemmTool, GemmForge):
      assert(target=="gpu")
      return GemmforgeGemmGen(arch, descr, gemmTool)
    elif gemmTool:
      return GemmGen(arch, descr, gemmTool)
  return Generic(arch, descr)
