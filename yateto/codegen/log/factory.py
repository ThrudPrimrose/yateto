import copy
from ..common import *
from .generic import Generic
from .GemmforgeLOG import GemmforgeLOG

class Description(object):
  def __init__(self,
               alpha,
               add: bool,
               result: IndexedTensorDescription,
               leftTerm: IndexedTensorDescription,
               rightTerm: IndexedTensorDescription,
               loopIndices,
               transA,
               transB,
               prefetchName):
    self.alpha = alpha
    self.add = add
    self.result = result
    self.leftTerm = leftTerm
    self.rightTerm = rightTerm
    self.loopIndices = loopIndices
    self.transA = transA
    self.transB = transB
    self.prefetchName = prefetchName
    
    rA = loopRanges(self.leftTerm, self.loopIndices)
    rB = loopRanges(self.rightTerm, self.loopIndices)
    rC = loopRanges(self.result, self.loopIndices)
    assert testLoopRangesEqual(rA, rB)
    assert testLoopRangesAContainedInB(rA, rC)
    assert testLoopRangesAContainedInB(rB, rC)
    
    rC.update(rA)
    rC.update(rB)

    self.loopRanges = rC
    
    self.innerLoopIndices = self.loopIndices - self.result.indices
    self.outerLoopIndices = self.loopIndices - self.innerLoopIndices
    
    self.assignLoopRanges = copy.deepcopy(self.loopRanges)
    self.addLoopRanges = copy.deepcopy(self.loopRanges)

    if len(self.innerLoopIndices) == 0:
      if self.add:
        self.assignLoopRanges = None
      else:
        self.addLoopRanges = None
    elif not self.add:
      peelOffIndex = str(self.innerLoopIndices.firstIndex())
      self.assignLoopRanges[peelOffIndex].stop = self.loopRanges[peelOffIndex].start+1
      self.addLoopRanges[peelOffIndex].start   = self.loopRanges[peelOffIndex].start+1
    else:
      self.assignLoopRanges = None

    def __str__(self):
        return f"Description: " + \
               f"alpha: {self.alpha}, " + \
               f"add: {self.add}, " + \
               f"result: {self.result}, " + \
               f"leftTerm: {self.leftTerm}, " + \
               f"rightTerm: {self.rightTerm}, " + \
               f"loopIndices: {self.loopIndices}, " + \
               f"transA: {self.transA}, " + \
               f"transB: {self.transB}, " + \
               f"prefetchName: {self.prefetchName}, " + \
               f"loopRanges: {self.loopRanges}, " + \
               f"innerLoopIndices: {self.innerLoopIndices}, " + \
               f"outerLoopIndices: {self.outerLoopIndices}, " + \
               f"assignLoopRanges: {self.assignLoopRanges}, " + \
               f"addLoopRanges: {self.addLoopRanges}"

    def __repr__(self):
      return self.__str__()

def generator(arch, descr, target):
  if target == "gpu":
    return GemmforgeLOG(arch, descr)
  else:
    return Generic(arch, descr, target)

