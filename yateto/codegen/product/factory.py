from ...memory import CSCMemoryLayout
from ..common import *
from .generic import Generic
from .GemmforgeProduct import GemmforgeProduct

class Description(object):
  def __init__(self, alpha, add: bool, result: IndexedTensorDescription, leftTerm: IndexedTensorDescription, rightTerm: IndexedTensorDescription):
    self.alpha = alpha
    self.add = add
    self.result = result
    self.leftTerm = leftTerm
    self.rightTerm = rightTerm

    self.isACsc = isinstance(self.leftTerm.memoryLayout, CSCMemoryLayout)
    self.isBCsc = isinstance(self.rightTerm.memoryLayout, CSCMemoryLayout)
    
    rA = loopRanges(self.leftTerm, self.result.indices)
    rB = loopRanges(self.rightTerm, self.result.indices)
    rC = loopRanges(self.result, self.result.indices)
    assert testLoopRangesEqual(rA, rB)
    assert testLoopRangesAContainedInB(rA, rC)
    assert testLoopRangesAContainedInB(rB, rC)
    
    rA.update(rB)

    self.loopRanges = rA

  def __str__(self):
    return (f"Description(\n"
            f"\talpha: {self.alpha}\n"
            f"\tadd: {self.add}\n"
            f"\tresult: {self.result}\n"
            f"\tleftTerm: {self.leftTerm}\n"
            f"\trightTerm: {self.rightTerm}\n"
            f"\tisACsc: {self.isACsc}\n"
            f"\tisBCsc: {self.isBCsc}\n"
            f"\tloopRanges: {self.loopRanges}\n"
            f")")

def generator(arch, descr, target):
  if target == 'cpu':
    return Generic(arch, descr)
  elif target == 'gpu':
    print("WARNING: Product operation is experimental and in early stages of development")
    return GemmforgeProduct(arch=arch, descr=descr)
