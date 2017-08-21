#include "dataflow.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <vector>

namespace llvm {

class CoherenceAnalysis {
private:
  void computeBB_Gen_Kill(Function &F,
                          DenseMap<const Instruction *, inOut> &i_InOut,
                          DenseMap<const BasicBlock *, genKill> &bb_GenKill);

  void computeS_Gen_Kill(Function &F,
                         DenseMap<const Instruction *, genKill> &i_GenKill);

public:
  bool hasKernel;
  bool hasDivergence;
  DenseMap<const Instruction *, genKill> i_GenKill;
  DenseMap<const Instruction *, inOut> i_InOut;
  std::map<Instruction *, std::map<Value *, tuple>> gpuKernel;
  std::map<Value*, std::set<Value*>> callee_functions;
  std::map<Value*, Value*> ptr_to_malloc;
  std::vector<Value *> domain;

  void ud_chain(Instruction *instr, 
                    std::set<Value *> &variables,
                    std::vector<Value *> domain);

  bool variableBelongs(Value *v, 
                       std::vector<Value *> domain,
                       int size);

  DenseMap<const BasicBlock *, genKill> bb_GenKill;
  DenseMap<const BasicBlock *, inOut> bb_InOut;

  void GPU_GEN(Function &F, 
               DenseMap<const Instruction *, genKill> &i_GenKill,
               std::vector<Instruction *> worklist_GPU);

  void CPU_GEN(DenseMap<const Instruction *,
               genKill> &i_GenKill,
               std::vector<Instruction *> worklist);


  void ud_chain_malloc(Instruction *instr, 
                       std::set<Value *> &buffers);
  
  void ud_chain_check(Instruction *instr, 
                      std::set<Value *> &buffers);


  void ud_chain_arg(Instruction *instr,
                    Value *argument, 
                    std::set<Value *> &buffers);


  bool runOnFunction(Function &F);
};
}
