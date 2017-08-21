#include "CoherenceAnalysis.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "llvm/Transforms/IPO.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Dominators.h"

#define DEBUG_TYPE "dcao"


namespace llvm{

  class clBufferInfos {
  public:
    int position;
    Value *size;
    Value *malloc;
  };

  namespace {
    struct DCAO : public ModulePass {
      static char ID;
      DCAO();

    public:
      std::map<Value*,clBufferInfos> buffersInfo;
      std::map<Value*,Value*> mallocToclCreateBuffer;
      std::map<BasicBlock*,std::map<Value*,CA>> CAMapUnmap;
      std::map<BasicBlock*,std::map<BasicBlock*,BasicBlock*>> newBB;
      std::map<Function*,std::map<int,Value*>> pointerCast;
      std::vector<Instruction*> mapInstructions;
      std::vector<Instruction*> unmapInstructions;

      int BBNumber;
      int debug;

      Instruction *_cl_device_init;
      CoherenceAnalysis *Analysis;
      bool isTail;
      int positionMalloc;
      void defUse_chain(Instruction *instr,
                        Value *malloc,
                        std::map<Value*,int> &mallocs_uses);

      void error_message(int msg);

      void du_chain(Instruction *instr, 
                    std::set<Value*> &GPUBuffer,
                    bool foundVar, 
                    std::vector<Value*> &var);

      void initSharedMemory(Module &M);

      void createBuffers(Module &M,
                         std::map<Value*,int> mallocs_uses);
      void createBuffer(Value *malloc,
                        Module &M,
                        std::string typeOfBuffer);

      void changeKernelArgs(Module &M);
      void updateDCAData(Module &M,
                         CoherenceAnalysis *Analysis);
      void applyMapAndUnmapInstructions(Module &M,
                                        CoherenceAnalysis *Analysis);
      void removeOffload(Module &M);
      void removeGPUUnnecessaryCalls(Module &M);
      int getNumberOfPredecessor(BasicBlock* BB);
      int getNumberOfSuccessor(BasicBlock* BB);

      void map(int buffer,
               Module &M,
               Instruction *before,
               int status);

      void unmap(int buffer,
                 Module &M,
                 Instruction *before);

      BranchInst *createNewBB(Module &M,
                              BasicBlock* predBB,
                              BasicBlock* succBB,
                              BasicBlock* newBB);

      bool predBBWereVisited(BasicBlock *BB,
                             std::set<BasicBlock*> visitedBB);

      void isCombination(BasicBlock* BB,
                         bool* hasCPU,
                         bool* hasGPU,
                         int *new_status,
                         Value* buffer);

      int getIdxBBSuccessor(BasicBlock* BBPred,
                            BasicBlock* BBSucc);

      void updateCANewBB(BasicBlock* BB,
                         Value* buffer,
                         int status,
                         int scope);

      void updatePHINode(BasicBlock* BB,
                         BasicBlock* BBSucc,
                         BasicBlock* newBB);

      void createBuffers2(Module &M,
                          std::map<Value*,int> mallocs_uses);

      std::map<Value*, std::set<Value*> > buildCG (Module &M);

      std::vector<Value *> buildDomain(Module &M);

      bool checkCallSetDevice(Instruction *instr);

      void tryRemoveMapAndUnmapFromLoop();
      bool tryMoveMapOutLoop(Instruction *mapInst, Loop *L, bool isMap);

      virtual bool runOnModule(Module &M);
      virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    };
  }
}
