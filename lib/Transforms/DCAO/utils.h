#include "llvm/IR/CFG.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/User.h"
#include "llvm/IR/InstIterator.h"
#include <set>
#include <vector>
#include <map>
#include <string>
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"

#include "llvm/Analysis/CallGraph.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/ADT/SCCIterator.h"


#define R_  1
#define W_  2
#define RW_ 3
#define RWnBlock_ 4

#define cpu_ 1
#define gpu_ 2
#define X_ 3

namespace llvm{

class tuple{
    public:
        int status;
        int scope;
};

class CA{
    public:
        int scope_top;
        int scope_bot;
        int status_top;
        int status_bot;
};

class genKill {
    public:
        llvm::DenseMap<const llvm::Value*, tuple> gen;
        llvm::DenseMap<const llvm::Value*, tuple> kill;
};

class inOut {
    public:
        llvm::DenseMap<const llvm::Value*, tuple> in;
        llvm::DenseMap<const llvm::Value*, tuple> out;
};
#ifdef __cplusplus
  extern "C" {
#endif

    void printSet(std::string setName,
                  llvm::DenseMap<const llvm::Value*,
                  tuple> set);

#ifdef __cplusplus
  }
#endif

}
