#include "utils.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"


namespace llvm {

class DataFlow {

    public:
    	void computeBB_In_Out(Function &F,
                      DenseMap<const BasicBlock*, genKill> &bb_GenKill,
                      DenseMap<const BasicBlock*, inOut> &bb_InOut, std::vector<Value *> domain);

    	void computeS_In_Out(Function &F, 
		      DenseMap<const Instruction*, genKill> &i_GenKill,
                      DenseMap<const Instruction*, inOut> &i_InOut,
		      DenseMap<const BasicBlock*, inOut> &bb_InOut, std::vector<Value *> domain);

        void operatorInNewTuple(tuple* gen_tuple,
                                tuple* out_tuple,
                                tuple* new_tuple);

        void operatorOutNewTuple(tuple* tuple1,
                                 tuple* tuple2,
                                 tuple* new_tuple);

        bool checkIfEqual(DenseMap<const llvm::Value*, tuple> &new_set,
                          DenseMap<const llvm::Value*, tuple> &old_set,
                          std::vector<Value *> &domain);


};

}
