#include "utils.h"

using namespace llvm;

#ifdef __cplusplus
  extern "C" {
#endif

void printSet(std::string setName, llvm::DenseMap<const llvm::Value*, tuple> set){
    if(!set.empty()){
        for (DenseMap<const llvm::Value*, tuple>::const_iterator inst_domain = set.begin(); 
    		inst_domain != set.end(); ++inst_domain) {

	    const Value* valueCheck = inst_domain->first;
	    const tuple tupleCheck = inst_domain->second;
	    errs()<< " "<<setName<< "------------> " <<*valueCheck ;
	    if(tupleCheck.status == R_ && tupleCheck.scope == cpu_) errs() << "       --> (R, cpu)\n";
	    else if(tupleCheck.status == W_ && tupleCheck.scope == cpu_) errs() << "       --> (W, cpu)\n";
	    else if(tupleCheck.status == RW_ && tupleCheck.scope == cpu_) errs() << "       --> (RW, cpu)\n";
	    else if(tupleCheck.status == R_ && tupleCheck.scope == gpu_) errs() << "       --> (R, gpu)\n";
	    else if(tupleCheck.status == W_ && tupleCheck.scope == gpu_) errs() << "       --> (W, gpu)\n";
	    else if(tupleCheck.status == RW_ && tupleCheck.scope == gpu_) errs() << "       --> (RW, gpu)\n";	
	    else if(tupleCheck.status == R_ && tupleCheck.scope == X_) errs() << "       --> (R, x)\n";
	    else if(tupleCheck.status == W_ && tupleCheck.scope == X_) errs() << "       --> (W, x)\n";
	    else if(tupleCheck.status == RW_ && tupleCheck.scope == X_) errs() << "       --> (RW, x)\n";		
	}
    }
}

#ifdef __cplusplus
}
#endif
