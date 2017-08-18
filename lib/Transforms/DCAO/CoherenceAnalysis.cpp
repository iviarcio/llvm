#include "CoherenceAnalysis.h"

using namespace llvm;

bool CoherenceAnalysis::runOnFunction(Function &F) {
  bool modified = false;

  DataFlow dataflow;


  computeS_Gen_Kill(F, i_GenKill);
  if(this->hasDivergence) return false;

  dataflow.computeS_In_Out(F, i_GenKill, i_InOut, bb_InOut, domain);
  if(this->hasDivergence) return false;

  computeBB_Gen_Kill(F, i_InOut, bb_GenKill);
  if(this->hasDivergence) return false;
  
  dataflow.computeBB_In_Out(F, bb_GenKill, bb_InOut, domain);
  if(this->hasDivergence) return false;

  dataflow.computeS_In_Out(F, i_GenKill, i_InOut, bb_InOut, domain);

  return modified;
}

/* Generate the GEN-KILL set of each Basic Block */
void CoherenceAnalysis::computeBB_Gen_Kill(Function &F,
                                           DenseMap<const Instruction *, inOut> &i_InOut,
                                           DenseMap<const BasicBlock *, genKill> &BB_GenKill) {
  for (Function::iterator BB = F.begin(); BB != F.end(); ++BB) {
    /* Using the First Sentence of the Basic Block to generate the GEN[B] */
    Instruction *BB_first = &(BB->front());
    inOut BB_FirstInOut = i_InOut.lookup(BB_first);
    DenseMap<const llvm::Value *, tuple> BB_FirstIn = BB_FirstInOut.in;

    /* Generating the GEN[B] */
    genKill BB_InOut;
    BB_InOut.gen = BB_FirstIn;
    BB_GenKill.insert(std::make_pair(&*BB, BB_InOut));
  }
}

/* Generate the GEN-KILL set of each Instructions */
void CoherenceAnalysis::computeS_Gen_Kill(Function &F,
                                          DenseMap<const Instruction *, genKill> &i_GenKill) {

  std::vector<Instruction *> worklist_CPU;
  std::vector<Instruction *> worklist_GPU;

  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    Instruction *instr = &*I;
    switch (instr->getOpcode()) {
    // CPU READ
    case Instruction::Load: {
      worklist_CPU.push_back(&*I);
      break;
    }
    // CPU Write
    case Instruction::Store: {
      worklist_CPU.push_back(&*I);
      break;
    }
    // GPU Read and Write
    case Instruction::Call: {
      CallInst *S = cast<CallInst>(instr);

      if (S->getCalledFunction()->getName() == "_cl_execute_kernel" ||
          S->getCalledFunction()->getName() == "_cl_execute_tiled_kernel") {
        worklist_GPU.push_back(&*I);
      } else if (!(S->getCalledFunction()->getName() == "free" ||
                   S->getCalledFunction()->getName() == "_cl_release_buffer" ||
                   S->getCalledFunction()->getName() == "_cl_create_read_only" ||
                   S->getCalledFunction()->getName() == "_cl_offloading_read_only" ||
                   S->getCalledFunction()->getName() == "_cl_create_write_only" ||
                   S->getCalledFunction()->getName() == "_cl_create_read_write" ||
                   S->getCalledFunction()->getName() == "_cl_read_buffer" ||
                   S->getCalledFunction()->getName() == "_cl_read_only" ||
                   S->getCalledFunction()->getName() == "_cl_offloading_read_write" ||
                   S->getCalledFunction()->getName() == "_cl_offloading_write_only")) {
        worklist_CPU.push_back(&*I);
      }
    }
    default:
      break;
    }
  }
  
  GPU_GEN(F, i_GenKill, worklist_GPU);
  
  CPU_GEN(i_GenKill, worklist_CPU);
}

/* For each CPU instruction, this function identify its malloc */
void CoherenceAnalysis::GPU_GEN(Function &F,
                                DenseMap<const Instruction *,
                                genKill> &i_GenKill,
                                std::vector<Instruction *> kernels) {

  std::map<Instruction *, std::vector<Instruction *>> kernelArgs;
  std::vector<Instruction *> args;

  /* Identify which buffers are used by a given kernel */
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    Instruction *instr = &*I;

    if (CallInst *S = dyn_cast<CallInst>(instr)) {

      /* Identify GPU buffers */
      if (S->getCalledFunction()->getName() == "_cl_create_write_only" ||
          S->getCalledFunction()->getName() == "_cl_create_read_only" ||
          S->getCalledFunction()->getName() == "_cl_offloading_read_only" ||
          S->getCalledFunction()->getName() == "_cl_offloading_write_only" ||
          S->getCalledFunction()->getName() == "_cl_create_read_write" ||
          S->getCalledFunction()->getName() == "_cl_offloading_read_write") {
        args.push_back(instr);
      }

      /* Identify release */
      if (S->getCalledFunction()->getName() == "_cl_release_buffer") {
        args.pop_back();
      }

      /* Identify kernel execution  */
      if (S->getCalledFunction()->getName() == "_cl_execute_tiled_kernel" ||
          S->getCalledFunction()->getName() == "_cl_execute_kernel") {
        kernelArgs.insert(std::make_pair(instr, args));
      }
    }
  }

  for (std::vector<Instruction *>::iterator it = kernels.begin(); it != kernels.end(); ++it) {
    Instruction *instr = *it;
    std::map<Instruction *, std::vector<Instruction *>>::iterator kernel;
    kernel = kernelArgs.find(instr);
    std::vector<Instruction *> kernelExecutionArgs = kernel->second;

    genKill ins_GenKill;
    llvm::DenseMap<const llvm::Value *, tuple> gen;
    ins_GenKill.gen = gen;

    /* Checking the corresponding ID of the buffer that was created as shared buffer*/
    int size = kernelExecutionArgs.size();
    std::vector<Value *> sharedBuffers;
    std::vector<tuple> typeBuffer;

    for (int i = 0; i < size; i++) {
      /* Check which variable is used by the Argument */
      CallInst *S = dyn_cast<CallInst>(kernelExecutionArgs[i]);
      Value *secondArgument;
      Value *mallocInst;

      if(S->getCalledFunction()->getName() == "cl_read_buffer")
        secondArgument = S->getArgOperand(2);
      else
        secondArgument = S->getArgOperand(1);

      /* Check the buffer ID and insert in a vector structure */
      if (isa<BitCastInst>(secondArgument)) {
        std::set<Value *> variables;
        Instruction *check = cast<Instruction>(secondArgument);

        /* This func returns the malloc's instruction */
        ud_chain_malloc(check, variables);

        /* It is expected to be found only one malloc instruction */
        std::set<Value *>::iterator variable;
        if (variables.size() > 1 || variables.size() == 0){
          hasDivergence = true;
          return;
        }

        variable = begin(variables);
        mallocInst = *variable;

      } else {
        mallocInst = secondArgument;
      }

      ptr_to_malloc.insert(std::make_pair(secondArgument, mallocInst));
      sharedBuffers.push_back(mallocInst);
      tuple type;

      /* GPU Write */
      if (S->getCalledFunction()->getName() == "_cl_create_write_only" ||
          S->getCalledFunction()->getName() == "_cl_create_read_write" ||
          S->getCalledFunction()->getName() == "_cl_read_buffer" ||
          S->getCalledFunction()->getName() == "_cl_offloading_read_write" ||
          S->getCalledFunction()->getName() == "_cl_offloading_write_only") {
        type.status = W_;
        type.scope = gpu_;
      }
      /* GPU Read */
      else if (S->getCalledFunction()->getName() == "_cl_create_read_only" ||
               S->getCalledFunction()->getName() == "_cl_offloading_read_only" ||
               S->getCalledFunction()->getName() == "_cl_read_only") {
        type.status = R_;
        type.scope = gpu_;
      }

      typeBuffer.push_back(type);
    }


    /* Identify which buffers are indeed used by the kernel */
    std::vector<Instruction *> argsToBeChanged;
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      Instruction *checkInstruction = &*I;
      if (CallInst *S = dyn_cast<CallInst>(checkInstruction)) {
        if (S->getCalledFunction()->getName() == "_cl_execute_kernel" ||
            S->getCalledFunction()->getName() == "_cl_execute_tiled_kernel") {
          if (checkInstruction == instr)
            break;
          else
            argsToBeChanged.clear();
        }
        if (S->getCalledFunction()->getName() == "_cl_set_kernel_arg")
          argsToBeChanged.push_back(checkInstruction);
      }
    }

    size = argsToBeChanged.size();
    std::map<Value *, tuple> buffersKernel;

    if (size == 0) {
      CallInst *argsCheck;
      bool hasArgInst = false;
      for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
        Instruction *checkInstruction = &*I;
        if (CallInst *S = dyn_cast<CallInst>(checkInstruction)){
          if (S->getCalledFunction()->getName() == "_cl_execute_kernel" ||
              S->getCalledFunction()->getName() == "_cl_execute_tiled_kernel"){
            if (checkInstruction == instr)
              break;
            else
              hasArgInst = false;
          }
        
          if (S->getCalledFunction()->getName() == "_cl_set_kernel_args"){
            argsCheck = S;
            hasArgInst = true;
          }
        }
      }

      if (hasArgInst){
        Value *bufferArg = argsCheck->getArgOperand(0);
        ConstantInt *bufferPosition = dyn_cast<ConstantInt>(bufferArg);
        uint64_t vectorPosition = bufferPosition->getZExtValue();

        for(int i = 0; i < (int)vectorPosition; i++){
          Value *buffer = sharedBuffers[i];
          tuple type = typeBuffer[i];
          buffersKernel.insert(std::make_pair(buffer, type));

          /* Preparing GEN set */
          ins_GenKill.gen.insert(std::make_pair(&*buffer, type));
        }
      } else {
        this->hasDivergence = true;
      }

    } else {
      for (int i = 0; i < size; i++) {
        CallInst *S = dyn_cast<CallInst>(argsToBeChanged[i]);

        Value *bufferSelect = S->getArgOperand(1);
        ConstantInt *bufferPosition = dyn_cast<ConstantInt>(bufferSelect);
        uint64_t vectorPosition = bufferPosition->getZExtValue();

        Value *buffer = sharedBuffers[vectorPosition];
        tuple type = typeBuffer[vectorPosition];
        buffersKernel.insert(std::make_pair(buffer, type));

        /* Preparing GEN set */
        ins_GenKill.gen.insert(std::make_pair(&*buffer, type));
      }
    }

    /* Inserting the kernel into GEN set */
    i_GenKill.insert(std::make_pair(&*instr, ins_GenKill));

    /* TODO: Check if it will be necessary */
    gpuKernel.insert(std::make_pair(instr, buffersKernel));
  }
}

/* For each GPU instruction, this function identify its malloc */
void CoherenceAnalysis::CPU_GEN(DenseMap<const Instruction *, genKill> &i_GenKill,
                                std::vector<Instruction *> worklist){

  for (std::vector<Instruction *>::iterator it_worklist = worklist.begin();
       it_worklist != worklist.end(); ++it_worklist) {

    if(hasDivergence) return;

    bool isCallInst = false;
    Instruction *instr = *it_worklist;
    genKill ins_GenKill;
    llvm::DenseMap<const llvm::Value *, tuple> gen;
    ins_GenKill.gen = gen;
    tuple ins_tuple;

    /* Used to store all Variables that affect the Instruction */
    std::set<Value *> variables_gen;

    if (isa<StoreInst>(instr)) {
      /* Defining scope and status */
      ins_tuple.status = W_;
      ins_tuple.scope = cpu_;

    } else if (isa<LoadInst>(instr)) {
      /* Used to generate the GEN[s] of all Instructions that affect DCA */
      ins_tuple.status = R_;
      ins_tuple.scope = cpu_;

    } else if (isa<CallInst>(instr)) {
      CallInst *S = cast<CallInst>(instr);

      Function *Fcalled = S->getCalledFunction();
      if (Fcalled->isDeclaration())
        continue;

      if (S->getCalledFunction()->getName() == "malloc") {
        ins_tuple.status = W_;
        ins_tuple.scope = cpu_;
        variables_gen.insert(instr);
      } else {
        isCallInst = true;
        ins_tuple.status = W_;
        ins_tuple.scope = cpu_;
       
        /* For each argument, call du_chain */
        for (User::op_iterator j = instr->op_begin(), e = instr->op_end(); j != e; ++j) {
          Value *arg = *j;
          if (isa<PointerType>(arg->getType()) && !isa<PHINode>(arg)){
            if (Instruction *vi = dyn_cast<Instruction>(arg)){
              if(isa<CallInst>(vi)) continue;

              ud_chain_malloc(vi, variables_gen); 
            }else if (isa<Argument>(arg)){ 

              ud_chain_arg(instr, arg, variables_gen);
            }
          }
        }

        if (!variables_gen.empty()) {
          for (std::set<Value *>::iterator it_variable = variables_gen.begin();
               it_variable != variables_gen.end(); it_variable++) {
            Value *var = *it_variable;
            ins_GenKill.gen.insert(std::make_pair(&*var, ins_tuple));
          }

          /* Insert GEN set for CPU */
          i_GenKill.insert(std::make_pair(&*instr, ins_GenKill));
        }
      }
    }

    if(!isCallInst){

      ud_chain_malloc(instr, variables_gen);

      if(variables_gen.size() > 1)
        this->hasDivergence = true;

      std::set<Value *>::iterator it = variables_gen.begin();
      Value *v = *it;
      ins_GenKill.gen.insert(std::make_pair(&*v, ins_tuple));
      i_GenKill.insert(std::make_pair(&*instr, ins_GenKill));
    }
  }
}


void CoherenceAnalysis::ud_chain_arg(Instruction *instr,
                                     Value *argument, 
                                     std::set<Value *> &buffers){
  int position = 0;

  /* Identify the Function's argument */
  Function *F = instr->getParent()->getParent();
  for (Function::arg_iterator it_arg = F->arg_begin(); it_arg != F->arg_end(); ++it_arg){
    Argument *arg = &*it_arg;
    if(arg == argument) break;
    position++;
  }

  std::map<Value*, std::set<Value *>>::iterator it;
  if (callee_functions.count(F) > 0){
    it = callee_functions.find(F);
    std::set<Value *> *checkFunctions = &it->second;

    for (std::set<Value *>::iterator it2=checkFunctions->begin(); it2!=checkFunctions->end(); ++it2){
      Value *v = *it2;
      Function *FCallee = cast<Function>(v);

      /* Look for caller instructions */
      for (inst_iterator I = inst_begin(FCallee), E = inst_end(FCallee); I != E; ++I) {
        Instruction *instrCallee = &*I;
        if (isa<CallInst>(instrCallee)) {
          CallInst *S = cast<CallInst>(instrCallee);
          if (!(S->getCalledFunction() == F)) continue;
          Value *argCalleeFunc = S->getArgOperand(position);
          if (isa<Argument>(argCalleeFunc)){
            ud_chain_arg(instrCallee, argCalleeFunc, buffers);
          }else{
            Instruction *j = dyn_cast<Instruction>(argCalleeFunc);
            ud_chain_malloc(j, buffers);
          }
        }
      }
    }
  }
}

void CoherenceAnalysis::ud_chain_check(Instruction *instr,
                                       std::set<Value *> &buffers){
  Value *firstOperand;
  if(isa<StoreInst>(instr))
    firstOperand = instr->getOperand(1);
  else
    firstOperand = instr->getOperand(0);

  if(isa<Argument>(firstOperand)){
    Value *arg = cast<Argument>(firstOperand);
    ud_chain_arg(instr, arg, buffers);
  
  }else if(isa<GlobalValue>(instr)){
    this->hasDivergence=true;
  }else{
    if(Instruction *j = dyn_cast<Instruction>(firstOperand))
      ud_chain_malloc(j, buffers);
  } 
}

/* This function identify the malloc of a given pointer */
void CoherenceAnalysis::ud_chain_malloc(Instruction *instr, 
                                        std::set<Value *> &buffers){

  std::map<Value*,int>::iterator it;

  if(isa<CallInst>(instr)){
    CallInst* S = cast<CallInst>(instr);

    if(S->getCalledFunction()->getName() == "malloc"){
      buffers.insert(instr);
    }else{
      this->hasDivergence = true;
    }
  }else if(isa<BitCastInst>(instr)){
    ud_chain_check(instr, buffers);
  }else if(isa<GetElementPtrInst>(instr)){
    ud_chain_check(instr, buffers);
  }else if(isa<LoadInst>(instr)){
    ud_chain_check(instr, buffers);
  }else if(isa<StoreInst>(instr)){
    ud_chain_check(instr, buffers);
  }else if(isa<PHINode>(instr)){
    this->hasDivergence = true;
  }
}
