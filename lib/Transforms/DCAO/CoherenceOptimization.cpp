#include "CoherenceOptimization.h"

static int dbg_dcao;

#define malloc_R 1
#define malloc_W 2
#define malloc_RW 3

using namespace llvm;

STATISTIC(clBuffersCreated, "The # of clCreatedBuffers instructions created");
STATISTIC(mapInstructionsCreated, "The # of map instructions created");
STATISTIC(unmapInstructionsCreated, "The # of unmap instructions created");

DCAO::DCAO() : ModulePass(ID){
  initializeDCAOPass(*PassRegistry::getPassRegistry());
  this->debug = dbg_dcao;
}

bool DCAO::runOnModule(Module &M) {
  std::map<Value *, int> mallocs_uses;
  positionMalloc = 0;
  BBNumber = 0;

  if(this->debug) errs() << "Run it in dbg mode \n";

  Analysis = new CoherenceAnalysis();
  Analysis->callee_functions = buildCG(M);
  Analysis->hasDivergence = false;
  Analysis->domain = buildDomain(M);

  for (Module::iterator FI = M.begin(), E = M.end(); FI != E; ++FI) {
    if (Analysis->hasDivergence) continue;

    if (!(FI->isDeclaration())) {
      Analysis->runOnFunction(*FI);
    }
  }

  /* DCA validate */
  if (Analysis->hasDivergence){
    errs() << "Data Coherence Analysis and Optimization cannot be applied on this program \n";
    return false;
  }

  /* Insert the instruction call _cl_init_shared_buffer */
  initSharedMemory(M);
  
  /* Creating clCreateBuffer by replacing mallloc's call */
  createBuffers2(M, mallocs_uses);

  /* Change kernel args */
  changeKernelArgs(M);

  /* Update DCA informations ghatered during the Analysis */
  updateDCAData(M, Analysis);

  /* Applying map and unmap instructions */
  applyMapAndUnmapInstructions(M, Analysis);

  /* Cleaning the code */
  removeGPUUnnecessaryCalls(M);

  /* Try to remove instructions inside a loop */
  tryRemoveMapAndUnmapFromLoop();

  return true;
}

/* This function create the domain of DCA.
 * The domain is formed by malloc instructions */
std::vector<Value *> DCAO::buildDomain(Module &M) {
  std::vector<Value *> domain;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (CallInst *S = dyn_cast<CallInst>(&I))
          if (S->getCalledFunction()->getName() == "malloc")
            domain.push_back(&I);

  return domain;
}

/* This function create a struct with the Call Graph that is used
 * when applying DCA's IPA */
std::map<Value *, std::set<Value *>> DCAO::buildCG(Module &M) {
  std::map<Value *, std::set<Value *>> callee_functions;
  CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
  
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    CallGraphNode *CallingNode = (CG)[I];
    Function *Caller = &*I;

    if (Caller->isDeclaration())
      continue;

    for (CallGraphNode::iterator CGNI = CallingNode->begin(),
         CGNE = CallingNode->end(); CGNI != CGNE; ++CGNI) {
      Function *Callee = CGNI->second->getFunction();
      
      if (Callee->isDeclaration())
        continue;

      if (callee_functions.count(Callee)) {
        std::map<Value *, std::set<Value *>>::iterator callee_elements;
        std::set<Value *> *callees;
        callee_elements = callee_functions.find(Callee);
        callees = &callee_elements->second;
        callees->insert(Caller);
      } else {
        std::set<Value *> callees;
        callees.insert(Caller);
        callee_functions.insert(std::make_pair(Callee, callees));
      }
    }
  }

  return callee_functions;
}

/* This function create a function that initialize the DCAO in the libmptogpu's lib */
void DCAO::initSharedMemory(Module &M) {
  Function *main = M.getFunction("main");
  BasicBlock *entry = &main->getEntryBlock();
  Instruction *next;

  for (BasicBlock::iterator I = entry->begin(), IE = entry->end(); I != IE; ++I) {
    Instruction *instr = &*I;

    if (isa<CallInst>(instr)) {
      CallInst *S = cast<CallInst>(instr);
      if (S->getCalledFunction()->getName() == "_cldevice_init") {
        _cl_device_init = instr;
        next = &*I++;
      }
    }
  }
  
  /* Define Arguments */
  std::vector<Type *> args;
  args.push_back(llvm::Type::getInt32Ty(M.getContext()));

  /* Defining the Type of the Function */
  FunctionType *FuncVoidTy = FunctionType::get(Type::getVoidTy(M.getContext()), args, false);

  /* Referencing the Instruction, that is in the shared library libmptogpu */
  Constant *createRefInit = M.getOrInsertFunction("_cl_init_shared_buffer", FuncVoidTy);

  /* Create constant type */
  Value *isdbg = ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), this->debug);
  std::vector<Value *> argumentsDbg;
  argumentsDbg.push_back(isdbg);

  /* Creating CallInst and setting it after _cldevice_init */
  Function *FuncInitSharedBuffer = cast<Function>(createRefInit);
  CallInst *createFuncInit = CallInst::Create(FuncInitSharedBuffer, argumentsDbg, "", next);

  /* Setting Attribute */
  AttributeSet gpuClangAttributes = cast<CallInst>(_cl_device_init)->getAttributes();
  createFuncInit->setAttributes(gpuClangAttributes);

  /* Setting Tail Call Optimization */
  if (cast<CallInst>(_cl_device_init)->isTailCall()) {
    createFuncInit->setTailCall(true);
    this->isTail = true;
  }
}

/* This function analize and create all shared buffers.
 * It uses the DCA's information, that runs an Alias Analysis to identify
 * all mallocs that are used in the GPU */
void DCAO::createBuffers2(Module &M,
                          std::map<Value *, int> mallocs_uses) {

  std::vector<Value*> mallocs_create;
  /* Identify all malloc's call */
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      Instruction *instr = &*I;
      if (CallInst *S = dyn_cast<CallInst>(instr)) {
        if (S->getCalledFunction()->getName() == "malloc"){
          mallocs_uses.insert(std::make_pair(S, 0));
          mallocs_create.push_back(S);
        }
      }
    }
  }

  /* Identify GPU bufers */
  std::vector<Instruction *> GPUBuffers;
  std::map<Value *, int> mallocs_type;
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    if (F->isDeclaration())
      continue;

    GPUBuffers.clear();
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      Instruction *instr = &*I;
      if (CallInst *S = dyn_cast<CallInst>(instr)) {
        if (S->getCalledFunction()->getName() == "_cl_offloading_write_only" ||
            S->getCalledFunction()->getName() == "_cl_offloading_read_write" ||
            S->getCalledFunction()->getName() == "_cl_offloading_read_only" ||
            S->getCalledFunction()->getName() == "_cl_read_buffer") {
          GPUBuffers.push_back(S);
        }
      }
    }

    /* Check which malloc is pointing to this buffer */
    int sizeBuffers = GPUBuffers.size();
    std::set<Value *> CPUBuffer;
    Value *checkInstruction;
    for (int i = 0; i < sizeBuffers; i++) {
      CallInst *bufferGPU = dyn_cast<CallInst>(GPUBuffers[i]);

      if (bufferGPU->getCalledFunction()->getName() == "_cl_read_buffer")
        checkInstruction = bufferGPU->getArgOperand(2);
      else
        checkInstruction = bufferGPU->getArgOperand(1);

      std::map<Value *, Value *>::iterator it;
      it = Analysis->ptr_to_malloc.find(checkInstruction);
      Value *vmalloc = it->second;

      std::map<Value *, int>::iterator it2;
      it2 = mallocs_uses.find(vmalloc);
      int *type = &it2->second;

      if (bufferGPU->getCalledFunction()->getName() == "_cl_offloading_write_only") {
        if (*type == 0)
          *type = malloc_W;
        else if (*type == malloc_R)
          *type = malloc_RW;
      }
      /* Buffer Read and Write */
      else if (bufferGPU->getCalledFunction()->getName() == "_cl_offloading_read_write" ||
               bufferGPU->getCalledFunction()->getName() == "_cl_read_buffer") {
        *type = malloc_RW;
      }
      /* Buffer Read Only */
      else if (bufferGPU->getCalledFunction()->getName() == "_cl_offloading_read_only") {
        if (*type == 0)
          *type = malloc_R;
        else if (*type == malloc_W)
          *type = malloc_RW;
      }
    }
  }

  int mallocs_size = mallocs_create.size();
  for (int i = 0; i < mallocs_size; i++){
    std::map<Value *, int>::iterator it = mallocs_uses.find(mallocs_create[i]);
    Value *instrMalloc = &*it->first;

    if (it->second == malloc_R)
      createBuffer(instrMalloc, M, "_cl_create_shared_buffer_read_only");
    else if (it->second == malloc_W)
      createBuffer(instrMalloc, M, "_cl_create_shared_buffer_write_only");
    else if (it->second == malloc_RW)
      createBuffer(instrMalloc, M, "_cl_create_shared_buffer_read_write");

  }
}

/* Create an OpenCL buffer */
void DCAO::createBuffer(Value *malloc,
                        Module &M,
                        std::string typeOfBuffer) {
  /* STATICS information */
  ++clBuffersCreated;

  CallInst *S = cast<CallInst>(malloc);
  Value *sizeMalloc = S->getArgOperand(0);
  Value *sharedBufferPosition = ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), positionMalloc);

  /* Defining function's arguments */
  std::vector<Type *> args;
  args.push_back(llvm::Type::getInt64Ty(M.getContext()));
  args.push_back(llvm::Type::getInt32Ty(M.getContext()));

  FunctionType *FuncTypeBuffer = FunctionType::get(Type::getVoidTy(M.getContext()), args, false);
  Constant *createBuffer = M.getOrInsertFunction(typeOfBuffer, FuncTypeBuffer);

  Function *FuncBuffer = cast<Function>(createBuffer);

  std::vector<Value *> argumentsBuffer;
  argumentsBuffer.push_back(sizeMalloc);
  argumentsBuffer.push_back(sharedBufferPosition);

  CallInst *createFuncBuffer = CallInst::Create(FuncBuffer, argumentsBuffer, "", cast<Instruction>(malloc));
  if (isTail)
    createFuncBuffer->setTailCall(true);

  /* Setting Attribute */
  AttributeSet gpuClangAttributes = cast<CallInst>(_cl_device_init)->getAttributes();

  createFuncBuffer->setAttributes(gpuClangAttributes);

  /*Starting map */
  args.clear();
  args.push_back(llvm::Type::getInt32Ty(M.getContext()));
  FunctionType *FuncTypeMap = FunctionType::get(Type::getInt8PtrTy(M.getContext()), args, false);
  Constant *createMap = M.getOrInsertFunction("_cl_map_buffer_write", FuncTypeMap);

  Function *FuncMap = cast<Function>(createMap);

  /* Creating a constant value */
  Value *newvalue = ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), positionMalloc);

  /* Creating CallInst */
  std::vector<Value *> argumentsMap;
  argumentsMap.push_back(newvalue);

  CallInst *createFuncMap = CallInst::Create(FuncMap, argumentsMap);
  if (isTail)
    createFuncMap->setTailCall(true);

  createFuncMap->setAttributes(gpuClangAttributes);

  /* Instruction that will be replaced */
  Instruction *instrToReplace = cast<Instruction>(malloc);
  BasicBlock::iterator ii(instrToReplace);

  ReplaceInstWithInst(instrToReplace->getParent()->getInstList(), ii, createFuncMap);

  /* Insert the variable on the structure */
  clBufferInfos clBufferInfo;
  clBufferInfo.position = positionMalloc;
  clBufferInfo.size = sizeMalloc;
  clBufferInfo.malloc = malloc;

  mallocToclCreateBuffer.insert(std::make_pair(malloc, createFuncMap));
  buffersInfo.insert(std::make_pair(createFuncMap, clBufferInfo));

  positionMalloc++;
}

/* Change all existing kernel argument linking the shared buffer created */
void DCAO::changeKernelArgs(Module &M){
  std::set<Instruction *> kernels;
  std::map<Instruction *, std::vector<Instruction *>> kernelArgs;
  std::vector<Instruction *> args;
  std::map<int, Value *> newPointer;
  std::map<Function *, std::map<int, Value *>>::iterator FuncPointers;

  /* Identifying how manny kernels is in the code */
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    if (F->isDeclaration())
      continue;

    /* Clear all Containers */
    kernels.clear();
    kernelArgs.clear();
    args.clear();
    newPointer.clear();

    Function *func = &*F;
    /* Identify all Functions that belongs to the Function F*/
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      Instruction *instr = &*I;
      if (CallInst *S = dyn_cast<CallInst>(instr))
        if (S->getCalledFunction()->getName() == "_cl_execute_kernel" ||
            S->getCalledFunction()->getName() == "_cl_execute_tiled_kernel")
          kernels.insert(instr);
    }

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

    for (std::set<Instruction *>::iterator it_kernel = kernels.begin();
         it_kernel != kernels.end(); ++it_kernel) {
      Instruction *instr = *it_kernel;
      std::map<Instruction *, std::vector<Instruction *>>::iterator kernel;
      kernel = kernelArgs.find(instr);
      Instruction *kernelExecution = kernel->first;
      std::vector<Instruction *> kernelExecutionArgs = kernel->second;

      /* Checking the corresponding ID of the buffer that was created as shared buffer. */
      int size = kernelExecutionArgs.size();
      std::vector<Value *> sharedBuffers;
      for (int i = 0; i < size; i++) {
        CallInst *S = dyn_cast<CallInst>(kernelExecutionArgs[i]);
        Value *secondArgument = S->getArgOperand(1);

        Value *v;
        if (CallInst *clBuffer = dyn_cast<CallInst>(secondArgument)){
          if (clBuffer->getCalledFunction()->getName() == "_cl_map_buffer_write" ||
              clBuffer->getCalledFunction()->getName() == "_cl_map_buffer_read" ||
              clBuffer->getCalledFunction()->getName() == "_cl_map_buffer_read_write"){
            v = secondArgument;
          }
        }else{
          std::map<Value *, Value *>::iterator it_instrToMalloc;
          it_instrToMalloc = Analysis->ptr_to_malloc.find(secondArgument);
          Value *malloc = it_instrToMalloc->second;
          std::map<Value *, Value *>::iterator it_mallocToclCreateBuffer;
          it_mallocToclCreateBuffer = mallocToclCreateBuffer.find(malloc);
          v = it_mallocToclCreateBuffer->second;
        }

        Value *instrMalloc = dyn_cast<CallInst>(v)->getArgOperand(0);

        sharedBuffers.push_back(instrMalloc);

        ConstantInt *bufferPosition = dyn_cast<ConstantInt>(instrMalloc);
        uint64_t vectorPosition = bufferPosition->getZExtValue();

        if (!newPointer.count(vectorPosition))
          newPointer.insert(std::make_pair(vectorPosition, secondArgument));
      }
      
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

      /* Changing _cl_set_kernel_arg to _cl_set_kernel_arg_shared_buffer */
      if(size > 0){
        for (int i = 0; i < size; i++) {
          CallInst *S = dyn_cast<CallInst>(argsToBeChanged[i]);

          std::vector<Type *> argsCallFunction;
          std::vector<Value *> argValues;
          argsCallFunction.push_back(llvm::Type::getInt32Ty(M.getContext()));
          argsCallFunction.push_back(llvm::Type::getInt32Ty(M.getContext()));

          Value *bufferSelect = S->getArgOperand(1);
          ConstantInt *bufferPosition = dyn_cast<ConstantInt>(bufferSelect);
          uint64_t vectorPosition = bufferPosition->getZExtValue();

          /* Arg Position */
          argValues.push_back(S->getArgOperand(0));

          /* Shared Buffer key */
          Value *sharedBufferPosition = sharedBuffers[vectorPosition];
          argValues.push_back(sharedBufferPosition);

          FunctionType *FuncTypeArg = FunctionType::get(
              Type::getInt32Ty(M.getContext()), argsCallFunction, false);
          Constant *funcArgKernel = M.getOrInsertFunction(
              "_cl_set_kernel_arg_shared_buffer", FuncTypeArg);
          Function *FuncArg = cast<Function>(funcArgKernel);
          CallInst *createFuncArg = CallInst::Create(FuncArg, argValues);

          if (isTail)
            createFuncArg->setTailCall(true);

          /* Setting Attribute */
          AttributeSet gpuClangAttributes =
              cast<CallInst>(_cl_device_init)->getAttributes();
          createFuncArg->setAttributes(gpuClangAttributes);

          Instruction *instrToReplace = argsToBeChanged[i];
          BasicBlock::iterator ii(instrToReplace);
          ReplaceInstWithInst(instrToReplace->getParent()->getInstList(), ii,
                              createFuncArg);

        }
      }else{
        CallInst *argsCheck;
        Instruction *nextInstr;
        bool findArgInst = false;
        for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
          Instruction *checkInstruction = &*I;
          if (CallInst *S = dyn_cast<CallInst>(checkInstruction)){
            if (S->getCalledFunction()->getName() == "_cl_execute_kernel" ||
                S->getCalledFunction()->getName() == "_cl_execute_tiled_kernel"){
              if (checkInstruction == instr)
                break;
              else
                findArgInst = false;
            }
        
            if (S->getCalledFunction()->getName() == "_cl_set_kernel_args"){
              argsCheck = S;
              findArgInst = true;
              nextInstr = &*++I;
            }
          }
        }

        if (findArgInst){
          Value *bufferSelect = argsCheck->getArgOperand(0);
          ConstantInt *bufferPosition = dyn_cast<ConstantInt>(bufferSelect);
          uint64_t vectorPosition = bufferPosition->getZExtValue();

          for(int i = 0; i < (int)vectorPosition; i++){
            std::vector<Type *> argsCallFunction;
            std::vector<Value *> argValues;
            argsCallFunction.push_back(llvm::Type::getInt32Ty(M.getContext()));
            argsCallFunction.push_back(llvm::Type::getInt32Ty(M.getContext()));

            /* Arg Position */
            Value *argPosition = ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), i);
            argValues.push_back(argPosition);

            /* Shared Buffer key */
            Value *bufferPosition = sharedBuffers[i];
            argValues.push_back(bufferPosition);

            FunctionType *FuncTypeArg = FunctionType::get(
                Type::getInt32Ty(M.getContext()), argsCallFunction, false);
            Constant *funcArgKernel = M.getOrInsertFunction(
                "_cl_set_kernel_arg_shared_buffer", FuncTypeArg);
            Function *FuncArg = cast<Function>(funcArgKernel);
            CallInst *createFuncArg = CallInst::Create(FuncArg, argValues);

            if (isTail)
              createFuncArg->setTailCall(true);

            /* Setting Attribute */
            AttributeSet gpuClangAttributes = cast<CallInst>(_cl_device_init)->getAttributes();
            createFuncArg->setAttributes(gpuClangAttributes);

            BasicBlock *BB = nextInstr->getParent();
            BB->getInstList().insert(nextInstr, createFuncArg);

          }
        }
      }
    }

    if (!func->isDeclaration()) {
      pointerCast.insert(std::make_pair(func, newPointer));
    }
  }
}

/* Update the DCA's information with the shared buffers created  */
void DCAO::updateDCAData(Module &M,
                         CoherenceAnalysis *Analysis) {
  std::map<Value *, Value *>::iterator it;
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {

    /*Updating i_InOut */
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      Instruction *instr = &*I;
      inOut iOld = Analysis->i_InOut.lookup(instr);
      inOut iNew;
      bool modified = 0;

      /* OUT-set */
      for (llvm::DenseMap<const llvm::Value *, tuple>::iterator u =
               iOld.out.begin(); u != iOld.out.end(); ++u) {

        Value *v = const_cast<Value *>(u->first);
        if (mallocToclCreateBuffer.count(v)) {
          it = mallocToclCreateBuffer.find(v);
          tuple t = u->second;
          iNew.out.insert(std::make_pair(it->second, t));
          modified = 1;
        }
      }

      /* IN-set */
      for (llvm::DenseMap<const llvm::Value *, tuple>::iterator u =
               iOld.in.begin(); u != iOld.in.end(); ++u) {

        Value *v = const_cast<Value *>(u->first);
        if (mallocToclCreateBuffer.count(v)) {
          it = mallocToclCreateBuffer.find(v);
          tuple t = u->second;
          iNew.in.insert(std::make_pair(it->second, t));
          modified = 1;
        }
      }

      /* Check and update */
      if (modified) {
        Analysis->i_InOut.erase(instr);
        Analysis->i_InOut.insert(std::make_pair(instr, iNew));
      }
    }
  }

  /* Updating bb_inOut */
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    Function *func = &*F;
    for (Function::iterator BB = func->begin(); BB != func->end(); ++BB) {
      BasicBlock *bb = &*BB;
      inOut iOld = Analysis->bb_InOut.lookup(bb);
      inOut iNew;
      bool modified = 0;

      /* Analysing Instruction OUT */
      for (llvm::DenseMap<const llvm::Value *, tuple>::iterator u =
               iOld.out.begin(); u != iOld.out.end(); ++u) {

        Value *v = const_cast<Value *>(u->first);
        if (mallocToclCreateBuffer.count(v)) {
          it = mallocToclCreateBuffer.find(v);
          tuple t = u->second;
          iNew.out.insert(std::make_pair(it->second, t));
          modified = 1;
        }
      }

      /* Analysing Instruction IN */
      for (llvm::DenseMap<const llvm::Value *, tuple>::iterator u =
               iOld.in.begin(); u != iOld.in.end(); ++u) {

        Value *v = const_cast<Value *>(u->first);
        if (mallocToclCreateBuffer.count(v)) {
          it = mallocToclCreateBuffer.find(v);
          tuple t = u->second;
          iNew.in.insert(std::make_pair(it->second, t));
          modified = 1;
        }
      }

      /* Check and update */
      if (modified) {
        Analysis->bb_InOut.erase(bb);
        Analysis->bb_InOut.insert(std::make_pair(bb, iNew));
      }
    }
  }

  /* i_GenKill */ 
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      Instruction *instr = &*I;

      if(Analysis->i_GenKill.count(instr) == 0)
        continue;

      genKill iOld = Analysis->i_GenKill.lookup(instr);
      genKill iNew;
      bool modified = 0;

      /* GEN-set */
      for (llvm::DenseMap<const llvm::Value *, tuple>::iterator u =
               iOld.gen.begin(); u != iOld.gen.end(); ++u) {

        Value *v = const_cast<Value *>(u->first);
        if (mallocToclCreateBuffer.count(v)) {
          it = mallocToclCreateBuffer.find(v);
          tuple t = u->second;
          iNew.gen.insert(std::make_pair(it->second, t));
          modified = 1;
        }
      }

      /* Check and update */
      if (modified) {
        Analysis->i_GenKill.erase(instr);
        Analysis->i_GenKill.insert(std::make_pair(instr, iNew));
      }
    }
  }
}
  
/* This function remove unnecessary functions */
void DCAO::removeGPUUnnecessaryCalls(Module &M) {
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      Instruction *instrReleaseBuffer = &*I;
      if (CallInst *S = dyn_cast<CallInst>(instrReleaseBuffer)) {
        Function *releaseFunction = S->getCalledFunction();
        bool rem = false;
        if (releaseFunction->getName() == "_cl_release_buffer" ||
            releaseFunction->getName() == "_cl_create_write_only" ||
            releaseFunction->getName() == "_cl_create_read_write" ||
            releaseFunction->getName() == "_cl_read_buffer" ||
            releaseFunction->getName() == "_cl_offloading_read_write" ||
            releaseFunction->getName() == "_cl_offloading_write_only" ||
            releaseFunction->getName() == "_cl_create_read_only" ||
            releaseFunction->getName() == "_cl_set_kernel_args" || 
            releaseFunction->getName() == "_cl_offloading_read_only" ||
            releaseFunction->getName() == "_cl_read_only") {
          rem = true;
        } else if (releaseFunction->getName() == "free") {
          Value *checkValue = S->getArgOperand(0);
          if(CallInst *checkFree = dyn_cast<CallInst>(checkValue)){
            if (checkFree->getCalledFunction()->getName() == "_cl_map_buffer_write"){
              //void _cl_release_shared_buffer(int index);
              std::vector<Type *> args;
              args.push_back(llvm::Type::getInt32Ty(M.getContext()));
              FunctionType *FuncTypeRelease = FunctionType::get(Type::getVoidTy(M.getContext()), args, false);
              Constant *createRelease = M.getOrInsertFunction("_cl_release_shared_buffer", FuncTypeRelease);
              Function *FuncRelease = cast<Function>(createRelease);
              Value *newValue = checkFree->getArgOperand(0);
              std::vector<Value *> argumentsRelease;
              argumentsRelease.push_back(newValue);
              CallInst *createFuncRelease = CallInst::Create(FuncRelease, argumentsRelease);

              if (isTail)
                createFuncRelease->setTailCall(true);

              AttributeSet gpuClangAttributes = cast<CallInst>(_cl_device_init)->getAttributes();
              createFuncRelease->setAttributes(gpuClangAttributes);

              BasicBlock *BB = instrReleaseBuffer->getParent();
              BB->getInstList().insert(instrReleaseBuffer, createFuncRelease);
              rem = true;

            }else if(!(checkFree->getCalledFunction()->getName() == "malloc")){
              rem = true;
            }
          }else{
            rem = true;
          }
        }

        if(rem){
          instrReleaseBuffer->eraseFromParent();
          I = inst_begin(F);
        }
      }
    }
  }
}

/* This function uses DCA's informations to set map and unmap instructions */
void DCAO::applyMapAndUnmapInstructions(Module &M, 
                                        CoherenceAnalysis *Analysis) {

  std::vector<BasicBlock *> reachableBB;
  std::set<BasicBlock *> visitedBB;
  std::set<BasicBlock *> checkAfter;

  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    Function *func = &*F;
    if (!(func->isDeclaration())) {
      int totalBB = func->getBasicBlockList().size();
      int totalBBVisited;

      for (std::map<Value *, clBufferInfos>::iterator bufferGPU =
               buffersInfo.begin(); bufferGPU != buffersInfo.end(); ++bufferGPU) {

        CA CAfirstBB;
        std::map<BasicBlock *, std::map<Value *, CA>>::iterator it;
        std::map<Value *, CA>::iterator it_varCA;
        std::map<Value *, CA> *ptr_varCA;
        std::map<Value *, CA> varCA;
        int counter = 0;
        int status, scope;

        /* Clean all sets before starting */
        reachableBB.clear();
        visitedBB.clear();
        checkAfter.clear();
        CAMapUnmap.clear();

        Value *buffer = bufferGPU->first;
        BasicBlock *firstBB = &func->front();

        /* Start all functions with CPU/W */
        CAfirstBB.scope_top = cpu_;
        CAfirstBB.status_top = W_;

        reachableBB.push_back(firstBB);
        varCA.insert(std::make_pair(buffer, CAfirstBB));

        /* Remove the if clausule */
        if (CAMapUnmap.count(firstBB)) {
          it = CAMapUnmap.find(firstBB);
          ptr_varCA = &it->second;
          ptr_varCA->insert(std::make_pair(buffer, CAfirstBB));
        } else
          CAMapUnmap.insert(std::make_pair(firstBB, varCA));

        /* If this current function creates the shared buffer, then it is
           necessary to start setting map and unmap function after the creation */
        bool isCreated = false;
        bool isBuffer = false;
        for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
          Instruction *instr = &*I;
          if (instr == buffer)
            isCreated = true;
        }

        /* The ID of the shared buffer */
        std::map<Value *, clBufferInfos>::iterator bufferInfo;
        bufferInfo = buffersInfo.find(buffer);
        clBufferInfos sharedBufferInfos;
        sharedBufferInfos = bufferInfo->second;
        int position = sharedBufferInfos.position;

        do {
          if ((int)reachableBB.size() == counter) {
            if(checkAfter.size() == 0) break;
            /* Choose the BB that all Predecessors were visited */
            BasicBlock *insertBB;
            bool checkAfterFound = false;

            for(std::set<BasicBlock*>::iterator it_checkAfter = checkAfter.begin(); 
                it_checkAfter != checkAfter.end(); ++it_checkAfter){
              BasicBlock *BBCheckAfterCheck = *it_checkAfter;
              if(predBBWereVisited(BBCheckAfterCheck, visitedBB)){
                insertBB = BBCheckAfterCheck;
                checkAfterFound = true;
                break;
              }
            }

            if(!checkAfterFound)
              insertBB = (*checkAfter.begin());

            checkAfter.erase(insertBB);
            reachableBB.push_back(insertBB);

            bool hasCPU = false, hasGPU = false;
            int new_status = 0;
            int new_scope;

            isCombination(insertBB, &hasCPU, &hasGPU, &new_status, buffer);
            if (hasCPU && hasGPU)
              new_scope = cpu_;
            else if (hasCPU)
              new_scope = cpu_;
            else
              new_scope = gpu_;

            updateCANewBB(insertBB, buffer, new_status, new_scope);
          }

          BasicBlock *BB = reachableBB[counter];
          visitedBB.insert(BB);
          it = CAMapUnmap.find(BB);
          ptr_varCA = &it->second;
          it_varCA = ptr_varCA->find(buffer);

          Instruction *instr = &BB->front();
          CA *ca = &it_varCA->second;
          status = ca->status_top;
          scope = ca->scope_top;

          inOut iInOut = Analysis->i_InOut.lookup(instr);

          /* Checking if there is transitions between instructions inside
             a basic block */
          for (BasicBlock::iterator I = BB->begin(); I != BB->end(); ++I) {
            instr = &*I;
            if (isCreated && !isBuffer) {
              if (instr == buffer)
                isBuffer = true;
              else
                continue;
            }

            /* It is possible to insert instructions only after all PHINodes */
            if(isa<PHINode>(instr)) continue;
            else if(isa<BitCastInst>(instr)) continue;
            else if(isa<AllocaInst>(instr)) continue;
            else if(checkCallSetDevice(instr)) continue;

            inOut i = Analysis->i_InOut.lookup(instr);
            tuple t = i.out.lookup(buffer);

            if (!instr->isTerminator()) {
              BasicBlock::iterator Inext = I;
              Inext++;
              instr = &*Inext;
            }

            /* IN[var] = (v,R,CPU) */
            if (scope == cpu_ && status == R_) {
              if (t.scope == cpu_ && t.status != R_)
                status = t.status;
              else if (t.status != R_ && t.scope == gpu_) {
                scope = gpu_;
                status = t.status;
                unmap(position, M, instr);
              }
              /* IN[var] = (v,W,CPU) */
            } else if (scope == cpu_ && status == W_) {
              if (t.scope == gpu_) {
                scope = gpu_;
                status = t.status;
                unmap(position, M, instr);
              } else if (t.scope == cpu_ && t.status == RW_)
                status = RW_;
              /* IN[var] = (v,RW,CPU) */
            } else if (scope == cpu_ && status == RW_) {
              if (t.scope == gpu_) {
                scope = gpu_;
                status = t.status;
                unmap(position, M, instr);
              } else if (t.scope == cpu_ && t.status != RW_)
                status = RW_;
              /* IN[var] = (v,R,GPU) */
            } else if (scope == gpu_ && status == R_) {
              if (t.scope == cpu_) {
                scope = cpu_;
                status = t.status;
                map(position, M, instr, status);
              } else if (t.scope == gpu_)
                status = t.status;
              /* IN[var] = (v,W,GPU) */
            } else if (scope == gpu_ && status == W_) {
              if (t.scope == cpu_) {
                scope = cpu_;
                status = t.status;
                map(position, M, instr, status);
              } else if (t.scope == gpu_ && t.status == RW_)
                status = RW_;
              /* IN[var] = (v,RW,GPU) */
            } else if (scope == gpu_ && status == RW_) {
              if (t.scope == cpu_) {
                scope = cpu_;
                status = t.status;
                map(position, M, instr, status);
              } else if (t.scope == gpu_ && t.status != RW_)
                status = RW_;
            }
          }

          ca->status_bot = status;
          ca->scope_bot = scope;

          /* Check if there is any BB succ that has x as Scope */
          for (succ_iterator succ_itBB = succ_begin(BB), succ_etBB = succ_end(BB); succ_itBB != succ_etBB; ++succ_itBB) {
            BasicBlock *BB_succ = *succ_itBB;
            inOut succInOutBB = Analysis->bb_InOut.lookup(BB_succ);
            tuple t = succInOutBB.in.lookup(buffer);

            std::map<BasicBlock *, std::map<Value *, CA>>::iterator itSucc;
            std::map<Value *, CA>::iterator it_varCASucc;
            std::map<Value *, CA> *ptr_varCASucc;
            BasicBlock *bbMap;

            int numberOfPredecessor = getNumberOfPredecessor(BB_succ);
            int numberOfSuccessor = getNumberOfSuccessor(BB);
            int new_scope, new_status;
            new_status = 0;

            int updateCA = true;
            bool hasCPU = false;
            bool hasGPU = false;

            if (t.scope == X_) {
              /* Check if all Predecessor of BB_succ were visited.
                 If they were, then include the BB_succ into the set checkAfter set. */
              if (numberOfPredecessor > 1) {
                bool wereVisited = true;

                if (!predBBWereVisited(BB_succ, visitedBB)) {
                  updateCA = false;
                  wereVisited = false;
                  checkAfter.insert(BB_succ);
                }

                /* Check if there is the following combination between at least two 
                 * predecessor of the BB_succ: CA_1 = gpu and CA_2 = cpu */
                if (wereVisited) {
                  isCombination(BB_succ, &hasCPU, &hasGPU, &new_status, buffer);

                  if (hasCPU && hasGPU) {
                    new_scope = cpu_;
                    new_status = RW_;

                    for (pred_iterator pred_it = pred_begin(BB_succ),
                         pred_et = pred_end(BB_succ); pred_it != pred_et; ++pred_it) {
                      BasicBlock *BB_pred = *pred_it;
                      itSucc = CAMapUnmap.find(BB_pred);
                      ptr_varCASucc = &itSucc->second;
                      it_varCASucc = ptr_varCASucc->find(buffer);
                      CA *caBB = &it_varCASucc->second;

                      /* Create a new basic block between BB_succ and BBpred_succ */
                      if (caBB->scope_bot == gpu_) {
                        int predBBNumberOfSucc = getNumberOfSuccessor(BB_pred);
                        if (predBBNumberOfSucc > 1) {
                          totalBB++;
                          BranchInst *BI = createNewBB(M, BB_pred, BB_succ, bbMap);
                          updatePHINode(BB_pred, BB_succ, BI->getParent());
                          map(position, M, BI, RW_);
                          updateCANewBB(BI->getParent(), buffer, new_status, new_scope);
                        } else {
                          TerminatorInst *terminator = BB_pred->getTerminator();
                          map(position, M, terminator, RW_);
                          caBB->status_bot = new_status;
                          caBB->scope_bot = new_scope;
                        }
                      }
                    }
                  } else {
                    if (hasCPU)
                      new_scope = cpu_;
                    else if (hasGPU)
                      new_scope = gpu_;
                  }

                  if(checkAfter.count(BB_succ)){
                    checkAfter.erase(BB_succ);
                  }
                }
              } else {
                new_status = status;
                new_scope = scope;
              }

              /* If IN[BB_succ] == cpu */
            } else if (t.scope == cpu_) {
              if (scope == gpu_) {
                new_status = t.status;
                new_scope = t.scope;
                if (numberOfPredecessor > 1 && numberOfSuccessor > 1) {
                  totalBB++;
                  BranchInst *BI = createNewBB(M, BB, BB_succ, bbMap);
                  updatePHINode(BB, BB_succ, BI->getParent());
                  map(position,  M, BI, t.status);
                  updateCANewBB(BI->getParent(), buffer, new_status, new_scope);
                } else if (numberOfSuccessor > 1) {
                  Instruction *firstInst = BB_succ->getFirstNonPHI(); 
                  map(position, M, firstInst, t.status);
                } else {
                  TerminatorInst *terminator = BB->getTerminator();
                  map(position, M, terminator, t.status);
                  itSucc = CAMapUnmap.find(BB);
                  ptr_varCASucc = &itSucc->second;
                  it_varCASucc = ptr_varCASucc->find(buffer);
                  CA *caBB = &it_varCASucc->second;
                  caBB->scope_bot = new_scope;
                  caBB->status_bot = new_status;
                }
              } else {
                new_status = status;
                new_scope = scope;
              }
              /* if IN[BB_succ] == gpu */
            } else if (t.scope == gpu_) {
              if ((scope == cpu_ && status != R_) || (scope == cpu_ && t.status != R_)) {
                new_status = t.status;
                new_scope = t.scope;
                if (numberOfPredecessor > 1 && numberOfSuccessor > 1) {
                  totalBB++;
                  BranchInst *BI = createNewBB(M, BB, BB_succ, bbMap);
                  updatePHINode(BB, BB_succ, BI->getParent());
                  unmap(position, M, BI);
                  updateCANewBB(BI->getParent(), buffer, new_status, new_scope);
                } else if (numberOfSuccessor > 1) { 
                  Instruction *firstInst = BB_succ->getFirstNonPHI(); 
                  unmap(position, M, firstInst);
                } else {
                  TerminatorInst *terminator = BB->getTerminator();
                  unmap(position, M, terminator);
                  itSucc = CAMapUnmap.find(BB);
                  ptr_varCASucc = &itSucc->second;
                  it_varCASucc = ptr_varCASucc->find(buffer);
                  CA *caBB = &it_varCASucc->second;
                  caBB->scope_bot = new_scope;
                  caBB->status_bot = new_status;
                }
              } else {
                new_status = status;
                new_scope = scope;
              }
              /* If IN[BB_succ] == "" */
            } else {
              if (!predBBWereVisited(BB_succ, visitedBB)) {
                updateCA = false;
                checkAfter.insert(BB_succ);
              } else {
                if (checkAfter.count(BB_succ))
                  checkAfter.erase(BB_succ);
                if (numberOfPredecessor > 1) {
                  isCombination(BB_succ, &hasCPU, &hasGPU, &new_status, buffer);
                  if (hasCPU && hasGPU)
                    new_scope = cpu_;
                  else if (hasCPU)
                    new_scope = cpu_;
                  else
                    new_scope = gpu_;
                } else {
                  new_status = status;
                  new_scope = scope;
                }
              }
            }

            if (updateCA) {
              /* Updating CA of BB_succ*/
              if (!CAMapUnmap.count(BB_succ) && !checkAfter.count(BB_succ)) {
                std::map<Value *, CA> newBBCA;
                CAMapUnmap.insert(std::make_pair(BB_succ, newBBCA));
              }

              itSucc = CAMapUnmap.find(BB_succ);
              ptr_varCASucc = &itSucc->second;

              if (!ptr_varCASucc->count(buffer)) {
                CA caBB;
                caBB.scope_top = new_scope;
                caBB.status_top = new_status;
                ptr_varCASucc->insert(std::make_pair(buffer, caBB));
              } else {
                it_varCASucc = ptr_varCASucc->find(buffer);
                CA *caBB = &it_varCASucc->second;
                caBB->scope_top = new_scope;
                caBB->status_top = new_status;
              }
            }
          }

          /* Iterate over all instructions that belongs to the blockbasic BB,
             looking for transictions between CPU-GPU */
          for (succ_iterator succ_it = succ_begin(BB), succ_et = succ_end(BB);
               succ_it != succ_et; ++succ_it) {
            BasicBlock *BB_succ = *succ_it;
            
            if (visitedBB.count(BB_succ))
              continue;
            if (checkAfter.count(BB_succ))
              continue;

            int sizeReachableBB = reachableBB.size();
            bool hasValue = false;
            for (int i = 0; i < sizeReachableBB; i++)
              if (reachableBB[i] == BB_succ)
                hasValue = true;

            if (!hasValue){
              reachableBB.push_back(BB_succ);
            }
          }

          counter++;
          totalBBVisited = visitedBB.size();
        } while (totalBBVisited != totalBB);

        /* Before starting the BB's sentences process, check if the IN of the
           first sentence is GPU, if so, then start unmapping */
        if (!isCreated) {
          BasicBlock *BB_back = &func->back();
          std::map<BasicBlock *, std::map<Value *, CA>>::iterator it_CAMapUnmap;
          std::map<Value *, CA> instCA;
          std::map<Value *, CA>::iterator it_instCA;
          CA caBB;

          it_CAMapUnmap = CAMapUnmap.find(BB_back);
          instCA = it_CAMapUnmap->second;
          it_instCA = instCA.find(buffer);
          caBB = it_instCA->second;
          if (caBB.scope_bot == gpu_) {
            map(position, M, BB_back->getTerminator(), caBB.status_bot);
          }
        }
      }
    }
  }
}

/* Return the number of predecessors */
int DCAO::getNumberOfPredecessor(BasicBlock *BB) {
  int number = 0;
  for (pred_iterator pred_it = pred_begin(BB), pred_et = pred_end(BB);
       pred_it != pred_et; ++pred_it)
    number++;

  return number;
}

/* Return the number of sucessors */
int DCAO::getNumberOfSuccessor(BasicBlock *BB) {
  int number = 0;
  for (succ_iterator succ_it = succ_begin(BB), succ_et = succ_end(BB);
       succ_it != succ_et; ++succ_it)
    number++;

  return number;
}

/* Create the map function */
void DCAO::map(int buffer,
               Module &M,
               Instruction *before,
               int status) {
  /* STATIC */
  ++mapInstructionsCreated;

  /* Defining the type of access */
  std::string typeOfAccess;
  if (status == R_)
    typeOfAccess = "_cl_map_buffer_read";
  else if (status == W_)
    typeOfAccess = "_cl_map_buffer_write";
  else
    typeOfAccess = "_cl_map_buffer_read_write";

  /* Defining function's arguments */
  std::vector<Type *> args;
  args.push_back(llvm::Type::getInt32Ty(M.getContext()));
  FunctionType *FuncTypeMap = FunctionType::get(Type::getInt8PtrTy(M.getContext()), args, false);
  Constant *createMap = M.getOrInsertFunction(typeOfAccess, FuncTypeMap);
  Function *FuncMap = cast<Function>(createMap);

  /* Creating a constant value */
  Value *newValue = ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), buffer);

  /* Creating CallInst */
  std::vector<Value *> argumentsMap;
  argumentsMap.push_back(newValue);

  CallInst *createFuncMap = CallInst::Create(FuncMap, argumentsMap);

  if (isTail)
    createFuncMap->setTailCall(true);

  /* Setting Attribute */
  AttributeSet gpuClangAttributes = cast<CallInst>(_cl_device_init)->getAttributes();

  createFuncMap->setAttributes(gpuClangAttributes);

  BasicBlock *BB = before->getParent();
  BB->getInstList().insert(before, createFuncMap);

  mapInstructions.push_back(cast<Instruction>(createFuncMap));
}

/* This function inserts an unmap function */
void DCAO::unmap(int buffer,
                 Module &M,
                 Instruction *before) {
  /* STATICS */
  ++unmapInstructionsCreated;

  /* Defining function's arguments */
  std::vector<Type *> args;
  args.push_back(llvm::Type::getInt32Ty(M.getContext()));
  FunctionType *FuncTypeUnMap = FunctionType::get(Type::getVoidTy(M.getContext()), args, false);
  Constant *createUnMap = M.getOrInsertFunction("_cl_unmap_buffer", FuncTypeUnMap);

  Function *FuncUnMap = cast<Function>(createUnMap);

  /* Creating a constant value */
  Value *newValue = ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), buffer);

  /* Creating CallInst */
  std::vector<Value *> argumentsUnMap;
  argumentsUnMap.push_back(newValue);
  CallInst *createFuncUnMap = CallInst::Create(FuncUnMap, argumentsUnMap);

  if (isTail)
    createFuncUnMap->setTailCall(true);

  /* Setting Attribute */
  AttributeSet gpuClangAttributes = cast<CallInst>(_cl_device_init)->getAttributes();
  createFuncUnMap->setAttributes(gpuClangAttributes);

  BasicBlock *BB = before->getParent();
  BB->getInstList().insert(before, createFuncUnMap);
  unmapInstructions.push_back(cast<Instruction>(createFuncUnMap));
}

/* Create BB between a critial edge */
BranchInst *DCAO::createNewBB(Module &M,
                              BasicBlock *BB_pred,
                              BasicBlock *BB_succ,
                              BasicBlock *bbMap) {

  std::string bbName = "DCO";
  bbName = bbName + std::to_string(BBNumber);
  BBNumber++;
  Function *F = BB_pred->getParent();
  bbMap = llvm::BasicBlock::Create(M.getContext(), bbName, F, BB_succ);
  int idx = getIdxBBSuccessor(BB_pred, BB_succ);
  TerminatorInst *terminator = BB_pred->getTerminator();
  BranchInst *branch = dyn_cast<BranchInst>(terminator);
  branch->setSuccessor(idx, bbMap);
  BranchInst *createBranch = llvm::BranchInst::Create(BB_succ, bbMap);

  return createBranch;
}

/* Identify the Branch Instruction's Index to change the CFG */
int DCAO::getIdxBBSuccessor(BasicBlock *BBPred,
                            BasicBlock *BBSucc) {
  int idx = 0;
  TerminatorInst *terminator = BBPred->getTerminator();
  BranchInst *branch = dyn_cast<BranchInst>(terminator);
  int n = branch->getNumSuccessors();
  for (; idx < n; idx++) {
    BasicBlock *isBBEqual = branch->getSuccessor(idx);
    if (isBBEqual == BBSucc)
      break;
  }

  return idx;
}

/* Update the CA of the Basic Block BB */
void DCAO::updateCANewBB(BasicBlock *BB,
                         Value *buffer,
                         int status,
                         int scope) {

  CA caBB;
  std::map<BasicBlock *, std::map<Value *, CA>>::iterator it;
  std::map<Value *, CA>::iterator it_varCA;
  std::map<Value *, CA> *ptr_varCA;
  std::map<Value *, CA> newBBCA;

  caBB.scope_top = scope;
  caBB.status_top = status;

  CAMapUnmap.insert(std::make_pair(BB, newBBCA));
  it = CAMapUnmap.find(BB);
  ptr_varCA = &it->second;
  ptr_varCA->insert(std::make_pair(buffer, caBB));
}

/* Identify if all predecessors of BB were updated */
bool DCAO::predBBWereVisited(BasicBlock *BB,
                             std::set<BasicBlock *> visitedBB) {

  int amountOfPredecessor = getNumberOfPredecessor(BB);
  int isSameBB = false;
  int notVisited = false;

  for (pred_iterator pred_it = pred_begin(BB), pred_et = pred_end(BB);
       pred_it != pred_et; ++pred_it) {
    BasicBlock *BBpred = *pred_it;
    
    if (BBpred == BB)
      isSameBB = true;
    if (visitedBB.count(BBpred) == 0) {
      notVisited = true;
    }
  }

  if (amountOfPredecessor == 2 && isSameBB)
    return true;
  else if (notVisited)
    return false;

  return true;
}

/* Identify if there are CPU and GPU combination */
void DCAO::isCombination(BasicBlock *BB, 
                         bool *hasCPU,
                         bool *hasGPU, 
                         int *new_status,
                         Value *buffer) {

  std::map<BasicBlock *, std::map<Value *, CA>>::iterator itSucc;
  std::map<Value *, CA>::iterator it_varCASucc;
  std::map<Value *, CA> *ptr_varCASucc;
  CA *caBBPred;

  for (pred_iterator pred_it = pred_begin(BB), pred_et = pred_end(BB);
       pred_it != pred_et; ++pred_it) {
    BasicBlock *BBpred = *pred_it;
    itSucc = CAMapUnmap.find(BBpred);
    ptr_varCASucc = &itSucc->second;

    if (!ptr_varCASucc->count(buffer))
      continue;

    it_varCASucc = ptr_varCASucc->find(buffer);
    caBBPred = &it_varCASucc->second;

    /* Check status combination */
    if (*new_status == 0)
      *new_status = caBBPred->status_bot;
    else if (*new_status == R_ && caBBPred->status_bot != R_)
      *new_status = RW_;
    else if (*new_status == W_ && caBBPred->status_bot != W_)
      *new_status = RW_;

    /* Check if it is GPU or CPU */
    if (caBBPred->scope_bot == gpu_)
      *hasGPU = true;
    else if (caBBPred->scope_bot == cpu_)
      *hasCPU = true;
  }
}

bool DCAO::checkCallSetDevice(Instruction *instr) {
  if (CallInst *S = dyn_cast<CallInst>(instr)){
    if(S->getCalledFunction()->getName() == "_set_default_device"){
      return true;
    }
  }
  return false;
}

/* Update the PHI Node when created an new Basic Block */
void DCAO::updatePHINode(BasicBlock *BB,
                         BasicBlock *BBSucc,
                         BasicBlock *newBB) {
  for (BasicBlock::iterator I = BBSucc->begin(); I != BBSucc->end(); ++I) {   
    if(isa<PHINode>(*I)){
      PHINode *phi = cast<PHINode>(I);
      int idx = phi->getBasicBlockIndex(BB);
      if(idx != -1){
        phi->setIncomingBlock(idx, newBB);
      }
    }
  }
}

bool DCAO::tryMoveMapOutLoop(Instruction *instToCheck, Loop *L, bool isMap) {  
  BasicBlock *preHeader = L->getLoopPreheader();
  
  ConstantInt *constPosition = dyn_cast<ConstantInt>(instToCheck->getOperand(0));
  int position = (int) constPosition->getZExtValue();

  Instruction *buffer;
  for (std::map<Value *, clBufferInfos>::iterator bufferGPU =
       buffersInfo.begin(); bufferGPU != buffersInfo.end(); ++bufferGPU) {
    Instruction *instBuffer = cast<Instruction>(bufferGPU->first);
    ConstantInt *mapBufferPosition = dyn_cast<ConstantInt>(instBuffer->getOperand(0));
    int bufferPosition = (int) mapBufferPosition->getZExtValue();

    if (bufferPosition == position){
      buffer = instBuffer;
      break;
    }
  }
  
  bool hasGPU, hasCPU;
  hasGPU = hasCPU = false; 
  for(Loop::block_iterator bb = L->block_begin(); bb != L->block_end(); ++bb){
    BasicBlock *BB = *bb;
    for(BasicBlock::iterator I = BB->begin() ; I != BB->end(); ++I){
      Instruction *instr = &*I;
      
      if(Analysis->i_GenKill.count(instr) == 0)
        continue;
    
      genKill instrGen = Analysis->i_GenKill.lookup(instr);
      DenseMap<const llvm::Value *, tuple> genCheck = instrGen.gen;
      for(DenseMap<const Value*, tuple>::iterator it = genCheck.begin(); it != genCheck.end() ; ++it){
        Value *genValue = const_cast<Value *>(it->first);
        if (!(genValue == buffer)) continue; 
        tuple instTuple = it->second;
        if (instTuple.scope == cpu_) hasCPU = true;
        else if (instTuple.scope == gpu_) hasGPU = true;
      }  
    }
  }

  /* Move to the header BB when: 
   * 1 - The loop has no one CPU and GPU instructions;
   * 2 - It is map and there is no one GPU instruction;
   * 3 - It is unmap and there is no one CPU instruction;
  */
  if((hasGPU && !hasCPU && !isMap) || 
     (hasCPU && !hasGPU && isMap) || 
     (!hasCPU && !hasGPU)){
    instToCheck->removeFromParent();
    Instruction *terminator = &preHeader->back();
    preHeader->getInstList().insert(terminator, instToCheck);
  } 

  return true;
}

/* This function attempt to remove intructions inside a loop */
void DCAO::tryRemoveMapAndUnmapFromLoop(){
  int size = mapInstructions.size();
  for(int i = 0; i < size; i++){
    Instruction *mapInst = mapInstructions[i];
    BasicBlock *mapBB = mapInst->getParent();
    Function *mapFunc = mapInst->getParent()->getParent();

    LoopInfo &LI = getAnalysis<LoopInfo>(*mapFunc);

    while (true) {
      if (Loop *L = LI.getLoopFor(mapBB)) {
        if(!tryMoveMapOutLoop(mapInst, L, true)){
          break;
        }
      } else {
        break;
      }
    
      break;
      mapBB = mapInst->getParent();
    }
  }

  size = unmapInstructions.size();
  for(int i = 0; i < size; i++){
    Instruction *unmapInst = unmapInstructions[i];
    BasicBlock *unmapBB = unmapInst->getParent();
    Function *unmapFunc = unmapInst->getParent()->getParent();

    LoopInfo &LI = getAnalysis<LoopInfo>(*unmapFunc);

    while (true) {
      if (Loop *L = LI.getLoopFor(unmapBB)) {
        if(!tryMoveMapOutLoop(unmapInst, L, false)){
          break;
        }
      } else {
        break;
      }
    
      break;
      unmapBB = unmapInst->getParent();
    }
  }
}

void DCAO::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<AliasAnalysis>();
  AU.addRequired<DominatorTreeWrapperPass>(); 
  AU.addRequired<LoopInfo>();
  AU.addRequired<CallGraphWrapperPass>();
}

char DCAO::ID = 0;
//static RegisterPass<DCAO> X("dcao", "Data Coherence Analysis and Optimization", false, false);

INITIALIZE_PASS(DCAO, "dcao",
                "Data Coherence Analysis and Optimization", false, false)

ModulePass *llvm::createDCAOPass(int dbg) { 
  dbg_dcao = dbg; 
  return new DCAO(); 
}
