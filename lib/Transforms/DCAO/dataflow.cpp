#include "dataflow.h"

using namespace llvm;

/* Compute IN-OUT sets of each Basic Block */
void DataFlow::computeBB_In_Out(
    Function &F, DenseMap<const BasicBlock *, genKill> &BB_GenKill,
    DenseMap<const BasicBlock *, inOut> &BB_InOut,
    std::vector<Value *> domain) {

  int BBNumber = 0;
  /* The first step is to do IN[B] = GEN[B]
     It means that the IN[B] start its value equal to GEN[B]. */
  for (Function::iterator BB = F.begin(); BB != F.end(); ++BB) {

    if (!BB->hasName()) {
      std::string bbName = "BB";
      bbName = bbName + std::to_string(BBNumber);
      BB->setName(bbName);
    }

    genKill BB_GenValue;
    BB_GenValue = BB_GenKill.lookup(BB);
    DenseMap<const llvm::Value *, tuple> BBInFirstValue = BB_GenValue.gen;

    /* Generating the IN[B] */
    inOut BB_FirstInValue;
    BB_FirstInValue.in = BBInFirstValue;
    BB_InOut.insert(std::make_pair(&*BB, BB_FirstInValue));

  }

  /* Iterate while no changes ocours between two iterations */
  bool check;
  do {
    check = false;
    for (Function::iterator BB = F.begin(); BB != F.end(); ++BB) {

      inOut BB_InOutValues;
      BB_InOutValues = BB_InOut.lookup(BB);

      genKill BB_GenKillValues;
      BB_GenKillValues = BB_GenKill.lookup(BB);

      /* Values used to identify convergence */
      DenseMap<const llvm::Value *, tuple> new_in;
      DenseMap<const llvm::Value *, tuple> old_in;
      DenseMap<const llvm::Value *, tuple> new_out;
      DenseMap<const llvm::Value *, tuple> old_out;

      old_in = BB_InOutValues.in;
      old_out = BB_InOutValues.out;

      int amountOfBBSucc = 0;
      bool isItselfSucc = false;
      bool checkPattern = false;
      
      /* Check the following pattern: If the BB has itself as Succ and any other */
      for (succ_iterator SI = succ_begin(BB), E = succ_end(BB); SI != E; ++SI) {
        BasicBlock *BB_suc = *SI;
        if (BB_suc == BB)
          isItselfSucc = true;

        amountOfBBSucc++;
      }

      if (amountOfBBSucc == 2 && isItselfSucc)
        checkPattern = true;

      /* It is generating the new OUT[B] */
      for (std::vector<Value *>::iterator instr_domain = domain.begin();
           instr_domain != domain.end(); ++instr_domain) {
        Value *v = *instr_domain;

        /* Traversing all BB Succ */
        int count = 0;
        int bb_succ = 0;
        tuple new_tuple;

        for (succ_iterator SI = succ_begin(BB), E = succ_end(BB); SI != E; ++SI) {
          BasicBlock *BB_suc = *SI;
          inOut BBSucc_InOutValues = BB_InOut.lookup(BB_suc);

          if (BBSucc_InOutValues.in.count(v)) {
            tuple tuple_succ = BBSucc_InOutValues.in.lookup(v);

            /* If it is the first BB Succ and match */
            if (count == 0 && bb_succ == 0) {
              new_tuple = tuple_succ;
            }
            /* If it is the first match but is not the first BB Succ */
            else if (count == 0 && bb_succ > 0) {
           //   new_tuple.scope = X_;
           //   new_tuple.status = tuple_succ.status;
              new_tuple = tuple_succ;
            }
            /* If it is not the first match and is not the first BB Succ*/
            else {
              tuple previous_tuple = new_tuple;
              operatorOutNewTuple(&tuple_succ, &previous_tuple, &new_tuple);
            }

            count++;
          }

          bb_succ++;
        }

        if (count) {
          new_out.erase(v);
          new_out.insert(std::make_pair(&*v, new_tuple));
        }

        if (checkPattern) {
          BasicBlock *BB_suc;
          for (succ_iterator SI = succ_begin(BB), E = succ_end(BB); SI != E; ++SI) {
            BasicBlock *BB_checkSucc = *SI;
            if (BB_checkSucc != BB)
              BB_suc = BB_checkSucc;
          }

          inOut BBSucc_CheckInOutValues = BB_InOut.lookup(BB_suc);
          inOut BB_CheckInOutValues = BB_InOut.lookup(BB);

          if (BBSucc_CheckInOutValues.in.count(v) && (!BB_CheckInOutValues.in.count(v))) {
            tuple tuple_succ = BBSucc_CheckInOutValues.in.lookup(v);
            new_out.erase(v);
            new_out.insert(std::make_pair(&*v, tuple_succ));
          }
        }
      }

      /* It is generating the new IN[B] */
      for (std::vector<Value *>::iterator instr_domain = domain.begin();
           instr_domain != domain.end(); ++instr_domain) {
        tuple tuple_gen, tuple_out, tuple_new;
        Value *v = *instr_domain;

        bool outFound, genFound;
        outFound = false;
        genFound = false;

        if (new_out.count(v)) {
          tuple_out = new_out.lookup(v);
          outFound = true;
        }

        if (BB_GenKillValues.gen.count(v)) {
          tuple_gen = BB_GenKillValues.gen.lookup(v);
          genFound = true;
        }

        if (outFound && genFound) {
          operatorInNewTuple(&tuple_gen, &tuple_out, &tuple_new);
          new_in.insert(std::make_pair(&*v, tuple_new));
        } else if (outFound) {
          new_in.insert(std::make_pair(&*v, tuple_out));
        } else if (genFound) {
          new_in.insert(std::make_pair(&*v, tuple_gen));
        }
      }

      BB_InOutValues.in = new_in;
      BB_InOutValues.out = new_out;
      BB_InOut.erase(BB);
      BB_InOut.insert(std::make_pair(&*BB, BB_InOutValues));

      /* Check if both are equal */
      if (!checkIfEqual(new_in, old_in, domain))
        check = true;
      if (!checkIfEqual(new_out, old_out, domain))
        check = true;
    }
  } while (check);
}

void DataFlow::computeS_In_Out(
    Function &F, DenseMap<const Instruction *, genKill> &i_GenKill,
    DenseMap<const Instruction *, inOut> &i_InOut,
    DenseMap<const BasicBlock *, inOut> &BB_InOut,
    std::vector<Value *> domain) {

  for (Function::iterator BB = F.begin(); BB != F.end(); ++BB) {
    const TerminatorInst *TInst = BB->getTerminator();

    Instruction *instrSucc;
    for (BasicBlock::iterator I = BB->end(); I != BB->begin();) {
      I--;

      Instruction *instr = &*I;
      genKill genKillSet = i_GenKill.lookup(instr);

      /* Use it to generate IN[s] and OUT[s] */
      inOut instrInOut;
      llvm::DenseMap<const llvm::Value *, tuple> in;
      llvm::DenseMap<const llvm::Value *, tuple> out;
      llvm::DenseMap<const llvm::Value *, tuple> in_succ;

      bool lastStep = false;

      /* Check if the out is empty:
              If it is, then it is necessary to copy the GEN[s] to its IN[s]
              If it is not, then it is necessary to use the operator.
      */
      if (instr == TInst) {

        instrInOut = i_InOut.lookup(instr);
        inOut BB_InOutValue = BB_InOut.lookup(BB);
        if (!BB_InOutValue.out.empty())
          instrInOut.out = BB_InOutValue.out;

        if (instrInOut.out.empty()) {
          if (!(genKillSet.gen.empty())) {
            instrInOut.in = genKillSet.gen;
          }
        } else {
          lastStep = true;
        }
      }

      if (!instr->isTerminator() || lastStep) {

        /* The OUT[s] is equal to its IN[Succ s]. */
        tuple tuple_gen, tuple_out, tuple_new;

        if (!lastStep) {
          instrInOut.in = in;
          instrInOut.out = out;

          inOut insInOutSucc = i_InOut.lookup(instrSucc);
          in_succ = insInOutSucc.in;

          if (!(in_succ.empty()))
            instrInOut.out = in_succ;
        }

        /* IN[s] = GEN[s] op (OUT[s] - KILL[s]) */
        for (std::vector<Value *>::iterator instr_domain = domain.begin();
             instr_domain != domain.end(); ++instr_domain) {
          Value *v = *instr_domain;

          bool outFound, genFound;
          outFound = false;
          genFound = false;

          if (instrInOut.out.count(v)) {
            tuple_out = instrInOut.out.lookup(v);
            outFound = true;
          }

          if (genKillSet.gen.count(v)) {
            tuple_gen = genKillSet.gen.lookup(v);
            genFound = true;
          }

          if (outFound && genFound) {
            operatorInNewTuple(&tuple_gen, &tuple_out, &tuple_new);
            instrInOut.in.insert(std::make_pair(&*v, tuple_new));
          } else if (outFound) {
            instrInOut.in.insert(std::make_pair(&*v, tuple_out));
          } else if (genFound) {
            instrInOut.in.insert(std::make_pair(&*v, tuple_gen));
          }
        }
      }

      // Before Insert it, I'll try to delete its last register
      i_InOut.erase(instr);
      i_InOut.insert(std::make_pair(&*instr, instrInOut));

      instrSucc = &*instr;
    }
  }
}

void DataFlow::operatorInNewTuple(tuple *gen_tuple, 
                                  tuple *out_tuple,
                                  tuple *new_tuple) {
  if (gen_tuple->scope == cpu_)
    new_tuple->scope = cpu_;
  else if (gen_tuple->scope == gpu_) {
    new_tuple->scope = gpu_;
  }

  if ((gen_tuple->status == R_) && (out_tuple->status == R_)) {
    new_tuple->status = R_;
  } else if ((gen_tuple->status == R_) && (out_tuple->status == W_)) {
    if ((gen_tuple->scope == cpu_) && !(out_tuple->scope == cpu_))
      new_tuple->status = R_;
    else
      new_tuple->status = RW_;
  } else if ((gen_tuple->status == R_) && (out_tuple->status == RW_)) {
    if ((gen_tuple->scope == cpu_) && !(out_tuple->scope == cpu_))
      new_tuple->status = R_;
    else
      new_tuple->status = RW_;
  } else if ((gen_tuple->status == W_) && (out_tuple->status == R_)) {
    new_tuple->status = RW_;
  } else if ((gen_tuple->status == W_) && (out_tuple->status == W_)) {
    new_tuple->status = W_;
  } else if ((gen_tuple->status == W_) && (out_tuple->status == RW_)) {
    new_tuple->status = RW_;
  } else if ((gen_tuple->status == RW_) && (out_tuple->status == R_)) {
    new_tuple->status = RW_;
  } else if ((gen_tuple->status == RW_) && (out_tuple->status == W_)) {
    new_tuple->status = RW_;
  } else if ((gen_tuple->status == RW_) && (out_tuple->status == RW_)) {
    new_tuple->status = RW_;
  }
}

void DataFlow::operatorOutNewTuple(tuple *tuple1,
                                   tuple *tuple2,
                                   tuple *new_tuple) {
  /* Here is created the new Scope */
  if (tuple1->scope == cpu_ && tuple2->scope == cpu_)
    new_tuple->scope = cpu_;
  else if ((tuple1->scope == cpu_ && tuple2->scope == gpu_) ||
           (tuple1->scope == gpu_ && tuple2->scope == cpu_))
    new_tuple->scope = X_;
  else if ((tuple1->scope == cpu_ && tuple2->scope == X_) ||
           (tuple1->scope == X_ && tuple2->scope == cpu_))
    new_tuple->scope = X_;
  else if ((tuple1->scope == gpu_ && tuple2->scope == X_) ||
           (tuple1->scope == X_ && tuple2->scope == gpu_))
    new_tuple->scope = X_;
  else if (tuple1->scope == gpu_ && tuple2->scope == gpu_)
    new_tuple->scope = gpu_;

  /* Here is created the new Status */
  if (tuple1->status == R_ && tuple2->status == R_)
    new_tuple->status = R_;
  else if ((tuple1->status == R_ && tuple2->status == W_) ||
           (tuple1->status == W_ && tuple2->status == R_))
    new_tuple->status = RW_;
  else if ((tuple1->status == R_ && tuple2->status == RW_) ||
           (tuple1->status == RW_ && tuple2->status == R_))
    new_tuple->status = RW_;
  else if (tuple1->status == W_ && tuple2->status == W_)
    new_tuple->status = W_;
  else if ((tuple1->status == W_ && tuple2->status == RW_) ||
           (tuple1->status == RW_ && tuple2->status == W_))
    new_tuple->status = RW_;
  else if (tuple1->status == RW_ && tuple2->status == RW_)
    new_tuple->status = RW_;
}

bool DataFlow::checkIfEqual(DenseMap<const llvm::Value *, tuple> &new_set,
                            DenseMap<const llvm::Value *, tuple> &old_set,
                            std::vector<Value *> &domain) {

  for (std::vector<Value *>::iterator instr_domain = domain.begin();
       instr_domain != domain.end(); ++instr_domain) {
    Value *v = *instr_domain;

    if (new_set.count(v) == old_set.count(v)) {
      tuple tuple_new = new_set.lookup(v);
      tuple tuple_old = old_set.lookup(v);
      if ((tuple_new.status != tuple_old.status) ||
          (tuple_new.scope != tuple_old.scope))
        return false;
    } else {
      return false;
    }
  }
  return true;
}
