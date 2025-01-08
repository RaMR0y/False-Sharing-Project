/*
 * Copyright 2002-2019 Intel Corporation.
 * 
 * This software is provided to you as Sample Source Code as defined in the accompanying
 * End User License Agreement for the Intel(R) Software Development Products ("Agreement")
 * section 1.L.
 * 
 * This software and the related documents are provided as is, with no express or implied
 * warranties, other than those that are expressly stated in the License.
 */

/*
 *  This file contains an ISA-portable PIN tool for tracing memory accesses.
 */

#include<iostream>
#include<stdio.h>
#include<vector>
#include "pin.H"

using std::endl;
using std::cout;
using std::cerr;
using std::string;


const string REF_INTERVAL_FUNC_NAME = "um_map::mark_cpu_ref_interval";

const unsigned int MAX_FUNC_NAME = 256;
const unsigned MAX_REFS = 1000;
unsigned ref_count = 0;

struct func_info {
  string name;
  int kernel_src_line;
  int kernel_id;
};

struct mem_trace_entry {
  unsigned long addr;
  char access;
  int interval;
};

struct kernel_call_entry {
  int kernel_id;
  int instance;
};

std::vector<struct kernel_call_entry> kernel_call_trace;
std::vector<struct mem_trace_entry> mem_trace;

unsigned *kernel_launch_counts;
int call_trace_size = -1;
bool start_cpu_profile = false;

FILE * trace;
FILE * kernel_trace_file;


bool isEqual(struct mem_trace_entry e1, struct mem_trace_entry e2) {
  /* ignore R/W */
  return (e1.addr == e2.addr && e1.interval == e2.interval);
}

void remove_duplicates() {
  int visited = 0;
  for (std::vector<struct mem_trace_entry>::iterator jiter = mem_trace.begin(); jiter != mem_trace.end(); jiter++) { 
    struct mem_trace_entry entry;
    entry.addr = (*jiter).addr;
    entry.interval = (*jiter).interval;
    for (std::vector<struct mem_trace_entry>::iterator iter = mem_trace.begin() + visited + 1;
	 iter != mem_trace.end(); iter++) { 
      if (isEqual(entry, *iter)) {
    	mem_trace.erase(iter);
      }
    }
    visited++;
  }
}

int is_candidate(string func_name, std::vector<struct func_info>* candidate_funcs) {
  for (unsigned i = 0; i < candidate_funcs->size(); i++) {
    if (func_name == (*candidate_funcs)[i].name)
      return i;
  }
  return -1;
}

bool containsFuncName(const char *real_name, string sym_name) {
  return (sym_name.find(real_name) != string::npos);
}


bool name_has_kernel_prefix(string name) {
  string kernel_prefix = "__kernel__";
  for (unsigned int i = 0; i < kernel_prefix.size(); i++) {
    if (name[i] != kernel_prefix[i])
      return false;
  }
  return true;
}

bool read_func_names(FILE* infile, std::vector<struct func_info>& all_func_info) {
  
  char *func = (char *) malloc(sizeof(char) * MAX_FUNC_NAME);
  int kernel_src_line;
  unsigned num_kernels = 0;
  while (fscanf(infile, "%s%d", func, &kernel_src_line) != EOF) {
    std::string func_str(func);
    struct func_info this_func_info;
    if (name_has_kernel_prefix(func_str)) {
      func_str = func_str.substr(10);
      this_func_info.kernel_id = kernel_src_line;
      num_kernels++;
    }
    else {
      this_func_info.kernel_id = -1;
    }
    this_func_info.name = func_str;
    this_func_info.kernel_src_line = kernel_src_line;
    all_func_info.push_back(this_func_info);
  }
  kernel_launch_counts = (unsigned *) malloc(sizeof(int) * num_kernels);
  for (unsigned i = 0; i < num_kernels; i++)
    kernel_launch_counts[i] = 0;
  return true;
}

VOID RecordMemRead(VOID * ip, VOID * addr) {
  //  if (ref_count < MAX_REFS) {
  //    ref_count++;
    if (start_cpu_profile) {
      struct mem_trace_entry entry;
      entry.addr = (unsigned long) addr;
      entry.access = 'R';
      entry.interval = call_trace_size;
      mem_trace.push_back(entry);
    }
    //  }
}

VOID RecordMemWrite(VOID * ip, VOID * addr) {
  //  if (ref_count < MAX_REFS) {
  //    ref_count++;
    if (start_cpu_profile) {
      struct mem_trace_entry entry;
      entry.addr = (unsigned long) addr;
      entry.access = 'W';
      entry.interval = call_trace_size;
      mem_trace.push_back(entry);
    }
    //  }
}

void TrackKernelLaunch(VOID *kernel) {
  int kernel_id = *((int *) kernel);
  struct kernel_call_entry entry;
  entry.kernel_id = kernel_id;
  entry.instance = kernel_launch_counts[kernel_id];
  kernel_call_trace.push_back(entry);
  kernel_launch_counts[kernel_id]++;
  return;
}

void TrackCPURefInterval(VOID *dummy) {
  dummy = 0;
  start_cpu_profile = true;
  call_trace_size++;
  return;
}

VOID Routine(RTN rtn, VOID *candidate_funcs) {
    
  string func_name;
  func_name = PIN_UndecorateSymbolName(RTN_Name(rtn), UNDECORATION_NAME_ONLY); 
  if (func_name == REF_INTERVAL_FUNC_NAME) {
    RTN_Open(rtn);
    INS ins = RTN_InsHead(rtn);
    int *dummy_ptr = (int *) malloc(sizeof(int));
    (*dummy_ptr) = 0;
    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR) TrackCPURefInterval,
		   IARG_PTR, dummy_ptr, IARG_END);
    RTN_Close(rtn);
  }
  int index = is_candidate(func_name, (std::vector<struct func_info>*) candidate_funcs);
  if (index >= 0) {
    int kernel_id = (* ((std::vector<struct func_info>*) candidate_funcs))[index].kernel_id;
    int target_src_line = (* ((std::vector<struct func_info>*) candidate_funcs))[index].kernel_src_line;
    
    RTN_Open(rtn);
    
    if (kernel_id >= 0) {
      
      /* we have encountered a GPU kernel call */
      INS ins = RTN_InsHead(rtn);
      int *kernel_id_ptr = (int *) malloc(sizeof(int));
      (*kernel_id_ptr) = kernel_id;
      INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)TrackKernelLaunch,
		     IARG_PTR, kernel_id_ptr, IARG_END);
    }
    else {
      for (INS ins = RTN_InsHead(rtn); INS_Valid(ins); ins = INS_Next(ins)) {
     	string filename;    // source file name.
    	INT32 line = 0;     // line number within the file.
     	PIN_GetSourceLocation(INS_Address(ins), NULL, &line, &filename);
     	if (line > target_src_line) { 
	  UINT32 memOperands = INS_MemoryOperandCount(ins);
          // iterate over each memory operand of the instruction.
          for (UINT32 memOp = 0; memOp < memOperands; memOp++) {
    	    if (INS_MemoryOperandIsRead(ins, memOp)) {
    	      INS_InsertPredicatedCall(
    				       ins, IPOINT_BEFORE, (AFUNPTR)RecordMemRead,
    				       IARG_INST_PTR,
    				       IARG_MEMORYOP_EA, memOp,
    				       IARG_END);
    	    }
    	    if (INS_MemoryOperandIsWritten(ins, memOp)) {
    	      INS_InsertPredicatedCall(
    	  			       ins, IPOINT_BEFORE, (AFUNPTR)RecordMemWrite,
    	  			       IARG_INST_PTR,
    	  			       IARG_MEMORYOP_EA, memOp,
    	  			       IARG_END);
    	    }
	  }
	}
      }
    }
    RTN_Close(rtn);
  }
}


VOID Fini(INT32 code, VOID *v) {
  //  remove_duplicates();
  for (unsigned i = 0; i < mem_trace.size(); i++)
    fprintf(trace, "%p %c %d\n", (void *) mem_trace[i].addr, mem_trace[i].access, mem_trace[i].interval);
  fclose(trace);

  for (unsigned i = 0; i < kernel_call_trace.size(); i++)
    fprintf(kernel_trace_file, "%d %d\n", kernel_call_trace[i].kernel_id, kernel_call_trace[i].instance);
  fclose(kernel_trace_file);
}

INT32 Usage() {
    PIN_ERROR( "This Pintool prints a trace of Managed Memory Pages accessed by CPU\n" 
              + KNOB_BASE::StringKnobSummary() + "\n");
    return -1;
}

int main(int argc, char *argv[]) {
    // Initialize pin & symbol manager
    PIN_InitSymbols();
    if (PIN_Init(argc, argv)) return Usage();

     // TODO: should come from command-line 
    string infile_name = "instrument_funcs.out";
    FILE* infile = fopen(infile_name.c_str(), "r");

    // gather information about functions to be instrumented 
    std::vector<struct func_info> all_func_info;
    if (infile) {
      if (!read_func_names(infile, all_func_info)) {
    	fprintf(stderr, "error reading input file %s. Exiting ..\n", infile_name.c_str());
    	exit(0);
      }
    }
    else {
    	fprintf(stderr, "could not open input file %s\n", infile_name.c_str());
    	exit(0);
    }
    fclose(infile);
    
    trace = fopen("cpu_mem_trace.out", "w");
    kernel_trace_file = fopen("kernel_trace.out", "w");

    // Register Routine to be called to instrument rtn
    void *ptr_to_func_info = &all_func_info;
    RTN_AddInstrumentFunction(Routine, ptr_to_func_info);

    PIN_AddFiniFunction(Fini, 0);

    // Never returns
    PIN_StartProgram();
    
    return 0;
}
