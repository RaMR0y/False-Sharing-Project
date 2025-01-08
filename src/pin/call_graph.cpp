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
 *  This file contains an ISA-portable PIN tool for generating a call graph
 */

#include<iostream>
#include<stdio.h>
#include<vector>
#include "pin.H"

using std::endl;
using std::cout;
using std::cerr;
using std::string;

#define MAX_FUNC_NAME 256

const unsigned MAX_FUNC_STACK = 1000;

const char* outfile_name = "cg.out";
const char* instrument_funcs_filename = "instrument_funcs.out";

struct func_info {
  string name;
  bool isCuda;
  int kernel_src_line;
};

struct cg_node {
  char* name;
  bool isCuda;
  int kernel_src_line;
};

std::vector<struct cg_node> cg;
int previous_index = -1;


int is_candidate(string func_name, std::vector<struct func_info>* candidate_funcs) {
  for (unsigned i = 0; i < candidate_funcs->size(); i++) {
    if (func_name == (*candidate_funcs)[i].name)
      return i;
  }
  return -1;
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
  while (fscanf(infile, "%s%d", func, &kernel_src_line) != EOF) {
    struct func_info this_func_info;
    std::string func_base_str(func);
    std::string func_str;
    if (name_has_kernel_prefix(func_base_str)) {
      this_func_info.isCuda = true;
      func_str = func_base_str.substr(10);
    }
    else {
      this_func_info.isCuda = false;
      func_str = func_base_str;
    }
    this_func_info.name = func_str;
    this_func_info.kernel_src_line = kernel_src_line;
    all_func_info.push_back(this_func_info);
  }
  return true;
}

bool containsFuncName(const char *real_name, string sym_name) {
  return (sym_name.find(real_name) != string::npos);
}

void PrintFuncName(struct cg_node *this_cg_node, int *index) {
  if (cg.size() < MAX_FUNC_STACK) 
    cg.push_back((*this_cg_node));
  return;
}

bool visited(char *name, std::vector<char*> funcs) {
  for (unsigned int i = 0; i < funcs.size(); i++) {
    if (!strcmp(name, funcs[i]))
      return true;
  }
  return false;
}

// Pin calls this function every time a new rtn is executed
VOID Routine(RTN rtn, VOID *candidate_funcs) {
    
  std::vector<struct func_info>* candidates = (std::vector<struct func_info>*) candidate_funcs;
  string func_name_str = PIN_UndecorateSymbolName(RTN_Name(rtn), UNDECORATION_NAME_ONLY);
  int index = is_candidate(func_name_str, candidates);
  int* index_ptr = (int *) malloc(sizeof(int));
  if (index >= 0) {
    (*index_ptr) = index;
    // get function name
    char *myname = (char *) malloc(sizeof(char) * MAX_FUNC_NAME);
    const char *func_name = func_name_str.c_str();
    strncpy(myname, func_name, func_name_str.size());
    myname[func_name_str.size()] = '\0';
    
    // fill up CG node 
    struct cg_node  *this_cg_node = (cg_node *) malloc(sizeof(struct cg_node));
    this_cg_node->name = myname; 
    this_cg_node->kernel_src_line = (*candidates)[index].kernel_src_line; 
    this_cg_node->isCuda = (*candidates)[index].isCuda;
    
    RTN_Open(rtn);
    INS ins = RTN_InsHead(rtn);
    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)PrintFuncName,
		   IARG_PTR, this_cg_node, IARG_PTR, index_ptr,
		   IARG_END);
    RTN_Close(rtn);
  }
}


VOID Fini(INT32 code, VOID *v) {
  FILE *outfile = fopen(outfile_name, "w");
  FILE *instrument_funcs_file = fopen(instrument_funcs_filename, "w");

  std::vector<char *> visited_funcs;

  bool kernel_launched = false;
  for (unsigned int i = 0; i < cg.size(); i++) {
    fprintf(outfile, "%s:%d\n", cg[i].name, cg[i].kernel_src_line);
    if (cg[i].isCuda)
      kernel_launched = true;
    if (!visited(cg[i].name, visited_funcs)) {
      if (kernel_launched || cg[i].kernel_src_line >= 0) {
	if (cg[i].isCuda)
	  fprintf(instrument_funcs_file, "__kernel__%s %d\n", cg[i].name, cg[i].kernel_src_line);
	else
	  fprintf(instrument_funcs_file, "%s %d\n", cg[i].name, cg[i].kernel_src_line);
	visited_funcs.push_back(cg[i].name);
      }
    }
  }
  fclose(outfile);
  fclose(instrument_funcs_file);
}

/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */
   
INT32 Usage() {
    PIN_ERROR( "This Pintool prints a call graph\n"
              + KNOB_BASE::StringKnobSummary() + "\n");
    return -1;
}

/* ===================================================================== */
/* Main                                                                  */
/* ===================================================================== */

int main(int argc, char *argv[]) {
    // initialize pin & symbol manager
    PIN_InitSymbols();
    if (PIN_Init(argc, argv)) return Usage();

    string infile_name = "functions.in";
    FILE* infile = fopen(infile_name.c_str(), "r");

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
    
    // register routine to be called to instrument rtn
    void *ptr_to_func_info = &all_func_info;
    RTN_AddInstrumentFunction(Routine, ptr_to_func_info);

    PIN_AddFiniFunction(Fini, 0);


    // Never returns
    PIN_StartProgram();
    
    return 0;
}
