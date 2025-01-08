#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<umap.h>

#include<iostream>
#include<fstream>
#include<string>

using namespace std;

class mem_access {

public:
  mem_access();
  ~mem_access();
  void set_address(ADDR  _addr) { addr = _addr; }
  void set_access(char _access_type) { access_type = _access_type; }
  void set_interval(int _interval) { interval = _interval; }
  ADDR  get_address() { return addr; }
  char get_access() { return access_type; }
  int get_interval() { return interval; }
  void dump();

private:
  ADDR  addr;
  char access_type;
  int interval;
};

mem_access::mem_access() {
  addr = NULL_ADDR;
  access_type = 'z';
  interval = -1;
}

mem_access::~mem_access() {}

void mem_access::dump() {
  printf("%p\t%c\t%d\n", addr, access_type, interval);
}

class mem_trace {

public:
  mem_trace();
  mem_trace(unsigned);
  ~mem_trace();

  unsigned long get_size() { return trace.size(); } 
  unsigned long get_size_for_ref_group(unsigned long i) { return trace[i].size(); }
  unsigned long get_total_refs();
  unsigned long get_total_refs_in_live_range(int i, int j);

  ADDR  get_address(unsigned long i, unsigned long j) { return trace[i][j].get_address(); }
  mem_access get_ref(unsigned long i, unsigned long j) { return trace[i][j]; }
  int get_interval(unsigned long i, unsigned long j) { return trace[i][j].get_interval(); }

  void add(mem_access mem);
  void dump(); 

private:
  vector<vector<mem_access>> trace;
  int ref_groups;
};

mem_trace::mem_trace() {}

mem_trace::mem_trace(unsigned _ref_groups) {
  ref_groups = _ref_groups;
  trace.resize(ref_groups);
}

mem_trace::~mem_trace() {
  trace.clear();
}

void mem_trace::add(mem_access mem) {
  trace[mem.get_interval()].push_back(mem);
  return;
}

unsigned long mem_trace::get_total_refs_in_live_range(int begin, int end) {
  if (begin == UNDEFINED && end == UNDEFINED)
    return 0;
  unsigned long total_refs = 0;
  end = (end == END_OF_PROG ? trace.size() : end);
  for (unsigned i = begin; i < end; i++) 
    total_refs = total_refs + trace[i].size();
  return total_refs;
}

unsigned long mem_trace::get_total_refs() {
  unsigned long total_refs = 0;
  for (unsigned i = 0; i < trace.size(); i++) 
    total_refs = total_refs + trace[i].size();
  return total_refs;
}

void mem_trace::dump() {
  for (unsigned i = 0; i < trace.size(); i++) {
    printf("Interval %d: No. of refs %d\n", i, trace[i].size());
    for (unsigned j = 0; j < trace[i].size(); j++) {
      trace[i][j].dump();
    }
  }
}

void dump_call_trace(std::vector<kernel_call> &ctrace) {
  for (unsigned i = 0; i < ctrace.size(); i++) {
    cout << ctrace[i].get_id() << ":" << ctrace[i].get_instance() << endl;
    for (unsigned j = 0; j < ctrace[i].spaces(); j++) {
      fprintf(stdout, "%d: %p %p\n",  ctrace[i].get_access_record_space(j),
	      ctrace[i].get_access_record_begin(j),
	      ctrace[i].get_access_record_end(j));
    }
  }
  return;
}
	
void read_call_trace(string ctrace_file, std::vector<kernel_call> &ctrace) {
  ifstream input;
  input.open(ctrace_file.c_str(), ios::in);
  if (!input) {
    cerr << "Error opening file, exiting ..." << endl;
    exit(0);
  }

  int id;
  int instance;

  while (input >> id) {
    if (!(input >> instance)) {
      cerr << "Error reading file: R/W access";
      exit(0);
    }  
    kernel_call entry;
    entry.set_id(id);
    entry.set_instance(instance);
    ctrace.push_back(entry);
  }
  input.close();
  return;
}

void read_mem_trace(string trace_file, mem_trace *trace) {
  ifstream input;
  input.open(trace_file.c_str(), ios::in);
  if (!input) {
    cerr << "Error opening file, exiting ...";
    exit(0);
  }
  
  mem_access mem;
  void *addr;
  char access_type;
  int interval;
  
  int i = 0;
  while (input >> addr) {
    if (!(input >> access_type)) {
      cerr << "Error reading file: R/W access";
      exit(0);
    }  
    if (!(input >> interval)) {
      cerr << "Error reading file: kernel ID";
      exit(0);
    }  
    mem.set_address((ADDR) addr);
    mem.set_access(access_type);
    mem.set_interval(interval);
    trace->add(mem);
  }
  input.close();
  return;
}

um_map read_umap(string um_map_file) {

  ifstream input;
  input.open(um_map_file.c_str(), ios::in);
  if (!input) {
    cerr << "Error opening file, exiting ..." << endl;
    exit(0);
  }

  um_map um;
  void *addr;

  /* read begin and end address for all managed memory */
  if (!(input >> addr)) {
    cerr << "Error reading file contents: managed memory begin addr" << endl;
    exit(0);
  }  
  um.set_global_begin((unsigned long) addr);

  if (!(input >> addr)) {
    cerr << "Error reading file contents: managed memory end addr" << endl;
    exit(0);
  }  
  um.set_global_end((unsigned long) addr);

  /* number of spaces in this map */
  unsigned spaces;
  if (!(input >> spaces)) {
    cerr << "Error reading file contents: number of spaces in UM" << endl;
    exit(0);
  }  

  for (unsigned int i = 0; i < spaces; i++) {
    um_space this_space(i);
    /* begin and end address for this space  */
    if (!(input >> addr)) {
      cerr << "Error reading begin addr for space " << i << endl;
      exit(0);
    }  
    this_space.set_begin((unsigned long) addr);

    if (!(input >> addr)) {
      cerr << "Error reading end addr for space " << i << endl;
      exit(0);
    }  
    this_space.set_end((unsigned long) addr);

    /* page range for this space */
    unsigned page_index;
    if (!(input >> page_index)) {
      cerr << "Error reading begin page index for space " << i << endl;
      exit(0);
    }
    this_space.set_begin_page_index(page_index);
    if (!(input >> page_index)) {
      cerr << "Error reading end page index for space " << i << endl;
      exit(0);
    }  
    this_space.set_end_page_index(page_index);

    /* number of kernels instances that touch this space */
    unsigned kernels;
    if (!(input >> kernels)) {
      cerr << "Error reading number of kernels for space " << i << endl;
      exit(0);
    }  

    /* collect access record for each kernel instance */
    for (unsigned j = 0; j < kernels; j++) { 
      /* number of times kernel is launched */
      unsigned kernel_id;
      unsigned kernel_instance;
      if (!(input >> kernel_id >> kernel_instance)) {
	cerr << "Error reading kernel instance for space" << i << endl;
	exit(0);
      }  
      
      void *begin_addr, *end_addr;
      if (!(input >> begin_addr)) {
	cerr << "Error reading GPU begin addr" << endl;
	exit(0);
      }  
      if (!(input >> end_addr)) {
	cerr << "Error reading GPU end addr" << endl;
	exit(0);
      }  
      this_space.set_gpu_access_record(kernel_id, kernel_instance, (unsigned long) begin_addr, (unsigned long) end_addr);
    }
    um.add_space(this_space);
  }
  
  input.close();
  return um;
}

bool ref_in_page(ADDR ref, ADDR page) {
  return ((ref >= page) && (ref <= (page + BASE_PAGE_END_OFFSET)));
}

bool get_cpu_access_range_in_page(mem_trace &cpu_refs, int begin, int end,
				  ADDR &lowest, ADDR &highest) {

  unsigned long lowest_index = 0;
  unsigned long highest_index = 0;
  ADDR  lowest_addr;
  ADDR  highest_addr;
  unsigned long i;
  bool found_ref_in_range = false;

  end = end == END_OF_PROG ? cpu_refs.get_size() : end;
  for (i = begin; i < end; i++) {
    if (cpu_refs.get_size_for_ref_group(i) > 0) { 
      lowest_addr = cpu_refs.get_address(i,0);
      highest_addr = cpu_refs.get_address(i,0);
      found_ref_in_range = true;
      break;
    }
  }
  if (!found_ref_in_range) {
    lowest = NULL_ADDR;
    highest = NULL_ADDR;
    return false;
  }

  for (; i < end; i++) {
    for (unsigned long j = 0; j < cpu_refs.get_size_for_ref_group(i); j++) {
      if (cpu_refs.get_address(i,j) < lowest_addr) {
	lowest_index = j;
	lowest_addr = cpu_refs.get_address(i, j);
      }
      if (cpu_refs.get_address(i, j) > highest_addr) {
	highest_index = j;
	highest_addr = cpu_refs.get_address(i, j);
      }
    }
  }

  lowest = lowest_addr;

  // address is begining position of data item; want the end address of the last data item accessed
  // incrementing by void size (4 bytes);
  // TODO: increment should be data type specific
  highest_addr = highest_addr + (SIZEOF_DATA_TYPE - 1);     
  highest = highest_addr;

  return true;
}

/* 
 * Filter out CPU references that land on a specific page 
 *
 */
void filter_cpu_refs_by_page_access(std::vector<fs_page> &candidates, ADDR page_addr, mem_trace *trace, mem_trace *cpu_refs) {
  for (unsigned i = 0; i < trace->get_size(); i++) {
    for (unsigned j = 0; j < trace->get_size_for_ref_group(i); j++) {
      ADDR ref = trace->get_address(i, j);
      if (ref_in_page(ref, page_addr)) {
	cpu_refs->add(trace->get_ref(i, j));
      }
    }
  }
  ADDR lowest, highest;
  if (get_cpu_access_range_in_page(*cpu_refs, 0, cpu_refs->get_size(), lowest, highest)) {
    for (unsigned i = 0; i < candidates.size(); i++) {
      candidates[i].set_cpu_begin(lowest - page_addr);
      candidates[i].set_cpu_end(highest - page_addr);
    }
  }
}



void check_for_false_sharing(mem_trace &cpu_refs, fs_page &candidate,
			     std::vector<int> &occurrence_pre,
			     std::vector<int> &occurrence_post,
			     std::vector<fs_type> &fs_pre, 
			     std::vector<fs_type> &fs_post) {

  ADDR lowest, highest;

  /* members of group have the same base addr; using candidate 0 to represent */
  unsigned long base_addr = candidate.get_base_addr();
  int kernel_instance = candidate.get_kernel();
  int global_index = candidate.get_index();
  ADDR lowest_gpu_offset, highest_gpu_offset;

  bool is_fs = true;

  int n = candidate.get_num_post_live_range();
  
  for (unsigned i = 0; i < n; i++) {
    int live_begin = candidate.get_post_live_range_begin(i);
    int live_end = candidate.get_post_live_range_end(i);
    
    int occurrence = cpu_refs.get_total_refs_in_live_range(live_begin, live_end);
    occurrence_post.push_back(occurrence);
    if (get_cpu_access_range_in_page(cpu_refs, live_begin, live_end, lowest, highest)) {

      candidate.set_post_live_range_cpu_begin(i, lowest - base_addr);
      candidate.set_post_live_range_cpu_end(i, highest - base_addr);

      if (!candidate.overlap_in_post_range(i)) {
	fs_post.push_back(MALICIOUS);
      }
      else
	fs_post.push_back(TRUE);
    }
    else
      fs_post.push_back(NONE);
      
  }

  n = candidate.get_num_pre_live_range();
  for (unsigned i = 0; i < n; i++) {

    int live_begin = candidate.get_pre_live_range_begin(i);
    int live_end = candidate.get_pre_live_range_end(i);
    int occurrence = cpu_refs.get_total_refs_in_live_range(live_begin, live_end);
    occurrence_pre.push_back(occurrence);
  
    if (get_cpu_access_range_in_page(cpu_refs, live_begin, live_end, lowest, highest)) {

      candidate.set_pre_live_range_cpu_begin(i, lowest - base_addr);
      candidate.set_pre_live_range_cpu_end(i, highest - base_addr);

      if (!candidate.overlap_in_pre_range(i)) {
	if (kernel_instance == candidate.get_first_touch())
	  fs_pre.push_back(BENIGN);
	else
	  fs_pre.push_back(MALICIOUS);
      }
      else
	fs_pre.push_back(TRUE);
    }
    else
      fs_pre.push_back(NONE);
  }
}

bool index_visited(std::vector<unsigned> indices, int this_index) {

  for (std::vector<unsigned>::iterator i = indices.begin(); i != indices.end(); i++) {
    if (*i == this_index)
      return true;
  }
  return false;
}

int main() {

  um_map um;
  um = read_umap("umap.out");

 std::vector <kernel_call> kernel_trace;
  read_call_trace("kernel_trace.out", kernel_trace);

  unsigned ref_groups = kernel_trace.size();
  mem_trace trace(ref_groups);

  read_mem_trace("cpu_mem_trace.out", &trace);

  um.calculate_padding();
  um.merge_access_record_to_kernel_trace(kernel_trace);
  um.mark_fs_candidates(kernel_trace);

  unsigned mal_fs_count = 0;
  unsigned ben_fs_count = 0;
  unsigned true_sharing_count = 0;
  unsigned mal_fs_occurrence_count = 0;
  unsigned ben_fs_occurrence_count = 0;
  unsigned true_sharing_occurrence_count = 0;
  unsigned all_cpu_refs_to_candidates = 0;

  /* candidates are the set of all pages in the managed space with partial access by GPU */
  std::vector<std::vector<fs_page>> candidates = um.get_candidates();

  std::vector<unsigned> page_indices;
  
  bool verbose = true;
  for (unsigned int i = 0; i < candidates.size(); i++) {
    int master = candidates[i].size() - 1; 
    unsigned global_page_index = candidates[i][master].get_index();
    mem_trace cpu_refs(ref_groups);
    printf("Candidate %d\n", i);
    printf("\033[1;37m[KERNEL = %d\033[0m,", candidates[i][master].get_kernel());
    printf("\033[1;37m PAGE = %u]\033[0m \n", global_page_index);

    if (! index_visited(page_indices, global_page_index)) {
      page_indices.push_back(global_page_index);
    }
    
    /* base addr and page index for a group are the same; using entry 0 to represent */
    filter_cpu_refs_by_page_access(candidates[i], candidates[i][0].get_base_addr(), &trace, &cpu_refs);

    /* last entry in the group is the master candidate */
    if (cpu_refs.get_total_refs() > 0) {
      std::vector<int> occurrence_pre;
      std::vector<int> occurrence_post;

      std::vector<fs_type> fs_pre;
      std::vector<fs_type> fs_post;

      check_for_false_sharing(cpu_refs, candidates[i][master], occurrence_pre, occurrence_post, fs_pre, fs_post);
      unsigned this_cpu_refs = cpu_refs.get_total_refs();
      all_cpu_refs_to_candidates += this_cpu_refs;
      for (unsigned j = 0; j < fs_pre.size(); j++) {
	candidates[i][master].dump_fs_info(j, fs_pre[j], occurrence_pre[j], true, verbose);
	if (fs_pre[j] == MALICIOUS) {
	  mal_fs_count++;
	  mal_fs_occurrence_count += occurrence_pre[j];
	}
	if (fs_pre[j] == BENIGN) {
	  ben_fs_count++;
	  ben_fs_occurrence_count += occurrence_pre[j];
	}
	if (fs_pre[j] == TRUE) {
	  true_sharing_count++;
	  true_sharing_occurrence_count += occurrence_pre[j];
	}
      }
      for (unsigned j = 0; j < fs_post.size(); j++) {
	candidates[i][master].dump_fs_info(j, fs_post[j], occurrence_post[j], false, verbose);
	if (fs_post[j] == MALICIOUS) {
	  mal_fs_count++;
	  mal_fs_occurrence_count += occurrence_post[j];
	}
	if (fs_post[j] == BENIGN) {
	  ben_fs_count++;
	  ben_fs_occurrence_count += occurrence_post[j];
	}
	if (fs_post[j] == TRUE) {
	  true_sharing_count++;
	  true_sharing_occurrence_count += occurrence_post[j];
	}
      }
      printf("\n");
    }
  }
  printf("\n\033[1;36m===================== [Summary] =====================\n\033[0m");
  if (mal_fs_count == 0 && ben_fs_count == 0)
    printf("Yay, No False Sharing\n\n");

  printf("Number of allocation to managed memory:\t\t%d\n", um.get_space_count()); 
  printf("Number of page in allocated in managed memory:\t%d\n", um.get_page_count()); 
  printf("False sharing candidates: \t\t\t%d\n", candidates.size()); 
  printf("Number of distinct candidate pages:\t\t%d\n", page_indices.size()); 
  printf("Number of CPU references to candidate pages: \t%d\n", all_cpu_refs_to_candidates);
  printf("Number of malicious false sharing cases: \t%d\n", mal_fs_count);
  printf("Number of malicious false sharing occurrences: \t%d\n", mal_fs_occurrence_count);
  printf("Number of benign false sharing cases: \t\t%d\n", ben_fs_count);
  printf("Number of benign false sharing occurrences: \t%d\n", ben_fs_occurrence_count);
  printf("Number of true sharing cases: \t\t\t%d\n", true_sharing_count);
  printf("Number of true sharing occurrences: \t\t%d\n", true_sharing_occurrence_count);
  
#ifdef DEBUG  
  um.print_offsets();
#endif
  return 0;
}
