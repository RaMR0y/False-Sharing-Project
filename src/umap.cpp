#include<umap.h>
#include<stdio.h>
#include<iostream>

using namespace std;

um_page::um_page() {
  gpu_begin_addr = INVALID_OFFSET;
  gpu_end_addr = INVALID_OFFSET; 
  cpu_begin_addr = INVALID_OFFSET; 
  cpu_end_addr = INVALID_OFFSET;   
  base_addr = 0;
}

um_page::um_page(ADDR addr) {
  gpu_begin_addr = INVALID_OFFSET;
  gpu_end_addr = INVALID_OFFSET; 
  cpu_begin_addr = INVALID_OFFSET; 
  cpu_end_addr = INVALID_OFFSET;   
  base_addr = addr;
}

um_page::~um_page() {
  pre_live_range.clear();
  post_live_range.clear();
}

void um_page::set_pre_live_range(int begin, int end, int addr_range_begin, int addr_range_end) {
  struct fs_live_range this_live_range;
  this_live_range.begin = begin;
  this_live_range.end = end;
  this_live_range.addr_range_begin = addr_range_begin;
  this_live_range.addr_range_end = addr_range_end;
  
  pre_live_range.push_back(this_live_range);
}

void um_page::set_post_live_range(int begin, int end, int addr_range_begin, int addr_range_end) {
  struct fs_live_range this_live_range;
  this_live_range.begin = begin;
  this_live_range.end = end;
  this_live_range.addr_range_begin = addr_range_begin;
  this_live_range.addr_range_end = addr_range_end;
  post_live_range.push_back(this_live_range);
}

void um_page::set_pre_live_range_addr_range(unsigned begin, unsigned end) {
  pre_live_range[0].addr_range_begin = begin;
  pre_live_range[0].addr_range_end = end;
}

void um_page::set_post_live_range_addr_range(unsigned begin, unsigned end) {
  post_live_range[0].addr_range_begin = begin;
  post_live_range[0].addr_range_end = end;
}

void um_page::dump(bool verbose) {
  if (verbose) {
    printf("base address: %p\n", (void *) base_addr);
    printf("UM page index: %u\n", index);

    if (gpu_begin_addr != INVALID_OFFSET)
      printf("gpu_begin_addr: %u\n", gpu_begin_addr);
    else
      printf("gpu_begin_addr: none\n");

    if (gpu_end_addr != INVALID_OFFSET)
      printf("gpu_end_addr: %u\n", gpu_end_addr);
    else
      printf("gpu_end_addr: none\n");

    if (cpu_begin_addr != INVALID_OFFSET)
      printf("cpu_begin_addr: %u\n", cpu_begin_addr);
    else
      printf("cpu_begin_addr: none\n");

    if (cpu_end_addr != INVALID_OFFSET)
      printf("cpu_end_addr: %u\n", cpu_end_addr);
    else
      printf("cpu_end_addr: none\n");
    for (unsigned i = 0; i < get_num_pre_live_range(); i++) {
      if (pre_live_range[i].end == END_OF_PROG) 
	printf("pre live range: %d -> EOF\n", pre_live_range[i].begin);
      else
	printf("pre live range: %d -> %d\n", pre_live_range[i].begin, pre_live_range[i].end);
    }
    for (unsigned i = 0; i < get_num_post_live_range(); i++) {
      if (post_live_range[i].end == END_OF_PROG) 
	printf("post live range: %d -> EOF\n", post_live_range[i].begin);
      else
	printf("post live range: %d -> %d\n", post_live_range[i].begin, post_live_range[i].end);
    }
  }
  else {
    printf("\033[1;37m PAGE = %u]\033[0m   ", index);

    if (gpu_begin_addr != INVALID_OFFSET)
      printf("(%u,", gpu_begin_addr);
    else
      printf("(none,");

    if (gpu_end_addr != INVALID_OFFSET)
      printf("%u),", gpu_end_addr);
    else
      printf("none),");

    if (cpu_begin_addr != INVALID_OFFSET)
      printf("(%u,", cpu_begin_addr);
    else
      printf("(none,");

    if (cpu_end_addr != INVALID_OFFSET)
      printf("%u),", cpu_end_addr);
    else
      printf("none),");

    unsigned n = get_num_pre_live_range();
    for (unsigned i = 0; i < n; i++) {
      if (pre_live_range[i].end == END_OF_PROG) 
	printf("[%d:EOF],", pre_live_range[i].begin);
      else 
	printf("[%d:%d],", pre_live_range[i].begin, pre_live_range[i].end);
    }
    n = get_num_post_live_range();
    for (unsigned i = 0; i < n; i++) {
      if (post_live_range[i].end == END_OF_PROG) 
	printf("[%d:EOF]", post_live_range[i].begin);
      else
	printf("[%d:%d]", post_live_range[i].begin, post_live_range[i].end);
      if (i == n - 1) 
	printf("\n");
      else 
	printf(", ");
    }
  }
}

void fs_page::dump_fs_info(int live_range, enum fs_type fs, unsigned count, bool pre, bool verbose) {

  int live_begin,live_end;
  int live_addr_range_begin, live_addr_range_end;
  int cpu_begin, cpu_end;

  if (pre) {
    live_begin = get_pre_live_range_begin(live_range);
    live_end = get_pre_live_range_end(live_range);
    live_addr_range_begin = get_pre_live_range_addr_range_begin(live_range);
    live_addr_range_end = get_pre_live_range_addr_range_end(live_range);
    cpu_begin = get_pre_live_range_cpu_begin(live_range);
    cpu_end = get_pre_live_range_cpu_end(live_range);
  }
  else {
    live_begin = get_post_live_range_begin(live_range);
    live_end = get_post_live_range_end(live_range);
    live_addr_range_begin = get_post_live_range_addr_range_begin(live_range);
    live_addr_range_end = get_post_live_range_addr_range_end(live_range);
    cpu_begin = get_post_live_range_cpu_begin(live_range);
    cpu_end = get_post_live_range_cpu_end(live_range);
  }
  if (fs == MALICIOUS) 
    printf("\033[1;31m*** MALICIOUS FALSE SHARING ***\033[0m\n");
  if (fs == BENIGN) 
    printf("\033[1;33m*** BENIGN FALSE SHARING ***\033[0m\n");
  if (fs == TRUE) 
    printf("\033[1;32m*** TRUE SHARING ***\033[0m\n");
  if (fs == NONE) 
    printf("NONE\n");
  if (fs != NONE) {
    printf("Live range and access: ");
    if (live_end == END_OF_PROG) 
      printf("[%d:EOF], (%d:%d) (%d:%d)\n", live_begin, live_addr_range_begin,
	     live_addr_range_end, cpu_begin, cpu_end);
    else 
      printf("[%d:%d], (%d:%d) (%d:%d)\n", live_begin, live_end,
	     live_addr_range_begin, live_addr_range_end, cpu_begin, cpu_end);
  }    
  printf("Number of occurrences: %d\n", count);
  return;
}

void fs_page::dump(bool verbose) {
  printf("\033[1;37m[KERNEL = %d\033[0m,", kernel);
  um_page::dump(verbose);
}

bool fs_page::overlap_in_post_range(unsigned i) {

  if (get_post_live_range_cpu_begin(i) > get_post_live_range_addr_range_end(i))
    return false;
  if (get_post_live_range_cpu_end(i) < get_post_live_range_addr_range_begin(i))
    return false;

  return true;
}

bool fs_page::overlap_in_pre_range(unsigned i) {

  if (get_pre_live_range_cpu_begin(i) > get_pre_live_range_addr_range_end(i))
    return false;
  if (get_pre_live_range_cpu_end(i) < get_pre_live_range_addr_range_begin(i))
    return false;

  return true;
}

fs_page::fs_page() {
  kernel = UNDEFINED;
}

#if 0
fs_page::fs_page(fs_page &other_page) {
  set_gpu_begin(other_page.get_gpu_begin());
  set_gpu_end(other_page.get_gpu_end());
  set_cpu_begin(other_page.get_cpu_begin());
  set_cpu_end(other_page.get_cpu_end());
  set_base_addr(other_page.get_base_addr());
  set_pre_live_range(other_page.get_pre_live_range_begin(), other_page.get_pre_live_range_end());
  set_post_live_range(other_page.get_post_live_range_begin(), other_page.get_post_live_range_end());
  kernel = other_page.get_kernel();
}
#endif 

fs_page::~fs_page() {}

void fs_page::set_kernel(int _kernel) {
  kernel = _kernel;
  return;
}

int fs_page::get_kernel() {
  return kernel;
}


void um_map::add_candidate(fs_page &page) {
  int kernel = page.get_kernel();
  unsigned global_index = page.get_index();
  int candidate_index = UNDEFINED;
  for (unsigned i = 0; i < candidates.size(); i++) {
    if (candidates[i].size() > 0) {
      if (candidates[i][0].get_kernel() == kernel &&
	  candidates[i][0].get_index() == global_index) {
	candidate_index = i;
	break;
      }
    }
    else {
      fprintf(stderr, "Fatal Error: candidates vector not populated correctly\n");
      return;
    }
  }

  if (candidate_index < 0) { // need a new entry
    std::vector<fs_page> candidates_for_this_index;
    candidates_for_this_index.push_back(page);
    candidates.push_back(candidates_for_this_index);
  }
  else
    candidates[candidate_index].push_back(page);
  return;
}

um_space::um_space(unsigned _id) {
  id = _id;
  padding = 0;
}

um_space::~um_space() {}

ADDR um_space::get_access_record_addr(unsigned i, access_type access) {
  if (access == GPU_BEGIN) 
   return access_record[i].gpu_begin_addr; 
  if (access == GPU_END) 
   return access_record[i].gpu_end_addr; 
  fprintf(stderr, "Fatal Error: request for address of invalid access type. Exiting\n");
  exit(0);
}

void um_space::set_gpu_access_record(unsigned kernel, unsigned instance,
				     ADDR begin_addr, ADDR end_addr) {
  struct gpu_access_record this_access_record;
  this_access_record.gpu_begin_addr = begin_addr;
  this_access_record.gpu_end_addr = end_addr;
  this_access_record.kernel_id = kernel;
  this_access_record.kernel_instance = instance;
  
  access_record.push_back(this_access_record);
}


bool um_space::is_aligned_begin(ADDR addr) {
  return !((addr - begin_addr) % BASE_PAGE_SIZE);
}


bool um_space::access_record_matches_kernel(struct gpu_access_record access_record, kernel_call &kernel) {
  return (access_record.kernel_id == kernel.get_id() && 
	  access_record.kernel_instance == kernel.get_instance());
}

void um_space::get_access_range_for_kernel(kernel_call &kernel) {
  for (unsigned i = 0; i < access_record.size(); i++) {
    if (access_record_matches_kernel(access_record[i], kernel)) {
      struct access_range kernel_access_range;
      kernel_access_range.begin = access_record[i].gpu_begin_addr;
      kernel_access_range.end = access_record[i].gpu_end_addr;
      kernel_access_range.space = get_id();
      kernel.add_access_range(kernel_access_range);
      return;
    }
  }
  return;
}

void um_space::dump(FILE *outfile, bool verbose) {

  if (verbose) {
    fprintf(outfile, "[ALLOC:%d] Begin address: %p\n", id, (void *) get_begin());
    fprintf(outfile, "[ALLOC:%d] End address: %p\n", id, (void *) get_end());
    fprintf(outfile, "[ALLOC:%d] size: %lu bytes\n", id, get_size());
    fprintf(outfile, "[ALLOC:%d] number of pages: %u\n", id, get_page_count());
    fprintf(outfile, "[ALLOC:%d] begin page index: %u\n", id, get_begin_page_index());
    fprintf(outfile, "[ALLOC:%d] end page index: %u\n", id, get_end_page_index());
    fprintf(outfile, "[ALLOC:%d] padding: %u\n", id, get_padding());
  }
  else {
    /* address range for entire space */ 
    fprintf(outfile, "%p %p\n", (void *) get_begin(), (void *) get_end());
    fprintf(outfile, "%u %u\n", get_begin_page_index(), get_end_page_index());

    /* number of kernel instances in which this space is touched */
    fprintf(outfile, "%lu\n", access_record.size());

    for (unsigned i = 0; i < access_record.size(); i++) {
      ADDR gpu_begin = get_access_record_addr(i, GPU_BEGIN);
      ADDR gpu_end = get_access_record_addr(i, GPU_END);
      int kernel = get_access_record_kernel(i);
      int instance = get_access_record_instance(i);
      fprintf(outfile, "%d %d %p %p\n", kernel, instance,
	      (void *) gpu_begin, (void *) gpu_end);
    }
  }
}


um_map::um_map() {
  global_begin_addr = 0;
  global_end_addr = 0;
  interval_start = false;
}

um_map::~um_map() {
  pages.clear();
  spaces.clear();
}

unsigned um_map::get_global_page_offset() {
  if (pages.size() == 0)
    return 0;
  else 
    return pages.back().get_gpu_end();
} 

bool um_map::last_page_filled() {
  if (pages.size() == 0)
    return true;
  else 
    return (get_global_page_offset() == (BASE_PAGE_SIZE - 1));
}

bool um_map::spill_over_to_new_page(unsigned offset) {
  if (offset == 0)
    return false;
  if (last_page_filled())
    return true;
  if (offset + get_global_page_offset() > BASE_PAGE_SIZE - 1)
    return true;
  else
    return false;
}

bool um_map::page_is_marked(std::vector<um_page> &pages, unsigned page_index) {
  for (unsigned i = 0; i < pages.size(); i++) {
    if (page_index == get_page_index(pages[i].get_base_addr()))
      return true;
  }
  return false;
}

void um_map::dump(bool verbose) {

  FILE *outfile;
  outfile = fopen("umap.out", "w");
  if (verbose) {
    /* global properties */  
    fprintf(outfile, "[UM] begin address: %p\n", (void *) global_begin_addr);
    fprintf(outfile, "[UM] end address: %p\n", (void *) global_end_addr);
    fprintf(outfile, "[UM] size: %lu bytes\n", get_global_size());
    fprintf(outfile, "[UM] number of pages: %u\n", get_page_count());
    fprintf(outfile, "[UM] number of spaces: %lu\n", spaces.size());
    
    if (!candidates.size())
      fprintf(outfile, "No candidates for false sharing\n");
    else {
      for (unsigned int i = 0; i < candidates.size(); i++) {
	for (unsigned int j = 0; j < candidates[i].size(); j++)
	  fprintf(outfile, "[PAGE:%d] address: %p\n", candidates[i][j].get_index(),
		  (void *) candidates[i][j].get_base_addr());
	fprintf(stderr, "\n");
      }
    }
    for (unsigned int i = 0; i < spaces.size(); i++)
      spaces[i].dump(outfile, true);
  }
  else {
    fprintf(outfile, "%p %p\n", (void *) global_begin_addr, (void *) global_end_addr);

    fprintf(outfile, "%lu\n", spaces.size());
    for (unsigned int i = 0; i < spaces.size(); i++)
      spaces[i].dump(outfile, false);
  }
  fclose(outfile);
}

/* is addr aligned at the beginning of a page in the UM space */
bool um_map::is_aligned_begin(ADDR addr) {
  return !((addr - global_begin_addr) % BASE_PAGE_SIZE);
}

/* is addr aligned at the end of a page in the UM space */
bool um_map::is_aligned_end(ADDR addr, unsigned padding) {

  if (padding == 0) {
    /* offset within the page */
    ADDR end_offset = (addr - global_begin_addr) % BASE_PAGE_SIZE;

    //    printf("page offset for end addr: %lu\n", end_offset);

    /* if offset is between 4091-4095 then it is aligned */
    if ((end_offset <= (BASE_PAGE_END_OFFSET)) &&
	(end_offset > (BASE_PAGE_END_OFFSET - 4)))
      return true; 
    else
      return false;
  }
  else {
    ADDR end_offset = ((addr + padding) - global_begin_addr) % BASE_PAGE_SIZE;
    //    printf("page offset for end addr (with padding): %lu\n", end_offset);
    if (end_offset == BASE_PAGE_END_OFFSET)
      return true; 
    else
      return false;
  }
}

ADDR um_map::get_page_base_addr(ADDR addr) {
  ADDR norm_addr = addr - global_begin_addr;

  unsigned offset = (norm_addr % BASE_PAGE_SIZE);
  return (addr - offset);
}

unsigned um_map::get_page_index(ADDR addr) {
  ADDR norm_addr = addr - global_begin_addr;

  return (norm_addr / BASE_PAGE_SIZE);
}

void um_map::update_gpu_access(unsigned kernel, unsigned instance, void *gpu_begin_addr,
			       int begin_offset, int extent) {
  
  for (int i = spaces.size() - 1; i >= 0; i--) {
    if (spaces[i].get_begin() == (ADDR) gpu_begin_addr) {
      ADDR begin_addr = spaces[i].get_begin();
      begin_addr = begin_addr + begin_offset;
      ADDR end_addr = (begin_addr + extent) - 1;
      spaces[i].set_gpu_access_record(kernel, instance, begin_addr, end_addr);
      break;
    }
  }
}


void um_map::update_gpu_access(unsigned kernel, unsigned instance, void *gpu_begin_addr) {
  for (int i = spaces.size() - 1; i >= 0; i--) {
    if (spaces[i].get_begin() == (ADDR) gpu_begin_addr) {
      ADDR begin_addr = spaces[i].get_begin();
      ADDR end_addr = spaces[i].get_end();
      spaces[i].set_gpu_access_record(kernel, instance, begin_addr, end_addr);
      break;
    }
  }
}


void um_map::calculate_padding() {
#if 0  
  ADDR begin_addr_prev = spaces[0].get_begin();
  ADDR size_prev = spaces[0].get_size();
  for (unsigned int i = 1; i < spaces.size(); i++) {
    ADDR begin_addr_next = spaces[i].get_begin();
    ADDR offset = begin_addr_next - begin_addr_prev;
    unsigned padding = offset - size_prev;
    spaces[i - 1].set_padding(padding);
    spaces[i - 1].set_end(spaces[i - 1].get_end() + padding);
    begin_addr_prev = begin_addr_next;
    size_prev = spaces[i].get_size();
  }
  #endif
}
void um_map::merge_access_record_to_kernel_trace(std::vector<kernel_call> &trace) {
  for (unsigned i = 0; i < trace.size(); i++) {
    for (std::vector<um_space>::iterator space = spaces.begin() ; space != spaces.end(); space++)
      space->get_access_range_for_kernel(trace[i]);
  }
}


enum page_access_pattern um_map::calculate_page_access_pattern(kernel_call &kernel,
							       ADDR page_addr,
							       int begin_offset, int end_offset,
							       int &new_begin_offset, int &new_end_offset) {
  
  bool access_lower = false;
  /* GPU access starts at beginning of page; only need to consider tail-end */
  if (begin_offset == 0)
    access_lower = true;
  bool access_higher = false;

  bool left_empty = false;
  bool right_empty = false;
  /* GPU access extends untils end of page; only need to consider front-end */
  if (end_offset == BASE_PAGE_END_OFFSET) {
    access_higher = true;
  }

  /* represents access in the candidate page by all spaces combined */
  int left_range_begin = UNDEFINED;
  int left_range_end = UNDEFINED;

  if (!access_lower) {
    for (unsigned j = 0; j < kernel.spaces(); j++) {
      ADDR begin_addr = kernel.get_access_record_begin(j);
      ADDR end_addr = kernel.get_access_record_end(j);
      
      int this_range_begin;
      int this_range_end;
      if (page_addr > end_addr || (page_addr + begin_offset <= begin_addr))
	continue;
      this_range_begin = (page_addr >= begin_addr ? page_addr : begin_addr);
      this_range_end = (end_addr < (page_addr + begin_offset) ? end_addr : (page_addr + begin_offset - 1));
      
      /* convert address range to offset from base page addr */
      this_range_begin = this_range_begin - page_addr;
      this_range_end = this_range_end - page_addr;
      
      if (this_range_begin > left_range_begin)
	left_range_begin = this_range_begin;
      if (this_range_end > left_range_end)
	left_range_end = this_range_end;
    }
  }
  
  /* this kernel accesses the entire front portion */
  if ((left_range_begin == 0) && (left_range_end == begin_offset - 1))
    access_lower = true;

  if ((left_range_begin == UNDEFINED) && (left_range_end == UNDEFINED))
    left_empty = true;
  /* represents access in the candidate page by all spaces combined */
  int right_range_begin = UNDEFINED;
  int right_range_end = UNDEFINED;

  ADDR begin_addr, end_addr;
  if (!access_higher) {
    for (unsigned j = 0; j < kernel.spaces(); j++) {
      begin_addr = kernel.get_access_record_begin(j);
      end_addr = kernel.get_access_record_end(j);
      
      int this_range_begin;
      int this_range_end;
      if ((page_addr + end_offset >= end_addr) || ((page_addr + BASE_PAGE_END_OFFSET) < begin_addr))
	continue;
      this_range_begin = ((page_addr + end_offset) >= begin_addr ? (page_addr + end_offset) : begin_addr);
      this_range_end = (end_addr < (page_addr + BASE_PAGE_END_OFFSET) ? end_addr : (page_addr + BASE_PAGE_END_OFFSET));
      
      /* convert address range to offset from base page addr */
      this_range_begin = this_range_begin - page_addr;
      this_range_end = this_range_end - page_addr;
      
      if (this_range_begin > right_range_begin)
	right_range_begin = this_range_begin;
      if (this_range_end > right_range_end)
	right_range_end = this_range_end;
    }
  }
  if ((right_range_begin == (end_offset + 1)) && (right_range_end == BASE_PAGE_END_OFFSET))
    access_higher = true;
  if ((right_range_begin == UNDEFINED) && (right_range_end == UNDEFINED))
    right_empty = true;
  
  /* both left and right segments fully accessed by all spaces combined) */
  if (access_lower && access_higher) 
    return FULL;
  
  /* no access to either the  left or right segment */
  if (left_empty && right_empty)
    return EMPTY;
  
  /* partial access to left and right segment; live range addr range is altered */
  new_begin_offset = left_range_end + 1;
  new_end_offset = begin_offset;

  if (left_range_begin < 0)
    new_begin_offset = begin_offset;
  else
    new_begin_offset = (left_range_begin < begin_offset ? left_range_begin : begin_offset);

  new_end_offset = (right_range_end > end_offset ? right_range_end : end_offset);

  return PARTIAL;
}
  
  
  void update_pre_live_range_for_cohort(std::vector<fs_page> &candidates) {
  unsigned n = candidates[candidates.size() - 1].get_num_pre_live_range(); 

  if (n > 0) {
    unsigned pre_begin = candidates[candidates.size() - 1].get_pre_live_range_begin(n - 1); 
    unsigned pre_end = candidates[candidates.size() - 1].get_pre_live_range_end(n - 1);
    unsigned pre_addr_begin = candidates[candidates.size() - 1].get_pre_live_range_addr_range_begin(n - 1); 
    unsigned pre_addr_end = candidates[candidates.size() - 1].get_pre_live_range_addr_range_end(n - 1);
    for (unsigned i = 0; i < candidates.size() - 1; i++) {
      candidates[i].set_pre_live_range(pre_begin, pre_end, pre_addr_begin, pre_addr_end);
    }
  }
  return;
}

void update_post_live_range_for_cohort(std::vector<fs_page> &candidates) {
  unsigned n = candidates[candidates.size() - 1].get_num_post_live_range(); 

  if (n > 0) {
    unsigned post_begin = candidates[candidates.size() - 1].get_post_live_range_begin(n - 1); 
    unsigned post_end = candidates[candidates.size() - 1].get_post_live_range_end(n - 1);
    unsigned post_addr_begin = candidates[candidates.size() - 1].get_post_live_range_addr_range_begin(n - 1); 
    unsigned post_addr_end = candidates[candidates.size() - 1].get_post_live_range_addr_range_end(n - 1);
    for (unsigned i = 0; i < candidates.size() - 1; i++) {
      candidates[i].set_post_live_range(post_begin, post_end, post_addr_begin, post_addr_end);
    }
  }
  return;
}


void um_map::calculate_live_range(std::vector<kernel_call> &trace) {

  for (unsigned i = 0; i < candidates.size(); i++) {

    int n = candidates[i].size();    

    /* extract the master page in this group */
    fs_page candidate = candidates[i][n - 1];    
    int kernel = candidate.get_kernel();
    ADDR page_addr = candidate.get_base_addr();
    int begin_offset = candidate.get_gpu_begin();
    int end_offset = candidate.get_gpu_end();

    int new_begin_offset = UNDEFINED;
    int new_end_offset = UNDEFINED;
    
    /* calculate pre live range */    
    int live_begin = 0;
    int live_end = kernel;
    bool live_range_open = true;
    for (int j = kernel - 1; j >= 0; j--) {
      enum page_access_pattern access = calculate_page_access_pattern(trace[j], page_addr,
							      begin_offset, end_offset,
							      new_begin_offset, new_end_offset);
      /* found the end of a live range */
      if (access == FULL || access == PARTIAL) {
	live_begin = j;
	candidates[i][n - 1].set_pre_live_range(live_begin, live_end, begin_offset, end_offset);
	update_pre_live_range_for_cohort(candidates[i]);
	live_range_open = false;
      }

      /* candidate is killed completely, no need to look further */
      if (access == FULL)
	break;
      
      /* continue to search for live range from this point with new offset range */
      if (access == PARTIAL) {
	begin_offset = new_begin_offset;
	end_offset = new_end_offset;
	live_end = live_begin;
	live_range_open = true;
      }

      /* nothing to do if EMPTY, just continue */
    }
    if (live_range_open && kernel > 0) {
	live_begin = 0;
	candidates[i][n - 1].set_pre_live_range(live_begin, live_end, begin_offset, end_offset);
	update_pre_live_range_for_cohort(candidates[i]);
    }

    /* reset offsets */
    begin_offset = candidate.get_gpu_begin();
    end_offset = candidate.get_gpu_end();
    new_begin_offset = UNDEFINED;
    new_end_offset = UNDEFINED;

    /* calculate post live range */
    live_begin = kernel;
    live_end = END_OF_PROG;
    live_range_open = true;
    for (unsigned j = kernel + 1; j < trace.size(); j++) {

      enum page_access_pattern access = calculate_page_access_pattern(trace[j], page_addr,
								      begin_offset, end_offset,
								      new_begin_offset, new_end_offset);


      /* found the end of a live range */
      if (access == FULL || access == PARTIAL) {
	live_end = j;
	candidates[i][n - 1].set_post_live_range(live_begin, live_end, begin_offset, end_offset);
	update_post_live_range_for_cohort(candidates[i]);
	live_range_open = false;
      }

      /* candidate is killed completely, no need to look further */
      if (access == FULL)
	break;
      
      /* continue to search for live range from this point with new offset range */
      if (access == PARTIAL) {
	begin_offset = new_begin_offset;
	end_offset = new_end_offset;
	live_begin = live_end;
	live_range_open = true;
      }

      /* nothing to do if EMPTY, just continue */
    }
    if (live_range_open) {
	live_end = END_OF_PROG;
	candidates[i][n - 1].set_post_live_range(live_begin, live_end, begin_offset, end_offset);
	update_post_live_range_for_cohort(candidates[i]);
    }
  }    
  return;
}

  
void um_map::calculate_first_touch(std::vector<kernel_call> &trace) { 
  first_touch.resize(get_page_count());
  for (unsigned i = 0; i < first_touch.size(); i++)
    first_touch[i] = UNTOUCHED;

  /* for each kernel invocation and each space touched in that invocation */
  for (unsigned i = 0; i < trace.size(); i++) {    
    for (unsigned j = 0; j < trace[i].spaces(); j++) {
      /* find index of first page */
      ADDR begin_addr = trace[i].get_access_record_begin(j);
      ADDR base_addr;
      base_addr = get_page_base_addr(begin_addr);
      unsigned global_begin_index = get_page_index(base_addr);

      /* find index of last page */
      ADDR end_addr = trace[i].get_access_record_end(j);
      base_addr = get_page_base_addr(end_addr);
      unsigned global_end_index = get_page_index(base_addr);

      /* if a page was untouched previously then designate this invocation as the one two touch it first */
      for (unsigned k = global_begin_index; k <= global_end_index; k++) {
	if (first_touch[k] == UNTOUCHED)
	  first_touch[k] = i;
      }
    }
  }
}

void um_map::dump_first_touch() {
  for (unsigned i = 0; i < first_touch.size(); i++)
    printf("page %d: %d\n", i, first_touch[i]);  
}

/* 
 * merge GPU access range in candidate pages that span multiple spaces 
 */
void um_map::merge_gpu_access_range_in_candidates() {
  for (unsigned i = 0; i < candidates.size(); i++) {
    /* only need to merge if candidate page is accessed in multiple spaces */
    if (candidates[i].size() > 1) {
      unsigned lowest = candidates[i][0].get_gpu_begin();
      unsigned highest = candidates[i][0].get_gpu_end();
      for (unsigned j = 0; j < candidates[i].size(); j++) {
	if (candidates[i][j].get_gpu_begin() < lowest)
	  lowest = candidates[i][j].get_gpu_begin();
	if (candidates[i][j].get_gpu_end() > highest)
	  highest = candidates[i][j].get_gpu_end();
      }
      fs_page merged_candidate = candidates[i][0];
      merged_candidate.set_base_addr(candidates[i][0].get_base_addr());
      merged_candidate.set_kernel(candidates[i][0].get_kernel());

      merged_candidate.set_gpu_begin(lowest);
      merged_candidate.set_gpu_end(highest);

      /* note: live-range info updated post-merge */
      
      candidates[i].push_back(merged_candidate);
    }
  }
  return;
}

void um_map::mark_fs_candidates(std::vector<kernel_call> &trace) { 
  calculate_first_touch(trace);
  //  dump_first_touch();
  for (unsigned i = 0; i < trace.size(); i++) {
    for (unsigned j = 0; j < trace[i].spaces(); j++) {
      int this_space = trace[i].get_access_record_space(j);
      ADDR begin_addr = trace[i].get_access_record_begin(j);
      ADDR end_addr = trace[i].get_access_record_end(j);
      ADDR base_addr;
      base_addr = get_page_base_addr(begin_addr);
      ADDR end_base_addr = get_page_base_addr(end_addr);
      unsigned global_index = get_page_index(base_addr);
      if (begin_addr && !is_aligned_begin(begin_addr)) {
	base_addr = get_page_base_addr(begin_addr);
	global_index = get_page_index(base_addr);

	/* don't add this page if it has already been marked as a candidate */
	//	if (!page_is_marked(candidates, global_index)) {
	  fs_page candidate;
	  int begin_offset = begin_addr - base_addr;
	  candidate.set_gpu_begin(begin_offset);

	  unsigned end_offset = end_addr - base_addr;	  
	  if (end_offset > BASE_PAGE_END_OFFSET)
	    end_offset = BASE_PAGE_END_OFFSET;

	  candidate.set_gpu_end((end_offset));
	  candidate.set_base_addr(base_addr);
	  candidate.set_index(global_index);
	  candidate.set_first_touch(first_touch[global_index]);
	  candidate.set_kernel(i);

	  add_candidate(candidate);
      }
      global_index = get_page_index(end_base_addr);
      if (end_addr && !is_aligned_end(end_addr, spaces[this_space].get_padding())) {
	end_base_addr = get_page_base_addr(end_addr);
	  fs_page candidate;
	  int begin_offset = begin_addr - end_base_addr;
	  if (begin_offset < 0) 
	    begin_offset = 0;
	  candidate.set_gpu_begin(begin_offset);
	  int end_offset = end_addr - end_base_addr;
	  candidate.set_gpu_end(end_offset);

	  candidate.set_base_addr(end_base_addr);
	  candidate.set_index(global_index);
	  candidate.set_first_touch(first_touch[global_index]);
	  candidate.set_kernel(i);

	  add_candidate(candidate);
      }
    }
  }

  merge_gpu_access_range_in_candidates();
  calculate_live_range(trace);
  return;
}

kernel_call::kernel_call() {}

kernel_call::kernel_call(int _id, int _instance) {
  id = _id;
  instance = _instance;
}

kernel_call::~kernel_call() {
  access_record.clear();
}


void um_map::update(unsigned id, void *begin, ADDR size) {

  ADDR begin_addr = (ADDR) begin;
  ADDR end_addr = begin_addr + size;

  end_addr = end_addr - 1;

  um_space this_space(id);
  if (id == 0)
    global_begin_addr = begin_addr;
  if (end_addr > global_end_addr)
    global_end_addr = end_addr;

  this_space.set_begin(begin_addr);
  this_space.set_end(end_addr);

  unsigned begin_page_index = get_page_index(begin_addr);
  unsigned end_page_index = get_page_index(end_addr);
  this_space.set_begin_page_index(begin_page_index);
  this_space.set_end_page_index(end_page_index);
  spaces.push_back(this_space);    

  return;
}


void um_map::print_offsets() {
  for (unsigned int i = 0; i  < spaces.size(); i++) {
    printf("Space %d\n", i); 
    printf("%p %p\n", (void *) spaces[i].get_begin(), (void *) spaces[i].get_end());
    printf("%u %u\n", spaces[i].get_begin_page_index(), spaces[i].get_end_page_index());
    ADDR end_addr = spaces[i].get_end();
    printf("Num bytes in last page = %lu\n", (end_addr % 4096) + 1);
  }
}
