#include<vector>
#include<fstream>
#include<climits>
#include<string>

const unsigned BASE_PAGE_SIZE = 4 * 1024;
const unsigned BASE_PAGE_END_OFFSET = BASE_PAGE_SIZE - 1;

const unsigned SIZEOF_DATA_TYPE = 4;
const unsigned END_OF_PROG = INT_MAX;

const int NULL_ADDR = 0;
const unsigned INVALID_OFFSET = BASE_PAGE_SIZE;
const int UNTOUCHED = -1;
const int UNDEFINED = -1;

enum access_type { GPU_BEGIN, GPU_END, CPU_BEGIN, CPU_END };
enum page_access_pattern { FULL, EMPTY, PARTIAL };
enum fs_type { NONE, BENIGN, MALICIOUS, TRUE }; 

const std::string access_names[3] = {"FULL", "EMPTY", "PARTIAL"};

typedef unsigned long ADDR;

struct gpu_access_record {
  ADDR gpu_begin_addr; 
  ADDR gpu_end_addr; 
  int kernel_id;
  int kernel_instance;
};

struct fs_live_range {
  int begin;
  int end;
  unsigned addr_range_begin;
  unsigned addr_range_end;
  int cpu_begin;
  int cpu_end;
};
  
struct access_range {
  ADDR begin; 
  ADDR end; 
  unsigned space;
};

class kernel_call {
 public:

  kernel_call();
  kernel_call(int id, int instance);
  ~kernel_call();

  int get_id() { return id; }
  int get_instance() { return instance; }
  int get_access_record_space(int i) { return access_record[i].space; }
  ADDR  get_access_record_begin(int i) { return access_record[i].begin; }
  ADDR  get_access_record_end(int i) { return access_record[i].end; }

  unsigned spaces() { return access_record.size(); }
  void set_id(int _id) { id = _id; }
  void set_instance(int _instance) { instance = _instance; }
  void add_access_range(struct access_range this_access_range) { access_record.push_back(this_access_range);}

 private:
  int id;
  int instance;
  std::vector<struct access_range> access_record;
};


class um_page {
 public:
  um_page();
  um_page(ADDR addr);
  ~um_page();

  ADDR get_base_addr() { return base_addr; }
  unsigned get_index() { return index; }
  int get_first_touch() { return first_touch; }
  void set_first_touch(int _first_touch) { first_touch = _first_touch; }
  unsigned get_gpu_begin() { return gpu_begin_addr; }
  unsigned get_gpu_end() { return gpu_end_addr; }
  unsigned get_cpu_begin() { return cpu_begin_addr; }
  unsigned get_cpu_end() { return cpu_end_addr; }

  void set_gpu_begin(ADDR addr) { gpu_begin_addr = addr; }
  void set_gpu_end(ADDR addr) { gpu_end_addr = addr; }
  void set_cpu_begin(ADDR addr) { cpu_begin_addr = addr; }
  void set_cpu_end(ADDR addr) { cpu_end_addr = addr; }

  void set_pre_live_range_cpu_begin(int i, int page_offset) { pre_live_range[i].cpu_begin = page_offset; }
  void set_pre_live_range_cpu_end(int i, int page_offset) { pre_live_range[i].cpu_end = page_offset; }
  void set_post_live_range_cpu_begin(int i, int page_offset) { post_live_range[i].cpu_begin = page_offset; }
  void set_post_live_range_cpu_end(int i, int page_offset) { post_live_range[i].cpu_end = page_offset; }

  unsigned get_pre_live_range_cpu_begin(int i) { return pre_live_range[i].cpu_begin; }
  unsigned get_pre_live_range_cpu_end(int i) { return pre_live_range[i].cpu_end; }
  unsigned get_post_live_range_cpu_begin(int i) { return post_live_range[i].cpu_begin; }
  unsigned get_post_live_range_cpu_end(int i) { return post_live_range[i].cpu_end; }

  
  void set_base_addr(ADDR addr) { base_addr = addr; }
  void set_index(unsigned _index) { index = _index; }

  void set_pre_live_range(int begin, int end, int addr_range_begin, int addr_range_end);
  void set_pre_live_range_addr_range(unsigned begin, unsigned end);

  int get_pre_live_range_begin(unsigned i) { return pre_live_range[i].begin; }
  int get_pre_live_range_end(unsigned i) { return pre_live_range[i].end; }
  unsigned get_pre_live_range_addr_range_begin(unsigned i) { return pre_live_range[i].addr_range_begin; }
  unsigned get_pre_live_range_addr_range_end(unsigned i) { return pre_live_range[i].addr_range_end; }

  void set_post_live_range(int begin, int end, int addr_range_begin, int addr_range_end);
  void set_post_live_range_addr_range(unsigned begin, unsigned end);

  int get_post_live_range_begin(unsigned i) { return post_live_range[i].begin; }
  int get_post_live_range_end(unsigned i) { return post_live_range[i].end; }
  unsigned get_post_live_range_addr_range_begin(unsigned i) { return post_live_range[i].addr_range_begin; }
  unsigned get_post_live_range_addr_range_end(unsigned i) { return post_live_range[i].addr_range_end; }

  unsigned get_num_pre_live_range () { return pre_live_range.size(); }
  unsigned get_num_post_live_range () { return post_live_range.size(); }

  void dump(bool verbose);
  
private:
  unsigned gpu_begin_addr; 
  unsigned gpu_end_addr; 
  unsigned cpu_begin_addr; 
  unsigned cpu_end_addr; 
  
  std::vector<struct fs_live_range> pre_live_range;
  std::vector<struct fs_live_range> post_live_range;
  ADDR base_addr;
  unsigned index;                // global index within UM space 
  int first_touch;
};

class fs_page : public um_page {
 public:
  fs_page();
  ~fs_page();

  void set_kernel(int _kernel);
  int get_kernel();
  bool overlap_in_post_range(unsigned i);
  bool overlap_in_pre_range(unsigned i);
  void dump(bool verbose);
  void dump_fs_info(int live_range, enum fs_type fs, unsigned count, bool pre, bool verbose);

 private:
  int kernel;
};

class um_space {
 public:
  um_space(unsigned);
  ~um_space();
  unsigned int get_id() { return id; }
  ADDR get_begin() { return begin_addr; }
  ADDR get_end() { return end_addr; }
  ADDR get_size() { return ((begin_addr - end_addr) + 1);}
  unsigned get_page_count() { return ((end_page_index - begin_page_index) + 1); }
  unsigned get_padding() { return padding; }

  unsigned get_begin_page_index() { return begin_page_index;}
  unsigned get_end_page_index() { return end_page_index; }

  unsigned kernel_instances() { return access_record.size(); }
    
  void set_begin(ADDR addr) { begin_addr = addr; }
  void set_end(ADDR addr) { end_addr = addr; }

  void set_padding(unsigned _padding) { padding = _padding; }
  void set_begin_page_index(unsigned index) { begin_page_index = index; }
  void set_end_page_index(unsigned index) { end_page_index = index; }

  bool is_aligned_begin(ADDR addr);
  bool is_aligned_end(ADDR addr);

  ADDR get_access_record_addr(unsigned i, access_type access);
  int get_access_record_kernel(unsigned i) { return access_record[i].kernel_id; }
  int get_access_record_instance(unsigned i) { return access_record[i].kernel_instance; }
  void set_gpu_access_record(unsigned kernel, unsigned instance, ADDR begin_addr, ADDR end_addr);

  bool access_record_matches_kernel(struct gpu_access_record access_record, kernel_call &kernel);
  void get_access_range_for_kernel(kernel_call &kernel);

  void dump(FILE *outfile, bool verbose);

 private:
  unsigned id;

  ADDR begin_addr;
  ADDR end_addr;

  unsigned begin_page_index;
  unsigned end_page_index;

  std::vector<gpu_access_record> access_record;
  
  unsigned padding;
};

class um_map {
 public:
  um_map();
  ~um_map();
  unsigned get_page_count() { return (get_page_index(global_end_addr) + 1); }
  unsigned get_space_count() { return spaces.size(); }
  ADDR get_global_size() { return ((global_end_addr - global_begin_addr) + 1);}
  unsigned get_global_page_offset();

  std::vector<std::vector<fs_page>> get_candidates() { return candidates; }

  bool page_is_marked(std::vector<um_page> &pages, unsigned page_index);
  bool last_page_filled();
  bool spill_over_to_new_page(unsigned offset);

  void update(unsigned, void *, ADDR);
  void update_gpu_access(unsigned kernel, unsigned instance, void *space_base_addr,
			 int begin_offset, int extent);
  void update_gpu_access(unsigned kernel, unsigned instance, void *gpu_begin_addr);

  void mark_cpu_ref_interval() { interval_start = true; return; }
  void dump(bool verbose);
  void dump_first_touch();
  bool is_aligned_begin(ADDR addr);
  bool is_aligned_end(ADDR addr, unsigned padding);

  ADDR get_page_base_addr(ADDR addr);
  unsigned get_page_index(ADDR addr);

  void set_global_begin(ADDR addr) { global_begin_addr = addr; }
  void set_global_end(ADDR addr) { global_end_addr = addr; }

  void add_space(um_space& space) { spaces.push_back(space); }

  void add_candidate(fs_page& page);
  void mark_fs_candidates(std::vector<kernel_call> &trace);
  void merge_gpu_access_range_in_candidates();

  void calculate_padding();
  void merge_access_record_to_kernel_trace(std::vector<kernel_call> &trace);

  enum page_access_pattern calculate_page_access_pattern(kernel_call &kernel,
							 ADDR page_addr,
							 int begin_offset, int end_offset,
							 int &new_begin_offset, int &new_end_offset);
  
  void calculate_first_touch(std::vector<kernel_call> &trace);
  void calculate_live_range(std::vector<kernel_call> &trace);

  void print_offsets();
 private:
  std::vector<um_page> pages; 
  std::vector<um_space> spaces; 
  ADDR global_begin_addr;
  ADDR global_end_addr;

  std::vector<int> first_touch;

  bool interval_start;

  std::vector<std::vector<fs_page>> candidates; 
};


/*** EXTERN DECLARATION FOR INSTRUMENTED SRC ***/ 
extern um_map map;

