/*
 *  Instrument CUDA source with calls to Raptor runtime 
 */
#include "clang/Driver/Options.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include<fstream>

using namespace std;
using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

Rewriter rewriter;
int numFunctions = 0;
int managed_spaces = 0;
unsigned kernels = 0;
unsigned instrumented_src_lines = 0;

bool have_main = false;
bool have_managed_alloc = false;

// llvm gives back positiion last - 1 with getEndLoc and also doesn't give back semi-colon
unsigned stmt_offset_adjustment = 2;

// info on each managed space in this source 
std::vector<string> managed_spaces_global_table;

// info on each kernel launch in this source file 
std::vector<std::vector<unsigned>> kernel_call_table;
std::vector<string> kernel_list;

// all instrumented code bracketed with these comment lines 
const string begin_instrument_comment = "\n/**** BEGIN CODE INSERTED BY AUTOFS  ****/";
const string end_instrument_comment = "\n/**** END CODE INSERTED BY AUTOFS  ****/\n";

// TODO: pass this as a command-line argument to this libtool */
const string UMAP_INCLUDE_PATH = "/home/Faculty/aq10/Raptor/src";

static llvm::cl::OptionCategory MyToolCategory("false-sharing-detect options");

int lookuptable(std::vector<string> &table, string name) {
  for (unsigned int i = 0; i < table.size(); i++) {
    if (table[i] == name)
      return i;
  }
  return -1;
}

int get_kernel_id(std::vector<string> kernels, std::string kernel) {
  for (unsigned int i = 0; i < kernels.size(); i++)
    if (kernel == kernels[i])
      return i;
  return -1;
}

bool exists(std::vector<string> kernels, std::string kernel) {
  for (unsigned int i = 0; i < kernels.size(); i++)
    if (kernel == kernels[i])
      return true;
  return false;
}



/*
 * This visitor will insert call to map.dump() at every exit point in a function 
 * This is only done for main(). Hence, a separate visitor is used 
 */
class MainReturnVisitor : public RecursiveASTVisitor<MainReturnVisitor> {
private:
  ASTContext *astContext; 

public:
  explicit MainReturnVisitor(CompilerInstance *CI) 
      : astContext(&(CI->getASTContext())) {
    rewriter.setSourceMgr(astContext->getSourceManager(), astContext->getLangOpts());
  }
  
  void insert_stmt_map_dump(ReturnStmt *ret, bool need_block) {
    const string map_dump_stmt = "map.dump(false);";
    if (need_block) {
      rewriter.InsertText(ret->getBeginLoc(), begin_instrument_comment + "\n{\n"
			  + "  " + map_dump_stmt + "\n  ", true, true); 
      rewriter.InsertText(ret->getEndLoc().getLocWithOffset(stmt_offset_adjustment),
			  "\n}" + end_instrument_comment, true, true);  
      instrumented_src_lines = instrumented_src_lines + 7; 
    }
    else {
      rewriter.InsertText(ret->getBeginLoc(), begin_instrument_comment
			  + "\n" + map_dump_stmt + end_instrument_comment, true, true); 
      instrumented_src_lines = instrumented_src_lines + 5; 
    }
    return;
  }

  virtual bool VisitReturnStmt(ReturnStmt *ret) {
    DynTypedNodeList parents = astContext->getParents(*ret);
    bool need_block = true;
    if ((parents.size() == 1) && (parents[0].getNodeKind().asStringRef() == "CompoundStmt"))
      need_block = false;
    insert_stmt_map_dump(ret, need_block);
    return true;
  }
   
};


class KernelArgVisitor : public RecursiveASTVisitor<KernelArgVisitor> {
private:
  ASTContext *astContext; 

public:
  explicit KernelArgVisitor(ASTContext *context) 
    : astContext(context) {
    rewriter.setSourceMgr(astContext->getSourceManager(), astContext->getLangOpts());
  }
  
  virtual bool VisitDeclRefExpr(DeclRefExpr *var_ref) {

    ValueDecl* decl = var_ref->getDecl();
    if (decl) {
      string alloc_name = decl->getDeclName().getAsString();
      int index = lookuptable(managed_spaces_global_table, alloc_name);
      if (index >= 0)
    	kernel_call_table[kernels].push_back(index);
    }
    return true;
  };

  virtual bool VisitMemberExpr(MemberExpr *var_ref) {

    ValueDecl* decl = var_ref->getMemberDecl();
    if (decl) {
      string alloc_name = decl->getDeclName().getAsString();
      int index = lookuptable(managed_spaces_global_table, alloc_name);
      if (index >= 0)
    	kernel_call_table[kernels].push_back(index);
    }
    return true;
  };
  
};

class FSVisitor : public RecursiveASTVisitor<FSVisitor> {
private:
  ASTContext *astContext; 
  
public:

  void insert_global_decls(FunctionDecl *func) {
    string umap_header = "\n#include<" + UMAP_INCLUDE_PATH + "/umap.h>";
    string umap_decl = "\num_map map;";
    string alloc_id_decl = "\nunsigned alloc_ID = 0;";
    string size_decl = "\nunsigned long __size;";
    string global_decls;

    string kernel_call_count_stmts;
    for (unsigned i = 0; i < kernel_list.size(); i++) {
      kernel_call_count_stmts = kernel_call_count_stmts  +
	"\nunsigned __kernel_" + std::to_string(i) + "_call_count = 0;";
    }
    global_decls = begin_instrument_comment + umap_header;    
    if (have_main)
      global_decls = global_decls + umap_decl; 
    if (have_managed_alloc) 
      global_decls = global_decls + alloc_id_decl + size_decl;
    global_decls = global_decls + kernel_call_count_stmts + end_instrument_comment + "\n";

    FullSourceLoc loc = astContext->getFullLoc(func->getBeginLoc());                                               
    loc = loc.getFileLoc();
#ifdef DEBUG
    if (loc.isValid())
      llvm::outs() << "Function " << func->getQualifiedNameAsString() << " allocation at "
    		   << loc.getSpellingLineNumber() << ":"                                                      
    		   << loc.getSpellingColumnNumber() << "\n";                                                  
#endif
    rewriter.InsertTextAfter(loc, global_decls); 
    instrumented_src_lines = instrumented_src_lines + 7; 
  }
  
  void insert_stmt_map_dump(ReturnStmt *ret) {
    const string map_dump_stmt = "\nmap.dump(false);";
    rewriter.InsertText(ret->getBeginLoc(), begin_instrument_comment
			+ map_dump_stmt + end_instrument_comment, true, true); 
    instrumented_src_lines = instrumented_src_lines + 5; 
    return;
  }

  void insert_stmt_to_mark_cpu_ref_interval(CUDAKernelCallExpr* kernel) {
    string mark_cpu_ref_interval_stmt = "\nmap.mark_cpu_ref_interval();";
    rewriter.InsertText(kernel->getBeginLoc(), begin_instrument_comment
			+ mark_cpu_ref_interval_stmt + end_instrument_comment, true, true); 
    instrumented_src_lines = instrumented_src_lines + 3; 
  }

  void insert_stmt_to_update_gpu_access(CUDAKernelCallExpr* kernel) {
    KernelArgVisitor *visitor = new KernelArgVisitor(astContext); 
    visitor->TraverseStmt(kernel);

    FunctionDecl *kernel_decl =  dyn_cast<FunctionDecl>(kernel->getCalleeDecl());
    string kernel_name = kernel_decl->getNameInfo().getName().getAsString();
    int kernel_id = get_kernel_id(kernel_list, kernel_name);
    if (kernel_id < 0) {
      llvm::errs() << "Fatal Error: Kernel name: " << kernel_name << " not found\n";
      exit(0);
    }
      
    string kernel_call_count = "__kernel_" + std::to_string(kernel_id) + "_call_count";
    string kernel_call_increment = "\n" + kernel_call_count + "++;";
    string gpu_access_update_stmts;
    for (unsigned int i = 0; i < kernel_call_table[kernels].size(); i++)  {
      /* handle is the name of the allocated space */
      /* at kernel call, the name represents the base address of the space */
      string managed_space_handle = managed_spaces_global_table[kernel_call_table[kernels][i]];
      
      gpu_access_update_stmts += "\nmap.update_gpu_access(" +
	std::to_string(kernel_id) + ", " + 
	kernel_call_count + ", " + 
	managed_space_handle + ");"; 
      //	std::to_string(kernel_call_table[kernels][i]) + ");"; 

    }
    gpu_access_update_stmts = "\n" + begin_instrument_comment
      + gpu_access_update_stmts + kernel_call_increment + end_instrument_comment;
    FullSourceLoc FullLocation = astContext->getFullLoc((kernel->getEndLoc()).getLocWithOffset(stmt_offset_adjustment));
    rewriter.InsertText(FullLocation, gpu_access_update_stmts, true, true); 

    // two lines for comments and one for each space accessed by GPU
    instrumented_src_lines = instrumented_src_lines + 4 + kernel_call_table[kernels].size(); 

    return;
  }

  void insert_stmts_for_umap_construction(CallExpr *call) {
    Expr *managed_ptr = call->getArg(0);
    Expr *alloc_size = call->getArg(1);
    if (UnaryOperator *op = dyn_cast<UnaryOperator>(managed_ptr)) {
      Expr *sub_expr = op->getSubExpr();
      ValueDecl* decl;
      if (DeclRefExpr *ref_expr = dyn_cast<DeclRefExpr>(sub_expr))
	decl = ref_expr->getDecl();
      if (MemberExpr *ref_expr = dyn_cast<MemberExpr>(sub_expr))
	decl = ref_expr->getMemberDecl();

      string alloc_name = decl->getDeclName().getAsString();
      managed_spaces_global_table.push_back(alloc_name);
      managed_spaces++;
      StringRef size = Lexer::getSourceText(CharSourceRange(alloc_size->getSourceRange(), true),
					    astContext->getSourceManager(),
					    astContext->getLangOpts());
	
      string begin_stmt = "\nvoid *begin_" + alloc_name + " = (void *)" + alloc_name + ";";
      // declare __size when we encounter first managed space in this source 
      string end_stmt;
      // if (managed_spaces == 1)
      // 	end_stmt = "\nunsigned long __size = " + size.str() + ";";
      // else
      end_stmt = "\n__size = " + size.str() + ";";
      
      const string ID = "alloc_ID";
      string update_stmt = "\nmap.update(" + ID + ", " + "begin_" + alloc_name + ", __size);";
      string inc_stmt = "\n" + ID + "++;";
      string range_stmts = "\n" + begin_instrument_comment +
	begin_stmt + end_stmt + update_stmt + inc_stmt
	+ end_instrument_comment;
      rewriter.InsertText(astContext->getFullLoc((call->getEndLoc()).getLocWithOffset(stmt_offset_adjustment)),
			  range_stmts, true, true); 
      instrumented_src_lines = instrumented_src_lines + 8; 
    }
  }


  explicit FSVisitor(CompilerInstance *CI) 
      : astContext(&(CI->getASTContext())) {
    rewriter.setSourceMgr(astContext->getSourceManager(), astContext->getLangOpts());
  }
  
  virtual bool VisitFunctionDecl(FunctionDecl *func) {
    if (numFunctions == 0)
      insert_global_decls(func);
    numFunctions++;
    return true;
  }
  
  virtual bool VisitCallExpr(CallExpr *call) {
    FunctionDecl *callee_decl =  dyn_cast<FunctionDecl>(call->getCalleeDecl());
    string callee_name = callee_decl->getNameInfo().getName().getAsString();
    if (callee_name == "cudaMallocManaged")
      insert_stmts_for_umap_construction(call);
    return true;
  }

  virtual bool VisitCUDAKernelCallExpr(CUDAKernelCallExpr *kernel_call) {
    std::vector<unsigned> args;
    kernel_call_table.push_back(args);
    insert_stmt_to_mark_cpu_ref_interval(kernel_call);
    insert_stmt_to_update_gpu_access(kernel_call);
    kernels++;
    return true;
  }

};


class GlobalDeclsInfoVisitor : public RecursiveASTVisitor<GlobalDeclsInfoVisitor> {
private:
  ASTContext *astContext; // used for getting additional AST info

public:
  explicit GlobalDeclsInfoVisitor(CompilerInstance *CI) 
    : astContext(&(CI->getASTContext())) {
    rewriter.setSourceMgr(astContext->getSourceManager(), astContext->getLangOpts());
  }
  
  virtual bool VisitCallExpr(CallExpr *call) {
    FunctionDecl *callee_decl =  dyn_cast<FunctionDecl>(call->getCalleeDecl());
    string callee_name = callee_decl->getNameInfo().getName().getAsString();
    if (callee_name == "cudaMallocManaged") {
      have_managed_alloc = true;
      return true;
    }
    return true;
  }

  virtual bool VisitCUDAKernelCallExpr(CUDAKernelCallExpr *kernel_call) {
    FunctionDecl *callee_decl =  dyn_cast<FunctionDecl>(kernel_call->getCalleeDecl());
    string callee_name = callee_decl->getNameInfo().getName().getAsString();
    if (!exists(kernel_list,callee_name))
      kernel_list.push_back(callee_name);
    return true;
  }
  
};


class FSASTConsumer : public ASTConsumer {
private:
  FSVisitor *visitor;        
  MainReturnVisitor *main_return_visitor;
  GlobalDeclsInfoVisitor *global_decls_info_visitor;
public:
  explicit FSASTConsumer(CompilerInstance *CI)
    : visitor(new FSVisitor(CI)), main_return_visitor(new MainReturnVisitor(CI)),
      global_decls_info_visitor(new GlobalDeclsInfoVisitor(CI))  { }

  void insert_stmt_map_dump_after_last(Stmt *last) {
    const string map_dump_stmt = "\nmap.dump(false);";
    rewriter.InsertText(last->getEndLoc().getLocWithOffset(stmt_offset_adjustment), begin_instrument_comment
			+ map_dump_stmt + end_instrument_comment, true, true); 
    instrumented_src_lines = instrumented_src_lines + 5; 
    return;
  }

  virtual void HandleTranslationUnit(ASTContext &Context) {
    SourceManager& SM = Context.getSourceManager();
    auto Decls = Context.getTranslationUnitDecl()->decls();
    // do a pre-pass over all decls to count the number of distinct kernels
    for (auto &Decl : Decls) {
      // ignore decls in header files 
      const auto& FileID = SM.getFileID(Decl->getLocation());
      if (FileID != SM.getMainFileID())
	continue;
      
      global_decls_info_visitor->TraverseDecl(Decl);
    }

    // do a pre-pass over all decls to determine if main() is defined in this file 
    for (auto &Decl : Decls) {
      const auto& FileID = SM.getFileID(Decl->getLocation());
      if (FileID == SM.getMainFileID()) {
	if (FunctionDecl *func = dyn_cast<FunctionDecl>(Decl)) {
	  string funcName = func->getNameInfo().getName().getAsString();
	  if (funcName == "main") {
	    have_main = true;
	    if (func->hasBody()) {
	      main_return_visitor->TraverseDecl(func);
	      Stmt* body = func->getBody();
	      if (CompoundStmt* stmts = dyn_cast<CompoundStmt>(body)) {
		Stmt* last_stmt = stmts->body_back();
		ReturnStmt *ret_stmt = dyn_cast<ReturnStmt>(last_stmt);
		if (!ret_stmt) {
		  insert_stmt_map_dump_after_last(last_stmt);
		}
	      }
	    }
	    break;
	  }
	}
      }
    }
    for (auto &Decl : Decls) {
      // ignore decls in header files 
      const auto& FileID = SM.getFileID(Decl->getLocation());
      if (FileID != SM.getMainFileID())
	continue;
      
      visitor->TraverseDecl(Decl);
    }
    return;
  }  
};


class FSFrontendAction : public ASTFrontendAction {
public:
  virtual std::unique_ptr<ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI,
							 llvm::StringRef file) {
    return std::unique_ptr<ASTConsumer>(new FSASTConsumer(&CI)); 
  }
};


int main(int argc, const char **argv) {
  // parse the command-line args passed to your code
  CommonOptionsParser op(argc, argv, MyToolCategory);        
  // create a new Clang Tool instance (a LibTooling environment)
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());

  // run the Clang Tool, creating a new FrontendAction
  int result = Tool.run(newFrontendActionFactory<FSFrontendAction>().get());
  
  // print out the rewritten source code ("rewriter" is a global var.)
  rewriter.getEditBuffer(rewriter.getSourceMgr().getMainFileID()).write(errs());
  return result;
}
