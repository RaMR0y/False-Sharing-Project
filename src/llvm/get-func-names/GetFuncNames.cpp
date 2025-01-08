/*
 *  Instrument CUDA source with calls to umap runtime 
 */

#include "clang/Basic/SourceManager.h"
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
unsigned instrumented_src_lines = 0;

const unsigned ATTR_GLOBAL = 101;
const unsigned ATTR_DEVICE = 98;

// llvm gives back positiion last - 1 with getEndLoc and also doesn't give back semi-colon
unsigned stmt_offset_adjustment = 2;

struct func_info_type {
  string name;
  std::vector<int> kernel_src_lines;
};

std::vector<struct func_info_type> all_func_info;
std::vector<string> kernels;

// all instrumented code bracketed with these comment lines 
const string begin_instrument_comment = "\n/**** BEGIN CODE INSERTED BY AUTOFS  ****/";
const string end_instrument_comment = "\n/**** END CODE INSERTED BY AUTOFS  ****/\n";

static llvm::cl::OptionCategory MyToolCategory("get-func-names options");

bool exists(std::vector<string> kernels, std::string kernel) {
  for (unsigned int i = 0; i < kernels.size(); i++)
    if (kernel == kernels[i])
      return true;
  return false;
}


class KernelCallVisitor : public RecursiveASTVisitor<KernelCallVisitor> {
private:
  ASTContext *astContext; // used for getting additional AST info

public:
  explicit KernelCallVisitor(ASTContext *context) 
    : astContext(context) // initialize private members
  {
    rewriter.setSourceMgr(astContext->getSourceManager(), astContext->getLangOpts());
  }
  
  virtual bool VisitCUDAKernelCallExpr(CUDAKernelCallExpr *kernel_call) {
    FullSourceLoc loc = astContext->getFullLoc(kernel_call->getBeginLoc());
    loc = loc.getFileLoc();
    FunctionDecl *callee_decl =  dyn_cast<FunctionDecl>(kernel_call->getCalleeDecl());
    string callee_name = callee_decl->getNameInfo().getName().getAsString();
    if (!exists(kernels,callee_name))
      kernels.push_back(callee_name);
    if (loc.isValid())
      all_func_info[numFunctions].kernel_src_lines.push_back(loc.getSpellingLineNumber());
    return true;
  }

  
};


// Checks if a function has the __global__ or __device__ attribute set (i.e., CUDA) 
bool isCUDAFunc(AttrVec& attrs) {
  for (unsigned int i = 0; i < attrs.size(); i++) {
    Attr this_attr = (*attrs[i]);
    attr::Kind attr_val = this_attr.getKind();
    if (attr_val == ATTR_GLOBAL)
      return true;
    if (attr_val == ATTR_DEVICE)
      return true;
  }
  return false;
}

class FSVisitor : public RecursiveASTVisitor<FSVisitor> {
private:
  ASTContext *astContext; // used for getting additional AST info
  
public:

  explicit FSVisitor(CompilerInstance *CI) 
      : astContext(&(CI->getASTContext())) {
    rewriter.setSourceMgr(astContext->getSourceManager(), astContext->getLangOpts());
  }
  
  virtual bool VisitFunctionDecl(FunctionDecl *func) {
    string funcName = func->getQualifiedNameAsString();
    struct func_info_type this_func_info;
    this_func_info.name = funcName;
    AttrVec& attrs = func->getAttrs();
    if (!isCUDAFunc(attrs)) {
      all_func_info.push_back(this_func_info);
      if (func->hasBody()){
	KernelCallVisitor *visitor = new KernelCallVisitor(astContext); 
	visitor->TraverseStmt(func->getBody());
      }
      numFunctions++;
    }
    return true;
  }
  
};


class FSASTConsumer : public ASTConsumer {
private:
    FSVisitor *visitor; // doesn't have to be private

public:
  explicit FSASTConsumer(CompilerInstance *CI)
    : visitor(new FSVisitor(CI)) { }

  // override this to call our FSVisitor on the entire source file
  virtual void HandleTranslationUnit(ASTContext &Context) {
   
    SourceManager& SM = Context.getSourceManager();
    auto Decls = Context.getTranslationUnitDecl()->decls();
    for (auto &Decl : Decls) {
      const auto& FileID = SM.getFileID(Decl->getLocation());
      if (FileID != SM.getMainFileID())
	continue;
      
      /* we can use ASTContext to get the TranslationUnitDecl, which is
       a single Decl that collectively represents the entire source file */
      //    visitor->TraverseDecl(Context.getTranslationUnitDecl());
      visitor->TraverseDecl(Decl);
    }
  }
};


class FSFrontendAction : public ASTFrontendAction {
public:
  virtual std::unique_ptr<ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI,
							 llvm::StringRef file) {
    // pass CI pointer to ASTConsumer
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

  
  for (unsigned int i = 0; i < kernels.size(); i++)
    llvm::outs() << "__kernel__" << kernels[i] <<  " " << i << "\n"; // line number ignored by Pin

  for (unsigned int i = 0; i < all_func_info.size(); i++) {
    llvm::outs() << all_func_info[i].name << " ";
    int kernel_src_line = -1;
    if (all_func_info[i].kernel_src_lines.size() > 0) {
      int max = all_func_info[i].kernel_src_lines[0];
      for (unsigned int j = 1; j < all_func_info[i].kernel_src_lines.size(); j++) {
	if (all_func_info[i].kernel_src_lines[j] > max)
	  max = all_func_info[i].kernel_src_lines[j];
      }
      kernel_src_line = max;
    }
    llvm::outs() << kernel_src_line << "\n";
  }
  return result;
}


