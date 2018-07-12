#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/tracer_state.h"
#include "torch/csrc/assertions.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/utils/variadic.h"
#include "torch/csrc/autograd/function_hook.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/auto_unique_ptr.h"
#include <memory>
#include <mutex>
#include <vector>
#include <iostream>
#include <cstdint>
#include <unordered_map>

namespace torch { namespace jit { namespace tracer {

using at::WeakTensor;
using torch::autograd::Variable;
using variable_list = std::vector<Variable>;

struct TracingState : public std::enable_shared_from_this<TracingState> {
  TracingState();
  ~TracingState();

  struct WeakTensorHasher {
    size_t operator()(const WeakTensor& t) const {
      return std::hash<void*>()(t.unsafeGetTensorImpl());
    }
  };

  struct WeakTensorEq {
    bool operator()(const WeakTensor& t1, const WeakTensor& t2) const {
      return t1.unsafeGetTensorImpl() == t2.unsafeGetTensorImpl();
    }
  };

  std::unordered_map<WeakTensor, Value*, WeakTensorHasher, WeakTensorEq> value_map;
  std::shared_ptr<Graph> graph;
};


namespace detail {

thread_local std::shared_ptr<TracingState> tracing_state;

} // namespace detail


// This is meant to be used as a thread local place, where we can store extra
// info that gets lost when we call into ATen from Python bindings. One example
// for when this happens is when we get an IntList argument with e.g. sizes for
// view. When tracing, those might be tensors, which let us encode extra data
// dependencies, but once they get to the ATen call where we actually have the
// tracing logic, they get converted into a raw IntList, and we loose all
// information. To prevent this, we temporarily stash it in here.
struct ArgumentStash {
  struct IntListTrace : std::vector<Value*> {
    IntListTrace(int size)
      : std::vector<Value*>(size, nullptr) {}
  };

  static bool empty() {
    return stash.intlists.empty();
  }

  static void stashIntListElem(const std::string& arg_name,
                               size_t size,
                               size_t idx,
                               const Variable& var);

  static bool hasIntList(const std::string& arg_name) {
    return stash.intlists.count(arg_name) > 0;
  }

  static IntListTrace popIntList(const std::string& arg_name) {
    auto info = std::move(stash.intlists.at(arg_name));
    stash.intlists.erase(arg_name);
    return info;
  }

private:
  static thread_local ArgumentStash stash;
  std::unordered_map<std::string, IntListTrace> intlists;
};

// Should a function which takes 'vars' as inputs be traced?
// It suffices for ONE variable to be tracing: any "untraced" variables
// are treated as constants.
//
// NB: This code lives in the hotpath; make sure it is fast
//
// NB: Variable overload is not variadic because we don't actually
// need it (in most cases if we have a variable_list it is already
// flattened).
inline bool isTracingVar(const Variable& var) {
  return static_cast<bool>(detail::tracing_state);
}

inline bool isTracingVar(at::ArrayRef<Variable> vars) {
  return static_cast<bool>(detail::tracing_state);
}

// To be called with Tensor arguments from generated code
template<typename... Args>
inline bool isTracing(Args&&... args) {
  return static_cast<bool>(detail::tracing_state);
}

// Retrieve the tracing state which a function applied with 'vars' should
// be recorded to.  Precondition: isTracing(vars) == true.  At the moment,
// we don't support mixing up variables from different traces; this code
// will need to be revisited if that ever becomes supported.
inline std::shared_ptr<TracingState> getTracingState(const variable_list& vars) {
  JIT_ASSERT(detail::tracing_state);
  return detail::tracing_state;
}

// Having finished adding a new 'node' to the graph IR owned by TracingState 'state',
// 'setValueTrace' associates this node with an output variable, so that further operations
// involving this variable know which node in the IR to reference.
inline void setValueTrace(const std::shared_ptr<TracingState>& state, const Variable& var, Value *value) {
  JIT_ASSERT(var.defined());
  detail::tracing_state->value_map[WeakTensor(var)] = value;
}

// Given a variable 'var', return the 'node' which represents the instruction
// which computes the value of this variable in the IR.  When 'mustExist' is
// false, we interpret untraced variables as constants that are just embedded
// in the graph.  This is useful to handle code which does things like this
// (from torch.autograd.variable):
//
//    def mm(self, matrix):
//      output = Variable(self.data.new(self.data.size(0), matrix.data.size(1)))
//      return Addmm.apply(output, self, matrix, 0, 1, True)
//
// Here, mm fakes up a dummy variable with uninitialized data to do an inplace
// update on, but subsequently ignores it because the alpha scaling factor is zero.
// This is one of the cases where a Variable can be created inside of a trace, and
// if we treat it as a constant, everything will work out.
inline Value* getValueTrace(const std::shared_ptr<TracingState>& state, const Variable& var) {
  if (!var.defined()) {
    Node *n = state->graph->createUndefined();
    return state->graph->appendNode(n)->output();
  }

  auto & value_map = detail::tracing_state->value_map;
  auto it = value_map.find(WeakTensor(var));
  if (it == value_map.end()) {
    Value *constant = state->graph->appendNode(state->graph->createConstant(var.data()))->output();
    constant->inferTypeFrom(var.data());
    it = value_map.emplace_hint(it, WeakTensor(var), constant);
  }
  return it->second;
}

inline Value* getOutputTrace(const std::shared_ptr<TracingState>& state, const Variable& var, size_t output_no) {
  if (!var.defined()) {
    Node *n = state->graph->createUndefined();
    return state->graph->appendNode(n)->output();
  }

  auto & value_map = detail::tracing_state->value_map;
  auto it = value_map.find(WeakTensor(var));
  if (it == value_map.end()) {
    std::ostringstream os;
    os << "output " << output_no << " of traced region did not have observable "
       << "data dependence with trace inputs; this probably indicates your program "
       << "cannot be understood by the tracer.";
    throw std::runtime_error(os.str());
  }
  return it->second;
}

// Start tracing, treating 'inputs' as inputs to the trace, which can be
// varied on subsequent invocations of the trace.  Any other variables
// will be treated as constants.
//
// NB: Why does this take an rvalue reference?  We need to get a non-const
// reference to at::Tensor buffer to call unsafeGetTH, but you can't get this
// out of a const vector (silly std::vector...)
inline std::pair<std::shared_ptr<TracingState>, variable_list> enter(
    variable_list inputs) {
  if (detail::tracing_state) {
    AT_ERROR("Tracing can't be nested");
  }
  auto & state = detail::tracing_state = std::make_shared<TracingState>();
  for (auto& input : inputs) {
    auto * value_state = state->value_map[WeakTensor(input)];
    if (value_state) {
      // See Note [Repeated inputs] in tracer.cpp
      input = input.view(input.sizes());
    }
    auto input_node = state->graph->addInput(input.name());
    input_node->inferTypeFrom(input.data());
    state->value_map[WeakTensor(input)] = input_node;
  }
  return std::make_pair(state, inputs);
}

// Exit a trace, treating 'outputs' as the outputs of the trace.  These
// are the variables whose values will be computed upon subsequent
// invocations of the trace.
inline void exit(const variable_list& outputs) {
  auto & state = detail::tracing_state;
  size_t i = 0;
  for (auto& output : outputs) {
    state->graph->registerOutput(getOutputTrace(state, output, i));
    i++;
  }
}

// Pre-recorded information about the trace before we actually carry
// out the trace
struct PreTraceInfo {
  Node *n;
};

PreTraceInfo preRecordTrace(Symbol op, at::ArrayRef<Variable> inputs);
void postRecordTrace(const PreTraceInfo& info, at::ArrayRef<Variable> outputs);

autograd::Variable getSizeOf(const autograd::Variable& var, int64_t dim);

void recordSourceLocation(Node* n);
void setRecordSourceLocation(void (*v)(Node*));

// We must record the nodes of inputs before we actually carry out
// the operation, because an inplace operation may destroy the information
// we're interested in.  See #4480.
template<typename F>
PreTraceInfo makePreTraceInfo(at::ArrayRef<Variable> inputs, F ctor) {
  PreTraceInfo info;
  auto & state = detail::tracing_state;
  auto& graph = state->graph;

  Node *n = ctor(state, *graph);
  recordSourceLocation(n);

  for (const Variable & input : inputs) {
    n->addInput(getValueTrace(state, input));
  }

  // NB: Order matters. This must append after inputs but before outputs.
  graph->appendNode(n);

  info.n = n;

  return info;
}

}}} // namespace torch::jit::tracer
