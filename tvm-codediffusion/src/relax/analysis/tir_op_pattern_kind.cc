/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/arith/iter_affine_map.h>
#include <tvm/relax/analysis.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace relax {

using namespace tir;

class PatternKindAnalyzer : public StmtExprVisitor {
 public:
  explicit PatternKindAnalyzer(const tir::PrimFunc& func) {
    for (const tir::Var& param : func->params) {
      Optional<Buffer> param_buf = func->buffer_map.Get(param);
      if (param_buf.defined()) {
        param_buffers_.insert(param_buf.value());
      }
    }
  }

 private:
  bool IsOutputBlock(const BlockNode* block) {
    for (const BufferRegion& write_region : block->writes) {
      if (param_buffers_.count(write_region->buffer)) {
        return true;
      }
    }
    return false;
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    // We only support one buffer store in a block (usually generated by TE compute)
    // If we have already seen buffer store in the current block, classify as Opaque.
    if (store_.defined() && !IsSameArray(op->indices, store_.value()->indices)) {
      kind_ = relay::kOpaque;
      return;
    }
    store_ = GetRef<BufferStore>(op);
    StmtVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    loads_.push_back(GetRef<BufferLoad>(op));
    ExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BlockNode* op) final {
    if (op->name_hint == "root") {
      // Skip the root block
      StmtVisitor::VisitStmt(op->body);
      return;
    }

    // Step 1. Clear loads and store
    loads_.clear();
    store_ = NullOpt;
    // Step 2. Visit block body.
    StmtVisitor::VisitStmt(op->body);

    // We support exactly one buffer store in a block (usually generated by TE compute)
    // If we have not seen any store in the current block, classify as Opaque.
    if (!store_.defined()) {
      kind_ = relay::kOpaque;
      return;
    }

    BufferStore store = store_.value();

    // Step 3. Checking load store indices pattern
    relay::OpPatternKind index_pair_pattern = relay::kElemWise;
    bool has_elem_wise = false;
    for (const BufferLoad& load : loads_) {
      // Since elemwise is stricter than broadcast and broadcast is stricter than injective,
      // while the order amount enums: kElemWise < kBroadcast < kInjective.
      // We can simply use `std::max` to detect these three patterns.
      // E.g Here is only one store node but two load nodes, like C[i, j] = A[i, j] + B[i]
      // Buffer C and A are elemwise but C and B are broadcast. So the whole block follows
      // broadcast pattern.
      if (IsElemwisePattern(store, load)) {
        index_pair_pattern = std::max(index_pair_pattern, relay::kElemWise);
        has_elem_wise = true;
      } else if (IsBroadcastPattern(store, load)) {
        index_pair_pattern = std::max(index_pair_pattern, relay::kBroadcast);
      } else if (IsInjectivePattern(store, load)) {
        index_pair_pattern = std::max(index_pair_pattern, relay::kInjective);
      } else {
        index_pair_pattern = relay::kOpaque;
        break;
      }
    }
    // If there is a index pair is kElemWise and others are kBroadcast, we regard it as kElemWise
    // e.g. A[i, j] = B[i, j] + C[i]
    if (index_pair_pattern == relay::kBroadcast && has_elem_wise) {
      index_pair_pattern = relay::kElemWise;
    }
    // If the block index pattern is not opaque, update kind.
    if (index_pair_pattern != relay::kOpaque) {
      // This rule for softmax: reduce + injective.
      if (IsOutputBlock(op) && kind_ == relay::kCommReduce) {
        kind_ = relay::kOutEWiseFusable;
      } else {
        kind_ = std::max(kind_, index_pair_pattern);
      }
      return;
    }

    // Step 4. Checking if the block contains reduce axis by looking into block iterators.
    bool has_reduction = false;
    Array<tir::Var> reduce_vars;
    for (const IterVar& it : op->iter_vars) {
      if (it->iter_type == kCommReduce) {
        has_reduction = true;
        reduce_vars.push_back(it->var);
      }
    }

    if (has_reduction) {
      if (IsFMA(op->body)) {
        // FMA is regards as kOutEWiseFusable, e.g. Matmul or Conv.
        kind_ = std::max(kind_, relay::kOutEWiseFusable);
        return;
      } else {
        for (size_t i = 0; i < loads_.size(); ++i) {
          // If it's not a pure reduce, regards as kOutEWiseFusable.
          // This rule works for pooling for now.
          if (!IsPureReducePattern(reduce_vars, loads_[i]->indices)) {
            kind_ = std::max(kind_, relay::kOutEWiseFusable);
            return;
          }
        }
      }
      kind_ = std::max(kind_, relay::kCommReduce);
    } else {
      kind_ = relay::kOpaque;
    }
  }

  /********** Helper Functions **********/

  /*! \brief Checking if two arrays contains same elements. */
  static bool IsSameArray(const Array<PrimExpr>& lhs, const Array<PrimExpr>& rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (!lhs[i].same_as(rhs[i])) {
        return false;
      }
    }
    return true;
  }

  /*!
   * \brief Checking the load indices and store indices follows elemwise pattern.
   * It's elemwise pattern iff load indices and store indices are the same.
   * E.g A[i, j] = B[i, j]
   */
  static bool IsElemwisePattern(const BufferStore& store, const BufferLoad& load) {
    return IsSameArray(store->indices, load->indices);
  }

  /*!
   * \brief Checking the load indices and store indices follows broadcast pattern.
   * It's broadcast pattern iff all load indices are in the store indices in order
   * E.g. A[i, j] = B[i] is broadcast since all load indices(`i`) are in the store indices
   *      A[i, j] = B[i, k] is not broadcast since `k` are not in the store indices.
   *      A[i, j] = B[j, i] is not broadcast the load indices are not in the same order as store's
   */
  static bool IsBroadcastPattern(const BufferStore& store, const BufferLoad& load) {
    size_t ndim_load_buf = load->buffer->shape.size();
    size_t ndim_store_buf = store->buffer->shape.size();

    for (size_t i = 0, j = 0; i < ndim_load_buf; ++i) {
      if (is_const_int(load->buffer->shape[i], 1) && is_const_int(load->indices[i], 0)) {
        // Skip unit load dimensions
        // E.g. A[i, j] = B[1, j] is still broadcast
        continue;
      }

      // Try to find the i-th load index in the store indices.
      while (j < ndim_store_buf && !store->indices[j].same_as(load->indices[i])) {
        ++j;
      }

      // It's not broadcast if we cannot find load indices in the store indices in order.
      if (j == ndim_store_buf) {
        return false;
      }
    }
    return true;
  }

  /*!
   * \brief Checking the load indices and store indices follows injective pattern.
   * It's injective pattern iff all load index vars are in the store indices, no matter orders.
   * Note that we only support store indices are direct vars so far, which can be enhance later.
   * E.g. A[i, j] = B[j, i] is injective.
   *      A[i, j] = B[i - j] is injective since the load index vars are only i, j
   */
  static bool IsInjectivePattern(const BufferStore& store, const BufferLoad& load) {
    std::unordered_set<const tir::VarNode*> vars;
    for (const PrimExpr& store_index : store->indices) {
      if (const auto* v = store_index.as<tir::VarNode>()) {
        vars.insert(v);
      } else {
        return false;
      }
    }
    for (const PrimExpr& load_index : load->indices) {
      // return false if there are vars used in load indices but not in store indices.
      if (tir::UsesVar(load_index, [&vars](const tir::VarNode* var) { return !vars.count(var); })) {
        return false;
      }
    }
    return true;
  }

  /*!
   * \brief Checking the load indices and store indices allow data reuse.
   * It allow data reuse iff there is any vars in load indices but they are not in store indices
   * E.g. Store = A[i, j] and Load = B[i, j, k] allow data reuse.
   *      Store = A[i, j] and Load = B[i, j + k] allow data reuse.
   */
  static bool IsAllowReusePattern(const BufferStore& store, const BufferLoad& load) {
    std::unordered_set<const tir::VarNode*> vars;
    for (const PrimExpr& index : store->indices) {
      if (const auto* v = index.as<tir::VarNode>()) {
        vars.insert(v);
      } else {
        return false;
      }
    }
    for (const PrimExpr& index : load->indices) {
      PreOrderVisit(index, [&](const ObjectRef& node) {
        if (const auto* v = node.as<tir::VarNode>()) {
          if (vars.count(v)) {
            vars.erase(v);
          }
        }
        return true;
      });
    }
    return !vars.empty();
  }

  static PrimExpr RemoveCast(PrimExpr e) {
    for (;;) {
      if (const auto* cast = e.as<tir::CastNode>()) {
        e = cast->value;
      } else {
        break;
      }
    }
    return e;
  }

  /*! \brief Checking if the stmt is multiply add. E.g. C[i, j] += A[i, k] * B[j, k] */
  static bool IsFMA(const Stmt& body) {
    if (const auto* store = body.as<BufferStoreNode>()) {
      if (const auto* add = RemoveCast(store->value).as<tir::AddNode>()) {
        if (const auto* mul = RemoveCast(add->b).as<tir::MulNode>()) {
          const auto* store_lhs = RemoveCast(add->a).as<tir::BufferLoadNode>();
          if (!store_lhs || !store->buffer.same_as(store_lhs->buffer) ||
              !IsSameArray(store->indices, store_lhs->indices)) {
            return false;
          }
          const auto* lhs = RemoveCast(mul->a).as<tir::BufferLoadNode>();
          const auto* rhs = RemoveCast(mul->b).as<tir::BufferLoadNode>();
          if (!lhs || !rhs) {
            return false;
          }
          return IsAllowReusePattern(GetRef<BufferStore>(store), GetRef<BufferLoad>(lhs)) &&
                 IsAllowReusePattern(GetRef<BufferStore>(store), GetRef<BufferLoad>(rhs));
        }
      }
    }
    return false;
  }

  /*!
   * \brief Checking if it is pure reduce pattern.
   * It's pure reduce pattern iff all reduces axis are directly reduce var
   * E.g. A[i] = sum(B[i, j]) is pure reduce
   *      A[i] = sum(B[i, j + k]) is not pure reduce
   *      pooling is not pure reduce
   */
  static bool IsPureReducePattern(Array<tir::Var> reduce_loops, Array<PrimExpr> indices) {
    for (const PrimExpr& e : indices) {
      int id = -1;
      if (UsesVar(e, [&](const tir::VarNode* var) {
            for (size_t i = 0; i < reduce_loops.size(); ++i) {
              if (reduce_loops[i].get() == var) {
                id = i;
                return true;
              }
            }
            return false;
          })) {
        if (!reduce_loops[id].same_as(e)) {
          return false;
        }
      }
    }
    return true;
  }

 private:
  /*!
   * \brief The BufferStore node in the current block.
   * \note We only support one BufferStore node in a block (usually generated by TE compute)
   */
  Optional<BufferStore> store_;
  /*! \brief The BufferLoad nodes in the current block. */
  Array<BufferLoad> loads_;
  /*! \brief The result of op pattern. */
  relay::OpPatternKind kind_ = relay::kElemWise;
  /*! \brief The buffers from function params. I.e. the input and output buffers. */
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> param_buffers_;

 public:
  relay::OpPatternKind GetResult() { return kind_; }
};

relay::OpPatternKind AnalyzeOpPatternKind(const PrimFunc& func) {
  PatternKindAnalyzer analyzer(func);
  analyzer(func->body);
  return analyzer.GetResult();
}

bool HasReshapePattern(const PrimFunc& func) {
  class ReshapeDetector : public StmtVisitor {
   public:
    static bool Detect(const Buffer& src_buffer, const Buffer& dst_buffer, Stmt stmt) {
      ReshapeDetector detector(src_buffer, dst_buffer);
      detector(stmt);
      return detector.is_reshape_;
    }

   private:
    explicit ReshapeDetector(const Buffer& src_buffer, const Buffer& dst_buffer)
        : is_reshape_(false), src_buffer_(src_buffer), dst_buffer_(dst_buffer) {}

    void VisitStmt_(const ForNode* loop) final {
      ana_.Bind(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
      // To detect the reshape pattern, we require each For to have
      // either another For or a BlockRealize as body.
      if (!(loop->body->IsInstance<ForNode>() || loop->body->IsInstance<BlockRealizeNode>())) {
        return;
      }
      this->VisitStmt(loop->body);
    }

    void VisitStmt_(const BlockRealizeNode* block_realize) final {
      // Constructing the mapping from block iterators to iterator
      // binding values. The mapping will be used in the substitution of
      // the flattened buffer access index.
      const Block& block = block_realize->block;
      const Array<IterVar>& block_iter = block->iter_vars;
      const Array<PrimExpr>& iter_values = block_realize->iter_values;
      ICHECK_EQ(block_iter.size(), iter_values.size());
      int n_iter = block_iter.size();
      for (int i = 0; i < n_iter; ++i) {
        // To detect the reshape pattern, we require each block iter to be data-parallel.
        if (block_iter[i]->iter_type != tir::IterVarType::kDataPar) {
          return;
        }
      }

      // Recurse into the block.
      this->VisitStmt(block);
    }

    void VisitStmt_(const BlockNode* block) final {
      // Step 0. If the block body is a ForNode, recurse into it.
      if (block->body->IsInstance<ForNode>()) {
        this->VisitStmt(block->body);
        return;
      }

      Map<tir::Var, Range> var_range;
      for (const IterVar& v : block->iter_vars) {
        ana_.Bind(v->var, Range::FromMinExtent(v->dom->min, v->dom->extent));
        var_range.Set(v->var, Range::FromMinExtent(v->dom->min, v->dom->extent));
      }

      // Step 1. Get the load/store pattern of the block body.
      // To detect the reshape pattern, we require the block body to be a
      // BufferStore, which has a BufferLoad as value.
      const auto* buffer_store = block->body.as<BufferStoreNode>();
      if (buffer_store == nullptr) {
        return;
      }
      const auto* buffer_load = buffer_store->value.as<BufferLoadNode>();
      if (buffer_load == nullptr) {
        return;
      }
      // Further, we require the buffer being stored and being loaded to
      // match the parameter of the PrimFunc, namely `dst_buffer_` and `src_buffer_`.
      if (!(buffer_store->buffer.same_as(dst_buffer_) &&
            buffer_load->buffer.same_as(src_buffer_))) {
        return;
      }

      // Apply check 1: use iter_map_simplify
      // This check requires at least one of the src/dst side is a trivial buffer
      // access (e.g., buf[ax0, ax1, ax2]).

      auto f_calc_flattened_idx = [&](const Buffer& buffer, const Array<PrimExpr>& indices) {
        ICHECK_EQ(indices.size(), buffer->shape.size());
        int ndim = indices.size();
        PrimExpr idx = 0;
        for (int i = 0; i < ndim; ++i) {
          idx = idx * buffer->shape[i] + indices[i];
        }
        idx = ana_.Simplify(idx);
        return arith::IterMapSimplify(
            /*indices=*/{idx},
            /*input_iters=*/var_range,
            /*input_pred=*/Bool(true),
            /*check_level=*/arith::IterMapLevel::Surjective,
            /*analyzer=*/&ana_,
            /*simplify_trivial_iterators=*/true)[0];
      };

      auto f_is_trivial_indices = [block, this](const Buffer& buffer,
                                                const Array<PrimExpr>& indices) {
        if (indices.size() != block->iter_vars.size()) {
          return false;
        }
        for (int i = 0; i < static_cast<int>(block->iter_vars.size()); ++i) {
          if (!(indices[i].same_as(block->iter_vars[i]->var) &&
                this->ana_.CanProveEqual(block->iter_vars[i]->dom->min,
                                         IntImm(DataType::Int(64), /*value=*/0)) &&
                this->ana_.CanProveEqual(buffer->shape[i], block->iter_vars[i]->dom->extent))) {
            return false;
          }
        }
        return true;
      };

      Array<PrimExpr> nontrivial_indices{nullptr};
      Buffer nontrivial_buffer{nullptr};
      if (f_is_trivial_indices(dst_buffer_, buffer_store->indices)) {
        nontrivial_indices = buffer_load->indices;
        nontrivial_buffer = src_buffer_;
      } else if (f_is_trivial_indices(src_buffer_, buffer_load->indices)) {
        nontrivial_indices = buffer_store->indices;
        nontrivial_buffer = dst_buffer_;
      }

      if (nontrivial_indices.defined()) {
        DataType dtype =
            !block->iter_vars.empty() ? block->iter_vars[0]->var->dtype : DataType::Int(64);
        tir::Var fused_var("fused", dtype);
        Map<tir::Var, PrimExpr> inverse_indices_map;
        PrimExpr stride = IntImm(dtype, /*value=*/1);
        for (int i = static_cast<int>(block->iter_vars.size()) - 1; i >= 0; --i) {
          inverse_indices_map.Set(
              block->iter_vars[i]->var,
              floormod(floordiv(fused_var, stride), block->iter_vars[i]->dom->extent));
          stride *= block->iter_vars[i]->dom->extent;
        }
        PrimExpr flattened_idx = f_calc_flattened_idx(nontrivial_buffer, nontrivial_indices);
        flattened_idx = Substitute(std::move(flattened_idx), inverse_indices_map);

        Array<PrimExpr> simplify_res = arith::IterMapSimplify(
            /*indices=*/{flattened_idx},
            /*input_iters=*/{{fused_var, Range(IntImm(dtype, /*value=*/0), stride)}},
            /*input_pred=*/Bool(true),
            /*check_level=*/arith::IterMapLevel::Surjective,
            /*analyzer=*/&this->ana_,
            /*simplify_trivial_iterators=*/true);
        ICHECK_EQ(simplify_res.size(), 1);

        if (simplify_res[0].same_as(fused_var)) {
          this->is_reshape_ = true;
          return;
        }
      }

      // Apply check 2 as followup when check 1 is not satisfied.
      // Calculate the flattened access index according to the load/store pattern.
      PrimExpr src_idx = f_calc_flattened_idx(src_buffer_, buffer_load->indices);
      PrimExpr dst_idx = f_calc_flattened_idx(dst_buffer_, buffer_store->indices);
      // Check if we can prove the equality of flattened indices.
      if (ana_.CanProveEqual(src_idx, dst_idx)) {
        this->is_reshape_ = true;
        return;
      }
    }

    bool is_reshape_;
    const Buffer& src_buffer_;
    const Buffer& dst_buffer_;
    arith::Analyzer ana_;
  };

  if (func->params.size() < 2) {
    return false;
  }
  Optional<Buffer> src_buffer = func->buffer_map.Get(func->params.front());
  Optional<Buffer> dst_buffer = func->buffer_map.Get(func->params.back());
  if (!(src_buffer.defined() && dst_buffer.defined())) {
    return false;
  }

  // To detect the reshape pattern, we require each For to have
  // either another For or a BlockRealize as body.
  ICHECK(func->body->IsInstance<BlockRealizeNode>());
  return ReshapeDetector::Detect(src_buffer.value(), dst_buffer.value(), func->body);
}

TVM_REGISTER_GLOBAL("relax.analysis.has_reshape_pattern").set_body_typed(HasReshapePattern);

}  // namespace relax
}  // namespace tvm