/* Copyright 1993-2016 NVIDIA Corporation.  All rights reserved.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to NVIDIA intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and
  * conditions of a form of NVIDIA software license agreement by and
  * between NVIDIA and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of NVIDIA is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  * OF THESE LICENSED DELIVERABLES.
  *
  * U.S. Government End Users.  These Licensed Deliverables are a
  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  * 1995), consisting of "commercial computer software" and "commercial
  * computer software documentation" as such terms are used in 48
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  * U.S. Government End Users acquire the Licensed Deliverables with
  * only those rights set forth herein.
  *
  * Any use of the Licensed Deliverables in individual and commercial
  * software must include, in the user documentation and internal
  * comments to the code, the above Disclaimer and U.S. Government End
  * Users Notice.
  */

#ifndef _CG_SCAN_H_
#define _CG_SCAN_H_

#include "info.h"
#include "helpers.h"
#include "functional.h"
#include "coalesced_scan.h"

_CG_BEGIN_NAMESPACE

namespace details {
    
    // Group support for reduce.
    template <class TyGroup> struct _scan_group_supported : public _CG_STL_NAMESPACE::false_type {};

    template <unsigned int Sz, typename TyPar>
    struct _scan_group_supported<cooperative_groups::thread_block_tile<Sz, TyPar>> : public _CG_STL_NAMESPACE::true_type {};
    template <unsigned int Sz, typename TyPar>
    struct _scan_group_supported<internal_thread_block_tile<Sz, TyPar>>            : public _CG_STL_NAMESPACE::true_type {};
    template <>
    struct _scan_group_supported<cooperative_groups::coalesced_group>              : public _CG_STL_NAMESPACE::true_type {};
    
    template <typename TyGroup>
    using scan_group_supported = _scan_group_supported<details::remove_qual<TyGroup>>;
    
    enum class ScanType { exclusive, inclusive };
    
    template <unsigned int GroupId,  ScanType scan>
    struct scan_dispatch;

    template <>
    struct scan_dispatch<details::coalesced_group_id, ScanType::inclusive> {
        template <typename TyGroup, typename TyVal, typename TyFn>
        _CG_STATIC_QUALIFIER auto scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
            return coalesced_inclusive_scan(group, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op));
        }
    };

    template <bool IsIntegralPlus>
    struct integral_optimized_scan;

    template <>
    struct integral_optimized_scan<true> {
        template <typename TyGroup, typename TyVal, typename TyFn>
        _CG_STATIC_QUALIFIER auto scan(TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
            auto ret = coalesced_inclusive_scan(group, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op));
            return op(ret, -val);
        }
    };

    template <>
    struct integral_optimized_scan<false> {
        template <typename TyGroup, typename TyVal, typename TyFn>
        _CG_STATIC_QUALIFIER auto scan(TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
            auto ret = coalesced_inclusive_scan(group, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op));
            ret = group.shfl_up(ret, 1);
            if (group.thread_rank() == 0) {
                return {};
            }
            else {
                return ret;
            }
        }
    };

    template <>
    struct scan_dispatch<details::coalesced_group_id, ScanType::exclusive> {

        template <typename TyGroup, typename TyVal, typename TyFn>
        _CG_STATIC_QUALIFIER auto scan(TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
            using optimized_impl = 
                integral_optimized_scan<_CG_STL_NAMESPACE::is_same<remove_qual<TyFn>, cooperative_groups::plus<remove_qual<TyVal>>>::value
                                        && _CG_STL_NAMESPACE::is_integral<remove_qual<TyVal>>::value>;
            return optimized_impl::scan(group, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op));
        }
    };

#if defined(_CG_CPP11_FEATURES) && defined(_CG_ABI_EXPERIMENTAL)
    template <ScanType TyScan>
    struct scan_dispatch<details::multi_tile_group_id, TyScan> {
        template <unsigned int Size, typename ParentT, typename TyVal, typename TyFn>
        _CG_STATIC_QUALIFIER auto scan(const thread_block_tile<Size, ParentT>& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
            using warpType = details::internal_thread_block_tile<32, __static_size_multi_warp_tile_base<Size>>;
            using TyRet = details::remove_qual<TyVal>;
            const unsigned int num_warps = Size / 32;
            // In warp scan result, calculated in warp_lambda
            TyRet warp_scan;

            auto warp_lambda = [&] (const warpType& warp, TyRet* warp_scratch_location) {
                warp_scan = 
                    details::coalesced_inclusive_scan(warp, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op));
                if (warp.thread_rank() + 1 == warp.size()) {
                    *warp_scratch_location = warp_scan;
                }
                if (TyScan == ScanType::exclusive) {
                    warp_scan = warp.shfl_up(warp_scan, 1);
                }
            };

            auto inter_warp_lambda =
                [&] (const details::internal_thread_block_tile<num_warps, warpType>& subwarp, TyRet* thread_scratch_location) {
                    *thread_scratch_location =
                        details::coalesced_exclusive_scan(subwarp, *thread_scratch_location, _CG_STL_NAMESPACE::forward<TyFn>(op));
            };
            
            TyRet previous_warps_sum = details::multi_warp_collectives_helper<TyRet>(group, warp_lambda, inter_warp_lambda);
            if (TyScan == ScanType::exclusive && warpType::thread_rank() == 0) {
                return previous_warps_sum;
            }
            if (warpType::meta_group_rank() == 0) {
                return warp_scan;
            }
            else {
                return op(warp_scan, previous_warps_sum);
            }
        }
    };
#endif
} // details

template <typename TyGroup, typename TyVal, typename TyFn>
_CG_QUALIFIER auto inclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
    static_assert(details::is_op_type_same<decltype(op(val, val)), TyVal>::value, "Operator input and output types differ");
    static_assert(details::scan_group_supported<TyGroup>::value, "This group does not exclusively represent a tile");

    using dispatch = details::scan_dispatch<TyGroup::_group_id, details::ScanType::inclusive>;
    return dispatch::scan(group, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op));
}

template <typename TyGroup, typename TyVal>
_CG_QUALIFIER details::remove_qual<TyVal> inclusive_scan(const TyGroup& group, TyVal&& val) {
    return inclusive_scan(group, _CG_STL_NAMESPACE::forward<TyVal>(val), cooperative_groups::plus<details::remove_qual<TyVal>>());
}

template <typename TyGroup, typename TyVal, typename TyFn>
_CG_QUALIFIER auto exclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
    static_assert(details::is_op_type_same<decltype(op(val, val)), TyVal>::value, "Operator input and output types differ");
    static_assert(details::scan_group_supported<TyGroup>::value, "This group does not exclusively represent a tile");

    using dispatch = details::scan_dispatch<TyGroup::_group_id, details::ScanType::exclusive>;
    return dispatch::scan(group, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op));
}

template <typename TyGroup, typename TyVal>
_CG_QUALIFIER details::remove_qual<TyVal> exclusive_scan(const TyGroup& group, TyVal&& val) {
    return exclusive_scan(group, _CG_STL_NAMESPACE::forward<TyVal>(val), cooperative_groups::plus<details::remove_qual<TyVal>>());
}

_CG_END_NAMESPACE

#endif // _CG_SCAN_H_
