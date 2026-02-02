/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef MSSERVICEPROFILER_H
#define MSSERVICEPROFILER_H

#include "Profiler.h"

#define SERVER_PROFILER

#define ProfLevel0 msServiceProfiler::L0
#define ProfLevel1 msServiceProfiler::L1
#define ProfLevel2 msServiceProfiler::L2

#define ProfilerL0 msServiceProfiler::Profiler<msServiceProfiler::L0>
#define ProfilerL1 msServiceProfiler::Profiler<msServiceProfiler::L1>
#define ProfilerL2 msServiceProfiler::Profiler<msServiceProfiler::L2>

#define ITER_TYPE(_VECTOR_) decltype((_VECTOR_).begin())

#define PRIVATE_MACRO_VAR_ARGS_IMPL_COUNT(_1, _2, N, ...) N
#define PRIVATE_MACRO_VAR_ARGS_IMPL(args) PRIVATE_MACRO_VAR_ARGS_IMPL_COUNT args
#define PRIVATE_COUNT_MACRO_VAR_ARGS(...) PRIVATE_MACRO_VAR_ARGS_IMPL((__VA_ARGS__, 2, 1))

#define PRIVATE_MACRO_CHOOSE_HELPER1(M, count) M##count
#define PRIVATE_MACRO_CHOOSE_HELPER(M, count) PRIVATE_MACRO_CHOOSE_HELPER1(M, count)

#define PRIVATE_PROF_STMT(_STMT) _STMT
#define PRIVATE_PROF_STMT_LEVEL(_LEVEL, _STMT) msServiceProfiler::Profiler<msServiceProfiler::_LEVEL>()._STMT

#define PRIVATE_PROF1 PRIVATE_PROF_STMT
#define PRIVATE_PROF2 PRIVATE_PROF_STMT_LEVEL

#define PROF(...) PRIVATE_MACRO_CHOOSE_HELPER(PRIVATE_PROF, PRIVATE_COUNT_MACRO_VAR_ARGS(__VA_ARGS__))(__VA_ARGS__)
#define MONITOR(...) PROF(__VA_ARGS__)

#endif  // MSSERVICEPROFILER_H
