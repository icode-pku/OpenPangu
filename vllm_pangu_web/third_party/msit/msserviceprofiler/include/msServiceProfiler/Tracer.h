// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef MS_SERVER_TRACER_H
#define MS_SERVER_TRACER_H

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <limits>
#include <cstdint>
#include <random>
#include <mutex>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/syscall.h>

#include "ServiceProfilerInterface.h"

#define INTERFACE_CALL(FUNC, ...)                                        \
    msServiceProfilerCompatible::ServiceProfilerInterface::GetInstance() \
    .Call<FUNC_NAME_VAR(FUNC), decltype(FUNC)>(__VA_ARGS__)

#define INTERFACE_CALL_WITH_DEFAULT_RET(FUNC, defValue, ...)             \
    msServiceProfilerCompatible::ServiceProfilerInterface::GetInstance() \
        .CallWithRet<FUNC_NAME_VAR(FUNC), decltype(defValue), decltype(FUNC)>(defValue, ##__VA_ARGS__)

namespace msServiceProfiler {
class TraceContext {
public:
    inline static uint32_t GetTid()
    {
        thread_local uint32_t tid = static_cast<uint32_t>(syscall(SYS_gettid));
        return tid;
    }

    inline static uint64_t GetCurrentTimeInNanoseconds()
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
        return static_cast<uint64_t>(nanoseconds.count());
    }
    /**
     * @brief 返回每个线程的 ctx，理论上，不允许多个线程混用
     */
    static MS_SERVICE_PROFILER_HIDDEN TraceContext &GetTraceCtx()
    {
        thread_local TraceContext ctx{TraceContext::GetTid()};
        return ctx;
    }

    explicit TraceContext(const uint32_t tid) : tid_(tid)
    {}

    /**
     * @brief 解析 Http 的 trace 信息，并 Attch 到当前 ctx 中
     * @param traceParentOfW3C [in] W3C 的协议
     * @param traceOfB3 [in] b3 的协议
     */
    MS_SERVICE_PROFILER_HIDDEN size_t ExtractAndAttach(
        const std::string &traceParentOfW3C, const std::string &traceOfB3)
    {
        static TraceContextInfo emptyCurrent = {{0, 0}, 0, false};
        traceCtx_.push_back(INTERFACE_CALL_WITH_DEFAULT_RET(ParseHttpCtx, emptyCurrent, traceParentOfW3C, traceOfB3));
        return traceCtx_.size() - 1;
    }

    /**
     * @brief 更新当前的环境，一般Span 进入的时候更新，或者 http 请求更新时调用
     * @param traceId [in] trace id
     * @param spanId [in] span id，后续span 的 父spanid
     * @param isSample [in] 是否采样
     * @return 返回index，在退出的时候传入
     */
    MS_SERVICE_PROFILER_HIDDEN size_t Attach(const TraceId traceId, const SpanId spanId, const bool isSample = true)
    {
        traceCtx_.push_back(TraceContextInfo{traceId, spanId, isSample});
        return traceCtx_.size() - 1;
    }

    /**
     * @brief 更新当前的环境，span 退出，或者 http 请求结束时调用  todo 这块逻辑再想一下，为啥要在当前tid再删除
     * @param index [in] 第几个需要删除
     */
    MS_SERVICE_PROFILER_HIDDEN void Unattach(const size_t index)
    {
        if (index >= traceCtx_.size()) {
            return;
        }
        if (TraceContext::GetTid() == tid_ && index == traceCtx_.size() - 1) {
            traceCtx_.pop_back();
        } else {
            // 如果不是同一个线程，或者不是按顺序释放的，就进入比较复杂的退出逻辑，先记录一下退出index，在当前 tid
            // 再进行真正的删除
            std::lock_guard<std::mutex> guard(mutex_);
            multiThread = true;
            multiThreadUnattachIndex_.insert(index);
        }
        if (multiThread && TraceContext::GetTid() == tid_) {
            std::lock_guard<std::mutex> guard(mutex_);

            for (size_t readPos = traceCtx_.size(); readPos > 0; --readPos) {
                if (multiThreadUnattachIndex_.find(readPos - 1) == multiThreadUnattachIndex_.end()) {
                    break;
                }
                multiThreadUnattachIndex_.erase(readPos - 1);
                traceCtx_.pop_back();
            }
            multiThread = !multiThreadUnattachIndex_.empty();
        }
        return;
    }

    /**
     * @brief 得到当前的 ctx
     * @return 返回当前 ctx, 包括：trace id, span id, 是否采样
     */
    MS_SERVICE_PROFILER_HIDDEN const TraceContextInfo &GetCurrent()
    {
        static TraceContextInfo emptyCurrent = {{0, 0}, 0, false};
        if (traceCtx_.empty()) {
            return emptyCurrent;
        }
        return traceCtx_.back();
    }

    /**
     * @brief 获取一个随机数，不要常用
     * @return 随机数
     */
    static MS_SERVICE_PROFILER_HIDDEN uint32_t GenRandom()
    {
        std::random_device rd;   // 用于获取真随机数种子
        std::mt19937 gen(rd());  // 使用Mersenne Twister算法
        std::uniform_int_distribution<uint32_t> dis;

        return dis(gen);
    }

    /**
     * @brief 生成一个 TraceID 的高位
     * @return TraceID 的高位
     */
    static MS_SERVICE_PROFILER_HIDDEN uint64_t GenTraceId()
    {
        static uint32_t uint32Random = GenRandom();
        static uint64_t traceHigh = static_cast<uint64_t>(uint32Random) << 32;
        static uint64_t traceHighTime = TraceContext::GetCurrentTimeInNanoseconds();

        return traceHigh ^ (TraceContext::GetCurrentTimeInNanoseconds() - traceHighTime);
    }

    /**
     * @brief 生成一个 SpanId
     * @return SpanId
     */
    static MS_SERVICE_PROFILER_HIDDEN uint64_t GenSpanId()
    {
        static uint64_t spanHigh = static_cast<uint64_t>(TraceContext::GetTid()) << 32;
        thread_local uint32_t spanID = GenRandom();

        return spanHigh | spanID++;  // 设计上就是自动翻转
    }

    /**
     * @brief 加一个Resource 的属性，全局的属性
     */
    static MS_SERVICE_PROFILER_HIDDEN void addResAttribute(const char *key, const char *value)
    {
        INTERFACE_CALL(ResAddAttr, key, value);
    }

private:
    bool multiThread = false;
    std::set<size_t> multiThreadUnattachIndex_ = {};
    std::mutex mutex_{};
    uint32_t tid_;
    std::vector<TraceContextInfo> traceCtx_ = {};
};

class Span {
public:
    Span(const char *spanName, TraceContext &ctx, bool isSampled = true, const char *moduleName = nullptr,
        bool autoEnd = true)
        : autoEnd_(autoEnd), isSampled_(isSampled), ctx_(ctx), moduleName_(moduleName), span_data(nullptr)
    {
        constexpr int TUPLE_SAMPLE_INDEX = 2;
        auto &ctxInfo = ctx_.GetCurrent();
        isSampled_ = isSampled_ && std::get<TUPLE_SAMPLE_INDEX>(ctxInfo);
        if (!isSampled_) {
            return;
        }
        span_data = INTERFACE_CALL_WITH_DEFAULT_RET(NewSpanData, span_data, spanName);
        setCtx(ctxInfo);
    }

    /**
     * @brief 设置激活Span
     * @param startTime [in] 开始事件，默认为0，表示当前事件。也可以用户主动输入一个时间
     */
    MS_SERVICE_PROFILER_HIDDEN inline Span &Activate(uint64_t startTime = 0)
    {
        if (!isSampled_) {
            return *this;
        }
        INTERFACE_CALL(SpanActivate, span_data, startTime);
        return *this;
    }

    /**
     * @brief 设置属性
     * @param attrName [in] 属性名
     * @param value [in] 属性值
     */
    MS_SERVICE_PROFILER_HIDDEN inline Span &SetAttribute(const char *attrName, const char *value)
    {
        if (!isSampled_) {
            return *this;
        }
        INTERFACE_CALL(SpanAddAttribute, span_data, attrName, value);
        return *this;
    }

    /**
     * @brief 设置状态
     * @param isSuccess [in] 是否成功
     * @param msg [in] 失败消息
     */
    MS_SERVICE_PROFILER_HIDDEN inline Span &SetStatus(const bool isSuccess, const std::string &msg)
    {
        if (!isSampled_) {
            return *this;
        }
        INTERFACE_CALL(SpanSetStatus, span_data, isSuccess, msg);
        return *this;
    }

    /**
     * @brief 记录一个过程的结束节点
     */
    MS_SERVICE_PROFILER_HIDDEN void End()
    {
        if (!isSampled_) {
            return;
        }

        INTERFACE_CALL(SpanEndAndFree, span_data, std::move(moduleName_));
        span_data = nullptr;
        ctx_.Unattach(ctxIndex_);
        autoEnd_ = false;
    }

    /**
     * @brief 析构函数
     */
    ~Span()
    {
        if (autoEnd_) {
            End();
        }
    }

private:
    MS_SERVICE_PROFILER_HIDDEN inline void setCtx(const TraceContextInfo &ctxInfo)
    {
        auto &currentTraceID = std::get<0>(ctxInfo);
        auto &currentSpanID = std::get<1>(ctxInfo);

        auto spanID = SpanId(TraceContext::GenSpanId());
        auto &traceID =
            currentSpanID.as_uint64 == 0 ? TraceId{TraceContext::GenTraceId(), spanID.as_uint64} : currentTraceID;

        INTERFACE_CALL(SpanFillCtxData, span_data, traceID, spanID, currentSpanID);

        ctxIndex_ = ctx_.Attach(traceID, spanID, isSampled_);
    }

private:
    bool autoEnd_ = false;
    bool isSampled_ = false;
    TraceContext &ctx_;
    size_t ctxIndex_ = 0;
    std::string moduleName_;
    TRACE_SPAN_DATA span_data;
};

class Tracer {
public:
    /**
     * @brief 生成一个Span
     * @param spanName [in] Span 名字
     * @param moduleName [in] （可选）模块名字，类似于 PROF 的 domain
     * @param autoEnd [in] （可选）是否自动调用End，默认自动调用
     * @return 返回Span对象
     */
    static MS_SERVICE_PROFILER_HIDDEN Span StartSpanAsActive(
        const char *spanName, const char *moduleName = nullptr, bool autoEnd = true)
    {
        auto isEnable = IsEnable();
        std::cout << isEnable << std::endl;
        auto span = Span{spanName, TraceContext::GetTraceCtx(), isEnable, moduleName, autoEnd};
        return span;
    }

    /**
     * @brief 判断是否使能采集数据，当入参级别小于配置的级别时，返回true
     * @return true表示使能数据采集，false表示未使能
     */
    static MS_SERVICE_PROFILER_HIDDEN inline bool IsEnable()
    {
        return INTERFACE_CALL_WITH_DEFAULT_RET(IsTraceEnable, false);
    };
};
}  // namespace msServiceProfiler

#endif
