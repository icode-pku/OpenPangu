/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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

#include <cstring>
#include <memory>
#include <iostream>
#include <vector>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdexcept>
#include "ge/ge_ir_build.h"
#include "ge/ge_api_error_codes.h"


constexpr const char PATH_SEPARATOR = '/';
constexpr mode_t MAX_PERMISSION = 0777;
constexpr mode_t WRITE_FILE_NOT_PERMITTED = S_IWGRP | S_IWOTH | S_IROTH | S_IXOTH;

static std::string GetAbsPath(const std::string &originPath);

static bool IsSameOwner(const std::string& path)
{
    std::string absPath = GetAbsPath(path);
    struct stat buf;
    if (stat(absPath.c_str(), &buf)) {
        std::cerr << "get file stat failed";
        return false;
    }
    if (buf.st_uid != getuid()) {
        std::cerr << "file owner is not process usr";
        return false;
    }
    return true;
}

static bool OthersWritable(const std::string& path)
{
    struct stat path_stat;
    if (stat(path.c_str(), &path_stat) != 0) {
        std::cerr << "file not exists";
        return false;
    }
    mode_t permissions = path_stat.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO);
    return ((permissions & WRITE_FILE_NOT_PERMITTED) > 0);
}

static bool ParentWritable(const std::string& path)
{
    if (!IsSameOwner(path) || OthersWritable(path)) {
        return false;
    }
    return true;
}

static std::string GetFullPath(const std::string &originPath)
{
    if (originPath.empty()) {
        return "";
    }
    if (originPath[0] == PATH_SEPARATOR) {
        return originPath;
    }

    // 获取当前工作目录，检查是否成功
    char* cwd_ptr = getcwd(nullptr, 0);
    if (!cwd_ptr) {  // 检查getcwd返回值是否为nullptr
        // 抛出异常
        throw std::runtime_error("Failed to get current working directory");
    }

    // 安全处理动态分配的内存
    std::string cwd(cwd_ptr);
    free(cwd_ptr);  // 释放getcwd分配的内存，避免泄露

    return std::move(cwd + PATH_SEPARATOR + originPath);
}

static std::vector<std::string> SplitPath(const std::string &path)
{
    std::vector<std::string> tokens;
    size_t len = path.length();
    size_t start = 0;

    while (start < len) {
        size_t end = path.find(PATH_SEPARATOR, start);
        if (end == std::string::npos) {
            end = len;
        }
        if (start != end) {
            tokens.push_back(path.substr(start, end - start));
        }
        start = end + 1;
    }
    return tokens;
}

static std::string GetAbsPath(const std::string &originPath)
{
    std::string fullPath = GetFullPath(originPath);
    if (fullPath.empty()) {
        return "";
    }

    std::vector<std::string> tokens = SplitPath(fullPath);
    std::vector<std::string> tokensRefined;

    for (std::string& token : tokens) {
        if (token.empty() || token == ".") {
            continue;
        } else if (token == "..") {
            if (tokensRefined.empty()) {
                return "";
            }
            tokensRefined.pop_back();
        } else {
            tokensRefined.emplace_back(token);
        }
    }

    if (tokensRefined.empty()) {
        return "/";
    }
    std::string resolvedPath("");
    for (std::string& token : tokensRefined) {
        resolvedPath.append("/").append(token);
    }
    return resolvedPath;
}

static std::string GetParentDir(const std::string& path)
{
    size_t found = path.find_last_of('/');
    if (found != std::string::npos) {
        return path.substr(0, found);
    }
    return ".";
}

static std::string GetRoot(const std::string& path, int max_depth = 200)
{
    if (max_depth <= 0) {
        throw std::runtime_error("Max recursion depth exceeded while searching for root directory.");
    }
    std::string parentDir = GetParentDir(path);
    // 如果父目录存在且已经创建了，就返回
    if (access(parentDir.c_str(), F_OK) == 0) {
        return GetParentDir(path);  // 返回需要创建的目录路径
    }
    return GetRoot(parentDir, max_depth - 1);  // 递归向上查找
}

static void ParentCheck(const std::string& file_path)
{
    std::string parent_dir = GetRoot(GetAbsPath(file_path));
    if (parent_dir.empty()) {
        return;
    }

    if (!ParentWritable(parent_dir)) {
        std::cerr << "Parent directory has incorrect permissions: " << parent_dir << std::endl;
        throw std::runtime_error("Parent directory permission check failed.");
    }
}

static bool SaveOM(const void *model, size_t length, const std::string &file_path)
{
    if (length <= 0 || model == nullptr) {
        return false;
    }

    uint8_t* om_buffer = nullptr;
    try {
        om_buffer = new uint8_t[length];
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        return false;
    }

    // 没有结束迭代器的话，使用copy_n更加规范
    std::copy_n(static_cast<const uint8_t*>(model), length, om_buffer);

    ge::ModelBufferData om_model;
    // std::shared_ptr默认使用delete来释放内存，会导致对new[]的ub行为。所以自定义为delete[]
    om_model.data = std::shared_ptr<uint8_t>(om_buffer, [](uint8_t* p) { delete[] p; });
    om_model.length = length;

    try {
        ParentCheck(file_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create parent directories: " << e.what() << std::endl;
        return false;
    }

    auto ret = ge::aclgrphSaveModel(file_path.c_str(), om_model);
    if (ret != ge::SUCCESS) {
        std::cerr << "Failed to save model to " << file_path << std::endl;
        return false;
    }

    return true;
}