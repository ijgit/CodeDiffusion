/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "cuda_utils.h"

namespace fastertransformer
{

enum class ActivationType
{
    Gelu,
    Relu,
    Silu,
    GeGLU,
    ReGLU,
    SiGLU,
    Identity,
    InvalidType
};

inline bool isGatedActivation(ActivationType activaiton_type)
{
    return activaiton_type == ActivationType::GeGLU || activaiton_type == ActivationType::ReGLU
        || activaiton_type == ActivationType::SiGLU;
}

inline ActivationType get_activation(const std::string& activation_name)
{
    if (activation_name == "identity")
        return ActivationType::Identity;
    if (activation_name == "relu")
        return ActivationType::Relu;
    if (activation_name == "silu")
        return ActivationType::Silu;
    if (activation_name == "gelu")
        return ActivationType::Gelu;
    // todo: more
    return ActivationType::InvalidType;
}

} // namespace fastertransformer