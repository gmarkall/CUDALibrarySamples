# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


#
# Example showing the use of LTO callbacks with CUFFT to perform
# truncation with zero padding, using a Python function compiled by Numba
#


from numba import cuda, types
from numba.core.extending import (make_attribute_wrapper, models,
                                  register_model)


# User code

class cb_params:
    def __init__(self, window_N, signal_size):
        self.window_N = window_N
        self.signal_size = signal_size


def cufftJITCallbackLoadComplex(cb_input, index, info, sharedmem):
    params = info[0]
    sample = index % params.signal_size

    if sample < params.window_N:
        return cb_input[index]
    else:
        return 0+0j


# Numba extensions

class CbParamsType(types.Type):
    def __init__(self):
        super().__init__(name='cb_params')


cb_params_type = CbParamsType()


@register_model(CbParamsType)
class IntervalModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('window_N', types.uint32),
            ('signal_size', types.uint32),
            ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(CbParamsType, 'window_N', 'window_N')
make_attribute_wrapper(CbParamsType, 'signal_size', 'signal_size')


# Code to compile the function using the extensions

cufftComplexPointer = types.CPointer(types.complex64)
signature = (
    cufftComplexPointer,
    types.uint64,
    types.CPointer(cb_params_type),
    types.voidptr
)

(ltoir,), resty = cuda.compile_ltoir(cufftJITCallbackLoadComplex, signature,
                                     device=True)

ptx, resty = cuda.compile_ptx_for_current_device(cufftJITCallbackLoadComplex,
                                                 signature, device=True)

print(ptx)

with open('r2c_c2r_lto_callback_numba.ltoir', 'wb') as f:
    f.write(ltoir)
