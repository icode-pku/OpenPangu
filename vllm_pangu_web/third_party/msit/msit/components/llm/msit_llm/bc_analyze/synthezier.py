# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from collections.abc import Iterable

import numpy as np
import pandas as pd

from msit_llm.bc_analyze.utils import get_timestamp
from msit_llm.common.log import logger
from msit_llm.common.constant import MSIT_BAD_CASE_FOLDER_NAME
from components.utils.file_open_check import ms_open
from components.utils.security_check import ms_makedirs


class Synthesizer(object):
    """This class is used for collecting llm evluation results under several datasets"""
    HEADER = ['queries', 'input_token_ids', 'output_token_ids', 'passed']
    SYNTHESIZER_FOLDER_NAME = os.path.join(MSIT_BAD_CASE_FOLDER_NAME, 'synthesizer')

    def __init__(self, *, queries=None, input_token_ids=None, output_token_ids=None, passed=None) -> None:
        """Create a synthesizer collecting information from Large Language Model under dataset evluation
        
        It will internally call `Synthesizer.from_args` for you, for the introduction to the parameters,
        please refer to `Synthesizer.from_args`

        Examples
        --------
        >>> from msit_llm import Synthesizer
        >>> synthesizer = Synthesizer(
        ...     queries='Question 1', 
        ...     input_token_ids=[1, 2, 3], 
        ...     output_token_ids=[4, 5, 6], 
        ...     passed='Correct')
        """
        self._info = dict(
            zip(self.HEADER, (np.array([], dtype=object), ) * len(self.HEADER))
        )
        
        self.from_args(
            queries=queries,
            input_token_ids=input_token_ids,
            output_token_ids=output_token_ids,
            passed=passed
        )
    
    def from_args(self, *, queries=None, input_token_ids=None, output_token_ids=None, passed=None) -> None:
        """Collecting dataset evaluation result from memory. User should take care of the evaluation details.
        
        If any parameter is `None`, it will not be added to the `Sythesizer`. This allows the user to add each
        item iteratively whenever it is ready, without extra effort on collecting each item as a whole.

        Parameters:
        -----------
        queries: str or iterable of str, optional
            The queries or questions from the dataset that the model is evaluated on. Default to `None`.
        input_token_ids: Any, optional
            The input ids of the queries or questions after encoding. Default to `None`.
        output_token_ids: Any, optional
            The output ids of the model outputs, the generated texts before decoding. Default to `None`.
        passed: {'Correct', 'Wrong'}, optional
            Indicate whether the generated texts from the model is expected with respect to the dataset label.
            `Correct` or `Wrong` depends on different evluation strategies under corresponding dataset.
            Default to `None`

        Examples
        --------
        >>> from msit_llm import Synthesizer
        >>> synthesizer = Synthesizer()
        >>> synthesizer.from_args(
        ...     queries='Question 1', 
        ...     input_token_ids=[1, 2, 3], 
        ...     output_token_ids=[4, 5, 6], 
        ...     passed='Correct')
        """
        self._update_attributes(
            queries=queries,
            input_token_ids=input_token_ids,
            output_token_ids=output_token_ids,
            passed=passed
        )

    def to_csv(self, *, errors='trunc'):
        """Archive the collected result to csv file. 
        
        Users should call `from_args` or implicitly call `from_args` through constructor before invoking this 
        method.

        Parameters
        ----------
        `errors` : {'trunc', 'pad', 'strict'}, optional
            Consistent strategies to the data frame. Since each item can be added to `Synthesizer` asynchronously,
            it is possible to have item with differnt sizes or lengths. If `trunc` is chosen, then the data frame 
            will be truncated to the shortest length among those items. If `pad` is chosen, `Synthesizer` will pad
            missing rows with `np.nan` to be consistent with the largest length. If `strict` is chosen, 
            `Synthesizer` will do nothing and let `pandas` to figure out all the items are consistent or not.

        Notes
        -----
        The resulting csv file name will be purely time stamp, for example: 20240720042235.csv`.
            
        Exceptions
        ----------
        `RuntimeError` : raises if user did not initalize (or implicitly) `Synthesizer` through `from_args`.
        `ValueError` : raises if unexpected value passed in `errors`, or if the `strict` is chosen, but the 
        `pandas` complains that the data frame is not consistent.

        Examples
        --------
        >>> from msit_llm import Synthesizer
        >>> synthesizer = Synthesizer()
        >>> synthesizer.to_csv()
        ERROR - 'from_args' should be called first before invoking this method
        """
        result_df = self.to_df(errors=errors)
        self._save_result(result_df)

    def to_df(self, *, errors):
        if not any(value.size for value in self._info.values()):
            logger.error("'from_args' should be called first before invoking this method")
            raise RuntimeError
        
        # Does not support match expression until Python >= 3.10 
        if errors == 'pad':
            self._padding()
        elif errors == 'trunc':
            self._truncating()
        elif errors == 'strict':
            pass
        else:
            logger.error(
                "Wrong value 'errors'. "
                "Expected to be either '%s', '%s' or '%s', got %r", 'pad', 'trunc', 'strict', errors)
            raise ValueError
        
        return pd.DataFrame(self._info)
    
    def _update_attributes(self, **kwargs):
        for key, value in kwargs.items():
            if value is not None:
                modified_value = self._sanitize_value(value)
                self._info[key] = np.append(self._info[key], modified_value, axis=0)

    def _sanitize_value(self, value):
        # str is a iterable
        if isinstance(value, str) or not isinstance(value, Iterable):
            return np.array([str(value)], dtype=object)

        return np.fromiter((str(item) for item in value), dtype=object)
    
    def _padding(self):
        max_len = max(map(len, self._info.values()))

        for key in self._info.keys():
            array = self._info[key]
            diff = max_len - len(array)
            if diff != 0:
                self._info[key] = np.append(array, [np.nan] * diff)
    
    def _truncating(self):
        min_len = min(map(len, self._info.values()))

        for key in self._info.keys():
            array = self._info[key]
            self._info[key] = array[:min_len]
    
    def _save_result(self, df_to_save: pd.DataFrame):
        if df_to_save.empty:
            logger.warning("'Synthesizer' detected that there is no data to save. Hence no result is saved")
            return
        
        os.makedirs(self.SYNTHESIZER_FOLDER_NAME, mode=0o700, exist_ok=True)
        path = os.path.join(self.SYNTHESIZER_FOLDER_NAME, f"{get_timestamp()}.csv")

        flags = os.O_WRONLY | os.O_CREAT
        modes = os.st.S_IRUSR | os.st.S_IWUSR | os.st.S_IRGRP
        with ms_open(path, 'w') as file:
            df_to_save.to_csv(file, encoding='utf-8', index=False)
        
        logger.info("'Sythesizer' has successfully finished the synthesis, the result is stored at '%s'", path)
