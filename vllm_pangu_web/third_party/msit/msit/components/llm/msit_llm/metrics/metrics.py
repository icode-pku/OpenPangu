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


import re
import statistics
import warnings
from itertools import islice
from abc import abstractmethod, ABCMeta

from msit_llm.common.log import logger
from msit_llm.common.validate import validate_parameters_by_func, validate_parameters_by_type

from tqdm import tqdm
import jieba
from nltk import bleu_score
from rouge_chinese import Rouge


class Metrics(metaclass=ABCMeta):
    _LEGAL_CHAR_PATTERN = r'^[\u4e00-\u9fa50-9a-zA-Z\s]+$'
    _EXCLUDE_LIST = ['.', ',', '。', '，', ' ', '(', ')', '"', "'"]

    def __init__(self, thr=None, ngrams=None):
        self._thr = thr
        self._default_thr = None

        self._ngrams = ngrams
        self._default_ngrams = None

    @abstractmethod
    def _quantify_word(self, word):
        raise NotImplementedError

    @abstractmethod
    def _compare_two_words(self, word1, word2):
        raise NotImplementedError

    @abstractmethod
    def _which_is_better(self, score, thr):
        raise NotImplementedError

    def compare_two_lists_of_words(self, target, refenece):
        if self._thr is None:
            self._thr = self._default_thr

        if self._ngrams is None:
            self._ngrams = self._default_ngrams

        score = None

        for i, (word1, word2) in enumerate(zip(tqdm(target), refenece)):
            try:
                score = self._compare_two_words(word1, word2)
            except Exception as e:
                logger.error("An error occured when trying to compare two strings `%s` and `%s` "
                             "inside the class `%s`. This error is caused by: %s",
                             word1, word2, self.__class__.__name__, e)
                continue
            
            if not self._which_is_better(score, self._thr):
                yield i, score


class Accuracy(Metrics):
    @validate_parameters_by_func(
        {
            "thr": [lambda thr: thr is None or 0 <= thr <= 1]
        },
        in_class=True
    )
    def __init__(self, thr=None, ngrams=None):
        super().__init__(thr, ngrams)
        self._default_thr = 1

    def _quantify_word(self, word):
        pass

    def _compare_two_words(self, word1, word2):
        return int(word1 == word2)

    # score >= thr is better, indicating thr < score is bad case
    def _which_is_better(self, score, thr):
        return score >= thr


class EditDistance(Metrics):
    @validate_parameters_by_func(
        {
            "thr": [lambda thr: thr is None or 0 <= thr]
        },
        in_class=True
    )
    def __init__(self, thr=None, ngrams=None):
        super().__init__(thr, ngrams)
        self._default_thr = 5

    def _quantify_word(self, word):
        pass

    def _compare_two_words(self, word1, word2):
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
        return dp[-1][-1]

    # score <= thr is better, meaning larger score is worse
    def _which_is_better(self, score, thr):
        return score <= thr


class RelativeAbnormalStringRate(Metrics):
    @validate_parameters_by_func(
        {
            "thr": [lambda thr: thr is None or 0 <= thr]
        },
        in_class=True
    )
    def __init__(self, thr=None, ngrams=None):
        super().__init__(thr, ngrams)
        self._default_thr = 1.2

    def _quantify_word(self, word):
        filtered_field = []
        
        try:
            filtered_field = [word for word in jieba.cut(word) if word not in self._EXCLUDE_LIST]
        except Exception:
            logger.error("Trying to tokenize `%s`, but failed.", word)
            raise
        
        if not filtered_field:
            logger.error("Process terminated due to invalid word value %s, please check if it is well-formatted.", word)
            raise ValueError("invalid word value.")
        
        rate = statistics.mean(
            not re.match(self._LEGAL_CHAR_PATTERN, word) for word in filtered_field
        )

        return 0.0001 if rate == 0 else rate

    def _compare_two_words(self, word1, word2):
        return self._quantify_word(word1) / self._quantify_word(word2)

    # score <= thr is better, meaning larger score is worse
    def _which_is_better(self, score, thr):
        return score <= thr


class BLEU(Metrics):
    @validate_parameters_by_func(
        {
            "thr": [lambda thr: thr is None or 0 <= thr <= 1]
        },
        in_class=True
    )
    def __init__(self, thr=None, ngrams=None):
        super().__init__(thr, ngrams)
        self._default_thr = 0.4
        self._default_ngrams = 1

    def _quantify_word(self, word):
        filtered_field = []
        
        try:
            filtered_field = [word for word in jieba.cut(word) if word not in self._EXCLUDE_LIST]
        except Exception:
            logger.error("Trying to tokenize `%s`, but failed.", word)
            raise
        
        return filtered_field
    
    def _compare_two_words(self, word1, word2):
        out_field = self._quantify_word(word1)
        ref_field = self._quantify_word(word2)

        weights = [0] * 4
        weights[self._ngrams - 1] = 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bleu = bleu_score.sentence_bleu([ref_field], out_field, weights=weights)

        return bleu

    # score >= thr is better, meaning smaller score is worse
    def _which_is_better(self, score, thr):
        return score >= thr


class ROUGE(Metrics):
    @validate_parameters_by_func(
        {
            "thr": [lambda thr: thr is None or 0 <= thr <= 1]
        },
        in_class=True
    )
    def __init__(self, thr=None, ngrams=None):
        super().__init__(thr, ngrams)
        self._default_thr = 0.4
        self._default_ngrams = 1

    def _quantify_word(self, word):
        filtered_field = ""
        
        try:
            filtered_field = " ".join(jieba.cut(word))
        except Exception:
            logger.error("Trying to tokenize `%s`, but failed.", word)
            raise
        
        return filtered_field
        
    def _compare_two_words(self, word1, word2):
        modified_out = self._quantify_word(word1)
        modified_ref = self._quantify_word(word2)

        rouge = Rouge()

        scores = rouge.get_scores(modified_out, modified_ref)
        return scores[0][f"rouge-{self._ngrams}"]['f']

    # score >= thr is better, meaning smaller score is worse
    def _which_is_better(self, score, thr):
        return score >= thr


class RelativeDistinctStringRate(Metrics):
    @validate_parameters_by_func(
        {
            "thr": [lambda thr: thr is None or 0 <= thr]
        },
        in_class=True
    )
    def __init__(self, thr=None, ngrams=None):
        super().__init__(thr, ngrams)
        self._default_thr = 0.8
        self._default_ngrams = 2

    def _quantify_word(self, word):
        unique = set()
        count = 0

        for contiguous_item in zip(
                *(
                        islice((word for word in jieba.cut(word) if word not in self._EXCLUDE_LIST), i, None)
                        for i in range(self._ngrams)
                )
        ):
            unique.add(contiguous_item)
            count += 1
        
        if not unique:
            logger.error("Process terminated due to invalid word value %s, please check if it is well-formatted.", word)
            raise ValueError("invalid word value.")
        
        # unique should contain at least one element when entering the loop
        # count should be greater or equal to 1 when entering the loop
        return len(unique) / count
    
    def _compare_two_words(self, word1, word2):
        return self._quantify_word(word1) / self._quantify_word(word2)

    # score >= thr is better, meaning smaller score is worse
    def _which_is_better(self, score, thr):
        return score >= thr


@validate_parameters_by_type(
    {
        "metric_name": [str],
        "thr": [int, float, None]
    }
)
def get_metric(metric_name, thr=None) -> Metrics:
    mapping = {
        "accuracy": Accuracy(thr),
        "rouge": ROUGE(thr),
        "rouge_1": ROUGE(thr, 1),
        "rouge_2": ROUGE(thr, 2),
        "rouge_l": ROUGE(thr, "l"),
        "bleu": BLEU(thr),
        "bleu_1": BLEU(thr, 1),
        "bleu_2": BLEU(thr, 2),
        "bleu_3": BLEU(thr, 3),
        "bleu_4": BLEU(thr, 4),
        "edit_distance": EditDistance(thr),
        "relative_abnormal": RelativeAbnormalStringRate(thr),
        "relative_distinct": RelativeDistinctStringRate(thr),
        "relative_distinct_1": RelativeDistinctStringRate(thr, 1),
        "relative_distinct_2": RelativeDistinctStringRate(thr, 2),
        "relative_distinct_3": RelativeDistinctStringRate(thr, 3),
        "relative_distinct_4": RelativeDistinctStringRate(thr, 4),
    }

    if metric_name not in mapping:
        logger.error("`%s` is not supported. Please choose from %s.", metric_name, list(mapping.keys()))
        raise KeyError

    return mapping[metric_name]
