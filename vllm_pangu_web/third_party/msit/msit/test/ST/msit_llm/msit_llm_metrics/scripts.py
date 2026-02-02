# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

import random
import logging
import os

LOG_LEVEL = os.environ.get("LOG_LEVEL", 'ERROR')
logging.basicConfig(
    level=LOG_LEVEL,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %I:%M:%S %p'
)

logger = logging.getLogger(__name__)

try:
    from msit_llm import CaseFilter
except Exception as e:
    logger.error("Error occured when loading `msit_llm` modules. Try to reinstall msit llm.")
    raise


def create_random_string(length):
    random_list = [chr(i) for i in range(97, 123)] + [" ", "$", "#", "@", "!", "%", "^"]

    output = []

    # `length` many sentences
    for _ in range(length):
        temp_str = ""

        # each sentences are made up of random words
        for _ in range(random.randint(0, 1024)):
            temp_str += random.choice(random_list)

        output.append(temp_str)

    return output


if __name__ == "__main__":
    case_filter = CaseFilter()
    logger.debug("Successfully created an instance from CaseFilter.")

    case_filter.add_metrics(
        accuracy=None,
        rouge=None,
        rouge_1=None,
        rouge_2=None,
        bleu=None,
        bleu_2=None,
        bleu_3=None,
        bleu_4=None,
        edit_distance=None,
        relative_abnormal=None,
        relative_distinct=None,
        relative_distinct_1=None,
        relative_distinct_2=None,
        relative_distinct_3=None,
        relative_distinct_4=None,
    )
    logger.debug("Successfully add metrics.")

    # 20 batches of sentences for each
    model_prompts = create_random_string(20)
    model_outputs = create_random_string(20)
    references = create_random_string(20)
    logger.debug("Successfully created random test inputs, outputs and references.")

    case_filter.apply(model_prompts, model_outputs, references)
    logger.debug("Successfully filtered bad cases.")

    for file in os.listdir():
        if file.endswith(".csv"):
            exit(0)

    exit(1)