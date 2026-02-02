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
from ms_service_profiler.exporters.base import ExporterBase

from ..common.utils import logger
from ..common.split_utils import get_service_type


class ExporterDecode(ExporterBase):
    name = "decode_data"
 
    @classmethod
    def initialize(cls, args):
        cls.args = args
 
    @classmethod
    def export(cls, data) -> None:
        cls.args.batch_size = cls.args.decode_batch_size
        cls.args.batch_num = cls.args.decode_number
        cls.args.rid = cls.args.decode_rid
        df = data.get('tx_data_df')
        if df is None:
            logger.error("The data is empty, please check")
            return
        processor = get_service_type(df)
        processor.initialize(cls.args)
        processor.run_split(df, "Decode")
        logger.info("Export decode data successfully.")
