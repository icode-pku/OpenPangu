# Copyright (c) 2023 Huawei Technologies Co., Ltd.
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
import shutil

import click


opt_type = click.option(
    '-t', '--type', 'query_type', type=click.Choice(['op', 'model']), required=True, help='fast query type.'
)

opt_opp_path = click.option('--opp_path', 'opp_path', type=str, help='opp path, required when type is op')

opt_out = click.option('-o', '--output', 'output', type=str, required=True, help='output file path')


@click.command()
@opt_type
@opt_opp_path
@opt_out
def fast_query(query_type, opp_path, output) -> None:
    cur_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    data_path = os.path.join(cur_dir, 'resource', 'analyze', 'dataset', 'opp', 'opp.json')

    out_path = os.path.realpath(output)
    if data_path == out_path:
        return

    shutil.copyfile(data_path, out_path)


if __name__ == '__main__':
    fast_query()
