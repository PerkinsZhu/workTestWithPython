"""
Created by PerkinsZhu on 2021/12/23 12:17
"""

# 从.csv文件中加载数据
# 导入 csvdir 和 pandas 包 (可以编辑 ~/.zipline/extension.py，也可以直接写在策略中)
import pandas as pd

from zipline.data.bundles import register
from zipline.data.bundles.csvdir import csvdir_equities

# 指定开始和结束时间：
start_session = pd.Timestamp('2016-1-1', tz='utc')
end_session = pd.Timestamp('2018-1-1', tz='utc')
# 然后我们可以传入 .csv 文件路径，用 register() 注册我们的自己编写的 bundle
register(
    'custom-csvdir-bundle',
    csvdir_equities(
        ['daily'],
        '/path/to/your/csvs',
    ),
    calendar_name='NYSE',  # US equities
    start_session=start_session,
    end_session=end_session
)