### 晴雨表指标使用说明书

#### 一、核心指标定义与解读​​

以下结合示例数据说明各指标含义：

markdown

复制

`【市场晴雨表 2023-10-05 15:30】
黄金变动: 0.75% → 因子: 1.5
波动率偏度: 左偏 → 因子: -2.0
市场情绪: 0.45
VIX指数: 22.5
综合得分: 0.85 → 风险评分: 0.0`

##### ​​1. 黄金变动因子​​

-   ​​数值示例​​：`0.75% → 因子1.5`
-   ​​含义​​：
    -   ​​黄金价格变动​​：当日黄金价格较前日上涨0.75%。
    -   ​​因子逻辑​​：
        -   若黄金短期（20日）动量向上且MACD金叉（趋势强劲），因子值>1.0（示例中1.5为强烈看涨）；
        -   若动量向下且MACD死叉，因子值<-1.0（强烈看跌）。
-   ​​实战意义​​：
    -   ​​因子>1.0​​：避险资产受追捧，可能预示市场风险偏好下降。

##### ​​2. 波动率偏度因子​​

-   ​​数值示例​​：`左偏 → 因子-2.0`
-   ​​含义​​：
    -   ​​左偏​​：期权市场显示​​低执行价波动率 > 高执行价波动率​​（看跌期权更贵），暗示市场担忧短期下跌风险。
    -   ​​因子逻辑​​：
        -   左偏时因子为负（示例中-2.0为显著风险信号），右偏时因子为正（看涨预期）。
-   ​​实战意义​​：
    -   ​​因子<-1.5​​：警惕市场恐慌情绪，可能触发对冲需求。

##### ​​3. 市场情绪值​​

-   ​​数值示例​​：`0.45`（范围-1.0至1.0）
-   ​​含义​​：
    -   ​​>0​​：市场情绪偏乐观（示例中0.45为中性偏多）；
    -   ​​<0​​：情绪偏悲观。
-   ​​数据来源​​：
    -   模拟舆情分析（新闻、社交媒体情绪指数），实际应用时可接入实时API。

##### ​​4. VIX指数​​

-   ​​数值示例​​：`22.5`
-   ​​分级解读​​：
    -   ​​<15​​：市场过度乐观，潜在风险累积；
    -   ​​15-25​​（示例22.5）：正常波动区间；
    -   ​​>25​​：恐慌情绪主导，可能出现急跌。

##### ​​5. 综合得分​​

-   ​​数值示例​​：`0.85`（范围-3.0至3.0）
-   ​​分级策略​​：
    -   ​​>1.0​​：强烈看涨信号，可增配风险资产；
    -   ​​0.5~1.0​​（示例0.85）：中性偏多，持仓观望；
    -   ​​<-1.0​​：触发风控机制，建议减仓。

##### ​​6. 风险评分​​

-   ​​数值示例​​：`0.0`（范围0-9，0为无风险）
-   ​​触发条件​​：
    -   ​​≥7.0​​（高风险）：`综合得分<-1.5` 或 `VIX>25`，示例中未触发；
    -   ​​≥5.0​​（中风险）：宏观经济数据恶化（如GDP负增长）。
-   ​​警报示例​​：

    markdown

    复制

    `[RISK ALERT] High Risk: Barometer score below -1.5`

* * * * *

#### ​​二、多因子动态权重机制​​

综合得分由四大因子加权计算，权重随VIX变化动态调整：

| VIX区间 | 黄金权重 | 波动率权重 | 情绪权重 | 模型预测权重 |
| --- | --- | --- | --- | --- |
| ​​<15​​ | 50% | 25% | 15% | 10% |
| ​​15-25​​ | 35% | 40% | 15% | 10% |
| ​​>25​​ | 20% | 45% | 25% | 10% |

​​示例计算​​（VIX=22.5，权重35%-40%-15%-10%）：

markdown

复制

`0.85 = 1.5 * 35% + (-2.0)*40% + 0.45 * 15% + ML预测值*10% `

* * * * *

#### ​​三、实战决策指南​​

1.  ​​正向信号组合​​（示例场景）：

    -   ​​黄金因子 >1.0​​ + ​​VIX<25​​ → 增持防御性资产（黄金、债券）；
    -   ​​综合得分0.85​​ + ​​风险评分0.0​​ → 维持当前仓位，监控波动率变化。
2.  ​​风险警报响应​​：

    -   ​​若综合得分<-1.5​​：立即启动对冲（买入看跌期权、增持现金）；
    -   ​​VIX>25且波动率左偏​​：降低杠杆，避免抄底。
3.  ​​情绪与波动率背离​​：

    -   ​​情绪>0.5但波动率左偏​​：市场"虚假繁荣"，需谨慎追涨。

* * * * *

#### ​​四、数据更新与校准​​

-   ​​频率​​：每15分钟更新实时数据（黄金、VIX）；
-   ​​模型再训练​​：每周更新LSTM模型参数，适配市场变化；
-   ​​人工复核​​：当风险评分>7.0时，需人工确认系统逻辑。

* * * * *

​​注​​：本说明书基于示例数据生成，实际数值可能因市场波动而变化，建议结合实时图表（波动率曲面、情绪热力图）综合判断。
