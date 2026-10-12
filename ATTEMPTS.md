# FACT 复现与实验记录（简版）

> 详细代码解读、命令、踩坑见 `FACT_NOTES.md`。本文只记"做过什么、结果是什么、下一步"。
> 时间：2026-10-04 ~ 10-05。机器：grogu-4-23（Blackwell 96GB）为主，grogu-4-8（3090 24GB）为辅。

## 一、已完成

| # | 尝试 | 结果 |
|---|---|---|
| 1 | clone `Bariona/FACT`，建 conda 环境 `fact` | 成功；Blackwell 需要 torch 2.7.1+cu128（官方 cu124 没有 sm_120 kernel） |
| 2 | 读代码，理清 policy 与 action-conditioned 世界模型如何共训、failure/success 如何共训 | 一个 Wan2.2-5B DiT，序列 `[state\|ref\|A\|G\|S\|V\|I]` + teacher-forcing mask；failure 只屏蔽 action loss，value 目标加惩罚 |
| 3 | 流程冒烟：数据预处理 → 40 步训练 → 推理 server | 全部跑通；单卡 ZeRO-2 + optimizer offload 约 0.4 s/step |
| 4 | 官方 `fact-wam` 闭环评测（RoboTwin, demo_clean, action-only） | beat_block_hammer 100 episode：**81%** |
| 5 | 官方 ckpt best-of-N（N=4）对照 N=1 | 见下表，**没有可见提升** |
| 6 | 自己采集 failure rollout（官方 ckpt, beat_block_hammer, 300 episode） | 240 成功 / **60 失败**，已转成 LeRobot v2.1（`datasets/RoboTwin/Fail/beat_block_hammer`） |
| 7 | 从官方 ckpt 微调 3000 步：只用 expert 对 expert+60 失败 | 无提升；加 failure 后 value 整体被抬高（见下） |
| 8 | 核对超参数与论文 | 8 项一致，3 项不一致（见下） |
| 9 | 推到 GitHub：`xiaochy/FACT-modified` main | 已 push（1 个提交，7 个文件） |
| 10 | 写 GAP 迁移设计（把 FACT 因果结构搬进 GAP） | 仅设计，`GAP-...-factcausal/FACT_ADAPTATION.md`，未实现；方向已被 #13 取代 |

## 二、关键结果

**best-of-N（官方 ckpt，N=1 对 N=4，成功 episode 数）**

| 任务 | episode 数 | N=1 | N=4 |
|---|---|---|---|
| beat_block_hammer | 100 | 81 | 81 |
| beat_block_hammer（早期） | 50 | 42 | 36 |
| blocks_ranking_size | 50 | 18 | 24 |
| hanging_mug | 50 | 7 | 7 |
| move_stapler_pad | 50 | 24 | 27 |
| handover_block | 50 | 24 | 19 |
| pick_diverse_bottles | 50 | 29 | 25 |
| turn_switch | 100 | 29 | 35 |
| put_bottles_dustbin | 100 | 86 | 83 |
| stamp_seal | 100 | 76 | 75 |
| place_mouse_pad | — | 失败（grogu-4-8 上 server 加载超时），无结果 | — |

合计（9 个任务，去掉重复的早期 beat_block_hammer 那行，共 650 个 episode）：N=1 为 374（57.5%），N=4 为 376（57.8%），逐任务方向不一致；单格标准误差 ±4~7 个百分点。

**微调对照（beat_block_hammer，100 episode，3000 步，从官方 ckpt 出发）**

| checkpoint | N=1 | N=4 |
|---|---|---|
| 官方 | 81% | 81% |
| 微调：expert | 77% | 82% |
| 微调：expert + 60 失败 | 74% | 76% |

value 分布（N=4 日志）：官方有 12% 的值 > 1.0（失败惩罚区间），中位数 0.19；加失败微调后 **100%** 的值 > 1.0（中位数 1.65）——value head 被整体抬高、失去区分。

## 三、结论与保留意见

- 仿真上 **best-of-N 没有提升**；论文也只在真机报告了 BoN（89%→92%，每格 20 次试验，统计误差同量级）。
- 官方 ckpt 的 value 分布像是见过失败惩罚（你在 HF 上看到它是 failure co-train 的），但我没在模型卡里找到这句话，属**间接证据**。
- 我的微调实验**不能检验"failure 共训是否有用"**：底座本身可能已见过 failure；单任务、60 个失败、3000 步，规模远小于论文（约 1.3K 失败、50 任务）；失败起始帧按作者代码默认整条标为失败。
- 论文没说明如何判断失败起始帧（"when available"）；代码缺失时默认 `failure_active_from_frame=0`，与我们一致。
- **超参数核对**：一致 — 初始化 Wan2.2、AdamW、LR 2e-5/2e-4、H=48、λ_fail=1、视频帧偏移、20 步去噪；**不一致** — action loss 权重论文 20 / 代码 10，value 权重论文 1 / 代码 0.4，代码多了论文没有的 future_state 项；另外论文的 value 目标截断为 0，代码是"剩余比例 + 1"（我对公式的解读，未验证）。
- value 的目标是训练时按 episode 长度现场算的（`(L−t_future−1)/(L−1)`，失败再 +1），不需要预先准备数据。

## 四、进行中：三个 place_bread_skillet 模型

目标：比较"future 用视频"和"future 用 3D track"。三个模型都从 Wan2.2 底座、单卡训练，评测都是 3 个种子（`SEED=0/1/2`）× 50 episode，只用 expert 数据（阶段一）。

| 模型 | 数据 | future | job / 状态 |
|---|---|---|---|
| **A** 视频版基线 | robotwin-v2：Clean 50 + Randomized 500（550 条） | 视频 | 训练中（job 3988939，24000 步，约 0.59 s/步）；评测 job 3988940–42 用依赖挂在其后 |
| **C** 视频版对照 | 你自己的 100 条 demo（`RoboTwin/data/place_bread_skillet/demo_clean_depth`） | 视频 | 已提交 job 3989088，20000 步 |
| **B** track 版 | 同样这 100 条 demo | **3D track**（48 步） | 代码已写好并通过单元检查；正在生成 track，之后打包、冒烟、提交 |

**只有 C 对 B 是"其他都不变"的对照**；A 的数据多 5 倍（550 条），两者不能直接比。

### 阶段一结果（2026-10-06，3 个种子 × 50 episode，N=1，demo_clean）

| 模型 | 数据 | 步数 | seed 0 / 1 / 2 | 平均 |
|---|---|---|---|---|
| A 视频版 | 550 条 | 24000 | 26% / 32% / 38% | **32.0%**（48/150） |
| C 视频版 | 你的 100 条 | 20000 | 2% / 4% / 2% | **2.7%**（4/150） |
| B track 版 | 同 100 条 | 20000 | 2% / 2% / 2% | **2.0%**（3/150） |

- **B 对 C 没有信息量**：两者都在地板上（3/150 对 4/150）。
- A 远低于官方 ckpt 在该任务上的成绩（论文约 80%），但 A 和 B/C 用同一套训练代码，说明差别主要来自数据与训练量（推断，未验证）：A 只见了约 19 万个样本（约 2 个 epoch，论文约 3800 万）；B/C 只有 100 个场景，约训 10 个 epoch，可能记住训练场景。
- 诊断评测（已提交，待结果）：官方 ckpt 在该任务上（确认评测流程）；C/B 在**训练用过的种子**上（`SEED=-1`，区分泛化问题和流程问题）；A 的 8000/16000 步 checkpoint（学习曲线）。

### track 版的做法（分支 `vera-dev/fact-future-track3d`，worktree `../FACT-track3d`）

- **数据**：track 是仿真器真值（`scripts/generate_gt_tracks_rgb.py`：重放 episode 的缓存关节轨迹，用每个点所属连杆的精确位姿得到未来位置，不是 CoTracker 估计）。你已有的 `GAP/data/tracks*` 文件和现在这批原始 episode **长度对不上**（应是另一批 demo 生成的），所以我**重新生成**：100 条 demo、**48 步**（和动作块一致，GAP 原来是 16 步）、不抽 Pi3/DINO 特征，4 个分片并行（每片约 5 分钟）。track 形状 `[T, 391, 48, 3]`，391 = 头部相机 17×23 的 patch；只有约 7.5% 的 patch 有效（机器人和可动物体，静态背景无效），loss 只算有效项。
- **转换**：`scripts/raw_to_staging.py` + `build_failure_dataset.py --include_success` 把原始 HDF5 转成 FACT 的 LeRobot 格式（14 维关节 state/action，`action[t]=state[t+1]` 与发布数据一致；三路 320×240 相机；`seen` 指令随机取一条）；T5、norm stats、VAE latent 已算完。
- **模型**：`future_image` token 换成 track token（每个 patch 一个：MLP 编码未来 48×3 位移 + MLP 编码 patch 中心 3D 坐标作位置；输出头解码回 48×3，无旋转位置编码）。teacher-forcing mask、per-token timestep、value、future_state 都不变；train 里 `track_loss` 取代 `visual_loss`（权重同为 1.0，只算有效项的 flow-matching 速度 MSE）。
- **推理**：Stage 1（出动作）和 Stage 2（算 value）都不经过 track token，推理代码不用改；track 头只是被加载时忽略。
- **检查**：`scripts/test_track_branch.py` 在小模型上验证了：改带噪 track/value/future_state → action 预测不变；改带噪 action → track 预测不变；改干净动作 → track 预测变化；梯度能到三个新模块。

### 阶段二（failure）

按论文 Algorithm 2：先用阶段一的模型 rollout，收集失败，再 expert + failure 继续训。评测用 `SEED=0/1/2`，所以采集 failure 要用 `SEED≥3`。视频版（A/C）用我写好的采集脚本即可；track 版（B）的 failure 样本也需要 track 真值，需要在仿真里重放被执行的动作来生成。你之前 GAP 的 rollout（`results_chain_pilot/place_bread_skillet`，30 个 episode、12 个失败）是末端位姿 16 维、按 16 步分块、只有单路视频，不能直接给 FACT 用。

## 五、踩坑（精简）

- 同节点 job 共享 `127.0.0.1`，固定端口会串到别人的 server → 现在端口按 job 号唯一。
- grogu-4-8 只有 24GB：5B 模型 + T5 放不下，T5 要放 CPU，首次请求慢，需把超时调到 600s；该节点上的 job 明显更慢，且仍可能因 server 加载超时失败。
- grogu-4-13 实际是 12GB 的 3080 Ti（memory 里已记录），不可用。
- `sbatch --export` 的值里不能有逗号。
- wandb 强制登录，配置里设 `log_with=None`。
- NFS 上大目录操作和 `git worktree add` 很慢，放后台。

## 六、下一步

1. 训练 place_bread_skillet 的视频版基线（等 VAE latent 完成）。
2. 弄清 track hdf5 的位置与结构，把这 100 条 demo 转成 FACT 的 LeRobot 格式并对齐 track，再实现 track 分支（token 化、decoder、loss、数据读取），同数据上训练并对比。
3. 评测：3 个种子 × 50 episode，报告均值 ± 标准差。
4. 视情况补：failure 采集（用自己训出的模型）、论文权重（action 20 / value 1）对照。
