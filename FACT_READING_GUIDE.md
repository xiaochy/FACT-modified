# FACT 代码阅读路线

> 行号对应主仓库 `FACT`（`main` 分支，未改动的原版代码）。`FACT-track3d` 分支改过几处，行号会有偏移。
> 如果时间很紧，只读 **第 1、3、4 步**（配置 → `forward_step` → mask 与 `_forward_train`），它们已经包含方法的全部核心。

## 0. 一张图：一个样本怎么流动

```
磁盘                                  数据层                                  训练
parquet(state/action)  ─┐
mp4(3 个相机)→ VAE latent 缓存 ─┼→ LeRobotDataset ─→ Sampler ─→ Transform ─→ collate ─→ CasualWATrainer.forward_step
T5 embedding(.pt)       ─┤   (读一帧 + 未来 48 步动作)   (按任务/episode 采样)  (归一化、delta、value、mask)      │
norm_stats_delta.json   ─┘                                                                                      ▼
                                                                       CasualWorldActionTransformer._forward_train
                                                                       (拼 token → 30 层 DiT → 各个 head) → 4 个 loss

推理： RoboTwin 仿真 → ModelClient(evaluation/robotwin) ──socket──▶ scripts/inference_server.py → WAPipeline
                                                                       → transformer 的 Stage 1（出动作）[+ Stage 2（value）]
```

## 1. 目录地图

| 目录 / 文件 | 行数 | 作用 |
|---|---|---|
| `world_action_model/configs/robotwin.py` | 424 | **所有训练配置**，读代码的第一站 |
| `world_action_model/models/transformer_wa_casual.py` | 2479 | **模型**：token 布局、因果 mask、各个 head、推理路径 |
| `world_action_model/trainer/wa_casual_trainer.py` | 959 | **训练一步**：加噪、拼 token、loss、failure mask |
| `world_action_model/transformers/wa_transforms_lerobot.py` | 555 | **数据变换**：归一化、delta action、value 目标、`action_loss_mask` |
| `world_action_model/pipeline/wa_pipeline.py` | 860 | 推理 pipeline：去噪循环、两阶段 |
| `world_action_model/pipeline/utils.py` | 187 | 归一化工具、逐 token 的 timestep |
| `fact_datasets/datasets/lerobot_dataset.py` | 668 | 读 LeRobot 数据；failure 标记；VAE latent 缓存 |
| `fact_datasets/samplers/episode_mixture_sampler.py` | 168 | 任务/episode 混合采样 |
| `fact_train/` | — | 通用训练框架（accelerate + DeepSpeed、checkpoint、scheduler），**可后看** |
| `scripts/inference_server.py` | 578 | 推理服务，含 best-of-N |
| `evaluation/robotwin/model2robotwin_interface.py` | 602 | 仿真侧 client：重规划、请求、执行动作 |
| `scripts/prepare_robotwin.py` 等 | — | 预处理：norm stats、T5、VAE latent |

## 2. 阅读顺序（8 步）

### 第 1 步　配置：先知道有哪些旋钮（`configs/robotwin.py`，约 15 分钟）
按 `§` 小节读：§2 数据混合（22–60 行附近）、§4 批大小与步数（60 行起）、§5 优化器（LR 2e-5、`ACTION_LR` 2e-4，81 行起）、**§6 模型（96–122 行：`NUM_FRAMES=48`、`ROBOT_ADAPTER_RANK=1024`、四个 loss 权重、`VALUE_PENALTY_SCALE`）**、§8 数据布局（136 行起：三个相机、画布大小）。
读完应能回答：action chunk 多长？四个 loss 的权重是多少？哪些参数用 2e-4？

### 第 2 步　数据：一个样本里有什么
1. `wa_transforms_lerobot.py:363` `__call__`：重点看 **state / delta action / 归一化（473–491 行：481 行做 delta，489–491 行归一化）**、**value 目标与 failure 惩罚（497–511 行）**、`action_loss_mask`（514 行）、最后输出的 key（523–553 行）。
2. `lerobot_dataset.py:601` `_get_local_v2_data`：取当前帧、未来 48 步动作、`future_state_index`（633 行）；**failure 标记**从 `meta/failure_rollouts.jsonl` 读入（527–535 行读取，646–647 行设置 `failure_episode / failure_active`）；`406` 行起是缓存的 VAE latent 的读取。
3. `episode_mixture_sampler.py:13` 类说明 + `147` `__iter__`：先按任务选、再选 episode、再选帧。
读完应能回答：`value` 怎么由 `episode_length` 算出？failure 样本和 expert 样本在数据层有什么区别？动作为什么是"delta"？（关节做差、夹爪用绝对值）

### 第 3 步　训练一步：`forward_step`（`wa_casual_trainer.py:658–866`，**最核心**）
- 658–700：取缓存的 latent（673 行），取 `action / state / future_state / value`。
- 703–718：**给四个目标加噪**（视频 704 行、动作 706 行、future_state、value 716 行），共用一个 σ；`876` `get_timestep_and_sigma`（σ 的采样与 shift）。
- 787 行：`ref_latents`（当前帧的干净 latent）；**790–802：拼逐 token 的 timestep**（`pipeline/utils.py:140` `build_teacher_forcing_per_token_timestep`：干净段为 0，带噪段为 `noise_t`）。
- 803–814：调用 transformer（注意同时传 `action`（带噪）和 `gt_action`（干净））。
- **838–866：四个 loss，`action_loss_mask` 在这里起作用**（`_masked_mean`；视觉 loss 在 846 行，loss 字典在 854 行）。
- `470` `parse_losses`：按 `loss_weights` 加权求和；`_aux_` 开头的只记录不训练。
读完应能回答：为什么 failure 样本只屏蔽 action loss？速度目标为什么是 `noise − x`？

### 第 4 步　模型的灵魂：token 布局与因果 mask（`transformer_wa_casual.py`）
1. `46` `_build_teacher_forcing_mask`：**必读**。序列 `[state | ref | pred_action | gt_action | future_state | value | future_image]`，谁能看谁（docstring 里有表）。
2. `1153` `_encode_robot_targets` → `1222` `_reorder_with_robot_tokens` → `1266` `_split_robot_outputs`：机器人 token 怎么编码、怎么插到视频 token 中间、怎么解码回去。
3. `2294` `_forward_train`：把上面串起来（2353 行 patch embedding → 拼序列 → 逐 token 时间嵌入 → 2393 行构造 mask → 30 层 block → 2449 行 `_split_robot_outputs` 解码各个 head → 2465 行视频输出头）。
读完应能回答：为什么 `pred_action` 看不到 `gt_action`？为什么 future_image 在序列最后？`gt_action` 为什么没有 decoder？

### 第 5 步　模型细节（可略读）
- `986` `__init__`：有哪些模块（`state_encoder / action_encoder / 三个 decoder / action_rope`）；`1092` `reset_action_heads`：新 head 在这里随机初始化。
- `662` `WanTransformerBlock`、`640` `RobotTokenAdapter`：机器人 token 在 FFN 后多一个 rank=1024 的 adapter（视频 token 不经过它）。
- `169` `WanAttnProcessor`：自注意力怎么带 mask（sdpa 或 flex）；`543 / 608` 两种旋转位置编码（视频用 3D，动作用 1D）。

### 第 6 步　推理：两阶段怎么实现
- `1352` `forward`：训练走 `_forward_train`，推理按参数分流。
- `1695` `_forward_inference_action_only`：**Stage 1**，序列只有 `[state | ref | pred_action]`，这是默认部署路径。
- `1528` `_forward_inference_future_action_only`：**Stage 2 的 value 部分**（把 Stage 1 的干净动作放进 `gt_action`，去噪 future_state 和 value，**不含视频**）；`1365` `_forward_inference_future`：含视频的完整版。
- prefix cache：`789 / 866`（共享前缀的 KV 缓存，best-of-N 加速）。
- `pipeline/wa_pipeline.py:372` `__call__`：去噪循环和两阶段的调度（参数 `action_only` 在 393 行、`skip_future_state_value` 在 401 行；约 572 行起按 `action_only` 分流）。

### 第 7 步　推理服务与仿真对接
- `scripts/inference_server.py:184` `get_policy`，里面的 `236` `inference`：加载权重（`146` `_load_transformer_checkpoint`：从 state dict 推断 adapter rank）、反归一化、**把 delta 加回 state（337 行 `add_state_to_action`）**；`297–430` **best-of-N**：采 N 个候选、算 value、428 行 `argmin` 选择、`_action_diversity`（353 行）。
- `evaluation/robotwin/model2robotwin_interface.py`：`210` `ModelClient`、`336` `needs_new_plan`（每 48 步重规划）、`500` `step`、`578` `eval`（仿真每一步调用的入口）。
- `world_action_model/sockets.py`：简单的 TCP 协议。

### 第 8 步　训练框架与预处理（想改训练流程时再看）
`scripts/train.py`（33 行）→ `fact_train/trainers/trainer.py`（1097 行通用训练循环）；预处理 `scripts/prepare_robotwin.py`、`compute_norm_stats.py`、`compute_vae_latents.py`、`compute_t5_embedding.py`。

## 3. 读完后的自测问题

1. 一个训练样本的输入有哪些？哪些是干净的、哪些带噪？
2. 四个 loss 分别监督什么？failure 样本哪些 loss 为 0？
3. `pred_action` 能看到什么、看不到什么？`future_image` 呢？
4. 默认部署（N=1）为什么只跑 Stage 1？best-of-N 为什么要 Stage 2，但不需要视频 token？
5. `value` 的数值是什么含义？best-of-N 选最大还是最小？
6. 为什么动作用 delta？推理时怎么还原成绝对关节目标？

## 4. 想改什么，看哪里

| 想做的事 | 改哪里 |
|---|---|
| 改 loss 权重 / 学习率 / 步数 | `configs/robotwin.py` §5–§6 |
| 改 failure 的处理（惩罚、起始帧） | `wa_transforms_lerobot.py:497–514`、`lerobot_dataset.py:641` |
| 改因果可见性 | `transformer_wa_casual.py:46` `_build_teacher_forcing_mask` |
| 把 future image 换成别的未来表示（比如 3D track） | `_forward_train`、`forward_step`、数据 transform（我们的 `vera-dev/fact-future-track3d` 分支就是这样做的，看 `git diff main` 最直观） |
| 改推理步数 / 候选个数 | `evaluation/robotwin/launch_config.yml`、`deploy_policy.yml` |
| 加相机 / 改画布 | `configs/robotwin.py` §8、`image_layouts.py` |

## 5. 建议的动手练习

1. `python -m scripts.test_dataloader --config world_action_model.configs.robotwin --num_workers 0 --batch_size 2 --num_batches 2`：打印一个 batch 的所有 key 和形状，对照第 2 步。
2. 在 `forward_step` 里打印各个 token 段的长度，对照第 4 步的布局。
3. （我们分支里有）`python -m scripts.test_track_branch`：用一个小模型验证"改什么输入、哪些输出会变"，是理解 mask 最快的办法。
4. 对着 `_build_teacher_forcing_mask` 在纸上画出 7×7 的可见性矩阵，再和代码里的 `-inf` 区域对一遍。

## 6. 已知的"论文与代码不一致"（读代码时别被误导）

- loss 权重：论文 `w_a=20, w_v=w_I=1`；代码 action 10、value 0.4、visual 1，另外多一个论文里没有的 `future_state`（0.4）。
- value 的定义：论文是进度（越大越好、`argmax`），代码是"剩余比例"（越小越好、`argmin`）；失败样本的目标论文是截断后的常数 0，代码是"剩余比例 + 1"。
- `ACTION_LR=2e-4` 在代码里作用于所有机器人 token 相关参数（encoder/decoder/adapter），论文只写了 "action FFN"。
