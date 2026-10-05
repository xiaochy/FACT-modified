# FACT 代码笔记：实现、理解方式与使用方法

> Repo: https://github.com/Bariona/FACT （Failure-Aware Causal Training for World-Action Models, CoRL 2026）
> 本地路径：`/project_data/held/vxiao/Bimanual_Manipulation_Project/FACT`
> 本文分三部分：① 具体实现 ② 怎么理解这个 repo ③ 怎么使用。最后一节记录本机跑 pipeline 的进度。

---

## 0. 一句话理解

**一个 Wan2.2-TI2V-5B 的 DiT，同时去噪「动作 + 未来状态 + value + 未来视频」。**
未来视频/状态/value 条件于**干净的真值动作**（teacher forcing），动作本身看不到未来，所以：

- policy 部署时只解码动作（不生成视频）；
- 世界模型的监督（视频/状态/value）反过来塑造共享表示，提升动作质量；
- failure 数据不教动作，只教「做错会怎样」（后果 + 更差的 value）。

---

## 1. 具体实现

### 1.1 文件地图

| 位置 | 作用 |
| --- | --- |
| `world_action_model/configs/robotwin.py` | **所有训练配置**（顶部 USER SETTINGS：路径、GPU、batch、LR、loss 权重、`VALUE_PENALTY_SCALE` 等） |
| `world_action_model/models/transformer_wa_casual.py` (2.5k 行) | `CasualWorldActionTransformer`：token 布局、teacher-forcing attention mask、各 head、prefix KV cache |
| `world_action_model/trainer/wa_casual_trainer.py` | `CasualWATrainer.forward_step`：加噪、拼 token、四个 loss、failure mask |
| `world_action_model/transformers/wa_transforms_lerobot.py` | 数据 transform：归一化、delta action、value 目标、`action_loss_mask` |
| `fact_datasets/datasets/lerobot_dataset.py` | LeRobot 数据读取；读取 `meta/failure_rollouts.jsonl` 并打 failure 标记 |
| `fact_datasets/samplers/episode_mixture_sampler.py` | 按「任务 multiplier × episode 数」→ episode → 帧 的混合采样 |
| `world_action_model/pipeline/wa_pipeline.py` | 推理 pipeline（动作 / 未来状态 / value / 视频分阶段去噪，best-of-N） |
| `scripts/` | `prepare_robotwin`、`compute_vae_latents`、`train`、`inference_server`、`test_dataloader` |
| `evaluation/robotwin/` | RoboTwin 闭环评测（server + client） |
| `fact_train/` | 通用训练框架（accelerate + DeepSpeed、EMA、checkpoint、scheduler） |

### 1.2 Token 布局与 attention mask

序列（`build_teacher_forcing_per_token_timestep` / `_build_teacher_forcing_mask`）：

```
[ state | ref(首帧) | pred_action | gt_action | future_state | value | future_image ]
   P       P            A             G            S            V          I
```

- **Timestep**：`state / ref / gt_action` 为 0（干净）；`pred_action / future_state / value / future_image` 共用同一个 `noise_t`。
- **Attention（Q 行 → 可看的 K）**：

| Query | 能看到 |
| --- | --- |
| P (state+ref) | P |
| A (pred_action) | P + A（A 是 sink，没人能看 A） |
| G (gt_action) | P + G |
| S/V (future_state, value) | P + G + S/V |
| I (future_image) | P + G + S/V + I |

  关键点：**除 A 自己外，所有 token 都看不到 `pred_action`**，因此不会通过 `G` 或后续 token 泄漏预测目标；而世界模型分支都能看到干净的 `gt_action`。
- `gt_action` 与 `pred_action` **共用 action encoder**；`gt_action` 只作 K/V 条件，没有 decoder head。
- 动作相关参数有额外的 FFN adapter（`ROBOT_ADAPTER_RANK=1024`），并且用独立的 `action_lr`（`build_optimizer_param_groups`）。

### 1.3 训练一步（`forward_step`）

1. 对 batch 采一个 `sigma ~ U(0,1)`，经 `flow_shift`(=3.0) 变换；timestep = round(σ·1000)。
2. 四路 flow-matching 加噪：`x_t = noise·σ + x·(1-σ)`，目标 `noise − x`。
   - 视频：latent（首帧作 ref，不加噪）
   - 动作：归一化后的 delta action（48 步 × 14 维）
   - 未来状态：`state[t+48]`（归一化）
   - value：标量 `[-1,1]`
3. 送入 transformer：`ref_latents, noisy_latents, action=noisy_action, gt_action=clean action, state, future_state=noisy, value=noisy`。
4. 四个 loss（MSE），权重见 `robotwin.py`：

| loss | 权重 | 说明 |
| --- | --- | --- |
| `visual_loss` | 1.0 | 未来视频（首帧位置被 `first_frame_mask` 排除） |
| `action_loss` | **10.0** | **masked mean，failure 样本 mask 为 0** |
| `future_state_loss` | 0.4 | 未来 proprio 状态 |
| `value_loss` | 0.4 | 任务进度/代价 |

   `parse_losses` 里按 `loss_weights` 加权求和；`_aux_*` 开头的 key 只记录不训练。
5. 默认使用预缓存 VAE latent（`visual_latents`），要求 `expand_timesteps=True`。

### 1.4 Policy 与 world model「怎么一起训」

没有两个网络、没有两阶段训练——**同一个前向、同一个 loss 求和、同一组参数**：

- 动作去噪（policy）= `pred_action` token 的 flow-matching；
- action-conditioned 世界模型 = 未来视频/状态/value 的 flow-matching，**条件为 `gt_action` token**；
- 梯度都回到同一个 DiT 主干，世界模型的预测任务充当对动作表示的辅助监督。
- 推理时分阶段（代码里 Stage1 只有 state+pred_action；Stage2 加入 gt_action(=Stage1 输出) 预测未来状态/value/视频）。`--skip_future_state_value` 只出动作。

### 1.5 Failure / Success 数据怎么一起训

数据标注：任务目录下的 `meta/failure_rollouts.jsonl`，每行
`{"episode_index": i, "failure_episode": true, "failure_active_from_frame": k}`。
`lerobot_dataset.py` 据此给每个样本设：

- `failure_episode`：整个 episode 是否为失败 rollout；
- `failure_active = failure_episode and frame_index >= failure_active_from_frame`。

`wa_transforms_lerobot.py` 中：

| | expert / success | failure rollout |
| --- | --- | --- |
| `action_loss_mask` | 1 | **0**（不模仿失败动作） |
| visual / future_state loss | 训练 | 训练（学习失败后的真实后果） |
| value 目标 | `(L − t_future − 1)/(L − 1)` ∈ [0,1]（剩余进度，越大越远离完成） | `failure_active` 时再 `+ VALUE_PENALTY_SCALE`(默认 1.0) → (1,2]，然后按 `value_min/max` 归一到 [-1,1] 并 clamp |

- 所以 `compute_norm_stats.py` 给 value 留了范围：`value.max ≥ 1 + VALUE_PENALTY_SCALE`，否则 failure 的 value 会被 clamp 抹平。
- 混合方式：`EpisodeMixtureSampler` 不区分 failure/success，比例完全取决于各任务目录中的 episode 数与 `dataset_multipliers`。想调 failure 比例 → 调 multiplier 或增删 failure episode。
- 监控：`_aux_action_loss_mask_rate` = batch 中 failure 的占比。
- 推理端的好处（论文）：value head 可为 N 个采样动作候选打分（best-of-N）。

---

## 2. 我怎么理解这个 repo（读码建议）

1. **先读 `forward_step`**（`wa_casual_trainer.py:658`）。整个方法的核心就在那 200 行：加噪、拼 token、四个 loss、mask。
2. 再读 `_build_teacher_forcing_mask`（`transformer_wa_casual.py:46`），理解「谁能看谁」。这是方法与普通 video DiT 最大的区别。
3. 读 transform（`wa_transforms_lerobot.py:440-555`）看每个监督信号是怎么从 raw data 构造的（delta action、归一化、value、failure mask）。
4. `configs/robotwin.py` 当作「实验开关面板」读：loss 权重、LR、数据集 multiplier、DeepSpeed stage 都在这。
5. 其余（`fact_train/`、sampler、collator）是通用训练基础设施，可后看。

**设计上值得记住的几点**

- 统一表示：动作/状态/value 都被当作「短的 token 序列」和视频 token 放进同一个 DiT，用 mask 控制信息流，而不是多个模块。
- 因果方向 act → imagine：未来依赖动作，动作不依赖未来。这保证部署时能丢掉视频分支。
- failure 不是「负样本」，而是「**有后果的样本**」：不学动作，但视频/状态/value 全部照常监督。这与 reward-weighted / 过滤式做法不同。
- 动作是 **delta action**（相对当前 state 的差，`delta_template["idx"]` 指定哪些维度做 delta），并 z-score 归一化；norm stats 来自 `prepare_robotwin`。

---

## 3. 怎么使用

### 3.1 环境（本机已完成）

本机 GPU 是 RTX PRO 6000 **Blackwell (sm_120)**，repo 固定的 `torch 2.6 + cu124` 不含 sm_120 kernel，所以改用：

```bash
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate fact          # python 3.11, torch 2.7.1+cu128, deepspeed 0.18.9, lerobot 0.3.2
```

安装脚本：`/scratch/vxiao/tmp/setup_fact.sh`（与官方 `setup_env.sh` 的区别仅为 torch 版本）。

### 3.2 模型与数据

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir ./models/Wan2.2-TI2V-5B-Diffusers   # 34GB
huggingface-cli download Bariona/robotwin-v2 robotwin-v2.tar --repo-type dataset --local-dir ./datasets   # 85GB
tar -xf datasets/robotwin-v2.tar -C datasets      # -> datasets/RoboTwin/{Clean,Randomized}/<task>/
huggingface-cli download Bariona/fact-wam --local-dir ./models/fact-wam                                    # 官方 checkpoint, 10GB（可选）
```

### 3.3 预处理 → 训练 → 推理 → 评测

```bash
# 1. norm stats + T5 缓存（可用 --dataset_glob 'Clean/*' 选子集）
python -m scripts.prepare_robotwin --robotwin_root ./datasets/RoboTwin \
  --wan_model_path ./models/Wan2.2-TI2V-5B-Diffusers --output_dir ./artifacts/robotwin
# 2. VAE latent 缓存（batch_size 与训练 BATCH_SIZE_PER_GPU 对齐）
python -m scripts.compute_vae_latents --batch_size 32
# 3. dataloader 检查（可选）
python -m scripts.test_dataloader --config world_action_model.configs.robotwin --num_workers 0 --batch_size 2 --num_batches 2
# 4. 训练：先改 configs/robotwin.py 顶部 USER SETTINGS
python -m scripts.train --config world_action_model.configs.robotwin.config
# 5. 推理服务
python -m scripts.inference_server --model_id ./models/Wan2.2-TI2V-5B-Diffusers \
  --transformer_path ./experiments/robotwin/models/<ckpt>/transformer \
  --stats_path ./artifacts/robotwin/norm_stats_delta.json --port 8093
#    用官方 ckpt: --transformer_path ./models/fact-wam/transformer --stats_path ./models/fact-wam/norm_stats_delta.json
# 6. 评测（RoboTwin 另建 conda env，并先 apply evaluation/robotwin/robotwin_test_num.patch）
bash evaluation/robotwin/launch_server.sh
bash evaluation/robotwin/launch_client.sh beat_block_hammer demo_clean
```

常用环境变量：`ROBOTWIN_DATASET_IDS` / `ROBOTWIN_DATASET_GLOBS`（选任务）、`FACT_USE_CACHED_VAE_LATENTS=0`（用原始视频训练）。

### 3.4 想自己加 failure 数据时

1. 在目标任务目录写 `meta/failure_rollouts.jsonl`（格式见 1.5），episode 本身按 LeRobot 格式并入该任务数据集；
2. 重新跑 `prepare_robotwin`（norm stats 的 value 范围需覆盖 `1 + VALUE_PENALTY_SCALE`）和 `compute_vae_latents`；
3. 用 `test_dataloader` 看 `failure_episode_mask / action_loss_mask`，训练时看 `_aux_action_loss_mask_rate` 是否符合预期；
4. 想调比例：改 `ROBOTWIN_DATASET_MULTIPLIERS`。

> 已确认：官方 `robotwin-v2.tar` **不含** `failure_rollouts.jsonl`（见 4.1）。采集 failure rollout 的脚本和流程见 4.7，我们已经自己做了一套。

---

## 4. 本机运行记录（2026-10-04 ~ 10-05）

### 4.0 总览

| 步骤 | 状态 | 备注 |
| --- | --- | --- |
| clone + conda env `fact` | 完成 | torch 2.7.1+cu128（Blackwell），脚本 `/scratch/vxiao/tmp/setup_fact.sh` |
| 下载 Wan2.2（34GB）/ `robotwin-v2.tar`（85GB）/ `fact-wam`（9.7GB） | 完成 | tar 里自带 T5 缓存，`prepare_robotwin` 可加 `--skip_t5` |
| 解压 | 仅 2 个任务 | `beat_block_hammer`、`click_bell`（每任务 50 episode，约 5.7k 帧） |
| 预处理（norm stats / VAE latent / dataloader） | 完成 | 见 4.2 |
| 训练冒烟（40 步）/ 推理冒烟 | 通过 | 见 4.3、4.4 |
| 官方 ckpt 闭环评测（含 best-of-N） | 完成 | 见 4.6 |
| failure rollout 采集（60 个） | 完成 | 见 4.7 |
| 从官方 ckpt 微调（expert / expert+failure） | 完成 | 见 4.8 |
| 微调后评测（N=1 / N=4） | 完成 | 见 4.9 |

### 4.1 发现：官方数据里没有 failure 数据

- `tar -tf robotwin-v2.tar` 里搜 `fail` / `rollout`：**0 个匹配**，没有 `failure_rollouts.jsonl`。发布数据是 expert demo：`Clean`（每任务 50 条，共 2500）+ `Randomized`（每任务 500 条，共 25000，论文数字）。
- HuggingFace 上作者只发布了 **一个模型** `Bariona/fact-wam`（多任务，50 个任务共用一个权重，没有按任务分开的权重），数据集还有 `Bariona/robotwin`（79.8GB）和 `Bariona/robotwin-v3.0`（84GB，LeRobot v3.0 格式？），从文件列表看不出里面有没有 failure 数据，**没有下载核实**；README 只支持 v2.1。
- `world_action_model/robotwin_utils.py` 里默认排除的顶层分组是 `EvalRollouts`、`Fail`，说明作者自己的布局是 `RoboTwin/Fail/<task>/`（含 `meta/failure_rollouts.jsonl`），选取用 `Fail/*`。我们沿用这个布局。
- 官方 ckpt 的 `transformer/config.json` 写 `robot_adapter_rank: 0`，但权重里**有** adapter（30 block × 4 个张量，rank 1024，总参数 5.19B）；`inference_server.py:156` 会从 state dict 推断 rank 并覆盖 config，所以评测用上了 adapter。**微调时要保持 `ROBOT_ADAPTER_RANK=1024`**。

### 4.2 预处理

```bash
export ROBOTWIN_ROOT=$PWD/datasets/RoboTwin
python -m scripts.prepare_robotwin --robotwin_root ./datasets/RoboTwin \
  --wan_model_path ./models/Wan2.2-TI2V-5B-Diffusers --output_dir ./artifacts/robotwin \
  --dataset_glob 'Clean/*' --skip_t5
python -m scripts.compute_vae_latents --devices cuda:0 --dataset_glob 'Clean/*' --batch_size 8   # 100 episode, 9537 样本, 503MB, 161s
python -m scripts.test_dataloader --config world_action_model.configs.robotwin --num_workers 0 --batch_size 2 --num_batches 2
```

- `artifacts/robotwin/norm_stats_delta.json` 只基于这 2 个任务，**仅供冒烟**。微调/评测官方 ckpt 一律用 `models/fact-wam/norm_stats_delta.json`（value 范围含 failure 惩罚，max=2.0）。
- 每样本 self-attn 仅 195 token（state 1 + ref 72 + action 48 + future_state 1 + value 1 + noisy 视频 72），T5 cross-attn 64。

### 4.3 训练冒烟（单卡）

配置 `world_action_model/configs/robotwin_smoke.py`（从 `robotwin.py` 用 sed 复制）：`GPU_IDS=[0]`、optimizer offload、batch 4、40 步、`log_with=None`。

- 约 **0.4 s/step（batch 4）**；NFS 上加载 5 个 shard 的 transformer 约 4.5 分钟。
- 40 步后各 loss 正常有值，`_aux_action_loss_mask_rate=0`；checkpoint 约 30GB/个。

### 4.4 推理冒烟

`python -m scripts.inference_server --transformer_path <ckpt>/transformer --stats_path <stats> --port 8093 --return_images`。
client 用 `BaseTcpInferenceClient.call_endpoint("inference", req)`，req：`observation.state`(14)、`observation.images.cam_high`(3×192×256, 0–1)、`cam_left_wrist` / `cam_right_wrist`(3×96×128)、`instruction`。
返回 `action [48,14]`、`future_state [1,14]`、`value [1,1]`、`images`、`values_per_sample`、`selected_index`、`action_diversity`。首次请求约 218s（CPU 上加载 T5），之后约 1.7s。

### 4.5 评测设置

- 补丁：`git -C RoboTwin apply FACT/evaluation/robotwin/robotwin_test_num.patch`（只改 `script/eval_policy.py` 一行，默认 100 不变；已对项目里的 `RoboTwin/` 打了）。
- 脚本：`eval_runs/fact_eval_pilot.sbatch`（同一 GPU 上起 server + client），参数用 `--export=ALL,TASK_NAME=..,TEST_NUM=..,SEED=..,TRANSFORMER_PATH=..,STATS_PATH=..`；BoN：`SKIP_FUTURE_STATE_VALUE=0,ENABLE_SAMPLE=true,BEST_OF_N=4`。
- 脚本会按 GPU 显存自动选 T5 放哪（>40GB 放 cuda，否则 cpu），端口按 job 号唯一（`20000 + JOBID % 20000`），server 超时默认 600s。
- 默认只做 **action-only 推理**（`--skip_future_state_value`），不跑世界模型分支；BoN 时才开 value。
- 结果：`RoboTwin/eval_result/<task>/evaluation.robotwin.model2robotwin_interface/<config>/fact/<时间戳>/_result.txt`。
- 评测种子从 `100000*(1+SEED)` 开始；**评测用 `SEED=0`，failure 采集用 `SEED≥1`，种子不重叠**。

### 4.6 官方 ckpt 评测结果（demo_clean）

**beat_block_hammer（100 episode，SEED=0）**：N=1 为 81/100，N=4 为 81/100。
（之前 50 episode 时是 84% 对 72%，样本加倍后差距消失，说明那是采样噪声。）

**多任务（官方 ckpt，50 episode/格，SEED=0）**

| 任务 | N=1 | N=4 | 差（N=4 − N=1，episode 数） |
| --- | --- | --- | --- |
| beat_block_hammer | 42/50 | 36/50 | −6 |
| blocks_ranking_size | 18/50 | 24/50 | +6 |
| hanging_mug | 7/50 | 7/50 | 0 |
| move_stapler_pad | 24/50 | 27/50 | +3 |
| handover_block | 24/50 | 19/50 | −5 |
| pick_diverse_bottles | 29/50 | 25/50 | −4 |
| **合计** | **144/300（48.0%）** | **138/300（46.0%）** | −6 |

方向不一致，每格标准误差约 ±7 个百分点。**官方 ckpt 上 best-of-N 没有可见提升。**

### 4.7 failure rollout 采集

- 采集模块：`evaluation/robotwin/collect_failure_interface.py`（作为 `POLICY_NAME` 传入）。包装 FACT 的 client，**关掉 low-frequency RGB**（否则只在每 48 步重规划时才渲染），每步记录三路 RGB、14 维 state、执行的 action；episode 结束若失败就写 staging 目录（3 个 mp4（ffmpeg）+ `data.npz` + `meta.json`）。最后一个 episode 靠 `atexit` 落盘。环境变量：`FACT_COLLECT_DIR`、`FACT_COLLECT_TAG`、`FACT_COLLECT_SAVE_SUCCESS`。
- 转换：`scripts/build_failure_dataset.py`（`fact` 环境，RoboTwin 环境没有 pyarrow）→ LeRobot v2.1：parquet、三路视频、`episodes.jsonl`、`tasks.jsonl`、`failure_rollouts.jsonl`（`failure_active_from_frame=0`，**整个 episode 都当失败，是简化**）。幂等，重复运行会跳过已导入的。
- 之后需要：`python -m scripts.compute_t5_embedding --root <dir> --wan_path models/Wan2.2-TI2V-5B-Diffusers --device cuda:0` 和 `python -m scripts.compute_vae_latents --dataset_id Fail/<task>`。
- 实际采集：官方 ckpt，N=1，beat_block_hammer，4 个 job（`SEED=1..4`，各 75 episode）。

| job | 成功率 | 失败 rollout |
| --- | --- | --- |
| s1 | 63/75（84.0%） | 12 |
| s2 | 61/75（81.3%） | 14 |
| s3 | 65/75（86.7%） | 10 |
| s4 | 51/75（68.0%） | 24 |
| **合计** | **240/300（80%）** | **60** |

- 输出：`datasets/RoboTwin/Fail/beat_block_hammer`（60 episode，24000 帧，726MB 视频，VAE latent 1.27GB）。失败 episode 平均约 400 帧（跑到步数上限才判失败），expert 约 114 帧。
- 采集视频是 320×240（仿真相机），expert 视频是 640×480；训练 transform 都会缩到同一画布，宽高比一致。

### 4.8 微调（从官方 ckpt 出发，单任务 beat_block_hammer）

- 配置：`world_action_model/configs/robotwin_ft.py`（从 `robotwin.py` 复制）：单卡、ZeRO-2 + optimizer offload、batch 8、3000 步、warmup 100、LR 1e-5 / action LR 1e-4、`log_with=None`；`CHECKPOINT` 指向 `models/fact-wam/transformer`（`load_checkpoint` 在 `reset_action_heads` 和 adapter 创建**之后**才加载，所以已训练的 head 不会丢）。环境变量：`FT_MAX_STEPS`、`FT_BATCH`、`FT_FAIL_MULT`（Fail 数据集的采样 multiplier，默认 0.3）、`FT_INIT_CKPT`。
- 提交：`eval_runs/fact_finetune.sbatch`，`FT_NAME=expert` 或 `expert_fail`。数据集 id 在脚本里按 `FT_NAME` 设置（**不能通过 `sbatch --export` 传，因为 id 里有逗号，会被拆开，悄悄变成只用 expert**）。
- 必须用官方 `models/fact-wam/norm_stats_delta.json`。
- 两个版本：
  - `expert`：只用 `Clean/beat_block_hammer`（50 episode）。
  - `expert_fail`：再加 `Fail/beat_block_hammer`（60 个失败），multiplier 0.3，失败样本约占 25%（日志里 `_aux_action_loss_mask_rate` 在 0.21–0.30 之间波动，验证了 action loss 屏蔽生效）。
- 各约 36 分钟、约 0.56 s/step；checkpoint 约 30GB/个：`experiments/ft_expert/models/checkpoint_epoch_5_step_3000/`、`experiments/ft_expert_fail/models/checkpoint_epoch_1_step_3000/`。
- 最后一步：expert 版 value_loss ≈ 0.0001；expert_fail 版 value_loss ≈ 0.02（**value head 没收敛**），action_loss 都约 0.001。

### 4.9 微调后评测（beat_block_hammer，100 episode，SEED=0）

| checkpoint | N=1 | N=4 |
| --- | --- | --- |
| 官方 `fact-wam` | 81% | 81% |
| 微调：只用 expert | 77% | 82% |
| 微调：expert + 60 个失败 rollout | 74% | 76% |

标准误差约 ±4 个百分点，**差异都在噪声范围内，没有证据显示 failure 共训或 BoN 有帮助**。已核对每个 server 加载的 `--transformer_path` 与预期一致。

N=4 时候选间 value 差距（server 日志，约 415–447 次调用）：

| checkpoint | 候选间差距中位数 | 差距均值 | value 均值 |
| --- | --- | --- | --- |
| 官方 | 0.018 | 0.140 | 0.365 |
| 微调 expert | 0.012 | 0.109 | 0.355 |
| 微调 expert+fail | **0.076** | 0.139 | **1.587** |

加了失败数据后候选间 value 的区分度确实变大，但 value 均值被整体抬高（失败 episode 帧数远多于 expert，且整个 episode 都被标为失败），而成功率没有因此提高。**没有分析过 value 与最终成败的相关性。**

**为什么这个实验不能当成对论文的检验**：单任务、60 个失败、3000 步（论文：50 任务、约 1.3K 个失败）；value head 没收敛；beat_block_hammer 已约 81%，提升空间小；失败标注粗糙；不知道官方 ckpt 是否用过失败数据。

### 4.10 踩坑汇总

- **wandb**：`scripts/train.py` 在 `log_with="wandb"` 时强制校验登录，`WANDB_MODE=offline` 绕不过；配置里设 `log_with=None`。
- **GPU 没有 cgroup 隔离**：`CUDA_VISIBLE_DEVICES` 为空，slurm 只分 1 卡也能看到别人的卡；一律显式 `cuda:0` / `GPU_IDS=[0]`。
- **端口冲突**：同一节点上的 job 共享 `127.0.0.1`，server 固定 8093 时，后启动的 job 绑定失败，client 却连上别人的 server，结果全部不可信。现在每个 job 用唯一端口。
- **T5 放 CPU 超时**：grogu-4-8（3090，24GB）放不下 5B 模型 + T5，T5 放 CPU 时第一次请求超过默认 120s 导致 job 失败；现在超时 600s，且尽量固定在 grogu-4-23（96GB，T5 放 GPU）。
- **节点 GPU 型号不一致**：申请 `grogu-4-13` 被分到 12GB 的 3080 Ti（sinfo 写 A6000），放不下。
- **OIDN 报错**：grogu-4-23 上 SAPIEN 日志里有大量 `OIDN Error: invalid handle`，但评测可以完整跑完；是否影响渲染质量没核实。
- **`sbatch --export` 的值里不能有逗号**。
- **server 日志要 `PYTHONUNBUFFERED=1`**，否则内部报错被缓冲吞掉。
- `launch_server.sh` 里 `conda env list | awk` 会打印一段 BrokenPipe 的 conda ERROR REPORT，是无害噪声，别当成真错误。
- 客户端：server 没起来时 `BaseTcpInferenceClient()` 构造函数就 connect 失败，需要重试循环。
- NFS 上 `git worktree add` 和大目录转换很慢，要放后台。
- 磁盘：`experiments/` 下每个 ckpt 约 30GB，`experiments/smoke` 用完可以删。

---

## 5. 论文要点与常见问题（arXiv 2608.10232）

- **多任务模型**：一个模型训 50 个 RoboTwin 任务，混合 clean（2500 条）和 randomized（25000 条）；任务靠 T5 指令区分。
- **主要数字（RoboTwin 平均）**：w/o video co-train 81.8%，Ours 85.6%，Ours w/ failure 87.5%（仿真加约 1.3K 个失败 rollout），Motus 87.8%。真机已见任务 82% → 89%（加 failure）→ 92%（再加 N=4 打分）；未见变体 67% → 77% → 82%；**没有 failure 数据时打分反而 79%**（不帮忙）。N 的扫描只在一个长任务上做过，N=4 折中。仿真没有报告 BoN 结果。
- **推理延迟**：action-only 约 380ms（RTX PRO 6000），比 Motus（1220ms）快约 3 倍。
- **ref**：只有**当前时刻**一帧的 VAE latent（干净、timestep 0、不算 loss），没有历史帧；state 也只有当前一个 token。部署时每 48 步重规划一次（`execute_actions_per_plan=48`）。
- **state / action**：14 维 = 左臂 6 关节 + 左夹爪 + 右臂 6 关节 + 右夹爪；动作对 12 个关节做 delta（减当前 state），两个夹爪用绝对值，再 z-score。
- **扩散**：flow matching，预测的是**速度** `noise − x0`（既不是 x0 预测也不是 ε 预测）；`x0 = x_t − σ·v`；推理用 Euler，20 步；不同候选靠不同初始噪声。
- **value**：代码里是"剩余时间比例" `(L−t_future−1)/(L−1)`，越小越好，failure 再加 `value_penalty_scale`；BoN 用 `argmin`。论文写成进度、`argmax`，只是符号翻转。
- **`action_diversity`**：N 个候选动作块（`[48,14]` 展平成 672 维，已反归一化并加回 state）两两 L2 距离的平均值；N=1 时恒为 0；单位是原始关节单位，**没有参照标准判断 0.15~0.18 算大还是小**。
- **T5 的作用**：把指令编码成 `[64,4096]` 的 embedding 做 cross-attention。训练用预先缓存的 `.pt`，不加载 T5；推理 server 要实时编码指令，所以要加载 T5（约 11GB）。
- **动作条件的作用**：主结果用 action-only 推理，部署时不跑世界模型分支；其价值主要在训练期（视频 co-training 是正则项，去掉后 RoboTwin 从 85.6% 降到 81.8%；causal-mask 消融只在真机上从 82% 到 77%，格子只有 20 次试验）和"让 failure 数据能安全加入训练"。

---

## 6. 迁移到 GAP 的设计（未实现）

- worktree：`../GAP-tte-flattenmlp-concatdit-actionrope-factcausal`，分支 `vera-dev/jointdiff-concatdit-actionrope-factcausal`（基于 bac550a，干净）。详细设计见该目录下的 `FACT_ADAPTATION.md`。
- 映射：future video → future 3D track；value 沿用 FACT 定义（argmin）；新增 `bridge_mode: fact_causal`，序列 `[场景 patch+state | 噪声 action | 干净 gt action | 噪声 track | 噪声 value]`，用 `src_mask` 控制：action 不看 track/value，track/value 看干净 gt action。
- 核心冲突：GAP 的 `concat_selfattn` 把"当前场景特征"和"带噪 future track"融合进同一个 token，action 会读到噪声的未来；必须拆开。GAP 已有的 `bridge="uni"` 是 track → action，与 FACT 方向相反。
- 另外：GAP 是 DDIM 离散 100 步 + 预测干净样本，action horizon 是 16（FACT 是 48）。
- **阻塞问题**：需要先确认已有的 success + failure rollout 有没有 3D track、DINO 特征，以及 `frame_index` / `episode_length` / failure 标记；`convert_tracks_to_zarr.py` 目前只对 expert demo 生成 track。

---

## 7. 新增/修改的文件

| 路径 | 说明 |
| --- | --- |
| `FACT/evaluation/robotwin/collect_failure_interface.py` | 失败 rollout 采集 policy adapter |
| `FACT/scripts/build_failure_dataset.py` | staging → LeRobot v2.1（含 `failure_rollouts.jsonl`） |
| `FACT/world_action_model/configs/robotwin_smoke.py` / `robotwin_ft.py` | 冒烟 / 微调配置（未改动原 `robotwin.py`） |
| `FACT/eval_runs/fact_eval_pilot.sbatch` | 评测 / 采集 sbatch（唯一端口、自动选 T5 设备） |
| `FACT/eval_runs/fact_finetune.sbatch` | 微调 sbatch |
| `FACT/datasets/RoboTwin/Fail/beat_block_hammer` | 60 个失败 rollout（LeRobot v2.1） |
| `FACT/datasets/collect_staging/` | 采集的原始 staging（已导入，可删） |
| `FACT/experiments/{smoke,ft_expert,ft_expert_fail}` | 各约 30GB，smoke 可删 |
| `RoboTwin/script/eval_policy.py` | 打了 `test_num` 一行补丁 |
| `GAP-...-factcausal/FACT_ADAPTATION.md` | GAP 迁移设计 |

## 8. 还没做的

- 其他任务上的 failure 采集与更大规模训练（50 任务、更多失败、更长训练，value head 收敛后再测 BoN）。
- 更精细的失败起始帧标注（现在是整条 episode）。
- value 与最终成败的相关性分析；N 的扫描（N=2/4/8）。
- GAP 迁移的实现。
