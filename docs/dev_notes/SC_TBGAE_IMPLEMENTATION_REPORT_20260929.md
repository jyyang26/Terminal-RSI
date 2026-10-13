# SC + TB-GAE 同时启用：可行性检查报告

**日期：2026-09-29。** 目的：回答"能否同时开启 TB-GAE 与 Score-Centering (SC) 跑 PPO"。

## 结论（TL;DR）

**不能同时启用。** 除了显式互斥守卫外，数据管线里有一个**具体的、会触发的对齐 bug**：训练上下文超长被裁剪时，SC 的 head 字段不会被同步裁剪/分块，导致 SC head 与 response token **错位** → 训练时 `sc_assert_alignment: full` 断言失败崩溃（或在宽松设置下静默产生错误的 SC 梯度）。在 256k + 终端任务上，一定会有轨迹超过 cap，所以这个 bug **必然在前几步触发**。

因此按你的指示"如果本身没有问题就直接都开然后拉起"——**本身有问题，故未拉起。** 下面是逐条依据（代码行号可核对）。

---

## 1. 显式互斥守卫（第一道硬墙）

`slime/utils/tb_gae_config.py:123` 的 `validate_tb_gae_config` 要求开 TB-GAE 时 `enable_score_centering == False`。实测：

```
BOTH ON rejected -> TB-GAE requires enable_score_centering=False, got True
```

SC 报告（`SC_IMPLEMENTATION_REPORT_20260929.md` 第 15 行）也写明二者不能同开。要"都开"必须先删掉这行守卫——**即移除一个安全检查去跑一个未验证的组合**。但即使删掉它，下面第 2 节的 bug 依然存在。

---

## 2. 关键 bug：cap 裁剪时 SC head 切片被 TB 早返回绕过（会崩溃/静默错误）

**位置：** `slime_plugins/harbor/sample_postprocess.py:119-123`

```python
def cap_total_tokens_allow_response_trim_in_samples(args, all_samples, data_source=None):
    ...
    if getattr(args, "enable_tb_gae", False) is True:
        from slime_plugins.harbor.tb_gae_chunks import split_tb_samples
        split_tb_samples(all_samples, cap)
        return                       # <-- 早返回：SC 切片分支永远到不了
    ...
    # 下面这段（162-179）只在 TB 关闭时执行：
    if getattr(args, "enable_score_centering", False):
        tracks = _slice_sc_tracks(... rollout_topk_token_ids, rollout_topk_log_probs, sc_response_token_ids ...)
        ...
        if not (sample.sc_response_token_ids == sample.tokens[-trim.response_length:]).all():
            raise ValueError("SC head/token identity failed after the hard response cap")
```

**机制：** 当 TB-GAE 开启，cap hook 走 `split_tb_samples` 后**立即 `return`**。负责裁剪 SC 三个 head 字段（`rollout_topk_token_ids` / `rollout_topk_log_probs` / `sc_response_token_ids`）的 SC 分支（162-179 行）在另一支，**永远执行不到**。

**后果：** 对任何超过 cap 的轨迹：
- `split_tb_samples`（`tb_gae_chunks.py:10-83`）只切了 `tokens` / `response_length` / `loss_mask` / `rollout_log_probs` / `turn_end_mask` / `is_real_done` / `tb_boundary`；
- **完全没碰 SC 的 head 字段**——它们仍是原始全长；
- 更糟：`split_tb_samples` 用 `copy.copy(sample)`（第 34 行）把一条轨迹拆成多个 chunk，每个 chunk **共享同一份原始全长 SC head 数组**，而各自的 response 只是一个 chunk 切片。

下游训练时，SC 的 `sc_terms.py` 按 `response_row_layout` 把 head `[R_原始, k]` 对齐到 response 行 `[R_chunk]` → 长度不符 → `sc_assert_alignment: full` 断言 **抛错崩溃**；若断言被放宽，则是**静默错位的 SC 梯度**。

**触发条件：** 轨迹总长 > cap（本 run 训练 cap 由 `HARBOR_MAX_TRAIN_TOTAL_TOKENS` / 256k 决定）。RUN2 曾有 76.6% 轨迹撞 max_turns 上限、终端任务普遍长——**必然有轨迹超 cap**，所以此 bug 在前几步就会命中，不是理论边角。

---

## 3. rollout_data 回退守卫同样被 TB 短路（次要）

`sample_postprocess.py:214-217`：TB 分支在 SC 分支（218-223）之前 `return`，所以 SC 的 "cap must run before DP" 回退校验在 TB 开启时也被跳过。两者都只是 assert-and-return，影响小，但印证了"TB 在管线各处短路 SC"这一模式。

---

## 4. 哪些其实能正常组合（为准确起见）

不是所有地方都冲突——为免夸大，明确列出**能共存**的部分：

- **loss 组合**：TB 在 `loss.py:731` 改的是 advantages（上游），SC 在 `train_loss.py:106` 用 advantages 加梯度中心化项（下游）。SC 把 A 当黑箱标量，机械上能消费 TB 的 advantages，不会因取值崩。
- **adapter 构造**：`atif_adapter.py:1027-1031` 里 `_attach_normalized_sc_head`（SC）与 `build_tb_masks`（TB）是并列 `if`，转换时两套字段**都会构造且对齐**。
- **DP 切分**：`slime/ray/rollout.py` 中 TB（935/979/1119-1122）与 SC（1018/1126）是独立 `if`，两套字段都会随 DP 传递。
- **actor 前向**：`actor.py` 里 TB 的 bootstrap 前向（457-461）与 SC 的 old-policy head（305/579-582）是独立顺序块，都会执行。

**唯一致命断点就是第 2 节的 cap hook**——但它在本工作负载上必然触发，所以整体结论仍是"不能开"。

---

## 5. 资源风险（即使修好第 2 节仍需评估）

两者各自额外做一次前向：
- TB：`compute_bootstrap_values` 的额外 256k old-critic 前向（`tb_gae_bootstrap.py`）；
- SC：old-policy head 的 gather 前向（`sc_terms.py`）。

在 256k 上，最后 PP 段的 vocab/CE 峰值本就是 OOM 的绑定项（TRAINING_INTERRUPTION_FAILURES 有记录）。两个额外前向叠加会进一步抬高 OOM 概率。这是资源风险，非正确性问题，但需在真实 GPU smoke 验证。

---

## 6. 数学层面：组合从未被验证

SC 的推导（SC 报告 §3.1）中心化的是"基于标准 PPO advantage 的策略梯度"；TB-GAE 把 advantage 换成**逐轮 bootstrap 的 turn-based GAE**。SC 把 A 当黑箱，所以不会因取值崩溃，但"在 turn-based advantage 上做 SC 中心化"这一组合**没有任何测试、没有推导、两份报告都明确禁止**。即使修好第 2 节的对齐 bug 让它能跑，也可能是**静默错误的目标函数**。

---

## 7. 要让二者真正共存，需要的工作（非本次范围）

1. 删除 `tb_gae_config.py:123` 的互斥守卫（移除安全检查）。
2. 让 `split_tb_samples` 在分块时**同步切分并按 chunk 分发** SC 的三个 head 字段（`rollout_topk_token_ids` / `rollout_topk_log_probs` / `sc_response_token_ids`，weighted 模式还有 `sc_old_head_log_probs`），并对每个 chunk 重新校验 `sc_response_token_ids == tokens[-response_length:]`。
3. 处理 `copy.copy` 导致的 head 数组共享（需 deep-copy 或按 chunk 切片新建）。
4. 为"TB advantage + SC 中心化"补数学推导与数值测试（CPU 逐 bit + 单机 GPU smoke）。
5. 评估两次额外 256k 前向的显存峰值。

这是一个**新的集成开发 + 验证任务**，不是配置开关。

---

## 8. 建议

- **不要同时开。** 存在必然触发的对齐 bug（第 2 节）+ 未验证的数学（第 6 节）。
- 这次 PPO（actor=`q38_27b_iter263_torch_dist`，critic=RESUME19 的 `iter_0000034`）**只开一个**：
  - **SC**：已有 2026-09-26 真实单机 GPU 冒烟背书，成熟度更高；或
  - **TB-GAE**：截断 bootstrap 已实现但 GPU 路径尚未跑通（三次冒烟都被 Daytona 配额拦在训练前）。
- 无论选哪个，还需先解决两个环境阻断：**Daytona 配额**（1 小时前 100% `Total CPU limit exceeded, Maximum allowed: 500`）与**磁盘**（1.9T/98%，`SAVE_INTERVAL=5` 只够存 1 次）。

> 说明：本文件按约定加 `.md` 后缀（与同目录 `SC_IMPLEMENTATION_REPORT_20260929.md` / `TB_GAE_IMPLEMENTATION_REPORT_20260927.md` 一致）。所有行号基于 2026-09-29 当前工作区代码，可直接核对。
