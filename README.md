## PCVAE-GAN
PCVAE-GAN 是一个用于二维相位展开的项目。当前仓库已经整理为一个**兼容优先的 hierarchical NVAE + z5 VQ 版本**：
`trainvae.py` -> `train/train.py` -> `st1_vae.py` -> `models/vae.py`。
### 1. 当前模型是什么
当前有效模型不是单层 VAE，而是一个 **5 层 hierarchical NVAE**：
- 编码器从输入 wrapped phase 开始，逐层下采样，输出 5 层后验分布参数：`z1 ~ z5`
- 解码器从最深层 `z5` 开始做 top-down 解码
- `z5` 上可以启用 **VQ（Vector Quantization）**，也就是先把最深层离散化，再向下生成 `z4 ~ z1`
- 最终输出连续相位图 `phi_hat`
#### hierarchical 具体是什么意思
这里的 hierarchical 指的是“**分层潜变量**”而不是单个 latent：
- **z5**：最深层，语义最强，负责全局相位结构
- **z4**：中深层，补充较大尺度的结构
- **z3**：中层，补充中尺度纹理
- **z2**：浅层，补充局部细节
- **z1**：最浅层，补充最细粒度的边缘和跳变
解码时是 **z5 → z4 → z3 → z2 → z1** 的 top-down 流程，逐层把高层语义细化成最终相位图。
---
### 2. 当前可用内容
- `trainvae.py`：默认训练入口
- `train/train.py`：Stage1 兼容训练包装器
- `st1_vae.py`：Stage1 训练实现
- `models/encoder.py`、`models/decoder.py`：5 层 hierarchical 编码器/解码器
- `models/vae.py`：兼容版 NVAE 包装器，支持 hierarchical / z5 VQ
- `models/vq.py`：Vector Quantizer
- `testvae.py`：兼容版测试脚本
- `config/pcvae_gan.yaml`：主配置文件
- `data/dataset.py`、`data/vaedata.py`：数据读取
> 说明：Stage2 / Stage3 目前没有接到默认入口里，仓库现在的重点是先保证当前 hierarchical VAE + VQ 链路能稳定运行。
---
### 3. 环境安装
建议使用 Python 3.10+，并安装与你 CUDA 版本匹配的 PyTorch。
项目当前实际用到的主要依赖包括：
- `torch`
- `numpy`
- `scipy`
- `omegaconf`
- `pyyaml`
- `matplotlib`（仅用于测试可视化，可选）
- `Pillow`
```bash
pip install -r requirements.txt
```
如果你需要指定版本的 PyTorch，请先安装 PyTorch，再安装其它依赖。
---
### 4. 数据目录
默认数据目录：
```text
dataset/
  train/
    train_wrapped/
    train_absolute/
  test/
    test_wrapped/
    test_absolute/
```
`.mat` 文件默认字段：
- wrapped phase：`input`
- absolute phase：`output`
如果你的字段名不同，请在 `config/pcvae_gan.yaml` 中修改：
- `data.wrap_key`
- `data.phi_key`
---
### 5. 训练
默认训练命令：
```bash
python trainvae.py --config config/pcvae_gan.yaml
```
也可以直接指定设备：
```bash
python trainvae.py --config config/pcvae_gan.yaml --device cuda:0
```
当前训练会使用：
- 重建损失：L1
- KL 损失：5 层 hierarchical KL
- VQ 损失：对 `z5` 做 VQ 时启用
---
### 6. 测试
兼容版测试脚本：
```bash
python testvae.py
```
它会自动：
- 读取 `config/pcvae_gan.yaml`
- 找到最新的 Stage1 checkpoint
- 优先读取 `test_wrap` / `test_true`，执行 wrapped → unwrapped 评测
- 也兼容 `test_wrapped` / `test_absolute`
- 如果只有 `test_phi`，会退回到 unwrapped fallback 模式，仅做重建评估
- 在 paired 模式下输出 MAE / RMSE / PSNR / SSIM / Wrap-MAE；fallback 模式下 Wrap-MAE 显示为 N/A
---
### 7. 配置说明
当前模型会优先识别这些字段：
- `model.latent_channels`
- `model.latent_levels`
- `model.output_channels`
- `model.strict_hierarchical`
- `model.use_z5_vq`
- `model.vq_num_embeddings`
- `model.vq_commitment_cost`
- `train.batch_size`
- `train.lr_encoder`
- `train.lr_decoder`
- `train.num_epochs_stage1`
- `data.sim_wrap` / `data.sim_true`
- `data.wrap_key` / `data.phi_key`
同时也兼容旧字段：
- `model.latent_dim`
- `data.sim_data`
- `data.sim_phi`
- `data.test_wrapped`
- `data.test_absolute`
- `test.checkpoint`
---
### 8. 常见问题
- **找不到数据字段**：检查 `.mat` 文件里的 key 是否和 `wrap_key` / `phi_key` 一致
- **checkpoint 找不到**：确认 `train.checkpoint_dir/stage1/` 下有 `checkpoint_epoch*.pth`
- **显存不足**：减小 `batch_size`
- **想恢复多阶段训练**：后面可以再把 Stage2 / Stage3 补回去
---
### 9. 说明
当前仓库保留的是“兼容优先”的最小可运行版本，但模型本身已经升级为 **hierarchical NVAE + z5 VQ**。
如果你愿意，我下一步可以继续帮你把训练日志、测试脚本和配置再整理得更干净一些。
