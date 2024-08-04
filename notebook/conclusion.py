### 解释一下Masked Autoencoder这个工作

# MAE是一种自监督学习方法，用于图像和其他类型的输入数据的表征学习。
# 主要思想是通过对输入数据的一部分进行mask，然后重建这些被mask的部分。

# 1. MAE基于encoder-decoder结构，这里的encoder和decoder都参考了Vision Transformer中的Transformer encoder。

# 2. 输入处理：
#    - 首先，图像经过encoder的处理。
#    - 其中，先进行随机mask操作。例如，mask_ratio = 0.5。
#    - 假设输入是224x224的图像，经过patch_embedding处理，patch_size = 16。
#    - 处理后得到的张量形状为[batch, 196, dim]。
#    - 使用mask_ratio = 0.5时，保留的信息为[batch, 196 * 0.5, dim]。
#    - 将这部分信息输入到Transformer encoder中。

# 3. Mask处理：
#    - 已被mask的patches用一个共享且可训练的token表示。
#    - 这样可以恢复张量的形状为[batch, 196, dim]。
#    - 然后，将这个张量输入到另一个Transformer encoder中。

# 4. 计算损失：
#    - 使用MSE-loss，计算预测值和原始输入之间的L2-loss。
#    - 损失仅在被mask的patches上计算，未被mask的patches不参与损失计算。
