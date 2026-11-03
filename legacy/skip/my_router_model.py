import torch
from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
# 导入你的 Router 和 Hook
from model_router import RouterTransformer
class SkipHook:
    def __call__(self, module, args, output):
        if isinstance(output, tuple):
            return (args[0],) + output[1:]
        return args[0]
class UniversalPrunedOPT(HFLM):
    def __init__(self, pretrained, strategy="baseline", router_weight_path=None, skip_layers=None, k_to_skip=6, **kwargs):
        """
        strategy: "baseline", "static", "sft", "rl"
        skip_layers: 供 static 策略使用的固定层列表
        """
        super().__init__(pretrained=pretrained, **kwargs)
        
        self.strategy = strategy
        self.k_to_skip = k_to_skip
        self.skip_layers = skip_layers if skip_layers else []
        self.transformer_layers = self.model.model.decoder.layers
        
        # 只有在策略是 sft 或 rl 时，才加载 router
        if self.strategy in ["sft", "rl"]:
            self.router = RouterTransformer(
                input_dim=5120, num_opt_layers=40, nhead=4, num_encoder_layers=2, d_hid=2048
            ).to(self.device)
            state = {k.replace('module.', ''): v for k, v in torch.load(router_weight_path, map_location=self.device).items()}
            self.router.load_state_dict(state)
            self.router.eval()

    def _model_call(self, inps):
        with torch.no_grad():
            layers_to_skip = []
            
            # 1. 判断跳层策略
            if self.strategy == "baseline":
                pass # 不跳层
            elif self.strategy == "static":
                layers_to_skip = self.skip_layers
            elif self.strategy in ["sft", "rl"]:
                embeddings = self.model.get_input_embeddings()(inps).to(torch.float16)
                logits = self.router(embeddings)
                _, skip_idx = torch.topk(logits, self.k_to_skip, dim=1, largest=False)
                layers_to_skip = skip_idx[0].tolist()
            
            # 2. 注册 Hooks
            handles = [self.transformer_layers[l].register_forward_hook(SkipHook()) for l in layers_to_skip]
            
            # 3. 前向传播
            outputs = self.model(inps)
            
            # 4. 卸载 Hooks
            for h in handles:
                h.remove()
                
            return outputs.logits
class DynamicPrunedOPT(HFLM):
    def __init__(self, pretrained, router_weight_path, k_to_skip=6, **kwargs):
        # 初始化基座模型
        super().__init__(pretrained=pretrained, **kwargs)
        
        self.k_to_skip = k_to_skip
        self.transformer_layers = self.model.model.decoder.layers
        
        # 加载你的 Router (比如 RL Router)
        self.router = RouterTransformer(
            input_dim=5120, num_opt_layers=40, nhead=4, num_encoder_layers=2, d_hid=2048
        ).to(self.device)
        rl_state = {k.replace('module.', ''): v for k, v in torch.load(router_weight_path, map_location=self.device).items()}
        self.router.load_state_dict(rl_state)
        self.router.eval()

    # 核心：拦截前向传播，插入动态剪枝逻辑
    def _model_call(self, inps):
        """
        lm-eval 会调用这个方法来获取 logits
        inps: 输入的 token_ids (batch_size, seq_len)
        """
        with torch.no_grad():
            # 1. 获取 Embedding 喂给 Router
            embeddings = self.model.get_input_embeddings()(inps).to(torch.float16)
            
            # 2. Router 预测需要跳过的层
            rl_logits = self.router(embeddings)
            _, rl_skip_idx = torch.topk(rl_logits, self.k_to_skip, dim=1, largest=False)
            
            # (可选) 针对 batch_size > 1 的情况，为了简单起见，可以强行设置 batch_size=1
            # 或者对 batch 取平均/第一个样本的决策
            rl_layers = rl_skip_idx[0].tolist() 
            
            # 3. 注册 Hooks
            handles = [self.transformer_layers[l].register_forward_hook(SkipHook()) for l in rl_layers]
            
            # 4. 执行原模型前向传播
            outputs = self.model(inps)
            
            # 5. 卸载 Hooks，恢复原状
            for h in handles:
                h.remove()
                
            return outputs.logits