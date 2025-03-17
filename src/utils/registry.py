optimizer = {
    "adam":    "torch.optim.Adam",
    "adamw":   "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd":     "torch.optim.SGD",
    "lamb":    "src.utils.optim.lamb.JITLamb",
}

scheduler = {
    "constant":        "transformers.get_constant_schedule",
    "plateau":         "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step":            "torch.optim.lr_scheduler.StepLR",
    "multistep":       "torch.optim.lr_scheduler.MultiStepLR",
    "cosine":          "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup":   "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup":   "transformers.get_cosine_schedule_with_warmup",
    "timm_cosine":     "src.utils.optim.schedulers.TimmCosineLRScheduler",
}

callbacks = {
    "timer":                 "src.callbacks.timer.Timer",
    "params":                "src.callbacks.params.ParamsLog",
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint":      "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping":        "pytorch_lightning.callbacks.EarlyStopping",
    "swa":                   "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary":    "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar":     "pytorch_lightning.callbacks.RichProgressBar",
    "progressive_resizing":  "src.callbacks.progressive_resizing.ProgressiveResizing",
    # "profiler": "pytorch_lightning.profilers.PyTorchProfiler",
}

model = {
    # Backbones from this repo
    "model":                 "src.models.sequence.backbones.model.SequenceModel",
}

layer = {
    "id":         "src.models.sequence.base.SequenceIdentity",
    "standalone": "models.s4.s4.S4Block",
    "s4d":        "models.s4.s4d.S4D",
    "pSpikeSSM":         "src.models.sequence.modules.s4block.S4Block",
    "fftconv":    "src.models.sequence.kernels.fftconv.FFTConv",
    "s4nd":       "src.models.sequence.modules.s4nd.S4ND",
    "mega":       "src.models.sequence.modules.mega.MegaBlock",
    "h3":         "src.models.sequence.experimental.h3.H3",
    "h4":         "src.models.sequence.experimental.h4.H4",
    # 'packedrnn': 'models.sequence.rnns.packedrnn.PackedRNN',
}

layer_decay = {
    'convnext_timm_tiny': 'src.models.baselines.convnext_timm.get_num_layer_for_convnext_tiny',
}

model_state_hook = {
    'convnext_timm_tiny_2d_to_3d': 'src.models.baselines.convnext_timm.convnext_timm_tiny_2d_to_3d',
    'convnext_timm_tiny_s4nd_2d_to_3d': 'src.models.baselines.convnext_timm.convnext_timm_tiny_s4nd_2d_to_3d',
}
