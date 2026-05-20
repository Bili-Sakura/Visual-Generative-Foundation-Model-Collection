import torch


class EDM2Loss:
    def __init__(self, p_mean: float = -0.4, p_std: float = 1.0, sigma_data: float = 0.5):
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data

    def __call__(self, model, images: torch.Tensor, labels: torch.Tensor = None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(images) * sigma
        denoised = model(sample=images + noise, sigma=sigma.flatten(), class_labels=labels, return_logvar=True)
        logvar = denoised.logvar
        loss = (weight / logvar.exp()) * ((denoised.sample - images) ** 2) + logvar
        return loss
