import torch

from ...adv_attack import AdversarialImageAttacks

class FastFGSM(AdversarialImageAttacks):
    # https://arxiv.org/abs/2001.03994
    def __init__(self, model, device=None, eps=0.03, alpha=0.04):
        super().__init__('FastFGSM', model, device)
        self.eps = eps
        self.alpha = alpha

    def forward(self, images, labels, loss_type="bce"):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        if loss_type == "bce":
            loss = torch.nn.BCELoss()
        else:
            loss = torch.nn.CrossEntropyLoss()

        adv_images = images + torch.randn_like(images).uniform_(-self.eps, self.eps)  # nopep8
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        adv_images.requires_grad = True

        outputs = self.get_logits(adv_images)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = adv_images + self.alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images