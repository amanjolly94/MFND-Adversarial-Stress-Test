import torch
import torch.nn as nn

from ...adv_attack import AdversarialImageAttacks

class NIFGSM(AdversarialImageAttacks):
    # https://arxiv.org/abs/1908.06281
    def __init__(self, model, device=None, eps=8/255, alpha=2/255, steps=10, decay=1.0):
        super().__init__('NIFGSM', model, device)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha

    def forward(self, images, labels, loss_type="bce"):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        if loss_type == "bce":
            loss = torch.nn.BCELoss()
        else:
            loss = torch.nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            nes_images = adv_images + self.decay*self.alpha*momentum
            outputs = self.get_logits(nes_images)
            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            grad = self.decay*momentum + grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            momentum = grad
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images