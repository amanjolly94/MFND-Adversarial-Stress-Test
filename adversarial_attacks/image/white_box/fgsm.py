import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from ...adv_attack import AdversarialImageAttacks

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image

def denorm(batch, mean, std, device):
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

class FGSMAttack:

    def __init__(self, model, epsilon=0.007, device=torch.device("cpu")):
        self.model = model
        self.epsilon = epsilon
        self.device = device

    def get_transform(self, size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
            transforms.Normalize(mean=mean, std=std),
        ])

        return transform

    def fgsm_attack(self, image, epsilon, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon*sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        return perturbed_image

    def denorm(self, batch, mean, std):
        if isinstance(mean, list):
            mean = torch.tensor(mean).to(self.device)
        if isinstance(std, list):
            std = torch.tensor(std).to(self.device)

        return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
    
    def get_attacked_features(self, img_path, size=299):
        trans_fn = self.get_transform(size)

        self.model.eval()

        img = Image.open(img_path).convert("RGB")
        img = trans_fn(img).unsqueeze(0)

        img.requires_grad = True
        output = self.model(img)

        init_pred = output.max(1, keepdim=True)[1]
        target = init_pred[0]

        loss = torch.nn.functional.nll_loss(output, target)
        self.model.zero_grad()
        loss.backward()

        img_grad = img.grad.data
        data_denorm = self.denorm(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        perturbed_data = fgsm_attack(data_denorm, self.epsilon, img_grad)
        perturbed_data_normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(perturbed_data)

        output = model(perturbed_data_normalized)
        final_pred = output.max(1, keepdim=True)[1]

        self.adv_ex = perturbed_data.squeeze().detach().cpu().numpy()

        return output, init_pred, final_pred
    
    def get_attack_img(self):
        return self.adv_ex

class FGSM(AdversarialImageAttacks):
    # https://arxiv.org/abs/1412.6572
    def __init__(self, model, device=None, eps=8/255):
        super().__init__('FGSM', model, device)
        self.eps = eps

    def forward(self, images, labels, loss_type="bce"):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        if loss_type == "bce":
            loss = torch.nn.BCELoss()
        else:
            loss = torch.nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.get_logits(images)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images

if __name__ == "__main__":

    # Test FGSM
    model = torchvision.models.inception_v3(pretrained=True)

    attack = FGSMAttack(model)

    img_path = "test_files/images/spot_3.png"
    output, init_pred, final_pred = attack.get_attacked_features(img_path)

    print(output.shape, init_pred, final_pred)