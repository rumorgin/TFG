import ast

from models.CVAEGAN import *
from models.resnet18_encoder import *
# from models.fscil_embedding_cvaegan.CVAEGAN import *
from utils import *


class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        if self.args.dataset in ['cifar100']:
            self.feature_extractor = resnet18(num_classes=100)
        if self.args.dataset in ['mini_imagenet']:
            self.feature_extractor = resnet18(False, args, num_classes=100)  # pretrained=False
        if self.args.dataset == 'cub200':
            self.feature_extractor = resnet18(False, args, num_classes=200)
            state_dict = torch.load(r'D:\fscil_lmu\pretrain\resnet18-5c106cde.pth', map_location=torch.device('cpu'))
            filtered_state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
            self.feature_extractor.load_state_dict(filtered_state_dict, strict=False)


        self.encoder = VAE_encoder(in_feature=512, out_feature=256, latent_dim=1024)
        self.decoder = VAE_decoder(in_feature=256, out_feature=512, latent_dim=256, class_dim=768)
        self.discriminator = Discriminator(in_feature=512, latent_dim=256, class_dim=768, output_cell=self.args.num_classes)


    def forward(self, *args):
        if self.mode == 'train_vaegan_classifier':
            image = args[0]
            labels = args[1]
            word_embedding = args[2]

            real_feature = self.feature_extractor(image)

            logits = F.linear(F.normalize(real_feature, p=2, dim=-1),
                              F.normalize(self.feature_extractor.fc.weight, p=2, dim=-1))
            logits = self.args.temperature * logits[:, :self.args.base_class]
            loss = F.cross_entropy(logits, labels)
            Acc = count_acc(logits, labels)

            return Acc, loss
        elif self.mode == 'train_vaegan_generator':
            image = args[0]
            labels = args[1]
            word_embedding = args[2]

            real_feature = self.feature_extractor(image)

            latent_z, mu, logvar = self.encoder(real_feature)

            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
            recon_feature = self.decoder(z=latent_z, label=word_embedding)

            noise = torch.randn((labels.size(0), self.args.output_length)).cuda()

            gen_feature = self.decoder(z=noise, label=word_embedding)

            real_adv, real_f = self.discriminator(image=real_feature)
            gen_adv, gen_f = self.discriminator(image=gen_feature.detach())
            recon_adv, recon_f = self.discriminator(image=recon_feature.detach())

            recons_loss = F.mse_loss(recon_f, real_f)

            encoder_loss = recons_loss + kld_loss

            gen_logits = F.linear(F.normalize(gen_feature, p=2, dim=-1),
                                  F.normalize(self.feature_extractor.fc.weight, p=2, dim=-1))
            gen_logits = self.args.temperature * gen_logits[:, :self.args.base_class]

            recon_logits = F.linear(F.normalize(recon_feature, p=2, dim=-1),
                                    F.normalize(self.feature_extractor.fc.weight, p=2, dim=-1))
            recon_logits = self.args.temperature * recon_logits[:, :self.args.base_class]

            real_logits = F.linear(F.normalize(real_feature, p=2, dim=-1),
                                   F.normalize(self.feature_extractor.fc.weight, p=2, dim=-1))
            real_logits = self.args.temperature * real_logits[:, :self.args.base_class]

            class_loss = F.cross_entropy(gen_logits, labels)

            g_disc_loss = -torch.log(gen_adv + 1e-8).mean() - torch.log(recon_adv + 1e-8).mean()

            generator_loss = g_disc_loss + class_loss + 0.01 * recons_loss

            dbinary_loss = -(torch.log(1 - gen_adv + 1e-8).mean() + torch.log(1 - recon_adv + 1e-8).mean() + torch.log(
                real_adv + 1e-8).mean())

            dclass_loss = F.cross_entropy(real_logits, labels) + F.cross_entropy(
                gen_logits[:, :self.args.base_class], labels) + F.cross_entropy(
                recon_logits[:, :self.args.base_class], labels)

            gp = self.gradient_penalty(self.discriminator, real_feature, gen_feature)

            d_loss = dbinary_loss + dclass_loss + gp

            Acc_gen = count_acc(gen_logits, labels)

            return Acc_gen, encoder_loss, generator_loss, d_loss

        elif self.mode == 'test_vaegan':
            image = args[0]
            labels = args[1]
            word_embedding = args[2]
            session = args[3]

            real_feature = self.feature_extractor(image)

            real_logits = F.linear(F.normalize(real_feature, p=2, dim=-1),
                                   F.normalize(self.feature_extractor.fc.weight, p=2, dim=-1))
            real_logits = self.args.temperature * real_logits[:, :self.args.base_class + session * self.args.way]
            Acc_real = count_acc(real_logits, labels)

            return Acc_real

        elif self.mode == 'test_vaegan_gen':
            image = args[0]
            labels = args[1]
            word_embedding = args[2]
            session = args[3]

            noise = torch.randn((labels.shape[0], self.args.output_length)).cuda()
            gen_feature = self.decoder(z=noise, label=word_embedding)

            real_logits = F.linear(F.normalize(gen_feature, p=2, dim=-1),
                                   F.normalize(self.feature_extractor.fc.weight, p=2, dim=-1))
            real_logits = self.args.temperature * real_logits[:, :self.args.base_class + session * self.args.way]
            Acc_real = count_acc(real_logits, labels)

            return Acc_real

        else:
            raise ValueError('Unknown mode')

    def update_fc_another(self, dataloader, class_list, session):
        num_class = self.args.base_class + (session) * self.args.way

        if self.args.dataset == 'mini_imagenet':
            wnids = dataloader.dataset.wnids
            word_embedding = torch.tensor([
                ast.literal_eval(dataloader.dataset.word_embedding[key]) for key in wnids[:num_class]]).cuda()
        else:
            word_embedding = torch.tensor([
                ast.literal_eval(dataloader.dataset.word_embedding[str(key)]) for key in range(num_class)]).cuda()
        word_embedding = word_embedding.repeat_interleave(self.args.episode_shot, 0)
        noise = torch.randn((word_embedding.shape[0], self.args.output_length)).cuda()
        gen_feature = self.decoder(z=noise, label=word_embedding)

        optimizer = torch.optim.SGD(self.feature_extractor.fc.parameters(), lr=0.0001, momentum=0.9)

        for batch in dataloader:
            data, label, _ = [_.cuda() for _ in batch]
            data = self.feature_extractor(data).squeeze().detach()

        for class_index in class_list:
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = data[data_index]
            proto = embedding.mean(0)
            self.feature_extractor.fc.weight.data[class_index] = proto

        with torch.enable_grad():
            for epoch in range(300):
                optimizer.zero_grad()
                combine_feature = torch.cat((data, gen_feature.detach()), dim=0)
                old_label = torch.arange(0, num_class).repeat_interleave(self.args.episode_shot, 0).cuda()
                combine_label = torch.cat((label, old_label), dim=0)
                new_logits = self.args.temperature * F.linear(F.normalize(combine_feature, p=2, dim=-1),
                                                              F.normalize(self.feature_extractor.fc.weight, p=2,
                                                                          dim=-1))
                Acc = count_acc(new_logits, combine_label)
                loss = F.cross_entropy(new_logits[:, :num_class], combine_label)
                loss.backward()
                optimizer.step()

    def gradient_penalty(self, f, real, fake=None):
        def interpolate(a, b=None):
            if b is None:  # interpolation in DRAGAN
                beta = torch.rand_like(a)
                b = a + 0.5 * a.var().sqrt() * beta
            alpha = torch.rand(a.size(0), 1)
            alpha = alpha.cuda()
            inter = a + alpha * (b - a)
            return inter

        x = interpolate(real, fake).requires_grad_(True)
        pred = f(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        grad = torch.autograd.grad(
            outputs=pred, inputs=x,
            grad_outputs=torch.ones_like(pred),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad = grad.view(grad.size(0), -1)
        norm = grad.norm(2, dim=1)
        gp = ((norm - 1.0) ** 2).mean()
        return gp
