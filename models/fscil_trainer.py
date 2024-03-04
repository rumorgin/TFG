import torch.nn as nn
from copy import deepcopy
from .Network import MYNET
import torch.nn.functional as F
# from .helper import *
from utils import *
from dataloader.data_utils import *
from tqdm import tqdm
import ast
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

class FSCILTrainer(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['train_loss_gen'] = []
        self.trlog['val_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['test_loss_gen'] = []
        self.trlog['train_acc'] = []
        self.trlog['train_acc_gen'] = []
        self.trlog['val_acc'] = []
        self.trlog['test_acc'] = []
        self.trlog['test_acc_gen'] = []
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc'] = [0.0] * args.sessions
        self.trlog['max_acc_epoch_gen'] = 0
        self.trlog['max_acc_gen'] = [0.0] * args.sessions
        self.model = MYNET(self.args, mode=self.args.base_mode)
        if args.use_gpu:
            self.model = self.model.cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def update_param(self, model, pretrained_dict):
        model_dict = model.state_dict()
        # pretrained_dict = {k.replace('module.encoder','feature_extractor'): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model
    def get_optimizer_base(self):
        optimizer_resnet=torch.optim.SGD(self.model.feature_extractor.parameters(),lr=self.args.lr_base,
                                    momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        optimizer_encoder = torch.optim.Adam(self.model.decoder.parameters(), self.args.lr_gan,betas=(self.args.beta, 0.999))

        optimizer_decoder = torch.optim.Adam(self.model.decoder.parameters(), self.args.lr_gan,betas=(self.args.beta, 0.999))

        optimizer_gan_discriminator=torch.optim.Adam(self.model.discriminator.parameters(), self.args.lr_gan,betas=(self.args.beta, 0.999))

        if self.args.schedule == 'Step':
            scheduler_resnet = torch.optim.lr_scheduler.StepLR(optimizer_resnet, step_size=self.args.step,
                                                           gamma=self.args.gamma)
            # scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=self.args.step,
            #                                                gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler_resnet = torch.optim.lr_scheduler.MultiStepLR(optimizer_resnet, milestones=self.args.milestones,
                                                           gamma=self.args.gamma)
            # scheduler_encoder = torch.optim.lr_scheduler.MultiStepLR(optimizer_encoder, milestones=self.args.milestones,
            #                                                gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler_resnet = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_resnet, T_max=self.args.epochs_base)
            # scheduler_encoder = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_encoder, T_max=self.args.epochs_base)


        optimizer=[optimizer_resnet,optimizer_encoder,optimizer_decoder,optimizer_gan_discriminator]

        return optimizer, scheduler_resnet

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        optimizer, scheduler = self.get_optimizer_base()

        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)

            self.model = self.update_param(self.model, self.best_model_dict)

            if session == 0:  # load base class train img label

                print('new classes for this session:\n', np.unique(train_set.targets))

                for epoch in range(args.epochs_base):
                    start_time = time.time()



                    tl, ta = classifier_train(self.model, trainloader, optimizer, scheduler, epoch, args, session)

                    # self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    # visualize(self.model, trainloader, epoch, args, session)
                    # test model with all seen class
                    tsl, tsa = test(self.model, testloader, epoch, args, session)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        # torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                if args.dataset == 'mini_imagenet' or args.dataset == 'cub200':
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    tl, ta = generator_train(self.model, trainloader, optimizer, scheduler, epoch, args, session)

                    # test model with all seen class
                    tsl, tsa = gan_test(self.model, testloader, epoch, args, session)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc_gen'][session]:
                        self.trlog['max_acc_gen'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch_gen'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        # torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test gen acc={:.3f}'.format(self.trlog['max_acc_epoch_gen'],
                                                                       self.trlog['max_acc_gen'][session]))

                    self.trlog['train_loss_gen'].append(tl)
                    self.trlog['train_acc_gen'].append(ta)
                    self.trlog['test_loss_gen'].append(tsl)
                    self.trlog['test_acc_gen'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))

            else:  # incremental learning sessions
                print("training session: [%d]" % session)

                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.update_fc_another(trainloader,np.unique(train_set.targets), session)

                tsl, tsa = test(self.model, testloader, 0, args, session)

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))


        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '/'

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)

        ## add the slurm process id
        job_id = os.environ.get('SLURM_JOB_ID')
        self.args.save_path = self.args.save_path + 'slurm_id_%s/' % str(job_id)

        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Cosine':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None

def classifier_train(model, trainloader, optimizer, scheduler, epoch, args, session):

    optimizer_resnet,optimizer_encoder, optimizer_decoder, optimizer_gan_discriminator = optimizer[0], optimizer[
        1], optimizer[2], optimizer[3]

    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        if args.use_gpu:
            data, train_label, word_embedding = [_.cuda() for _ in batch]
        else:
            data, train_label, word_embedding = [_ for _ in batch]

        model.mode = 'train_vaegan_classifier'

        Acc ,loss = model(data.squeeze(),train_label,word_embedding)

        optimizer_resnet.zero_grad()
        loss.backward()
        optimizer_resnet.step()

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'epoch {}, lr={:.4f}  classify_loss= {:.4f},  Acc_real={:.4f}'.format(epoch,lrc, loss.item(), Acc))

        tl.add(loss.item())
        ta.add(Acc)

    tl = tl.item()
    ta = ta.item()
    return tl, ta

def generator_train(model, trainloader, optimizer, scheduler, epoch, args, session):

    optimizer_resnet,optimizer_encoder, optimizer_decoder, optimizer_gan_discriminator = optimizer[0], optimizer[
        1], optimizer[2], optimizer[3]

    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)

    for i, batch in enumerate(tqdm_gen, 1):

        if args.use_gpu:
            data, train_label, word_embedding = [_.cuda() for _ in batch]
        else:
            data, train_label, word_embedding = [_ for _ in batch]

        model.mode = 'train_vaegan_generator'
        Acc_gen, encoder_loss, generator_loss, d_loss = model( data, train_label, word_embedding)

        optimizer_encoder.zero_grad()
        encoder_loss.backward(retain_graph=True)
        optimizer_encoder.step()

        optimizer_decoder.zero_grad()
        generator_loss.backward(retain_graph=True)
        optimizer_decoder.step()

        optimizer_gan_discriminator.zero_grad()
        d_loss.backward(retain_graph=True)
        optimizer_gan_discriminator.step()

        total_loss=generator_loss+d_loss

        tqdm_gen.set_description(
            'Session {},epoch {},  Loss_Enc= {:.4f}  Loss_D= {:.4f}  Loss_Dec= {:.4f}, Acc_fake={:.4f}'.format(
                session,epoch,encoder_loss.item(), d_loss.item(), generator_loss.item(),  Acc_gen))

        tl.add(total_loss.item())
        ta.add(Acc_gen)

    tl = tl.item()
    ta = ta.item()

    return tl, ta

def combine_train(model, trainloader, optimizer, scheduler, epoch, args, session):

    optimizer_resnet,optimizer_encoder, optimizer_decoder, optimizer_gan_discriminator = optimizer[0], optimizer[
        1], optimizer[2], optimizer[3]

    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)

    for i, batch in enumerate(tqdm_gen, 1):

        if args.use_gpu:
            data, train_label, word_embedding = [_.cuda() for _ in batch]
        else:
            data, train_label, word_embedding = [_ for _ in batch]

        real_feature = model.feature_extractor(data)

        logits = F.linear(F.normalize(real_feature, p=2, dim=-1),
                          F.normalize(model.feature_extractor.fc.weight, p=2, dim=-1))
        logits = args.temperature * logits[:, :args.base_class]
        loss = F.cross_entropy(logits, train_label)
        Acc = count_acc(logits, train_label)

        optimizer_resnet.zero_grad()
        loss.backward()
        optimizer_resnet.step()

        real_feature = model.feature_extractor(data)

        latent_z, mu, logvar = model.encoder(real_feature)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        recon_feature = model.decoder(z=latent_z, label=word_embedding)

        noise = torch.randn((train_label.size(0), 256)).cuda()

        gen_feature = model.decoder(z=noise, label=word_embedding)

        real_adv, real_f = model.discriminator(image=real_feature)
        gen_adv, gen_f = model.discriminator(image=gen_feature.detach())
        recon_adv, recon_f = model.discriminator(image=recon_feature.detach())

        recons_loss = F.mse_loss(recon_f, real_f)

        encoder_loss = recons_loss + kld_loss

        optimizer_encoder.zero_grad()
        encoder_loss.backward()
        optimizer_encoder.step()

        gen_logits = F.linear(F.normalize(gen_feature, p=2, dim=-1),
                              F.normalize(model.feature_extractor.fc.weight, p=2, dim=-1))
        gen_logits = args.temperature * gen_logits[:, :args.base_class]

        recon_logits = F.linear(F.normalize(recon_feature, p=2, dim=-1),
                                F.normalize(model.feature_extractor.fc.weight, p=2, dim=-1))
        recon_logits = args.temperature * recon_logits[:, :args.base_class]

        real_logits = F.linear(F.normalize(real_feature, p=2, dim=-1),
                               F.normalize(model.feature_extractor.fc.weight, p=2, dim=-1))
        real_logits = args.temperature * real_logits[:, :args.base_class]

        class_loss = F.cross_entropy(gen_logits, train_label)

        g_disc_loss = -torch.log(gen_adv + 1e-8).mean() - torch.log(recon_adv + 1e-8).mean()

        generator_loss = g_disc_loss + class_loss

        optimizer_decoder.zero_grad()
        generator_loss.backward()
        optimizer_decoder.step()

        dbinary_loss = -(torch.log(1 - gen_adv + 1e-8).mean() + torch.log(1 - recon_adv + 1e-8).mean() + torch.log(
            real_adv + 1e-8).mean())

        dclass_loss = F.cross_entropy(real_logits, train_label) + F.cross_entropy(
            gen_logits[:, :args.base_class], train_label) + F.cross_entropy(
            recon_logits[:, :args.base_class], train_label)

        gp = model.gradient_penalty(model.discriminator, real_feature, gen_feature)

        d_loss = dbinary_loss + dclass_loss + gp

        Acc_gen = count_acc(gen_logits, train_label)


        optimizer_gan_discriminator.zero_grad()
        d_loss.backward()
        optimizer_gan_discriminator.step()

        total_loss = generator_loss + d_loss

        tqdm_gen.set_description(
            'Session {},epoch {},  Loss_Enc= {:.4f}  Loss_D= {:.4f}  Loss_Dec= {:.4f}, Acc_fake={:.4f}, Acc_real={:.4f}'.format(
                session,epoch,encoder_loss.item(), d_loss.item(), generator_loss.item(),  Acc_gen, Acc))

        tl.add(total_loss.item())
        ta.add(Acc)

    tl = tl.item()
    ta = ta.item()

    return tl, ta


def gan_test(model, testloader, epoch, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            if args.use_gpu:
                data, test_label, word_embedding = [_.cuda() for _ in batch]
            else:
                data, test_label, word_embedding = [_ for _ in batch]

            model.mode = 'test_vaegan_gen'

            acc = model(data, test_label, word_embedding, session)

            tqdm_gen.set_description(
                'epoch {},  Acc_com={:.4f}'.format(epoch,  acc))

            vl.add(0)
            va.add(acc)

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va

def test(model, testloader, epoch, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            if args.use_gpu:
                data, test_label, word_embedding = [_.cuda() for _ in batch]
            else:
                data, test_label, word_embedding = [_ for _ in batch]

            model.mode = 'test_vaegan'

            acc = model(data, test_label, word_embedding, session)

            tqdm_gen.set_description(
                'epoch {},  Acc_com={:.4f}'.format(epoch,  acc))

            vl.add(0)
            va.add(acc)

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va

def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.test_batch_size,
                                              num_workers=args.num_workers, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label, _ = [_.cuda() for _ in batch]

            embedding = model.feature_extractor(data).squeeze()

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.feature_extractor.fc.weight.data[:args.base_class] = proto_list

    return model


def visualize(model, testloader, epoch, args, session):
    # replace fc.weight with the embedding average of train data
    model = model.eval()
    dataset_embeddings = []
    fake_embeddings = []
    true_labels = []
    fake_labels = []
    # data_list=[]
    # with torch.no_grad():
    for i, batch in enumerate(testloader):
        data, label, word_embedding = [_.cuda() for _ in batch]

        real_embedding = model.feature_extractor(data).squeeze()

        # Convert embeddings and labels to numpy arrays
        dataset_embeddings.append(real_embedding.detach().cpu().numpy())
        true_labels.append(label.cpu().numpy())

        noise = torch.randn((label.size(0), 256)).cuda()
        gen_feature = model.decoder(z=noise, label=word_embedding)

        fake_embeddings.append(gen_feature.detach().cpu().numpy())
        fake_labels.append(label.cpu().numpy())

    dataset_embeddings = np.concatenate(dataset_embeddings, axis=0)
    dataset_embeddings = (dataset_embeddings - dataset_embeddings.mean(axis=0)) / dataset_embeddings.std(axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    fake_embeddings = np.concatenate(fake_embeddings, axis=0)
    fake_embeddings = (fake_embeddings - fake_embeddings.mean(axis=0)) / fake_embeddings.std(axis=0)
    fake_labels = np.concatenate(fake_labels, axis=0)

    class_indices = np.where((true_labels >= 1) & (true_labels <= 10))[0]
    dataset_embeddings = dataset_embeddings[class_indices]
    true_labels = true_labels[class_indices]
    fake_embeddings = fake_embeddings[class_indices]
    fake_labels = fake_labels[class_indices]

    # Reduce the dimensionality of real embeddings using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    real_embeddings_2d = tsne.fit_transform(dataset_embeddings)

    # Create a scatter plot of the t-SNE embeddings for real images
    plt.scatter(real_embeddings_2d[:, 0], real_embeddings_2d[:, 1], c=true_labels)
    plt.title("t-SNE Visualization of Real Images")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

    # Concatenate real and fake embeddings and labels
    combined_embeddings = np.concatenate([dataset_embeddings, fake_embeddings], axis=0)
    combined_labels = np.concatenate([true_labels, fake_labels], axis=0)

    # Reduce the dimensionality of combined embeddings using t-SNE
    combined_embeddings_2d = tsne.fit_transform(combined_embeddings)

    # Create a scatter plot of the t-SNE embeddings for real and fake images
    plt.scatter(combined_embeddings_2d[:, 0], combined_embeddings_2d[:, 1], c=combined_labels)
    plt.title("t-SNE Visualization of Real and Fake Images")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()
    print('aa')
