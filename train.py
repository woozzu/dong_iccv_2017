import argparse
import fasttext
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import VisualSemanticEmbedding
from model import Generator, Discriminator
from data import ReedICML2016


parser = argparse.ArgumentParser()
parser.add_argument('--img_root', type=str, required=True,
                    help='root directory that contains images')
parser.add_argument('--caption_root', type=str, required=True,
                    help='root directory that contains captions')
parser.add_argument('--trainclasses_file', type=str, required=True,
                    help='text file that contains training classes')
parser.add_argument('--fasttext_model', type=str, required=True,
                    help='pretrained fastText model (binary file)')
parser.add_argument('--text_embedding_model', type=str, required=True,
                    help='pretrained text embedding model')
parser.add_argument('--save_filename', type=str, required=True,
                    help='checkpoint file')
parser.add_argument('--num_threads', type=int, default=4,
                    help='number of threads for fetching data (default: 4)')
parser.add_argument('--num_epochs', type=int, default=600,
                    help='number of threads for fetching data (default: 600)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size (default: 64)')
parser.add_argument('--learning_rate', type=float, default=0.0002,
                    help='learning rate (dafault: 0.0002)')
parser.add_argument('--lr_decay', type=float, default=0.5,
                    help='learning rate decay (dafault: 0.5)')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='beta1 for Adam optimizer (dafault: 0.5)')
parser.add_argument('--embed_ndim', type=int, default=300,
                    help='dimension of embedded vector (default: 300)')
parser.add_argument('--max_nwords', type=int, default=50,
                    help='maximum number of words (default: 50)')
parser.add_argument('--use_vgg', action='store_true',
                    help='use pretrained VGG network for image encoder')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
args = parser.parse_args()

if not args.no_cuda and not torch.cuda.is_available():
    print('Warning: cuda is not available on this machine.')
    args.no_cuda = True


def preprocess(img, desc, len_desc, txt_encoder):
    img = Variable(img.cuda() if not args.no_cuda else img)
    desc = Variable(desc.cuda() if not args.no_cuda else desc)

    len_desc = len_desc.numpy()
    sorted_indices = np.argsort(len_desc)[::-1]
    original_indices = np.argsort(sorted_indices)
    packed_desc = nn.utils.rnn.pack_padded_sequence(
        desc[sorted_indices, ...].transpose(0, 1),
        len_desc[sorted_indices]
    )
    _, txt_feat = txt_encoder(packed_desc)
    txt_feat = txt_feat.squeeze()
    txt_feat = txt_feat[original_indices, ...]

    txt_feat_np = txt_feat.data.cpu().numpy() if not args.no_cuda else txt_feat.data.numpy()
    txt_feat_mismatch = torch.Tensor(np.roll(txt_feat_np, 1, axis=0))
    txt_feat_mismatch = Variable(txt_feat_mismatch.cuda() if not args.no_cuda else txt_feat_mismatch)
    txt_feat_np_split = np.split(txt_feat_np, [txt_feat_np.shape[0] // 2])
    txt_feat_relevant = torch.Tensor(np.concatenate([
        np.roll(txt_feat_np_split[0], -1, axis=0),
        txt_feat_np_split[1]
    ]))
    txt_feat_relevant = Variable(txt_feat_relevant.cuda() if not args.no_cuda else txt_feat_relevant)
    return img, txt_feat, txt_feat_mismatch, txt_feat_relevant


if __name__ == '__main__':
    print('Loading a pretrained fastText model...')
    word_embedding = fasttext.load_model(args.fasttext_model)

    print('Loading a dataset...')
    train_data = ReedICML2016(args.img_root,
                              args.caption_root,
                              args.trainclasses_file,
                              word_embedding,
                              args.max_nwords,
                              transforms.Compose([
                                  transforms.Scale(74),
                                  transforms.RandomCrop(64),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor()
                              ]))
    vgg_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_threads)

    word_embedding = None

    # pretrained text embedding model
    print('Loading a pretrained text embedding model...')
    txt_encoder = VisualSemanticEmbedding(args.embed_ndim)
    txt_encoder.load_state_dict(torch.load(args.text_embedding_model))
    txt_encoder = txt_encoder.txt_encoder
    for param in txt_encoder.parameters():
        param.requires_grad = False

    G = Generator(use_vgg=args.use_vgg)
    D = Discriminator()

    if not args.no_cuda:
        txt_encoder.cuda()
        G.cuda()
        D.cuda()

    g_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, G.parameters()),
                                   lr=args.learning_rate, betas=(args.momentum, 0.999))
    d_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, D.parameters()),
                                   lr=args.learning_rate, betas=(args.momentum, 0.999))
    g_lr_scheduler = lr_scheduler.StepLR(g_optimizer, 100, args.lr_decay)
    d_lr_scheduler = lr_scheduler.StepLR(d_optimizer, 100, args.lr_decay)

    for epoch in range(args.num_epochs):
        d_lr_scheduler.step()
        g_lr_scheduler.step()

        # training loop
        avg_D_real_loss = 0
        avg_D_real_m_loss = 0
        avg_D_fake_loss = 0
        avg_G_fake_loss = 0
        avg_kld = 0
        for i, (img, desc, len_desc) in enumerate(train_loader):
            img, txt_feat, txt_feat_mismatch, txt_feat_relevant = \
                preprocess(img, desc, len_desc, txt_encoder)
            img_norm = img * 2 - 1
            img_G = Variable(vgg_normalize(img.data)) if args.use_vgg else img_norm

            ONES = Variable(torch.ones(img.size(0)))
            ZEROS = Variable(torch.zeros(img.size(0)))
            if not args.no_cuda:
                ONES, ZEROS = ONES.cuda(), ZEROS.cuda()

            # UPDATE DISCRIMINATOR
            D.zero_grad()
            # real image with matching text
            real_logit = D(img_norm, txt_feat)
            real_loss = F.binary_cross_entropy_with_logits(real_logit, ONES)
            avg_D_real_loss += real_loss.data[0]
            real_loss.backward()
            # real image with mismatching text
            real_m_logit = D(img_norm, txt_feat_mismatch)
            real_m_loss = 0.5 * F.binary_cross_entropy_with_logits(real_m_logit, ZEROS)
            avg_D_real_m_loss += real_m_loss.data[0]
            real_m_loss.backward()
            # synthesized image with semantically relevant text
            fake, _ = G(img_G, txt_feat_relevant)
            fake_logit = D(fake.detach(), txt_feat_relevant)
            fake_loss = 0.5 * F.binary_cross_entropy_with_logits(fake_logit, ZEROS)
            avg_D_fake_loss += fake_loss.data[0]
            fake_loss.backward()
            d_optimizer.step()

            # UPDATE GENERATOR
            G.zero_grad()
            fake, (z_mean, z_log_stddev) = G(img_G, txt_feat_relevant)
            kld = torch.mean(-z_log_stddev + 0.5 * (torch.exp(2 * z_log_stddev) + torch.pow(z_mean, 2) - 1))
            avg_kld += kld.data[0]
            fake_logit = D(fake, txt_feat_relevant)
            fake_loss = F.binary_cross_entropy_with_logits(fake_logit, ONES)
            avg_G_fake_loss += fake_loss.data[0]
            G_loss = fake_loss + kld
            G_loss.backward()
            g_optimizer.step()

            if i % 10 == 0:
                print('Epoch [%d/%d], Iter [%d/%d], D_real: %.4f, D_mis: %.4f, D_fake: %.4f, G_fake: %.4f, KLD: %.4f'
                      % (epoch + 1, args.num_epochs, i + 1, len(train_loader), avg_D_real_loss / (i + 1),
                      avg_D_real_m_loss / (i + 1), avg_D_fake_loss / (i + 1), avg_G_fake_loss / (i + 1), avg_kld / (i + 1)))

        save_image((fake.data + 1) * 0.5, './examples/epoch_%d.png' % (epoch + 1))
        torch.save(G.state_dict(), args.save_filename)
