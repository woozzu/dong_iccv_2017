import argparse
import fasttext

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

from model import VisualSemanticEmbedding
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
parser.add_argument('--save_filename', type=str, required=True,
                    help='checkpoint file')
parser.add_argument('--num_threads', type=int, default=4,
                    help='number of threads for fetching data (default: 4)')
parser.add_argument('--num_epochs', type=int, default=300,
                    help='number of threads for fetching data (default: 300)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size (default: 64)')
parser.add_argument('--learning_rate', type=float, default=0.0002,
                    help='learning rate (dafault: 0.0002)')
parser.add_argument('--margin', type=float, default=0.2,
                    help='margin for pairwise ranking loss (default: 0.2)')
parser.add_argument('--embed_ndim', type=int, default=300,
                    help='dimension of embedded vector (default: 300)')
parser.add_argument('--max_nwords', type=int, default=50,
                    help='maximum number of words (default: 50)')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
args = parser.parse_args()

if not args.no_cuda and not torch.cuda.is_available():
    print('Warning: cuda is not available on this machine.')
    args.no_cuda = True


def pairwise_ranking_loss(margin, x, v):
    zero = torch.zeros(1)
    diag_margin = margin * torch.eye(x.size(0))
    if not args.no_cuda:
        zero, diag_margin = zero.cuda(), diag_margin.cuda()
    zero, diag_margin = Variable(zero), Variable(diag_margin)

    x = x / torch.norm(x, 2, 1, keepdim=True)
    v = v / torch.norm(v, 2, 1, keepdim=True)
    prod = torch.matmul(x, v.transpose(0, 1))
    diag = torch.diag(prod)
    for_x = torch.max(zero, margin - torch.unsqueeze(diag, 1) + prod) - diag_margin
    for_v = torch.max(zero, margin - torch.unsqueeze(diag, 0) + prod) - diag_margin
    return (torch.sum(for_x) + torch.sum(for_v)) / x.size(0)


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
                                  transforms.Scale(256),
                                  transforms.RandomCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])
                              ]))

    word_embedding = None

    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_threads)

    model = VisualSemanticEmbedding(args.embed_ndim)
    if not args.no_cuda:
        model.cuda()

    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                 lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        avg_loss = 0
        for i, (img, desc, len_desc) in enumerate(train_loader):
            img = Variable(img.cuda() if not args.no_cuda else img)
            desc = Variable(desc.cuda() if not args.no_cuda else desc)
            len_desc, indices = torch.sort(len_desc, 0, True)
            indices = indices.numpy()
            img = img[indices, ...]
            desc = desc[indices, ...].transpose(0, 1)
            desc = nn.utils.rnn.pack_padded_sequence(desc, len_desc.numpy())

            optimizer.zero_grad()
            img_feat, txt_feat = model(img, desc)
            loss = pairwise_ranking_loss(args.margin, img_feat, txt_feat)
            avg_loss += loss.data[0]
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                      % (epoch + 1, args.num_epochs, i + 1, len(train_loader), avg_loss / (i + 1)))

        torch.save(model.state_dict(), args.save_filename)
