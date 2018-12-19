import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn
from tqdm import tqdm_notebook
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Code.GAN.mixture_gaussian import data_generator
plt.style.use('ggplot')


def plot(points, title):
    plt.scatter(points[:, 0], points[:, 1], s=10, c='b', alpha=0.5)
    plt.scatter(dset.centers[:, 0], dset.centers[:, 1], s=100, c='g', alpha=0.5)
    plt.title(title)
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.show()
    plt.close()

def noise_sampler(N, z_dim):
    return np.random.normal(size=[N, z_dim]).astype('float32')


def d_loop():
    # 1. Train D on real+fake
    d_optimizer.zero_grad()

    #  1A: Train D on real
    d_real_data = torch.from_numpy(dset.sample(minibatch_size))
    if cuda:
        d_real_data = d_real_data.cuda()
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision)
    if cuda:
        target = target.cuda()
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  1B: Train D on fake
    d_gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    if cuda:
        d_gen_input = d_gen_input.cuda()

    with torch.no_grad():
        d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision)
    if cuda:
        target = target.cuda()
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_loss = d_real_error + d_fake_error
    d_loss.backward()
    d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
    return d_real_error.cpu().item(), d_fake_error.cpu().item()


def d_unrolled_loop(d_gen_input=None):
    # 1. Train D on real+fake
    d_optimizer.zero_grad()

    #  1A: Train D on real
    d_real_data = torch.from_numpy(dset.sample(minibatch_size))
    if cuda:
        d_real_data = d_real_data.cuda()
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision)
    if cuda:
        target = target.cuda()
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  1B: Train D on fake
    if d_gen_input is None:
        d_gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    if cuda:
        d_gen_input = d_gen_input.cuda()

    with torch.no_grad():
        d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision)
    if cuda:
        target = target.cuda()
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_loss = d_real_error + d_fake_error
    d_loss.backward(create_graph=True)
    d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
    return d_real_error.cpu().item(), d_fake_error.cpu().item()


def g_loop():
    # 2. Train G on D's response (but DO NOT train D on these labels)
    g_optimizer.zero_grad()

    gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    if cuda:
        gen_input = gen_input.cuda()

    if unrolled_steps > 0:
        backup = copy.deepcopy(D)
        for i in range(unrolled_steps):
            d_unrolled_loop(d_gen_input=gen_input)

    g_fake_data = G(gen_input)
    dg_fake_decision = D(g_fake_data)
    target = torch.ones_like(dg_fake_decision)
    if cuda:
        target = target.cuda()
    g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
    g_error.backward()
    g_optimizer.step()  # Only optimizes G's parameters

    if unrolled_steps > 0:
        D.load(backup)
        del backup

    return g_error.cpu().item()

def g_sample():
    with torch.no_grad():
        gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
        if cuda:
            gen_input = gen_input.cuda()
        g_fake_data = G(gen_input)
        return g_fake_data.cpu().numpy()


if torch.cuda.is_available():
    cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
else:
    cuda = False

## choose uniform mixture gaussian or weighted mixture gaussian
dset = data_generator()
dset.random_distribution()
# dset.uniform_distribution()

plt.plot(dset.p)
plt.title('Weight of each gaussian')
plt.show()
plt.close()



sample_points = dset.sample(100)
plot(sample_points, 'Sampled data points')


# Model params (most of hyper-params follow the original paper: https://arxiv.org/abs/1611.02163)
z_dim = 256
g_inp = z_dim
g_hid = 128
g_out = dset.size

d_inp = g_out
d_hid = 128
d_out = 1

minibatch_size = 512

unrolled_steps = 10
d_learning_rate = 1e-4
g_learning_rate = 1e-3
optim_betas = (0.5, 0.999)
num_iterations = 3000
log_interval = 300
d_steps = 1
g_steps = 1

prefix = "unrolled_steps-{}-prior_std-{:.2f}".format(unrolled_steps, np.std(dset.p))
print("Save file with prefix", prefix)


###### MODELS: Generator model and discriminator model
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = F.tanh
        # self.activation_fn = F.relu

    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        return self.map3(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = F.relu

    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        return F.sigmoid(self.map3(x))

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

G = Generator(input_size=g_inp, hidden_size=g_hid, output_size=g_out)
D = Discriminator(input_size=d_inp, hidden_size=d_hid, output_size=d_out)
if cuda:
    G = G.cuda()
    D = D.cuda()
criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)

samples = []
for it in range(num_iterations):
    d_infos = []
    for d_index in range(d_steps):
        d_info = d_loop()
        d_infos.append(d_info)
    d_infos = np.mean(d_infos, 0)
    d_real_loss, d_fake_loss = d_infos

    g_infos = []
    for g_index in range(g_steps):
        g_info = g_loop()
        g_infos.append(g_info)
    g_infos = np.mean(g_infos)
    g_loss = g_infos

    if it % log_interval == 0:
        g_fake_data = g_sample()
        samples.append(g_fake_data)
        plot(g_fake_data, title='[{}] Iteration {}'.format(prefix, it))
        print(d_real_loss, d_fake_loss, g_loss)


# plot the samples through iterations
def plot_samples(samples):
    xmax = 5
    cols = len(samples)
    bg_color = seaborn.color_palette('Greens', n_colors=256)[0]
    plt.figure(figsize=(2 * cols, 2))
    for i, samps in enumerate(samples):
        if i == 0:
            ax = plt.subplot(1, cols, 1)
        else:
            plt.subplot(1, cols, i + 1, sharex=ax, sharey=ax)
        ax2 = seaborn.kdeplot(samps[:, 0], samps[:, 1], shaded=True, cmap='Greens', n_levels=20,
                              clip=[[-xmax, xmax]] * 2)
        plt.xticks([])
        plt.yticks([])
        plt.title('step %d' % (i * log_interval))

    ax.set_ylabel('%d unrolling steps' % unrolled_steps)
    plt.gcf().tight_layout()
    plt.savefig(prefix + '.png')
    plt.show()
    plt.close()

plot_samples(samples)