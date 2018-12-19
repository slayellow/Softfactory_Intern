import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import Code.utils.common_function as cf
import torch.utils.data
print(torch.cuda.is_available())
# Device configuration
class GAN():
    """Generate Adversarial Network Class

   Parameters
   ----------

   Returns
   -------
   """

    def __init__(self, latent_size, hidden_size, wave_size, total_epoch, batch_size, learning_rate, sample_path):
        """Class Constructor

       Parameters
       ----------
        latent_size: generator input size
        hidden_size: hidden_layer size
        wave_size: discriminator input size
        total_epoch: The number of epoch
        batch_size: mini batch size
        learning_rate: optimizer hyper parameter
        sample_path: image save path
       Returns
       -------
       """
        self.cf = cf.CommonFunc()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.wave_size = wave_size
        self.num_epochs = total_epoch
        self.batch_size = batch_size
        self.sample_dir = sample_path
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cf.make_dir(self.sample_dir)
        self.process()

    def min_max_denorm(self, origin_X, norm_X):
        """Denormalize data

       Parameters
       ----------
       origin_X: Original Data
       norm_X: Normalized Data
       Returns
       -------
       Denormalized Data of list type
       """
        max_data = max(origin_X)
        min_data = min(origin_X)
        return [int((data * (max_data - min_data))) + min_data for data in norm_X]

    def reset_grad(self, d_optimizer, g_optimizer):
        """Reset Gradient Descent Parameter

        Parameters
        ----------
        d_optimizer: Discriminator Optimizer
        g_optimizer: Generator Optimizer
        Returns
        -------
        """
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

    def min_max_norm(self, X):
        """Min-Max Normalize Data

        Parameters
        ----------
        X: Data to normalize
        Returns
        -------
        norm_X: Normalized Data of list type
        """
        norm_X = []
        for x_data in X:
            max_data = max(x_data)
            min_data = min(x_data)
            temp = []
            for x in x_data:
                norm_x = (x - min_data) / (max_data - min_data)
                temp.append(norm_x)
            norm_X.append(temp)
            temp = []
        return norm_X

    def process(self):
        """Process Flow

        Parameters
        ----------

        Returns
        -------
        """

        origin_data = self.cf.open_csv('../../Swell_Data.csv')
        data = self.min_max_norm(origin_data)
        data = torch.Tensor(data).to(self.device)
        data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=self.batch_size, shuffle=True)
        D = self.Discriminator().to(self.device)
        G = self.Generator().to(self.device)
        criterion = self.Cross_Entropy_Criterion()
        d_optimizer = self.Adam_Optimizer(D.parameters(), self.learning_rate)
        g_optimizer = self.Adam_Optimizer(G.parameters(), self.learning_rate)
        self.Training(origin_data, data_loader, D, G, criterion, d_optimizer, g_optimizer)

    def Discriminator(self):
        """Discriminator Model

        Parameters
        ----------
        Returns
        -------
        D: Neural Network Model
        """

        D = nn.Sequential(
            nn.Linear(self.wave_size, self.hidden_size),
            nn.LeakyReLU(0.2),
            # nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid())
        return D

    def Generator(self):
        """Generator Model

        Parameters
        ----------

        Returns
        -------
        G: Neural Network Model
        """

        G = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_size),
            # nn.ReLU(),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size, self.wave_size),
            nn.Tanh())
        return G

    def Cross_Entropy_Criterion(self):
        """Cross Entropy Loss Function

        Parameters
        ----------
        Returns
        -------
        Cross_Entropy Function
        """

        return nn.BCELoss()

    def Adam_Optimizer(self, hypothesis, learning_rate):
        """Adam Optimizer Function

        Parameters
        ----------
        hypothesis: Neural Network Model
        learning_rate: Learning Rate
        Returns
        -------
        Optimizer Function
        """
        return torch.optim.Adam(hypothesis, lr=learning_rate)

    def Training(self, origin_data, data_loader, D, G, criterion, d_optimizer, g_optimizer):
        """GAN Train

        Parameters
        ----------
        origin_data: original data
        data_loader: load data function
        D: Discriminator Model
        G: Generator Model
        criterion: Loss Function
        d_optimizer: Discriminator Adam Optimizer
        g_optimizer: Generator Adam Optimizer
        Returns
        -------

        """
        d_loss_list = []
        g_loss_list = []
        d_accuracy = []
        dg_accuracy = []
        total_step = len(data_loader)

        for epoch in range(self.num_epochs):
            for i, images in enumerate(data_loader):
                # Create the labels which are later used as input for the BCE loss
                real_labels = torch.ones(self.batch_size, 1).to(self.device)
                fake_labels = torch.zeros(self.batch_size, 1).to(self.device)
                outputs = D(images)
                d_loss_real = criterion(outputs, real_labels)
                real_score = outputs
                z = torch.randn(self.batch_size, self.latent_size).to(self.device)
                fake_images = G(z)
                outputs = D(fake_images)
                d_loss_fake = criterion(outputs, fake_labels)
                fake_score = outputs
                d_loss = d_loss_real + d_loss_fake
                self.reset_grad(d_optimizer, g_optimizer)
                d_loss.backward()
                d_optimizer.step()
                z = torch.randn(self.batch_size, self.latent_size).to(self.device)
                fake_images = G(z)
                outputs = D(fake_images)
                g_loss = criterion(outputs, real_labels)
                self.reset_grad(d_optimizer, g_optimizer)
                g_loss.backward()
                g_optimizer.step()
                if (i + 1) % 4 == 0:
                    d_loss_list.append(d_loss.item())
                    g_loss_list.append(g_loss.item())
                    d_accuracy.append(real_score.mean().item())
                    dg_accuracy.append(fake_score.mean().item())
                    print(
                        'Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'.format(
                            epoch, self.num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                            real_score.mean().item(), fake_score.mean().item()))
                if (epoch + 1) == 1:
                    plt.figure()
                    image = images[0].cpu().numpy()
                    image = self.min_max_denorm(origin_data[0], image)
                    plt.plot(range(32), image)
                    plt.savefig(os.path.join(self.sample_dir, 'real_image.png'))
                    plt.close()
                if (epoch + 1) % 20 == 0:
                    plt.figure()
                    fake_image = fake_images[0].cpu().detach().numpy()
                    fake_image = self.min_max_denorm(origin_data[50], fake_image)
                    plt.plot(range(32), fake_image)
                    plt.savefig(os.path.join(self.sample_dir, 'fake_images_{}.jpg'.format(epoch + 1)))
                    plt.close()
        plt.figure()
        plt.plot(range(len(d_loss_list)), d_loss_list)
        plt.savefig('D_LOSS.png')
        plt.clf()
        plt.plot(range(len(g_loss_list)), g_loss_list)
        plt.savefig('G_LOSS.png')
        plt.clf()
        plt.plot(range(len(d_accuracy)), d_accuracy)
        plt.savefig('D_ACCURACY.png')
        plt.clf()
        plt.plot(range(len(dg_accuracy)), dg_accuracy)
        plt.savefig('DG_ACCURACY.png')
        plt.close()
        # Save the model checkpoints
        torch.save(G.state_dict(), 'G.ckpt')
        torch.save(D.state_dict(), 'D.ckpt')


gan = GAN(4, 16, 32, 200, 60, 0.009, 'sample/')



