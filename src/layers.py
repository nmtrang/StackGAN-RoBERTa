"""Assortment of layers for use in models.py.
Refer to StackGAN paper: https://arxiv.org/pdf/1612.03242.pdf 
for variable names and working.
"""
import torch
import torch.nn as nn

from torch.autograd import Variable

__all__ = ['Stage1Generator', 'Stage1Discriminator',
           'Stage2Generator', 'Stage2Discriminator']


def conv3x3(in_channels, out_channels):
    """3x3 conv with same padding"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    )


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x) -> nn.ReLU:
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


def _downsample(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _upsample(in_channels, out_channels):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class CAug(nn.Module):
    """Module for conditional augmentation.
    Takes input as roberta embeddings of annotations and sends output to Stage 1 and 2 generators.
    """

    def __init__(self, emb_dim=768, n_g=128, device="cuda"):  # ! CHANGE THIS TO CUDA
        """
        @param emb_dim (int)            : Size of annotation embeddings.
        @param n_g      (int)           : Dimension of mu, epsilon and c_0_hat
        @param device   (torch.device)  : cuda/cpu
        """
        super(CAug, self).__init__()
        self.emb_dim = emb_dim
        self.n_g = n_g
        self.fc = nn.Linear(
            self.emb_dim, self.n_g * 2, bias=True
        )  # To split in mu and sigma
        self.relu = nn.ReLU()
        self.device = device

    def forward(self, text_emb):
        """
        @param   text_emb (torch.tensor): Text embedding.                 (batch, emb_dim)
        @returns c_0_hat  (torch.tensor): Gaussian conditioning variable. (batch, n_g)
        """
        enc = self.relu(self.fc(text_emb)).squeeze(1)  # (batch, n_g*2)

        mu = enc[:, : self.n_g]  # (batch, n_g)
        logvar = enc[:, self.n_g:]  # (batch, n_g)

        sigma = (logvar * 0.5).exp_()
        # exp(logvar * 0.5) = exp(log(var^0.5)) = sqrt(var) = std

        epsilon = Variable(torch.FloatTensor(sigma.size()).normal_())

        c_0_hat = epsilon.to(self.device) * sigma + mu  # (batch, n_g)

        return c_0_hat, mu, logvar


######################### STAGE 1 #########################


class Stage1Generator(nn.Module):
    """
    Stage 1 generator.
    Takes in input from Conditional Augmentation and outputs 64x64 image to Stage1Discrimantor.
    """

    def __init__(self, n_g=128, n_z=100, emb_dim=768):
        """
        @param n_g (int) : Dimension of c_0_hat.
        @param n_z (int) : Dimension of noise vector.
        """
        super(Stage1Generator, self).__init__()
        self.n_g = n_g
        self.n_z = n_z
        self.emb_dim = emb_dim
        self.inp_ch = self.n_g * 8

        # (batch, bert_size) -> (batch, n_g)
        self.caug = CAug(emb_dim=self.emb_dim)

        # (batch, n_g + n_z) -> (batch, inp_ch * 4 * 4)
        self.fc = nn.Sequential(  # feature extractor
            nn.Linear(self.n_g + self.n_z, self.inp_ch * 4 * 4, bias=False),
            nn.BatchNorm1d(self.inp_ch * 4 * 4),
            nn.ReLU(True),
        )

        # (batch, inp_ch, 4, 4) -> (batch, inp_ch//2, 8, 8)
        self.up1 = _upsample(self.inp_ch, self.inp_ch // 2)
        # -> (batch, inp_ch//4, 16, 16)
        self.up2 = _upsample(self.inp_ch // 2, self.inp_ch // 4)
        # -> (batch, inp_ch//8, 32, 32)
        self.up3 = _upsample(self.inp_ch // 4, self.inp_ch // 8)
        # -> (batch, inp_ch//16, 64, 64)
        self.up4 = _upsample(self.inp_ch // 8, self.inp_ch // 16)

        # -> (batch, 3, 64, 64)
        self.img = nn.Sequential(conv3x3(self.inp_ch // 16, 3), nn.Tanh())

    def forward(self, text_emb, noise):
        """
        @param   c_0_hat (torch.tensor) : Output of Conditional Augmentation (batch, n_g)
        @returns out     (torch.tensor) : Generator 1 image output           (batch, 3, 64, 64)
        """
        c_0_hat, mu, logvar = self.caug(text_emb)

        # -> (batch, n_g + n_z) (batch, 128 + 100)
        c_z = torch.cat((c_0_hat, noise), dim=1)

        # -> (batch, 1024 * 4 * 4)
        inp = self.fc(c_z)

        # -> (batch, 1024, 4, 4)
        inp = inp.view(-1, self.inp_ch, 4, 4)

        inp = self.up1(inp)  # (batch, 512, 8, 8)
        inp = self.up2(inp)  # (batch, 256, 16, 16)
        inp = self.up3(inp)  # (batch, 128, 32, 32)
        inp = self.up4(inp)  # (batch, 64, 64, 64)

        fake_img = self.img(inp)  # (batch, 3, 64, 64)
        
        return None, fake_img, mu, logvar
    
    # @property
    # def fake_img(self):
    #     return self._fake_img


class Stage1Discriminator(nn.Module):
    """
    Stage 1 discriminator
    """

    def __init__(self, n_d=128, m_d=4, emb_dim=768, img_dim=64):
        super(Stage1Discriminator, self).__init__()
        self.n_d = n_d
        self.m_d = m_d
        self.emb_dim = emb_dim

        self.fc_for_text = nn.Linear(self.emb_dim, self.n_d)
        self.down_sample = nn.Sequential(
            # (batch, 3, 64, 64) -> (batch, img_dim, 32, 32)
            nn.Conv2d(
                3, img_dim, kernel_size=4, stride=2, padding=1, bias=False
            ),  # (batch, 64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),
            # -> (batch, img_dim * 2, 16, 16)
            _downsample(img_dim, img_dim * 2),  # (batch, 128, 16, 16)
            # -> (batch, img_dim * 4, 8, 8)
            _downsample(img_dim * 2, img_dim * 4),  # (batch, 256, 8, 8)
            # -> (batch, img_dim * 8, 4, 4)
            _downsample(img_dim * 4, img_dim * 8),  # (batch, 512, 4, 4)
        )

        self.out_logits = nn.Sequential(
            # (batch, img_dim*8 + n_d, 4, 4) -> (batch, img_dim*8, 4, 4)
            conv3x3(img_dim * 8 + self.n_d, img_dim * 8),
            nn.BatchNorm2d(img_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (batch, 1)
            nn.Conv2d(img_dim * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid(),
        )

    def forward(self, text_emb, img):
        # image encode
        enc = self.down_sample(img)

        # text emb
        compressed = self.fc_for_text(text_emb)
        compressed = (
            compressed.unsqueeze(2).unsqueeze(
                3).repeat(1, 1, self.m_d, self.m_d)
        )

        con = torch.cat((enc, compressed), dim=1)

        output = self.out_logits(con)
        return output.view(-1)


######################### STAGE 2 #########################
class Stage2Generator(nn.Module):
    """
    Stage 2 generator.
    Takes in input from Conditional Augmentation and outputs 256x256 image to Stage2Discrimantor.
    """

    def __init__(self, stage1_gen, n_g=128, n_z=100, ef_size=128, n_res=4, emb_dim=768):
        """
        @param n_g (int) : Dimension of c_0_hat.
        """
        super(Stage2Generator, self).__init__()
        self.n_g = n_g
        self.n_z = n_z
        self.ef_size = ef_size
        self.n_res = n_res
        self.emb_dim = emb_dim

        self.stage1_gen = stage1_gen
        # Freezing the stage 1 generator:
        for param in self.stage1_gen.parameters():
            param.requires_grad = False

        # (batch, bert_size) -> (batch, n_g)
        self.caug = CAug(emb_dim=self.emb_dim)

        # -> (batch, n_g*4, 16, 16)
        self.encoder = nn.Sequential(
            conv3x3(3, n_g),  # (batch, 128, 64, 64)
            nn.LeakyReLU(0.2, inplace=True),  # ? Paper: leaky, code: relu
            _downsample(n_g, n_g * 2),  # (batch, 256, 32, 32)
            _downsample(n_g * 2, n_g * 4),  # (batch, 512, 16, 16)
        )

        # (batch, ef_size + n_g * 4, 16, 16) -> (batch, n_g * 4, 16, 16)
        # (batch, 128 + 512, 16, 16) -> (batch, 512, 16, 16)
        self.cat_conv = nn.Sequential(
            conv3x3(self.ef_size + self.n_g * 4, self.n_g * 4),
            nn.BatchNorm2d(self.n_g * 4),
            nn.ReLU(inplace=True),
        )

        # -> (batch, n_g * 4, 16, 16)
        # (batch, 512, 16, 16)
        self.residual = nn.Sequential(
            *[ResBlock(self.n_g * 4) for _ in range(self.n_res)]
        )

        # -> (batch, n_g * 2, 32, 32)
        self.up1 = _upsample(n_g * 4, n_g * 2)  # (batch, 256, 32, 32)
        # -> (batch, n_g, 64, 64)
        self.up2 = _upsample(n_g * 2, n_g)  # (batch, 128, 64, 64)
        # -> (batch, n_g // 2, 128, 128)
        self.up3 = _upsample(n_g, n_g // 2)  # (batch, 64, 128, 128)
        # -> (batch, n_g // 4, 256, 256)
        self.up4 = _upsample(n_g // 2, n_g // 4)  # (batch, 32, 256, 256)

        # (batch, 3, 256, 256)
        self.img = nn.Sequential(conv3x3(n_g // 4, 3), nn.Tanh())

    def forward(self, text_emb, noise):
        """
        @param   c_0_hat  (torch.tensor) : Output of Conditional Augmentation (batch, n_g)
        @param   s1_image (torch.tensor) : Ouput of Stage 1 Generator         (batch, 3, 64, 64)
        @returns out      (torch.tensor) : Generator 2 image output           (batch, 3, 256, 256)
        """
        _, stage1_img, _, _ = self.stage1_gen(text_emb, noise)
        stage1_img = stage1_img.detach()

        encoded_img = self.encoder(stage1_img)

        c_0_hat, mu, logvar = self.caug(text_emb)
        c_0_hat = c_0_hat.unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16)

        # -> (batch, ef_size + n_g * 4, 16, 16) # (batch, 640, 16, 16)
        concat_out = torch.cat((encoded_img, c_0_hat), dim=1)

        # -> (batch, n_g * 4, 16, 16)
        h_out = self.cat_conv(concat_out)
        h_out = self.residual(h_out)

        h_out = self.up1(h_out)
        h_out = self.up2(h_out)
        h_out = self.up3(h_out)
        # -> (batch, ng // 4, 256, 256)
        h_out = self.up4(h_out)

        # -> (batch, 3, 256, 256)
        fake_img = self.img(h_out)

        return stage1_img, fake_img, mu, logvar


class Stage2Discriminator(nn.Module):
    """
    Stage 2 discriminator
    """

    def __init__(self, n_d=128, m_d=4, emb_dim=768, img_dim=256):
        super(Stage2Discriminator, self).__init__()
        self.n_d = n_d
        self.m_d = m_d
        self.emb_dim = emb_dim

        self.fc_for_text = nn.Linear(self.emb_dim, self.n_d)
        self.down_sample = nn.Sequential(
            # (batch, 3, 64, 64) -> (batch, img_dim//4, 128, 128)
            nn.Conv2d(
                3, img_dim // 4, kernel_size=4, stride=2, padding=1, bias=False
            ),  # (batch, 64, 128, 128)
            nn.LeakyReLU(0.2, inplace=True),
            # -> (batch, img_dim//2, 64, 64)
            _downsample(img_dim // 4, img_dim // 2),  # (batch, 128, 64, 64)
            # -> (batch, img_dim, 32, 32)
            _downsample(img_dim // 2, img_dim),  # (batch, 256, 32, 32)
            # -> (batch, img_dim*2, 16, 16)
            _downsample(img_dim, img_dim * 2),  # (batch, 512, 16, 16)
            # -> (batch, img_dim*4, 8, 8)
            _downsample(img_dim * 2, img_dim * 4),  # (batch, 1024, 8, 8)
            # -> (batch, img_dim*8, 4, 4)
            _downsample(img_dim * 4, img_dim * 8),  # (batch, 2096, 4, 4)
            # -> (batch, img_dim*4, 4, 4)
            conv3x3(img_dim * 8, img_dim * 4),  # (batch, 1024, 4, 4)
            nn.BatchNorm2d(img_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (batch, img_dim*2, 4, 4)
            conv3x3(img_dim * 4, img_dim * 2),  # (batch, 512, 4, 4)
            nn.BatchNorm2d(img_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out_logits = nn.Sequential(
            # (batch, img_dim*2 + n_d, 4, 4) -> (batch, img_dim*2, 4, 4)
            conv3x3(img_dim * 2 + self.n_d, img_dim * 2),
            nn.BatchNorm2d(img_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (batch, 1)
            nn.Conv2d(img_dim * 2, 1, kernel_size=4, stride=4),
            nn.Sigmoid(),
        )

    def forward(self, text_emb, img):
        # image encode
        enc = self.down_sample(img)

        # text emb
        compressed = self.fc_for_text(text_emb)
        compressed = (
            compressed.unsqueeze(2).unsqueeze(
                3).repeat(1, 1, self.m_d, self.m_d)
        )

        con = torch.cat((enc, compressed), dim=1)

        output = self.out_logits(con)
        return output.view(-1)


#########################         #########################


if __name__ == "__main__":
    batch_size = 2
    n_z = 100
    emb_dim = 768  # 768
    emb = torch.randn((batch_size, emb_dim))
    noise = torch.empty((batch_size, n_z)).normal_()

    generator1 = Stage1Generator(emb_dim=emb_dim)
    generator2 = Stage2Generator(generator1, emb_dim=emb_dim)

    discriminator1 = Stage1Discriminator(emb_dim=emb_dim)
    discriminator2 = Stage2Discriminator(emb_dim=emb_dim)

    _, gen1, _, _ = generator1(emb, noise)
    print("output1 image dimensions :", gen1.size())  # (batch_size, 3, 64, 64)
    assert gen1.shape == (batch_size, 3, 64, 64)
    print()

    disc1 = discriminator1(emb, gen1)
    print("output1 discriminator", disc1.size())  # (batch_size)
    # assert disc1.shape == (batch_size)
    print()

    _, gen2, _, _ = generator2(emb, noise)
    # (batch_size, 3, 256, 256)
    print("output2 image dimensions :", gen2.size())
    assert gen2.shape == (batch_size, 3, 256, 256)
    print()

    disc2 = discriminator2(emb, gen2)
    print("output2 discriminator", disc2.size())  # (batch_size)
    # assert disc2.shape == (batch_size)
    print()

    ca = CAug(emb_dim=emb_dim, n_g=128, device="cpu")
    out_ca, _, _ = ca(emb)
    print("Conditional Aug output size: ", out_ca.size())  # (batch_size, 128)
    assert out_ca.shape == (batch_size, 128)

    # * Checking init weights
    # import engine
    # netG = Stage1Generator()
    # netG.apply(engine.weights_init)
    pass
