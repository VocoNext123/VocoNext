import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset import Dataset, mel_spectrogram, get_dataset_filelist
from models import Generator, MultiPeriodDiscriminator, feature_loss, generator_loss, discriminator_loss
from utils import AttrDict, build_env, scan_checkpoint, load_checkpoint, save_checkpoint
from models import Generator, MultiPeriodDiscriminator, MultiResolutionDiscriminator  # Ensure this is correct


torch.backends.cudnn.benchmark = True


def train(h):
    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(0))

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)


    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(mrd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay)

    training_filelist, validation_filelist = get_dataset_filelist(h.input_training_wav_list, h.input_validation_wav_list)

    trainset = Dataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                       h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, h.meloss, n_cache_reuse=0,
                       shuffle=True, device=device)
    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=None,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    sw = SummaryWriter(os.path.join(h.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    mrd.train()

    steps = 0
    start = time.time()
    for epoch in range(h.training_epochs):
        for i, batch in enumerate(train_loader):
            x, logamp, pha, rea, imag, y, meloss = batch
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            meloss = meloss.to(device)

            logamp_g, pha_g, rea_g, imag_g, y_g = generator(x)

            # Compute mel spectrogram of generated audio
            y_g_mel = mel_spectrogram(
                y_g.squeeze(1).detach(),
                h.n_fft,
                h.num_mels,
                h.sampling_rate,
                h.hop_size,
                h.win_size,
                h.fmin,
                h.meloss
            )

            # Debugging types and content
            print("meloss type:", type(meloss))
            print("y_g_mel type:", type(y_g_mel))
            print("meloss shape:", meloss.shape)
            print("y_g_mel shape:", y_g_mel.shape)

            # Convert to tensors if necessary
            if not isinstance(meloss, torch.Tensor):
                meloss = torch.tensor(meloss).to(device)
            if not isinstance(y_g_mel, torch.Tensor):
                y_g_mel = torch.tensor(y_g_mel).to(device)

            # Ensure shapes match
            if meloss.shape != y_g_mel.shape:
                print(f"Shape mismatch: meloss {meloss.shape}, y_g_mel {y_g_mel.shape}")
                meloss = meloss.view_as(y_g_mel)

            # Compute L1 loss
            L_Mel = F.l1_loss(meloss, y_g_mel)
            print(f"L1 Loss: {L_Mel.item()}")

            # Continue training loop...

            L_GAN_G = generator_loss(mpd(y, y_g)[1])[0] + 0.1 * generator_loss(mrd(y, y_g)[1])[0]
            L_G = 45 * L_Mel + L_GAN_G

            optim_g.zero_grad()
            L_G.backward()
            optim_g.step()

            # Discriminator loss
            optim_d.zero_grad()
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g.detach())
            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
            y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_g.detach())
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            L_D = loss_disc_s * 0.1 + loss_disc_f
            L_D.backward()
            optim_d.step()

            # Logging
            if steps % h.stdout_interval == 0:
                print(f"Steps: {steps}, Gen Loss: {L_G.item()}, Mel Loss: {L_Mel.item()}")

            # Tensorboard summary
            if steps % h.summary_interval == 0:
                sw.add_scalar("Generator_Loss", L_G, steps)
                sw.add_scalar("Mel_Spectrogram_Loss", L_Mel, steps)

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        print(f"Epoch {epoch + 1} completed. Time taken: {int(time.time() - start)} seconds")
        start = time.time()


def main():
    print("Initializing Training Process...")

    config_file = 'config.json'

    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(config_file, 'config.json', h.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)

    train(h)


if __name__ == '__main__':
    main()
