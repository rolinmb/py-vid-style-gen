import logging
import shutil
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

IN = 'vid-in/'
FRAMES = 'frames/'
RESULTFRAMEDIR = FRAMES + 'result/'
OUT = 'vid-out/'
OUTNAME = OUT + 'output.mp4'
MODELS = 'models/'
MODELNAME = MODELS + 'pvsg.pth'
SAVEMODEL = True

TARGETRES = (64, 64)
NFRAMES = 150
FPS = 30

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers = [logging.StreamHandler()]
)

# Extract frames from a specific video for training the generator
def extract_frames(vidin, outdirname):
    if not os.path.exists(vidin):
        logging.error(f'src/main.py : extract_frames() :: ERROR ::: The file {vidin} does not exist.')
        return
    
    if not vidin.endswith('.mp4'):
        logging.error(f'src/main.py : extract_frames() :: ERROR ::: The file {vidin} does not have a .mp4 extension.')
        return

    if os.path.exists(outdirname):
        shutil.rmtree(outdirname)

    class_folder = os.path.join(outdirname, 'class0')
    os.makedirs(class_folder, exist_ok=True)

    cap = cv2.VideoCapture(vidin)
    framenum = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        framefile = os.path.join(class_folder, f'{framenum:04d}.png')
        cv2.imwrite(framefile, frame)
        framenum += 1

    cap.release()
    logging.info(f'src/main.py : extract_frames() :: Extracted {framenum} frames from {vidin}.')

class Generator(nn.Module):
    def __init__(self, noise_dim=100, output_channels=3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_channels*TARGETRES[0]*TARGETRES[1]),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 3, TARGETRES[0], TARGETRES[1])
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_channels*TARGETRES[0]*TARGETRES[1], 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 3*TARGETRES[0]*TARGETRES[1])
        return self.model(x)

def train(generator, discriminator, dataloader, epochs, device):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(epochs):
        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Train Discriminator
            optimizer_d.zero_grad()
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)

            noise = torch.randn(batch_size, 100, device=device)
            fake_images = generator(noise)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()

        logging.info(f'src/main.py : train() :: Train Epoch {epoch+1}/{epochs} -> D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

# Generate new frames using the post-training generator
def generate_frames(generator, num_frames, output_folder, device):
    os.makedirs(output_folder, exist_ok=True)
    generator.eval()

    for i in range(num_frames):
        noise = torch.randn(1, 100, device=device)

        with torch.no_grad():
            generated_frame = generator(noise).cpu()
            generated_frame = (generated_frame * 0.5 + 0.5).clamp(0, 1)
            generated_frame = generated_frame.mul(255).byte()
            frame_path = os.path.join(output_folder, f'frame_{i:04d}.png')
            save_image(generated_frame, frame_path)

    logging.info(f'src/main.py : generate_frames() :: Generated frames saved to:', output_folder)

# Convert generated frames into a video file (.mp4 specifically)
def frames_to_video(output_folder, video_path, frame_rate=FPS):
    frame_files = sorted([os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.png')])

    if not frame_files:
        logging.info('src/main.py : frames_to_video() :: No frames found to convert in ', output_folder)
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # maybe put *'mp4v' back in?
    video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, TARGETRES)

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)

        if frame.shale[1::-1] != TARGETRES:
            frame = cv2.resize(frame, TARGETRES)

        video_writer.write(frame)

    video_writer.release()
    logging.info(f'src/main.py : frames_to_video() :: Video saved as {video_path}')

if __name__ == '__main__':
    os.makedirs(IN, exist_ok=True)
    os.makedirs(FRAMES, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)

    # Iterate through videos in the IN directory
    for vid in os.listdir(IN):
        if vid.endswith('.mp4'):
            video_path = os.path.join(IN, vid)
            subfolder_name = os.path.splitext(vid)[0]
            framedirname = os.path.join(FRAMES, subfolder_name)
            extract_frames(video_path, framedirname)

    tsfm = transforms.Compose([
        transforms.Resize(TARGETRES),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Load and preprocess frames output folders from input videos for training
    dataset = ImageFolder(root=FRAMES, transform=tsfm)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Train GAN, generate new frames as specified, then create .mp4 video
    train(generator, discriminator, dataloader, epochs=100, device=device)

    if SAVEMODEL:
        torch.save(generator.state_dict(), MODELNAME)

    generate_frames(generator, NFRAMES, RESULTFRAMEDIR, device=device)
    frames_to_video(RESULTFRAMEDIR, OUTNAME)
