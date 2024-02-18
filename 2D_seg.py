import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
# Check for GPU availability and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configuration
cf = {
    "image_size": 1000,
    "num_layers": 12,
    "hidden_dim": 768,
    "mlp_dim": 3072,
    "num_heads": 12,
    "dropout_rate": 0.1,
    "patch_size": 40,
    "num_channels": 1,  # Grayscale
    "batch_size": 8,
    "learning_rate": 0.001
}
cf["num_patches"] = (cf["image_size"] // cf["patch_size"]) * (cf["image_size"] // cf["patch_size"])

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            # Apply any necessary transformations
            image = self.transform(image)
            mask = self.transform(mask)

        return {'image': image, 'mask': mask}

# Specify the path to the folder containing images and masks
image_folder = './train/images'
mask_folder = './train/mask'
# Get all file paths for images and masks
all_image_paths = [os.path.join(image_folder, image) for image in os.listdir(image_folder) if image.endswith('.jpg')]
all_mask_paths = [os.path.join(mask_folder, mask) for mask in os.listdir(mask_folder) if mask.endswith('.jpg')]


# Split the dataset into training and validation sets (80% training, 20% validation)
train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(
    all_image_paths, all_mask_paths, test_size=0.2, random_state=42
)

# Define transformations (adjust as needed)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create dataset instances
train_dataset = CustomDataset(train_image_paths, train_mask_paths, transform=transform)
val_dataset = CustomDataset(val_image_paths, val_mask_paths, transform=transform)

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=cf["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=cf["batch_size"], shuffle=False)

# Custom Dice Loss with Adaptive Smoothing Factor
class CustomDiceLoss(nn.Module):
    def __init__(self):
        super(CustomDiceLoss, self).__init__()
        self.smooth = nn.Parameter(torch.tensor(1e-5))

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_pred_f = y_pred.view(-1)
        y_true_f = y_true.view(-1)
        intersection = (y_pred_f * y_true_f).sum()
        return 1 - ((2. * intersection + self.smooth) / (y_pred_f.sum() + y_true_f.sum() + self.smooth))


# UNETR Model Definition
class UNETR_2D(nn.Module):
    def __init__(self, cf):
        super().__init__()
        self.cf = cf

        # Patch + Position Embeddings
        self.patch_embed = nn.Linear(
            cf["patch_size"] * cf["patch_size"] * cf["num_channels"],
            cf["hidden_dim"]
        )
        self.positions = torch.arange(start=0, end=cf["num_patches"], step=1, dtype=torch.int32)
        self.pos_embed = nn.Embedding(cf["num_patches"], cf["hidden_dim"])

        # Transformer Encoder
        self.trans_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cf["hidden_dim"],
                nhead=cf["num_heads"],
                dim_feedforward=cf["mlp_dim"],
                dropout=cf["dropout_rate"],
                activation='gelu'
            ) for _ in range(cf["num_layers"])
        ])

        # CNN Decoder
        # Decoder 1
        self.d1 = self._deconv_block(cf["hidden_dim"], 512)
        self.s1 = nn.Sequential(
            self._deconv_block(cf["hidden_dim"], 512),
            self._conv_block(512, 512)
        )
        self.c1 = nn.Sequential(
            self._conv_block(512 + 512, 512),
            self._conv_block(512, 512)
        )
        # Decoder 2
        self.d2 = self._deconv_block(512, 256)
        self.s2 = nn.Sequential(
            self._deconv_block(cf["hidden_dim"], 256),
            self._conv_block(256, 256),
            self._deconv_block(256, 256),
            self._conv_block(256, 256)
        )
        self.c2 = nn.Sequential(
            self._conv_block(256 + 256, 256),
            self._conv_block(256, 256)
        )

        # Decoder 3
        self.d3 = self._deconv_block(256, 128)
        self.s3 = nn.Sequential(
            self._deconv_block(cf["hidden_dim"], 128),
            self._conv_block(128, 128),
            self._deconv_block(128, 128),
            self._conv_block(128, 128),
            self._deconv_block(128, 128),
            self._conv_block(128, 128)
        )
        self.c3 = nn.Sequential(
            self._conv_block(128 + 128, 128),
            self._conv_block(128, 128)
        )

        # Decoder 4
        self.d4 = self._deconv_block(128, 64)
        self.s4 = nn.Sequential(
            self._conv_block(1, 64),
            self._conv_block(64, 64)
        )
        self.c4 = nn.Sequential(
            self._conv_block(64 + 64, 64),
            self._conv_block(64, 64)
        )

        # Output
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def _deconv_block(self, in_c, out_c):
        return nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)

    def forward(self, inputs):
        """ Patch + Position Embeddings """
        batch_size, C, H, W = inputs.shape
        patch_H, patch_W = self.cf["patch_size"], self.cf["patch_size"]

        # Calculate required padding
        pad_H = (patch_H - H % patch_H) % patch_H
        pad_W = (patch_W - W % patch_W) % patch_W

        # Zero-pad the input
        inputs = F.pad(inputs, (0, pad_W, 0, pad_H))
        new_H = H + pad_H
        new_W = W + pad_W
        # Reshape for patches
        inputs = inputs.unfold(2, patch_H, patch_H).unfold(3, patch_W, patch_W)
        inputs = inputs.contiguous().view(batch_size, C, -1, patch_H * patch_W)
        inputs = inputs.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, patch_H * patch_W * C)

        actual_num_patches = inputs.shape[1]  # Update based on the actual shape after unfolding

        # Update position embeddings
        self.positions = torch.arange(start=0, end=actual_num_patches, step=1, dtype=torch.int32).to(inputs.device)
        self.pos_embed = nn.Embedding(actual_num_patches, self.cf["hidden_dim"]).to(inputs.device)

        patch_embed = self.patch_embed(inputs)

        positions = self.positions
        pos_embed = self.pos_embed(positions)  # Should now be [actual_num_patches, hidden_dim]

        x = patch_embed + pos_embed  # Should now match in size

        """ Transformer Encoder """
        skip_connection_index = [3, 6, 9, 12]
        skip_connections = []

        for i in range(self.cf["num_layers"]):
            layer = self.trans_encoder_layers[i]
            x = layer(x)

            if (i + 1) in skip_connection_index:
                skip_connections.append(x)

        """ CNN Decoder """
        z3, z6, z9, z12 = skip_connections

        # Reshaping
        batch = inputs.shape[0]
        z0 = inputs.view((batch, self.cf["num_channels"], new_H, new_W))

        # Calculate the new shape based on the actual number of patches
        new_shape = (batch, self.cf["hidden_dim"], int(np.sqrt(actual_num_patches)), int(np.sqrt(actual_num_patches)))

        # Reshape the tensor
        z3 = z3.view(new_shape)
        z6 = z6.view(new_shape)
        z9 = z9.view(new_shape)
        z12 = z12.view(new_shape)

        ## Decoder 1
        x = self.d1(z12)
        s = self.s1(z9)
        x = torch.cat([x, s], dim=1)
        x = self.c1(x)

        ## Decoder 2
        x = self.d2(x)
        s = self.s2(z6)
        x = torch.cat([x, s], dim=1)
        x = self.c2(x)

        ## Decoder 3
        x = self.d3(x)
        s = self.s3(z3)
        x = torch.cat([x, s], dim=1)
        x = self.c3(x)

        ## Decoder 4
        x = self.d4(x)
        s = self.s4(z0)
        x_upsampled = F.interpolate(x, size=(1000, 1000), mode='bilinear', align_corners=True)  # Upsample the tensor
        x = torch.cat([x_upsampled, s], dim=1)
        x = self.c4(x)

        """ Output """
        output = self.output(x)

        return output


# Initialize the model
model = UNETR_2D(cf).to(device)

# Example optimizer and loss function (adjust as needed)
optimizer = torch.optim.Adam(model.parameters(), lr=cf["learning_rate"])
criterion = CustomDiceLoss()

def predict(model, input_image):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.Tensor(input_image).unsqueeze(0).unsqueeze(0).to(device)
        output = model(input_tensor)
        predicted_mask = torch.sigmoid(output).cpu().numpy()
    return predicted_mask

# Optionally, implement evaluation function
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        # Iterate through the dataloader and calculate metrics
        # You can use the same metrics as in the training loop
        for batch in dataloader:
            inputs, targets = batch['image'].to(device), batch['mask'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

        average_loss = total_loss / num_batches
        return average_loss
            # Calculate metrics or save results as needed

# Training loop
num_epochs = 5
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        inputs, targets = batch['image'].to(device), batch['mask'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = evaluate(model, val_loader, criterion)
    print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

torch.save(model, './model.pth')
# Load the model (assuming the same model class is defined)
loaded_model = UNETR_2D(cf=cf)  # Make sure to create an instance of the model first
loaded_model = torch.load('./model.pth')
loaded_model.eval()  # Set the model to evaluation mode after loading

def predict_folder(model, folder_path, output_folder, threshold=0.5):
    model.eval()

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all images in the folder
    for image_name in os.listdir(folder_path):
        if image_name.endswith('.jpg'):  # Assuming images have a '.jpg' extension
            image_path = os.path.join(folder_path, image_name)
            input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Predict the mask using the model
            predicted_mask = predict(model, input_image)

            #apply threshold
            predicted_mask[predicted_mask > threshold] = 1
            predicted_mask[predicted_mask <= threshold] =0

            # Save or process the predicted mask as needed
            output_path = os.path.join(output_folder, f"predicted_{image_name}")
            cv2.imwrite(output_path, (predicted_mask[0, 0] * 255).astype(np.uint8))


# Example usage
input_folder = './test/images'
output_folder = './test/predictions'
predict_folder(model, input_folder, output_folder)

