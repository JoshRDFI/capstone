# Model for the Anime Recommender system
# Import statements
import pandas as pd
import time
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn

# Load dataset
print("Loading dataframes")
anime_df = pd.read_csv('clean_anime.csv')
scores_df = pd.read_csv('clean_scores.csv')

# Misc variables
num_users = scores_df['user_id'].nunique()
num_animes = anime_df['anime_id'].nunique()

class AnimeRatingDataset(Dataset):
    def __init__(self, user_tensor, anime_tensor, rating_tensor):
        self.user_tensor = user_tensor
        self.anime_tensor = anime_tensor
        self.rating_tensor = rating_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.anime_tensor[index], self.rating_tensor[index]

    def __len__(self):
        return len(self.user_tensor)

# Convert the data into tensors
print("Converting data to tensors then creating dataset and splitting into training and validation sets")
user_tensor = torch.tensor(scores_df['user_id'].values, dtype=torch.long)
anime_tensor = torch.tensor(scores_df['anime_id'].values, dtype=torch.long)
rating_tensor = torch.tensor(scores_df['rating'].values, dtype=torch.float32)

# Create the dataset
dataset = AnimeRatingDataset(user_tensor, anime_tensor, rating_tensor)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class NCF(nn.Module):
    def __init__(self, num_users, num_animes, embedding_dim=50):
        super(NCF, self).__init__()
        
        # User and Anime Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.anime_embedding = nn.Embedding(num_animes, embedding_dim)
        
        # Neural network architecture
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, user, anime):
        user_emb = self.user_embedding(user)
        anime_emb = self.anime_embedding(anime)
        
        # Concatenate user and anime embeddings
        x = torch.cat((user_emb, anime_emb), 1)
        x = self.fc_layers(x)
        return x
    
# Create the model
model = NCF(num_users, num_animes)
criterion = nn.MSELoss()

if __name__ == '__main__':
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create device on GPU, if not available then on CPU (will significantly increase runtime).
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    model.to(device)

    # Number of training epochs
    num_epochs = 5

    # Move the criterion to the same device as the model
    criterion.to(device)

    # Training loop
    training_start = time.time()
    print("Training loop start time (local 24h): ", time.strftime('%H:%M', time.localtime(training_start)))
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        # Get loop start time
        start = time.time()
        # Display the start time in hours:minutes format
        print("Start time (local 24h): ", time.strftime('%H:%M', time.localtime(start)))
        model.train()  # Set the model to training mode
        train_loss = 0.0
    
        for user, anime, rating in train_loader:
            # Move data to the device
            user, anime, rating = user.to(device), anime.to(device), rating.to(device)
        
            # Zero out any previous gradients
            optimizer.zero_grad()
        
            # Forward pass
            predictions = model(user, anime)
            loss = criterion(predictions.squeeze(), rating)
        
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()
     
        # Print average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}")
        # Calculate the elapsed time in hours:minutes format
        end = time.time() # Get loop end time
        elapsed_seconds = end - start
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        print(f"Epoch took {int(hours)} hours and {int(minutes):02} minutes to run.")

    training_end = time.time()
    total_elapsed = training_end - training_start
    total_hours, total_remainder = divmod(total_elapsed, 3600)
    total_minutes, _ = divmod(total_remainder, 60)
    print(f"Total training runtime was {int(total_hours)} hours and {int(total_minutes):02} minutes")
    
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for user, anime, rating in val_loader:
            user, anime, rating = user.to(device), anime.to(device), rating.to(device)
            predictions = model(user, anime)
            loss = criterion(predictions.squeeze(), rating)
            val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), 'recommender_model.pth')
    print("Model saved")
