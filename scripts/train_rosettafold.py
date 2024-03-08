import torch
from rosettafold import RoseTTAFold, utils
from sidechainnet import load_data
from esm.inverse_folding import get_encoder_output, get_attention_map

def train_rosettafold():
    # Load and preprocess the training data
    data_dir = "data/sidechainnet"
    train_data = load_data(data_dir, "train", with_pytorch="dataloaders")
    val_data = load_data(data_dir, "valid", with_pytorch="dataloaders")

    # Initialize and train the RoseTTAFold model
    model = RoseTTAFold.from_pretrained("roberta-base")
    model.fit(train_data, epochs=10, validation_data=val_data)
    model.save_pretrained("models/rosettafold")

    # Get encoder output and attention maps for interpretability analysis
    encoder_output = get_encoder_output(model, train_data)
    attention_map = get_attention_map(model, train_data)

    # Save encoder output and attention maps
    torch.save(encoder_output, "results/encoder_output.pt")
    torch.save(attention_map, "results/attention_map.pt")